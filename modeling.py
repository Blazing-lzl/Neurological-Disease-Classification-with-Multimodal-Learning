import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationEncoder(nn.Module):
    """
    Encodes a time-series segment into a fixed-size embedding vector.
    This network is shared between the VGRF and IMU modalities.
    A simple 1D CNN is used here as an example.
    """

    def __init__(self, input_features, embedding_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, features)
        # Conv1d expects (batch_size, features, seq_len)
        x = x.permute(0, 2, 1)
        embedding = self.network(x)
        return F.normalize(embedding, p=2, dim=1)  # L2 normalize embeddings


class AutoCorrelation(nn.Module):
    """The Auto-Correlation mechanism from the iTransformer paper."""

    def __init__(self, factor=1, dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def time_delay_agg(self, V, corr):
        """
        Calculates aggregation based on time-delayed series.
        V shape: [Batch, Head, Features, SeqLen]
        corr shape: [Batch, Head, Features, SeqLen]
        """
        batch, head, num_features, seq_len = V.shape

        # Find top k delays (indices with highest correlation)
        top_k = int(self.factor * np.log(seq_len))
        top_k = max(1, top_k)  # Ensure top_k is at least 1
        # Use top_k to get the most correlated delays
        weights, delay = torch.topk(corr, top_k, dim=-1)

        # Aggregate values based on these delays
        V_repeat = V.repeat(1, 1, 1, 2)  # Repeat V to handle circular shifts
        tmp_V = torch.zeros_like(V).to(V.device)

        for i in range(top_k):
            # Get the delays for this k
            delay_i = delay[..., i]
            # Shift the values in V according to the delay
            pattern = V_repeat[
                torch.arange(batch)[:, None, None],
                torch.arange(head)[None, :, None],
                torch.arange(num_features)[None, None, :],
                delay_i,
            ]
            tmp_V += pattern * (weights[..., i].unsqueeze(-1))

        return tmp_V

    def forward(self, Q, K, V):
        # Note: In iTransformer, Q, K, and V are the same tensor
        # Input shape: (Batch, SeqLen, Features)
        # We apply attention across features, so permute
        Q = Q.permute(0, 2, 1)  # -> (Batch, Features, SeqLen)
        K = K.permute(0, 2, 1)
        V = V.permute(0, 2, 1)

        batch, num_features, seq_len = Q.shape
        # Add a "head" dimension for compatibility
        Q = Q.unsqueeze(1)  # -> (Batch, Head=1, Features, SeqLen)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        # Autocorrelation via FFT
        Q_fft = torch.fft.rfft(Q, dim=-1)
        K_fft = torch.fft.rfft(K, dim=-1)
        corr = torch.fft.irfft(Q_fft * torch.conj(K_fft), n=seq_len, dim=-1)

        # Time delay aggregation
        output = self.time_delay_agg(V, corr)
        # -> (Batch, SeqLen, Features)
        output = output.squeeze(1).permute(0, 2, 1)
        return self.dropout(output)


class iTransformerBlock(nn.Module):
    """A single block of the iTransformer."""

    def __init__(self, input_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = AutoCorrelation(dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention -> Add & Norm
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward -> Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class NDClassifier(nn.Module):
    """The main classifier based on iTransformer."""

    def __init__(self, embedding_dim, num_classes, ff_dim=256, num_blocks=2):
        super().__init__()
        # The iTransformer expects a "sequence" of features.
        # Our embedding is a flat vector, so we'll treat it as a sequence
        # of length 1. This simplifies the model greatly. A more complex
        # approach might reshape the embedding, but we'll follow a simple
        # interpretation.
        self.embedding_projection = nn.Linear(embedding_dim, embedding_dim)

        # Simplified Transformer Encoder
        layers = [
            iTransformerBlock(input_dim=embedding_dim, ff_dim=ff_dim)
            for _ in range(num_blocks)
        ]
        self.transformer_blocks = nn.Sequential(*layers)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        # x is the embedding from the encoder, shape: (batch_size, embedding_dim)

        # Project and add a "sequence length" dimension of 1
        # -> (batch_size, 1, embedding_dim)
        x = self.embedding_projection(x).unsqueeze(1)

        # Pass through transformer blocks
        x = self.transformer_blocks(x)

        # Remove the sequence length dimension
        x = x.squeeze(1)  # -> (batch_size, embedding_dim)

        # Final classification
        output = self.classification_head(x)
        return output


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (Noise-contrastive estimation) loss function.
    Pulls positive pairs (vgrf_i, imu_i) together and pushes apart negative pairs.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, vgrf_embeddings, imu_embeddings):
        # vgrf_embeddings, imu_embeddings shape: [batch_size, embedding_dim]

        # Calculate cosine similarity
        # The logits are the similarity scores between each VGRF sample
        # and all IMU samples.
        logits = (vgrf_embeddings @ imu_embeddings.T) / self.temperature

        # The positive pairs are on the diagonal. The labels for
        # CrossEntropyLoss are therefore just the indices of the diagonal.
        batch_size = vgrf_embeddings.shape[0]
        labels = torch.arange(batch_size, device=vgrf_embeddings.device)

        # Calculate loss in both directions (VGRF->IMU and IMU->VGRF)
        loss_vgrf_imu = self.criterion(logits, labels)
        loss_imu_vgrf = self.criterion(logits.T, labels)

        return (loss_vgrf_imu + loss_imu_vgrf) / 2
