# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

################################################################################
## 1. DATA SIMULATION & PREPROCESSING
################################################################################


# Simulate data since we don't have the real datasets
# VGRF: Vertical Ground Reaction Force data
# IMU: Inertial Measurement Unit data (e.g., accelerometer, gyroscope)
def generate_dummy_data(num_samples, seq_len, vgrf_features, imu_features, num_classes):
    """Generates random time-series data and labels."""
    # VGRF data for 4 classes (HC, PD, HD, ALS)
    vgrf_data = torch.randn(num_samples, seq_len, vgrf_features)
    vgrf_labels = torch.randint(0, num_classes - 1, (num_samples,))  # Exclude 'Stroke'

    # IMU data for 4 classes (HC, PD, Stroke, plus some overlap)
    imu_data = torch.randn(num_samples, seq_len, imu_features)
    # Make labels for IMU: HC=0, PD=1, HD=2, Stroke=4 (to match diagram)
    # We'll map HD -> Stroke for simplicity in IMU data generation
    imu_labels_raw = torch.randint(0, num_classes - 1, (num_samples,))
    # Let's say label '2' (HD) becomes '4' (Stroke) in the IMU dataset
    imu_labels = torch.where(imu_labels_raw == 2, 4, imu_labels_raw)

    print(f"Generated {num_samples} samples.")
    print(
        f"VGRF data shape: {vgrf_data.shape}, Labels: {np.unique(vgrf_labels.numpy())}"
    )
    print(f"IMU data shape: {imu_data.shape}, Labels: {np.unique(imu_labels.numpy())}")

    return vgrf_data, vgrf_labels, imu_data, imu_labels


class MultimodalPairDataset(Dataset):
    """
    Creates pairs of (VGRF, IMU) samples for contrastive learning.
    A pair is "positive" if both samples belong to the same class (e.g., HC).
    """

    def __init__(self, vgrf_data, vgrf_labels, imu_data, imu_labels):
        self.vgrf_data = vgrf_data
        self.vgrf_labels = vgrf_labels
        self.imu_data = imu_data
        self.imu_labels = imu_labels

        # Group indices by class for easy pairing
        self.vgrf_class_indices = {
            i: np.where(vgrf_labels == i)[0] for i in np.unique(vgrf_labels)
        }
        self.imu_class_indices = {
            i: np.where(imu_labels == i)[0] for i in np.unique(imu_labels)
        }

        # Find common classes between the two modalities
        self.common_classes = sorted(
            list(
                set(self.vgrf_class_indices.keys()) & set(self.imu_class_indices.keys())
            )
        )

        # We will create pairs on the fly in __getitem__
        # The length is based on the smaller dataset to ensure pairs can be made
        self.length = min(len(vgrf_data), len(imu_data))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Select a random common class to ensure a positive pair can be formed
        label = np.random.choice(self.common_classes)

        # Pick a random VGRF sample from that class
        vgrf_idx = np.random.choice(self.vgrf_class_indices[label])
        vgrf_sample = self.vgrf_data[vgrf_idx]

        # Pick a random IMU sample from the same class (positive pair)
        imu_idx = np.random.choice(self.imu_class_indices[label])
        imu_sample = self.imu_data[imu_idx]

        # The data pre-processing steps (filtering, normalization, etc.) from the
        # diagram would happen here or before creating the dataset.
        # For this dummy data, we'll just return as is.
        # The data should be in shape (seq_len, features)
        return vgrf_sample, imu_sample, torch.tensor(label, dtype=torch.long)


class ClassifierDataset(Dataset):
    """Dataset for the classifier, using pre-computed embeddings."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


################################################################################
## 2. MODEL DEFINITION
################################################################################


###
### Part A: Representation Encoder for Contrastive Learning
###
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


###
### Part B: The iTransformer-based Classifier
###
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
        mean_value = torch.mean(corr, dim=-1, keepdim=True)
        # Use top_k if it's a valid index, otherwise use mean as a fallback
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
        output = output.squeeze(1).permute(0, 2, 1)  # -> (Batch, SeqLen, Features)
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
        # Our embedding is a flat vector, so we'll treat it as a sequence of length 1.
        # This simplifies the model greatly. A more complex approach might reshape
        # the embedding, but we'll follow a simple interpretation.
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
        x = self.embedding_projection(x).unsqueeze(
            1
        )  # -> (batch_size, 1, embedding_dim)

        # Pass through transformer blocks
        x = self.transformer_blocks(x)

        # Remove the sequence length dimension
        x = x.squeeze(1)  # -> (batch_size, embedding_dim)

        # Final classification
        output = self.classification_head(x)
        return output


################################################################################
## 3. LOSS FUNCTION
################################################################################


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
        # The logits are the similarity scores between each VGRF sample and all IMU samples.
        logits = (vgrf_embeddings @ imu_embeddings.T) / self.temperature

        # The positive pairs are on the diagonal. The labels for CrossEntropyLoss
        # are therefore just the indices of the diagonal.
        batch_size = vgrf_embeddings.shape[0]
        labels = torch.arange(batch_size, device=vgrf_embeddings.device)

        # Calculate loss in both directions (VGRF->IMU and IMU->VGRF)
        loss_vgrf_imu = self.criterion(logits, labels)
        loss_imu_vgrf = self.criterion(logits.T, labels)

        return (loss_vgrf_imu + loss_imu_vgrf) / 2


################################################################################
## 4. MAIN TRAINING SCRIPT
################################################################################

if __name__ == "__main__":
    # --- Hyperparameters ---
    NUM_SAMPLES = 1000
    SEQ_LEN = 128
    VGRF_FEATURES = 2
    IMU_FEATURES = 12
    NUM_CLASSES = 5  # HC, PD, HD, ALS, Stroke (0, 1, 2, 3, 4)
    EMBEDDING_DIM = 64
    BATCH_SIZE = 64
    CONTRASTIVE_EPOCHS = 10
    CLASSIFIER_EPOCHS = 100
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    vgrf_data, vgrf_labels, imu_data, imu_labels = generate_dummy_data(
        NUM_SAMPLES, SEQ_LEN, VGRF_FEATURES, IMU_FEATURES, NUM_CLASSES
    )
    pair_dataset = MultimodalPairDataset(vgrf_data, vgrf_labels, imu_data, imu_labels)
    pair_loader = DataLoader(pair_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Stage 1: Contrastive Learning ---
    print("\n--- STAGE 1: CONTRASTIVE PRE-TRAINING OF ENCODER ---")

    # Initialize shared encoder for both modalities
    # Note: We need two encoders initially to handle different input feature sizes
    # If feature sizes were the same, one encoder instance would suffice.
    encoder_vgrf = RepresentationEncoder(VGRF_FEATURES, EMBEDDING_DIM).to(device)
    encoder_imu = RepresentationEncoder(IMU_FEATURES, EMBEDDING_DIM).to(device)

    # Combine parameters for the optimizer
    all_encoder_params = list(encoder_vgrf.parameters()) + list(
        encoder_imu.parameters()
    )
    optimizer_contrastive = optim.Adam(all_encoder_params, lr=LR)
    contrastive_loss_fn = ContrastiveLoss()

    for epoch in range(CONTRASTIVE_EPOCHS):
        total_loss = 0
        for vgrf_batch, imu_batch, _ in pair_loader:
            vgrf_batch, imu_batch = vgrf_batch.to(device), imu_batch.to(device)

            optimizer_contrastive.zero_grad()

            # Get embeddings for the batch
            vgrf_embed = encoder_vgrf(vgrf_batch)
            imu_embed = encoder_imu(imu_batch)

            # Calculate loss
            loss = contrastive_loss_fn(vgrf_embed, imu_embed)
            loss.backward()
            optimizer_contrastive.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(pair_loader)
        print(f"Epoch {epoch+1}/{CONTRASTIVE_EPOCHS}, Contrastive Loss: {avg_loss:.4f}")

    print("Contrastive pre-training finished.")

    # --- Stage 2: Classifier Training ---
    print("\n--- STAGE 2: TRAINING THE ND CLASSIFIER ---")

    # Freeze the encoders
    encoder_vgrf.eval()
    encoder_imu.eval()
    for param in encoder_vgrf.parameters():
        param.requires_grad = False
    for param in encoder_imu.parameters():
        param.requires_grad = False

    # Create embeddings for the full dataset using the trained encoders
    with torch.no_grad():
        all_vgrf_embeds = encoder_vgrf(vgrf_data.to(device)).cpu()
        all_imu_embeds = encoder_imu(imu_data.to(device)).cpu()

    # Combine embeddings and labels for classifier training
    all_embeddings = torch.cat([all_vgrf_embeds, all_imu_embeds], dim=0)
    all_labels = torch.cat([vgrf_labels, imu_labels], dim=0)

    classifier_dataset = ClassifierDataset(all_embeddings, all_labels)
    classifier_loader = DataLoader(
        classifier_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Initialize the classifier
    classifier = NDClassifier(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES).to(
        device
    )
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=LR)
    classification_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(CLASSIFIER_EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for embed_batch, label_batch in classifier_loader:
            embed_batch, label_batch = embed_batch.to(device), label_batch.to(device)

            optimizer_classifier.zero_grad()

            # Get predictions
            outputs = classifier(embed_batch)

            # Calculate loss and accuracy
            loss = classification_loss_fn(outputs, label_batch)
            loss.backward()
            optimizer_classifier.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += label_batch.size(0)
            total_correct += (predicted == label_batch).sum().item()

        avg_loss = total_loss / len(classifier_loader)
        accuracy = 100 * total_correct / total_samples
        print(
            f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}, Classifier Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    print("Classifier training finished.")
