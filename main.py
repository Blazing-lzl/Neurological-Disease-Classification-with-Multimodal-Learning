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
# Three different datasets:
# 1. NDD Gait: 2-channel VGRF, 4 classes (HC, PD, HD, ALS)
# 2. PD Gait: 18-channel VGRF, 2 classes (HC, PD)
# 3. IMU Gait: 12-channel IMU, 3 classes (HC, PD, Stroke)

def generate_ndd_gait_data(num_samples, seq_len):
    """
    Generates NDD Gait dataset: 2-channel VGRF data with 4 classes.
    Classes: 0=HC, 1=PD, 2=HD, 3=ALS
    """
    data = torch.randn(num_samples, seq_len, 2)  # 2 channels
    labels = torch.randint(0, 4, (num_samples,))  # 4 classes
    
    print(f"Generated NDD Gait: {num_samples} samples, 2 channels, 4 classes")
    print(f"  Shape: {data.shape}, Labels: {np.unique(labels.numpy())}")
    
    return data, labels


def generate_pd_gait_data(num_samples, seq_len):
    """
    Generates PD Gait dataset: 18-channel VGRF data with 2 classes.
    Classes: 0=HC, 1=PD
    """
    data = torch.randn(num_samples, seq_len, 18)  # 18 channels
    labels = torch.randint(0, 2, (num_samples,))  # 2 classes
    
    print(f"Generated PD Gait: {num_samples} samples, 18 channels, 2 classes")
    print(f"  Shape: {data.shape}, Labels: {np.unique(labels.numpy())}")
    
    return data, labels


def generate_imu_gait_data(num_samples, seq_len):
    """
    Generates IMU Gait dataset: 12-channel IMU data with 3 classes.
    Classes: 0=HC, 1=PD, 2=Stroke
    """
    data = torch.randn(num_samples, seq_len, 12)  # 12 channels
    labels = torch.randint(0, 3, (num_samples,))  # 3 classes
    
    print(f"Generated IMU Gait: {num_samples} samples, 12 channels, 3 classes")
    print(f"  Shape: {data.shape}, Labels: {np.unique(labels.numpy())}")
    
    return data, labels


class MultimodalPairDataset(Dataset):
    """
    Creates pairs of samples from two modalities for contrastive learning.
    A pair is "positive" if both samples belong to the same class (e.g., HC).
    Handles label mapping between different modalities with different label spaces.
    """

    def __init__(self, data1, labels1, data2, labels2, label_mapping=None):
        """
        Args:
            data1: First modality data
            labels1: Labels for first modality
            data2: Second modality data  
            labels2: Labels for second modality
            label_mapping: Dict mapping labels1 values to labels2 values (for finding common classes)
                          If None, assumes same label space
        """
        self.data1 = data1
        self.labels1 = labels1
        self.data2 = data2
        self.labels2 = labels2
        self.label_mapping = label_mapping

        # Group indices by class for easy pairing
        self.data1_class_indices = {
            i: np.where(labels1 == i)[0] for i in np.unique(labels1)
        }
        self.data2_class_indices = {
            i: np.where(labels2 == i)[0] for i in np.unique(labels2)
        }

        # Find common classes between the two modalities
        if label_mapping is not None:
            # Map labels from data1 to data2 space and find intersection
            mapped_labels1 = set(label_mapping.get(k, -1) for k in self.data1_class_indices.keys())
            data2_labels = set(self.data2_class_indices.keys())
            common_in_data2_space = sorted(list(mapped_labels1 & data2_labels))
            
            # Store pairs of (data1_label, data2_label) for common classes
            self.common_pairs = [
                (k, v) for k, v in label_mapping.items() 
                if v in common_in_data2_space and k in self.data1_class_indices
            ]
        else:
            # No mapping, use direct label matching
            data1_keys = set(self.data1_class_indices.keys())
            data2_keys = set(self.data2_class_indices.keys())
            common = sorted(list(data1_keys & data2_keys))
            self.common_pairs = [(c, c) for c in common]

        if len(self.common_pairs) == 0:
            raise ValueError("No common classes found between the two modalities!")

        # The length is based on the smaller dataset to ensure pairs can be made
        self.length = min(len(data1), len(data2))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Select a random common class pair
        label1, label2 = self.common_pairs[np.random.randint(len(self.common_pairs))]

        # Pick a random sample from data1 with label1
        idx1 = np.random.choice(self.data1_class_indices[label1])
        sample1 = self.data1[idx1]

        # Pick a random sample from data2 with label2 (positive pair)
        idx2 = np.random.choice(self.data2_class_indices[label2])
        sample2 = self.data2[idx2]

        # Return the pair with the shared semantic label (using label1 as reference)
        return sample1, sample2, torch.tensor(label1, dtype=torch.long)


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
    NUM_SAMPLES = 30000
    SEQ_LEN = 128
    EMBEDDING_DIM = 64
    BATCH_SIZE = 256
    CONTRASTIVE_EPOCHS = 10
    CLASSIFIER_EPOCHS = 100
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("\n=== Generating Data ===")
    # NDD Gait: 2 channels, 4 classes (HC=0, PD=1, HD=2, ALS=3)
    ndd_data, ndd_labels = generate_ndd_gait_data(NUM_SAMPLES, SEQ_LEN)
    
    # PD Gait: 18 channels, 2 classes (HC=0, PD=1)
    pd_data, pd_labels = generate_pd_gait_data(NUM_SAMPLES, SEQ_LEN)
    
    # IMU Gait: 12 channels, 3 classes (HC=0, PD=1, Stroke=2)
    imu_data, imu_labels = generate_imu_gait_data(NUM_SAMPLES, SEQ_LEN)

    # --- Create Label Mappings for Contrastive Learning ---
    # Mapping: source_label -> target_label (for common classes)
    # NDD (4 classes) to PD (2 classes): HC=0->0, PD=1->1
    ndd_to_pd_mapping = {0: 0, 1: 1}  # Only HC and PD are common
    
    # NDD (4 classes) to IMU (3 classes): HC=0->0, PD=1->1
    ndd_to_imu_mapping = {0: 0, 1: 1}  # Only HC and PD are common
    
    # PD (2 classes) to IMU (3 classes): HC=0->0, PD=1->1
    pd_to_imu_mapping = {0: 0, 1: 1}  # HC and PD are common

    # --- Stage 1: Contrastive Learning for Different Pairs ---
    print("\n=== Stage 1: Contrastive Pre-training ===")
    
    # Initialize encoders for each modality
    encoder_ndd = RepresentationEncoder(2, EMBEDDING_DIM).to(device)    # 2 channels
    encoder_pd = RepresentationEncoder(18, EMBEDDING_DIM).to(device)    # 18 channels
    encoder_imu = RepresentationEncoder(12, EMBEDDING_DIM).to(device)   # 12 channels

    contrastive_loss_fn = ContrastiveLoss()

    # Pair 1: NDD VGRF + PD VGRF (both VGRF modality, different channels)
    print("\n--- Training NDD-PD Pair ---")
    pair_dataset_ndd_pd = MultimodalPairDataset(
        ndd_data, ndd_labels, pd_data, pd_labels, label_mapping=ndd_to_pd_mapping
    )
    pair_loader_ndd_pd = DataLoader(pair_dataset_ndd_pd, batch_size=BATCH_SIZE, shuffle=True)
    
    all_encoder_params = list(encoder_ndd.parameters()) + list(encoder_pd.parameters())
    optimizer_contrastive = optim.Adam(all_encoder_params, lr=LR)
    
    for epoch in range(CONTRASTIVE_EPOCHS):
        total_loss = 0
        for batch1, batch2, _ in pair_loader_ndd_pd:
            batch1, batch2 = batch1.to(device), batch2.to(device)
            optimizer_contrastive.zero_grad()
            embed1 = encoder_ndd(batch1)
            embed2 = encoder_pd(batch2)
            loss = contrastive_loss_fn(embed1, embed2)
            loss.backward()
            optimizer_contrastive.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(pair_loader_ndd_pd)
        print(f"Epoch {epoch+1}/{CONTRASTIVE_EPOCHS}, Contrastive Loss: {avg_loss:.4f}")

    # Pair 2: PD VGRF + IMU (cross-modal)
    print("\n--- Training PD-IMU Pair ---")
    pair_dataset_pd_imu = MultimodalPairDataset(
        pd_data, pd_labels, imu_data, imu_labels, label_mapping=pd_to_imu_mapping
    )
    pair_loader_pd_imu = DataLoader(pair_dataset_pd_imu, batch_size=BATCH_SIZE, shuffle=True)
    
    all_encoder_params = list(encoder_pd.parameters()) + list(encoder_imu.parameters())
    optimizer_contrastive = optim.Adam(all_encoder_params, lr=LR)
    
    for epoch in range(CONTRASTIVE_EPOCHS):
        total_loss = 0
        for batch1, batch2, _ in pair_loader_pd_imu:
            batch1, batch2 = batch1.to(device), batch2.to(device)
            optimizer_contrastive.zero_grad()
            embed1 = encoder_pd(batch1)
            embed2 = encoder_imu(batch2)
            loss = contrastive_loss_fn(embed1, embed2)
            loss.backward()
            optimizer_contrastive.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(pair_loader_pd_imu)
        print(f"Epoch {epoch+1}/{CONTRASTIVE_EPOCHS}, Contrastive Loss: {avg_loss:.4f}")

    # Pair 3: NDD VGRF + IMU (cross-modal)
    print("\n--- Training NDD-IMU Pair ---")
    pair_dataset_ndd_imu = MultimodalPairDataset(
        ndd_data, ndd_labels, imu_data, imu_labels, label_mapping=ndd_to_imu_mapping
    )
    pair_loader_ndd_imu = DataLoader(pair_dataset_ndd_imu, batch_size=BATCH_SIZE, shuffle=True)
    
    all_encoder_params = list(encoder_ndd.parameters()) + list(encoder_imu.parameters())
    optimizer_contrastive = optim.Adam(all_encoder_params, lr=LR)
    
    for epoch in range(CONTRASTIVE_EPOCHS):
        total_loss = 0
        for batch1, batch2, _ in pair_loader_ndd_imu:
            batch1, batch2 = batch1.to(device), batch2.to(device)
            optimizer_contrastive.zero_grad()
            embed1 = encoder_ndd(batch1)
            embed2 = encoder_imu(batch2)
            loss = contrastive_loss_fn(embed1, embed2)
            loss.backward()
            optimizer_contrastive.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(pair_loader_ndd_imu)
        print(f"Epoch {epoch+1}/{CONTRASTIVE_EPOCHS}, Contrastive Loss: {avg_loss:.4f}")

    print("Contrastive pre-training finished.")

    # --- Stage 2: Train Separate Classifiers for Each Dataset ---
    print("\n=== Stage 2: Training Classifiers ===")
    
    # Freeze all encoders
    for encoder in [encoder_ndd, encoder_pd, encoder_imu]:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    # Create embeddings for all datasets
    with torch.no_grad():
        ndd_embeds = encoder_ndd(ndd_data.to(device)).cpu()
        pd_embeds = encoder_pd(pd_data.to(device)).cpu()
        imu_embeds = encoder_imu(imu_data.to(device)).cpu()

    classification_loss_fn = nn.CrossEntropyLoss()

    # Train classifier for NDD Gait (4 classes)
    print("\n--- Training NDD Classifier (4 classes) ---")
    ndd_classifier_dataset = ClassifierDataset(ndd_embeds, ndd_labels)
    ndd_classifier_loader = DataLoader(ndd_classifier_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ndd_classifier = NDClassifier(embedding_dim=EMBEDDING_DIM, num_classes=4).to(device)
    optimizer_classifier = optim.Adam(ndd_classifier.parameters(), lr=LR)
    
    for epoch in range(CLASSIFIER_EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for embed_batch, label_batch in ndd_classifier_loader:
            embed_batch, label_batch = embed_batch.to(device), label_batch.to(device)
            optimizer_classifier.zero_grad()
            outputs = ndd_classifier(embed_batch)
            loss = classification_loss_fn(outputs, label_batch)
            loss.backward()
            optimizer_classifier.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += label_batch.size(0)
            total_correct += (predicted == label_batch).sum().item()
        avg_loss = total_loss / len(ndd_classifier_loader)
        accuracy = 100 * total_correct / total_samples
        print(f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Train classifier for PD Gait (2 classes)
    print("\n--- Training PD Classifier (2 classes) ---")
    pd_classifier_dataset = ClassifierDataset(pd_embeds, pd_labels)
    pd_classifier_loader = DataLoader(pd_classifier_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pd_classifier = NDClassifier(embedding_dim=EMBEDDING_DIM, num_classes=2).to(device)
    optimizer_classifier = optim.Adam(pd_classifier.parameters(), lr=LR)
    
    for epoch in range(CLASSIFIER_EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for embed_batch, label_batch in pd_classifier_loader:
            embed_batch, label_batch = embed_batch.to(device), label_batch.to(device)
            optimizer_classifier.zero_grad()
            outputs = pd_classifier(embed_batch)
            loss = classification_loss_fn(outputs, label_batch)
            loss.backward()
            optimizer_classifier.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += label_batch.size(0)
            total_correct += (predicted == label_batch).sum().item()
        avg_loss = total_loss / len(pd_classifier_loader)
        accuracy = 100 * total_correct / total_samples
        print(f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Train classifier for IMU Gait (3 classes)
    print("\n--- Training IMU Classifier (3 classes) ---")
    imu_classifier_dataset = ClassifierDataset(imu_embeds, imu_labels)
    imu_classifier_loader = DataLoader(imu_classifier_dataset, batch_size=BATCH_SIZE, shuffle=True)
    imu_classifier = NDClassifier(embedding_dim=EMBEDDING_DIM, num_classes=3).to(device)
    optimizer_classifier = optim.Adam(imu_classifier.parameters(), lr=LR)
    
    for epoch in range(CLASSIFIER_EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for embed_batch, label_batch in imu_classifier_loader:
            embed_batch, label_batch = embed_batch.to(device), label_batch.to(device)
            optimizer_classifier.zero_grad()
            outputs = imu_classifier(embed_batch)
            loss = classification_loss_fn(outputs, label_batch)
            loss.backward()
            optimizer_classifier.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += label_batch.size(0)
            total_correct += (predicted == label_batch).sum().item()
        avg_loss = total_loss / len(imu_classifier_loader)
        accuracy = 100 * total_correct / total_samples
        print(f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("\n=== Training Complete ===")
    print("Models trained:")
    print("  - NDD Gait: 2-channel VGRF, 4 classes (HC, PD, HD, ALS)")
    print("  - PD Gait: 18-channel VGRF, 2 classes (HC, PD)")
    print("  - IMU Gait: 12-channel IMU, 3 classes (HC, PD, Stroke)")
