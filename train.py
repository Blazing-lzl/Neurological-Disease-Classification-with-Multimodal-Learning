import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import (
    ClassifierDataset,
    MultimodalPairDataset,
    generate_imu_gait_data,
    generate_ndd_gait_data,
    generate_pd_gait_data,
)
from modeling import ContrastiveLoss, NDClassifier, RepresentationEncoder


def train_contrastive_learning(
    encoder_vgrf, encoder_imu, pair_loader, num_epochs, lr, device
):
    """Train the encoders using contrastive learning."""
    print("\n--- STAGE 1: CONTRASTIVE PRE-TRAINING OF ENCODER ---")

    # Combine parameters for the optimizer
    all_encoder_params = list(encoder_vgrf.parameters()) + list(
        encoder_imu.parameters()
    )
    optimizer_contrastive = optim.Adam(all_encoder_params, lr=lr)
    contrastive_loss_fn = ContrastiveLoss()

    for epoch in range(num_epochs):
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
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Contrastive Loss: {avg_loss:.4f}"
        )

    print("Contrastive pre-training finished.")
    return encoder_vgrf, encoder_imu


def train_classifier(classifier, classifier_loader, num_epochs, lr, device):
    """Train the ND classifier."""
    print("\n--- STAGE 2: TRAINING THE ND CLASSIFIER ---")

    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr)
    classification_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for embed_batch, label_batch in classifier_loader:
            embed_batch = embed_batch.to(device)
            label_batch = label_batch.to(device)

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
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Classifier Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("Classifier training finished.")
    return classifier


def main():
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

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
    encoder_ndd = RepresentationEncoder(
        2, EMBEDDING_DIM
    ).to(device)  # 2 channels
    encoder_pd = RepresentationEncoder(
        18, EMBEDDING_DIM
    ).to(device)  # 18 channels
    encoder_imu = RepresentationEncoder(
        12, EMBEDDING_DIM
    ).to(device)  # 12 channels

    # Pair 1: NDD VGRF + PD VGRF (both VGRF modality, different channels)
    print("\n--- Training NDD-PD Pair ---")
    pair_dataset_ndd_pd = MultimodalPairDataset(
        ndd_data, ndd_labels, pd_data, pd_labels,
        label_mapping=ndd_to_pd_mapping
    )
    pair_loader_ndd_pd = DataLoader(
        pair_dataset_ndd_pd, batch_size=BATCH_SIZE, shuffle=True
    )
    encoder_ndd, encoder_pd = train_contrastive_learning(
        encoder_ndd, encoder_pd, pair_loader_ndd_pd,
        CONTRASTIVE_EPOCHS, LR, device
    )

    # Pair 2: PD VGRF + IMU (cross-modal)
    print("\n--- Training PD-IMU Pair ---")
    pair_dataset_pd_imu = MultimodalPairDataset(
        pd_data, pd_labels, imu_data, imu_labels,
        label_mapping=pd_to_imu_mapping
    )
    pair_loader_pd_imu = DataLoader(
        pair_dataset_pd_imu, batch_size=BATCH_SIZE, shuffle=True
    )
    encoder_pd, encoder_imu = train_contrastive_learning(
        encoder_pd, encoder_imu, pair_loader_pd_imu,
        CONTRASTIVE_EPOCHS, LR, device
    )

    # Pair 3: NDD VGRF + IMU (cross-modal)
    print("\n--- Training NDD-IMU Pair ---")
    pair_dataset_ndd_imu = MultimodalPairDataset(
        ndd_data, ndd_labels, imu_data, imu_labels,
        label_mapping=ndd_to_imu_mapping
    )
    pair_loader_ndd_imu = DataLoader(
        pair_dataset_ndd_imu, batch_size=BATCH_SIZE, shuffle=True
    )
    encoder_ndd, encoder_imu = train_contrastive_learning(
        encoder_ndd, encoder_imu, pair_loader_ndd_imu,
        CONTRASTIVE_EPOCHS, LR, device
    )

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

    # Train classifier for NDD Gait (4 classes)
    print("\n--- Training NDD Classifier (4 classes) ---")
    ndd_classifier_dataset = ClassifierDataset(ndd_embeds, ndd_labels)
    ndd_classifier_loader = DataLoader(
        ndd_classifier_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    ndd_classifier = NDClassifier(
        embedding_dim=EMBEDDING_DIM, num_classes=4
    ).to(device)
    ndd_classifier = train_classifier(
        ndd_classifier, ndd_classifier_loader, CLASSIFIER_EPOCHS, LR, device
    )

    # Train classifier for PD Gait (2 classes)
    print("\n--- Training PD Classifier (2 classes) ---")
    pd_classifier_dataset = ClassifierDataset(pd_embeds, pd_labels)
    pd_classifier_loader = DataLoader(
        pd_classifier_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    pd_classifier = NDClassifier(
        embedding_dim=EMBEDDING_DIM, num_classes=2
    ).to(device)
    pd_classifier = train_classifier(
        pd_classifier, pd_classifier_loader, CLASSIFIER_EPOCHS, LR, device
    )

    # Train classifier for IMU Gait (3 classes)
    print("\n--- Training IMU Classifier (3 classes) ---")
    imu_classifier_dataset = ClassifierDataset(imu_embeds, imu_labels)
    imu_classifier_loader = DataLoader(
        imu_classifier_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    imu_classifier = NDClassifier(
        embedding_dim=EMBEDDING_DIM, num_classes=3
    ).to(device)
    imu_classifier = train_classifier(
        imu_classifier, imu_classifier_loader, CLASSIFIER_EPOCHS, LR, device
    )

    print("\n=== Training Complete ===")
    print("Models trained:")
    print("  - NDD Gait: 2-channel VGRF, 4 classes (HC, PD, HD, ALS)")
    print("  - PD Gait: 18-channel VGRF, 2 classes (HC, PD)")
    print("  - IMU Gait: 12-channel IMU, 3 classes (HC, PD, Stroke)")


if __name__ == "__main__":
    main()
