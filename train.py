import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ClassifierDataset, MultimodalPairDataset, generate_dummy_data
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
        print(f"Epoch {epoch+1}/{num_epochs}, " f"Contrastive Loss: {avg_loss:.4f}")

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
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Classifier Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    print("Classifier training finished.")
    return classifier


def main():
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Hyperparameters ---
    NUM_SAMPLES = 1000
    SEQ_LEN = 128
    VGRF_FEATURES = 2
    IMU_FEATURES = 12
    NUM_CLASSES = 5  # HC, PD, HD, ALS, Stroke (0, 1, 2, 3, 4)
    EMBEDDING_DIM = 64
    BATCH_SIZE = 128
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
    # Initialize shared encoder for both modalities
    # Note: We need two encoders initially to handle different input feature sizes
    # If feature sizes were the same, one encoder instance would suffice.
    encoder_vgrf = RepresentationEncoder(VGRF_FEATURES, EMBEDDING_DIM).to(device)
    encoder_imu = RepresentationEncoder(IMU_FEATURES, EMBEDDING_DIM).to(device)

    encoder_vgrf, encoder_imu = train_contrastive_learning(
        encoder_vgrf, encoder_imu, pair_loader, CONTRASTIVE_EPOCHS, LR, device
    )

    # --- Stage 2: Classifier Training ---
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
    classifier = train_classifier(
        classifier, classifier_loader, CLASSIFIER_EPOCHS, LR, device
    )


if __name__ == "__main__":
    main()
