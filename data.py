import numpy as np
import torch
from torch.utils.data import Dataset


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
        f"VGRF data shape: {vgrf_data.shape}, "
        f"Labels: {np.unique(vgrf_labels.numpy())}"
    )
    print(
        f"IMU data shape: {imu_data.shape}, " f"Labels: {np.unique(imu_labels.numpy())}"
    )

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
        vgrf_keys = set(self.vgrf_class_indices.keys())
        imu_keys = set(self.imu_class_indices.keys())
        self.common_classes = sorted(list(vgrf_keys & imu_keys))

        # We will create pairs on the fly in __getitem__
        # The length is based on the smaller dataset to ensure pairs can be made
        self.length = min(len(vgrf_data), len(imu_data))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Select a random common class to ensure positive pair can be formed
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
