import numpy as np
import torch
from torch.utils.data import Dataset


def generate_ndd_gait_data(num_samples, seq_len):
    """
    Generates NDD Gait dataset: 2-channel VGRF data with 4 classes.
    Classes: 0=HC, 1=PD, 2=HD, 3=ALS
    """
    data = torch.randn(num_samples, seq_len, 2)  # 2 channels
    labels = torch.randint(0, 4, (num_samples,))  # 4 classes

    print(
        f"Generated NDD Gait: {num_samples} samples, "
        f"2 channels, 4 classes"
    )
    print(f"  Shape: {data.shape}, Labels: {np.unique(labels.numpy())}")

    return data, labels


def generate_pd_gait_data(num_samples, seq_len):
    """
    Generates PD Gait dataset: 18-channel VGRF data with 2 classes.
    Classes: 0=HC, 1=PD
    """
    data = torch.randn(num_samples, seq_len, 18)  # 18 channels
    labels = torch.randint(0, 2, (num_samples,))  # 2 classes

    print(
        f"Generated PD Gait: {num_samples} samples, "
        f"18 channels, 2 classes"
    )
    print(f"  Shape: {data.shape}, Labels: {np.unique(labels.numpy())}")

    return data, labels


def generate_imu_gait_data(num_samples, seq_len):
    """
    Generates IMU Gait dataset: 12-channel IMU data with 3 classes.
    Classes: 0=HC, 1=PD, 2=Stroke
    """
    data = torch.randn(num_samples, seq_len, 12)  # 12 channels
    labels = torch.randint(0, 3, (num_samples,))  # 3 classes

    print(
        f"Generated IMU Gait: {num_samples} samples, "
        f"12 channels, 3 classes"
    )
    print(f"  Shape: {data.shape}, Labels: {np.unique(labels.numpy())}")

    return data, labels


class MultimodalPairDataset(Dataset):
    """
    Creates pairs of samples from two modalities for contrastive learning.
    A pair is "positive" if both samples belong to the same class
    (e.g., HC). Handles label mapping between different modalities with
    different label spaces.
    """

    def __init__(self, data1, labels1, data2, labels2, label_mapping=None):
        """
        Args:
            data1: First modality data
            labels1: Labels for first modality
            data2: Second modality data
            labels2: Labels for second modality
            label_mapping: Dict mapping labels1 to labels2
                          (for finding common classes)
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
            mapped_labels1 = set(
                label_mapping.get(k, -1)
                for k in self.data1_class_indices.keys()
            )
            data2_labels = set(self.data2_class_indices.keys())
            common_in_data2_space = sorted(
                list(mapped_labels1 & data2_labels)
            )

            # Store pairs of (data1_label, data2_label)
            # for common classes
            self.common_pairs = [
                (k, v) for k, v in label_mapping.items()
                if v in common_in_data2_space
                and k in self.data1_class_indices
            ]
        else:
            # No mapping, use direct label matching
            data1_keys = set(self.data1_class_indices.keys())
            data2_keys = set(self.data2_class_indices.keys())
            common = sorted(list(data1_keys & data2_keys))
            self.common_pairs = [(c, c) for c in common]

        if len(self.common_pairs) == 0:
            raise ValueError(
                "No common classes found between the two modalities!"
            )

        # The length is based on the smaller dataset
        self.length = min(len(data1), len(data2))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Select a random common class pair
        pair_idx = np.random.randint(len(self.common_pairs))
        label1, label2 = self.common_pairs[pair_idx]

        # Pick a random sample from data1 with label1
        idx1 = np.random.choice(self.data1_class_indices[label1])
        sample1 = self.data1[idx1]

        # Pick a random sample from data2 with label2 (positive pair)
        idx2 = np.random.choice(self.data2_class_indices[label2])
        sample2 = self.data2[idx2]

        # Return the pair with the shared semantic label
        # (using label1 as reference)
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
