# Neurological Disease Classification with Multimodal Learning

This project implements a multimodal contrastive learning framework for neurological disease classification using three different gait datasets: NDD Gait (2-channel VGRF), PD Gait (18-channel VGRF), and IMU Gait (12-channel IMU). The system consists of two main stages: contrastive pre-training of representation encoders across modalities and supervised training of dataset-specific iTransformer-based classifiers.

## Datasets

The framework supports three different gait analysis datasets:

| Dataset | Sensor Modality | Channels | Classes | Description |
|---------|----------------|----------|---------|-------------|
| **NDD Gait** | VGRF | 2 | 4 | HC, PD, HD, ALS |
| **PD Gait** | VGRF | 18 | 2 | HC, PD |
| **IMU Gait** | IMU | 12 | 3 | HC, PD, Stroke |

- **NDD Gait**: Neurodegenerative disease dataset with 2-channel VGRF data
- **PD Gait**: Parkinson's disease focused dataset with 18-channel VGRF data
- **IMU Gait**: Inertial measurement unit dataset for stroke detection

## Project Structure

```
├── data.py          # Data generation and dataset classes
├── modeling.py      # Model definitions and loss functions
├── train.py         # Training pipeline and main execution
├── main.py          # Standalone implementation (all-in-one)
└── README.md        # This file
```

## Model Architecture

### Stage 1: Contrastive Learning
- **Representation Encoders**: Separate 1D CNN-based encoders for each dataset that convert time-series data into fixed-size embeddings
- **Contrastive Loss**: InfoNCE loss function that pulls positive pairs (same class) together and pushes negative pairs apart
- **Cross-Modal Pairs**: Creates positive pairs across different modalities/datasets:
  - NDD VGRF ↔ PD VGRF (common classes: HC, PD)
  - PD VGRF ↔ IMU (common classes: HC, PD)
  - NDD VGRF ↔ IMU (common classes: HC, PD)

### Stage 2: Classification
- **iTransformer-based Classifier**: Uses Auto-Correlation mechanism for sequence modeling
- **NDClassifier**: Dataset-specific multi-class classifiers
  - NDD Classifier: 4 classes (HC, PD, HD, ALS)
  - PD Classifier: 2 classes (HC, PD)
  - IMU Classifier: 3 classes (HC, PD, Stroke)

## Key Features

- **Multi-Dataset Learning**: Supports three different gait datasets with varying channels and classes
- **Cross-Modal Contrastive Learning**: Learns shared representations across VGRF and IMU modalities
- **Label Mapping**: Handles different label spaces across datasets (common classes: HC, PD)
- **iTransformer Architecture**: Advanced time-series modeling with auto-correlation
- **Modular Design**: Clean separation of data, modeling, and training components

## Dependencies

```bash
pip install torch>=1.9.0
pip install numpy>=1.21.0
```

### Detailed Requirements

- **PyTorch**: Deep learning framework for model implementation
  - `torch`: Core PyTorch library
  - `torch.nn`: Neural network modules
  - `torch.optim`: Optimization algorithms
  - `torch.utils.data`: Data loading utilities
- **NumPy**: Numerical computing for data manipulation

## Usage

### Quick Start

```bash
python train.py
```

### Custom Training

```python
from train import main
from data import generate_dummy_data
from modeling import RepresentationEncoder, NDClassifier

# Run the complete training pipeline
main()
```

### Individual Components

```python
# Data generation for each dataset
from data import (
    generate_ndd_gait_data, 
    generate_pd_gait_data, 
    generate_imu_gait_data,
    MultimodalPairDataset
)

ndd_data, ndd_labels = generate_ndd_gait_data(num_samples=1000, seq_len=100)
pd_data, pd_labels = generate_pd_gait_data(num_samples=1000, seq_len=100)
imu_data, imu_labels = generate_imu_gait_data(num_samples=1000, seq_len=100)

# Create cross-modal pairs with label mapping
label_mapping = {0: 0, 1: 1}  # Map HC=0->0, PD=1->1
pair_dataset = MultimodalPairDataset(
    ndd_data, ndd_labels, 
    pd_data, pd_labels,
    label_mapping=label_mapping
)

# Model initialization for each dataset
from modeling import RepresentationEncoder, NDClassifier, ContrastiveLoss

encoder_ndd = RepresentationEncoder(input_features=2, embedding_dim=64)
encoder_pd = RepresentationEncoder(input_features=18, embedding_dim=64)
encoder_imu = RepresentationEncoder(input_features=12, embedding_dim=64)

classifier_ndd = NDClassifier(embedding_dim=64, num_classes=4)
classifier_pd = NDClassifier(embedding_dim=64, num_classes=2)
classifier_imu = NDClassifier(embedding_dim=64, num_classes=3)

contrastive_loss = ContrastiveLoss(temperature=0.1)
```

## Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `NUM_SAMPLES` | 30000 | Number of samples per dataset |
| `SEQ_LEN` | 128 | Sequence length for time-series data |
| `EMBEDDING_DIM` | 64 | Dimensionality of learned embeddings |
| `BATCH_SIZE` | 256 | Training batch size |
| `CONTRASTIVE_EPOCHS` | 10 | Epochs for contrastive pre-training |
| `CLASSIFIER_EPOCHS` | 100 | Epochs for classifier training |
| `LR` | 1e-3 | Learning rate |

**Dataset-Specific Parameters:**
- NDD Gait: 2 channels, 4 classes
- PD Gait: 18 channels, 2 classes
- IMU Gait: 12 channels, 3 classes

## Training Pipeline

1. **Data Generation**: Creates synthetic data for all three datasets (NDD, PD, IMU)
2. **Label Mapping**: Defines common classes across datasets (HC, PD)
3. **Contrastive Pre-training**: Trains encoders on three pairs:
   - NDD VGRF ↔ PD VGRF
   - PD VGRF ↔ IMU
   - NDD VGRF ↔ IMU
4. **Embedding Extraction**: Generates embeddings for all samples using trained encoders
5. **Classifier Training**: Trains three separate classifiers (NDD, PD, IMU)
6. **Evaluation**: Reports classification accuracy and loss for each dataset

## Output

The training process outputs:
- Data generation summary for each dataset
- Contrastive loss progression for each encoder pair
- Classification loss and accuracy for each classifier
- Final model performance metrics

Example output:
```
Using device: cuda

=== Generating Data ===
Generated NDD Gait: 1000 samples, 2 channels, 4 classes
  Shape: torch.Size([1000, 100, 2]), Labels: [0 1 2 3]
Generated PD Gait: 1000 samples, 18 channels, 2 classes
  Shape: torch.Size([1000, 100, 18]), Labels: [0 1]
Generated IMU Gait: 1000 samples, 12 channels, 3 classes
  Shape: torch.Size([1000, 100, 12]), Labels: [0 1 2]

=== Stage 1: Contrastive Pre-training ===

--- Training NDD-PD Pair ---
Epoch 1/10, Contrastive Loss: 2.3456
...

--- Training PD-IMU Pair ---
Epoch 1/10, Contrastive Loss: 1.8432
...

--- Training NDD-IMU Pair ---
Epoch 1/10, Contrastive Loss: 2.1234
...

=== Stage 2: Training Classifiers ===

--- Training NDD Classifier (4 classes) ---
Epoch 1/15, Classifier Loss: 1.3456, Accuracy: 42.50%
...

--- Training PD Classifier (2 classes) ---
Epoch 1/15, Classifier Loss: 0.6789, Accuracy: 58.30%
...

--- Training IMU Classifier (3 classes) ---
Epoch 1/15, Classifier Loss: 1.0123, Accuracy: 48.75%
...

=== Training Complete ===
Models trained:
  - NDD Gait: 2-channel VGRF, 4 classes (HC, PD, HD, ALS)
  - PD Gait: 18-channel VGRF, 2 classes (HC, PD)
  - IMU Gait: 12-channel IMU, 3 classes (HC, PD, Stroke)
```

## Notes

- This implementation uses synthetic data for demonstration purposes
- Real gait data should be preprocessed according to specific dataset requirements
- The framework handles different channel numbers and class counts automatically
- Label mapping ensures proper alignment of common classes (HC, PD) across datasets
- GPU acceleration is automatically enabled when available
- Each dataset has its own encoder and classifier optimized for its specific characteristics

## License

This project is provided as-is for research and educational purposes.
