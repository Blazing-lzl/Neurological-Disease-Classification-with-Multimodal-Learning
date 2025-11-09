# Neurological Disease Classification with Multimodal Learning

This project implements a multimodal contrastive learning framework for neurological disease classification using VGRF (Vertical Ground Reaction Force) and IMU (Inertial Measurement Unit) data. The system consists of two main stages: contrastive pre-training of representation encoders and supervised training of an iTransformer-based classifier.

## Project Structure

```
├── data.py          # Data generation and dataset classes
├── modeling.py      # Model definitions and loss functions
├── train.py         # Training pipeline and main execution
├── main.py          # Original monolithic implementation (for reference)
└── README.md        # This file
```

## Model Architecture

### Stage 1: Contrastive Learning
- **Representation Encoders**: 1D CNN-based encoders that convert time-series data into fixed-size embeddings
- **Contrastive Loss**: InfoNCE loss function that pulls positive pairs (same class) together and pushes negative pairs apart
- **Multimodal Pairs**: Creates positive pairs from VGRF and IMU samples belonging to the same neurological condition

### Stage 2: Classification
- **iTransformer-based Classifier**: Uses Auto-Correlation mechanism for sequence modeling
- **NDClassifier**: Multi-class classifier for distinguishing between neurological diseases
- **Classes**: Healthy Control (HC), Parkinson's Disease (PD), Huntington's Disease (HD), ALS, and Stroke

## Key Features

- **Multimodal Learning**: Combines VGRF and IMU sensor data
- **Contrastive Pre-training**: Learns shared representations across modalities
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
# Data generation
from data import generate_dummy_data, MultimodalPairDataset

vgrf_data, vgrf_labels, imu_data, imu_labels = generate_dummy_data(
    num_samples=1000, seq_len=128, 
    vgrf_features=2, imu_features=6, 
    num_classes=5
)

# Model initialization
from modeling import RepresentationEncoder, NDClassifier, ContrastiveLoss

encoder = RepresentationEncoder(input_features=6, embedding_dim=64)
classifier = NDClassifier(embedding_dim=64, num_classes=5)
contrastive_loss = ContrastiveLoss(temperature=0.1)
```

## Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `NUM_SAMPLES` | 1000 | Number of samples per modality |
| `SEQ_LEN` | 128 | Sequence length for time-series data |
| `VGRF_FEATURES` | 2 | Number of features in VGRF data |
| `IMU_FEATURES` | 6 | Number of features in IMU data |
| `NUM_CLASSES` | 5 | Number of neurological disease classes |
| `EMBEDDING_DIM` | 64 | Dimensionality of learned embeddings |
| `BATCH_SIZE` | 256 | Training batch size |
| `CONTRASTIVE_EPOCHS` | 10 | Epochs for contrastive pre-training |
| `CLASSIFIER_EPOCHS` | 15 | Epochs for classifier training |
| `LR` | 1e-3 | Learning rate |

## Training Pipeline

1. **Data Generation**: Creates synthetic VGRF and IMU time-series data with class labels
2. **Contrastive Pre-training**: Trains representation encoders to learn shared embeddings
3. **Embedding Extraction**: Generates embeddings for all samples using trained encoders
4. **Classifier Training**: Trains iTransformer-based classifier on the embeddings
5. **Evaluation**: Reports classification accuracy and loss metrics

## Output

The training process outputs:
- Contrastive loss progression during pre-training
- Classification loss and accuracy during classifier training
- Final model performance metrics

Example output:
```
Using device: cuda
Generated 1000 samples.
VGRF data shape: torch.Size([1000, 128, 2]), Labels: [0 1 2 3]
IMU data shape: torch.Size([1000, 128, 6]), Labels: [0 1 3 4]

--- STAGE 1: CONTRASTIVE PRE-TRAINING OF ENCODER ---
Epoch 1/10, Contrastive Loss: 2.3456
...

--- STAGE 2: TRAINING THE ND CLASSIFIER ---
Epoch 1/15, Classifier Loss: 1.2345, Accuracy: 45.67%
...
```

## Notes

- This implementation uses synthetic data for demonstration purposes
- Real VGRF and IMU data should be preprocessed according to your specific requirements
- The model architecture can be customized by modifying the hyperparameters
- GPU acceleration is automatically enabled when available

## License

This project is provided as-is for research and educational purposes.
