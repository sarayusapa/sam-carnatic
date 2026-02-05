# SAM-Audio for Carnatic Raga Classification

A SAM-Audio (Segment Anything Model for Audio) implementation for classifying Carnatic ragas from audio waveforms.

## Overview

The SAM-Audio architecture adapts the Segment Anything Model concept for audio processing:

1. **Temporal Segmentation** - Learns meaningful segments in audio signals
2. **Self-supervised Learning** - Masked segment prediction for representation learning
3. **Contrastive Learning** - Segment-level embeddings that cluster by raga
4. **Efficient Training** - Mixed precision, torch.compile, Flash Attention


## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/sam-carnatic.git
cd sam-carnatic

# Create conda environment with CUDA 12.1 support
conda env create -f environment.yml

# Activate the environment
conda activate sam-audio-conda-env

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Option 2: pip with venv

```bash
# Clone the repository
git clone https://github.com/yourusername/sam-carnatic.git
cd sam-carnatic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

## Dataset
The model uses the Hugging Face dataset `sarayusapa/carnatic-ragas`:

```python
from datasets import load_dataset
dataset = load_dataset("sarayusapa/carnatic-ragas")
print(dataset)
```
**Dataset Structure:**
- `audio`: Audio waveform at 16kHz
- `raga`: String label (one of 8 ragas)

**Ragas Included:**
Melakarta Ragas: Kalyani, Kharaharapriya, Mayamalavagoulai, Todi, 
Janya Ragas: Amritavarshini, Hamsanaadam, Varali, Sindhubhairavi

## Training

### Quick Start

```bash
python train.py --config config.yaml
```

### Custom Training

```bash
# Override specific parameters
python train.py --config config.yaml --batch-size 64 --epochs 100 --lr 1e-4

# Disable torch.compile (useful for debugging)
python train.py --config config.yaml --no-compile

# Set random seed for reproducibility
python train.py --config config.yaml --seed 123
```

### Configuration Options

Edit `config.yaml` to customize training:

```yaml
# Training parameters
num_epochs: 50
batch_size: 32                    # Increase for more VRAM (64 for 24GB)
learning_rate: 3e-4
warmup_epochs: 2                  # LR warmup epochs
patience: 10                      # Early stopping patience

# Loss weights
mask_weight: 0.5                  # Masked prediction loss weight
contrastive_weight: 0.3           # Contrastive loss weight

# Model architecture
encoder_dims: [64, 128, 256, 512]
num_segments: 64
mask_ratio: 0.15


### Monitoring Training

Training metrics are logged to Weights & Biases:

```

Logged metrics:
- `train_loss`, `train_accuracy`
- `val_loss`, `val_accuracy`, `val_f1_score`
- `learning_rate`

## Model Architecture

```
Input Audio (5s @ 16kHz)
        │
        ▼
┌───────────────────┐
│   Audio Encoder   │  CNN layers: 64→128→256→512 channels
│   (Conv1D + GELU) │  Stride 2, LayerNorm
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Segment Tokens   │  64 learnable tokens
│  + Attention      │  Flash Attention (SDPA)
└───────────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Masked     │   │ Contrastive  │   │Classification│
│  Prediction  │   │   Learning   │   │    Head      │
└──────────────┘   └──────────────┘   └──────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
   MSE Loss          InfoNCE Loss      CrossEntropy
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                    Total Loss = CE + 0.5*MSE + 0.3*InfoNCE
```


## Output Files

After training:
- `best_model.pth` - Best validation accuracy checkpoint
- `final_model.pth` - Final epoch weights

Checkpoint contents:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'val_accuracy': float,
    'config': dict
}
```

## Inference

```python
import torch
from train import SAMAudioModel, setup_cuda_optimizations

# Setup device
device, dtype = setup_cuda_optimizations()

# Load model
checkpoint = torch.load('best_model.pth')
config = checkpoint['config']

model = SAMAudioModel(
    encoder_config={
        'hidden_dims': config['encoder_dims'],
        'kernel_size': config['kernel_size'],
        'stride': config['stride'],
        'dropout_rate': config['dropout_rate']
    },
    num_classes=8  # Number of ragas
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Inference
with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
    audio_tensor = ...  # Your audio tensor (batch, 80000)
    outputs = model(audio_tensor.to(device))
    predictions = outputs['raga_logits'].argmax(dim=-1)
```

