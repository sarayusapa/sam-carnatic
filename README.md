# SAM-Audio for Carnatic Raga Classification

This repository contains a complete implementation of a SAM-Audio style model for Carnatic raga classification using the Hugging Face dataset "sarayusapa/carnatic-ragas".

## Overview

The SAM-Audio (Segment Anything Model for Audio) architecture adapts the concept of the Segment Anything Model (SAM) for visual tasks to audio processing. The model is designed for:

1. Learning meaningful temporal segments in audio signals
2. Self-supervised learning through masked segment prediction
3. Weakly-supervised learning for downstream classification tasks
4. Efficient fine-tuning using LoRA (Low-Rank Adaptation)

## Architecture Components

### 1. Audio Encoder
- CNN-based frontend similar to wav2vec2-style architectures
- Multiple convolutional layers with batch normalization and ReLU activation
- Extracts feature representations from raw audio waveforms

### 2. Latent Segmentation Tokens
- Learnable tokens that represent temporal segments in the audio
- Multi-head attention mechanism between audio features and segmentation tokens
- Captures local temporal patterns in the audio sequence

### 3. Masked Segment Prediction
- Randomly masks portions of the input audio
- Predicts masked segments based on unmasked context
- Enables self-supervised learning of audio representations

### 4. Contrastive Segment Learning
- Contrastive learning module for segment-level representations
- Encourages similar segments within the same raga to be closer in embedding space
- Uses a temperature parameter for controlling the similarity distribution

### 5. Classification Head
- Lightweight classifier that takes segment-level representations
- Averages segment representations for global context
- Outputs probabilities for each raga class

### 6. LoRA Integration
- Low-Rank Adaptation for efficient fine-tuning
- Reduces the number of trainable parameters
- Allows adaptation to new domains without full retraining

## Dataset

The model uses the Hugging Face dataset "sarayusapa/carnatic-ragas" which contains:
- Audio waveforms in the "audio" column
- String labels representing ragas in the "raga" column
- A reproducible 90/10 train-validation split with a fixed random seed

## Training Process

The training process combines:
1. **Masked Segment Prediction Loss**: Reconstruction loss for masked segments
2. **Contrastive Loss**: Encourages similar segments to be close in embedding space
3. **Classification Loss**: Cross-entropy loss for raga classification

The total loss is weighted combination:
```
total_loss = classification_loss + mask_weight * masked_lm_loss + contrastive_weight * contrastive_loss
```

## Configuration

The model is highly configurable through the `config.yaml` file:

## Usage

### Environment Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate sam-audio-conda-env
```

2. Install additional requirements if needed:
```bash
pip install -r requirements.txt  # if requirements.txt file exists
```

### Training

Run the training script:
```bash
python train.py --config config.yaml
```

Or customize training parameters:
```bash
python train.py --batch-size 16 --epochs 100 --lr 0.001 --seed 42
```

### Logging
metrics are logged to Weights & Biases.

## Model Architecture Diagram

```
Input Audio -> Audio Encoder -> Feature Sequence -> Segmentation Module -> Segment Representations -> Classifier -> Raga Probabilities
                    |                    |                    |                      |
                    v                    v                    v                      v
              Feature Maps        Attention Weights    Masked Prediction    Contrastive Learning
```

The model processes audio through the encoder to extract feature representations, then uses segmentation tokens to capture temporal patterns, with both masked prediction and contrastive learning objectives guiding the representation learning process.
