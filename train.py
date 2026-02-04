"""
SAM-Audio Style Model for Carnatic Raga Classification

This module implements a SAM-Audio style architecture for classifying Carnatic ragas
from audio waveforms. The model consists of:
- Audio encoder (CNN-based frontend)
- Latent segmentation tokens for temporal modeling
- Masked segment prediction for self-supervised learning
- Lightweight classification head for raga prediction
- LoRA integration for efficient fine-tuning
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoFeatureExtractor, Wav2Vec2Processor, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from huggingface_hub import HfApi
import wandb
from sklearn.metrics import accuracy_score, f1_score
import argparse
import yaml
from typing import Optional, Tuple, Union
import math
from peft import LoraConfig, get_peft_model, TaskType

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class AudioEncoder(nn.Module):
    """
    Audio encoder using a CNN-based frontend similar to wav2vec2-style architectures.
    Extracts feature representations from raw audio waveforms.
    """
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 256, 512], kernel_size=3, stride=2, dropout_rate=0.1):
        super(AudioEncoder, self).__init__()
        
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(hidden_dims):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=kernel_size//2)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output dimension after convolutions
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        # x shape: (batch, length) -> (batch, channels, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        features = self.conv_layers(x)
        return features.transpose(1, 2)  # (batch, length, channels)


class SegmentTokenPredictor(nn.Module):
    """
    Implements latent segmentation tokens and masked segment prediction.
    These tokens capture local temporal patterns in the audio sequence.
    """
    def __init__(self, hidden_size, num_segments=64, mask_ratio=0.15):
        super(SegmentTokenPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_segments = num_segments
        self.mask_ratio = mask_ratio
        
        # Learnable segmentation tokens
        self.segment_tokens = nn.Parameter(torch.randn(num_segments, hidden_size))
        
        # Segment attention mechanism
        self.segment_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Segment reconstruction head for masked prediction
        self.segment_predictor = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, audio_features, mask_indices=None):
        batch_size, seq_len, feat_dim = audio_features.shape
        segment_tokens = self.segment_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply segment attention between audio features and segmentation tokens
        attended_features, attn_weights = self.segment_attention(
            query=segment_tokens,
            key=audio_features,
            value=audio_features
        )
        
        # Generate predictions for masked segments if provided
        if mask_indices is not None and self.training:
            # Apply masking to audio features
            masked_features = audio_features.clone()
            for b, mask_idx in enumerate(mask_indices):
                masked_features[b, mask_idx, :] = 0.0
                
            # Recompute attention with masked features
            attended_masked, _ = self.segment_attention(
                query=segment_tokens,
                key=masked_features,
                value=masked_features
            )
            
            # Predict masked segments
            segment_predictions = self.segment_predictor(attended_masked)
            target_segments = attended_features  # Target is the unmasked representation
            
            return attended_features, segment_predictions, target_segments, attn_weights
        else:
            return attended_features, None, None, attn_weights
            
    def generate_mask_indices(self, batch_size, seq_len):
        """Generate random mask indices for masked segment prediction."""
        num_masked = int(seq_len * self.mask_ratio)
        mask_indices = []
        
        for _ in range(batch_size):
            indices = torch.randperm(seq_len)[:num_masked]
            mask_indices.append(indices)
            
        return mask_indices


class ContrastiveSegmentModule(nn.Module):
    """
    Contrastive learning module for segment-level representations.
    Encourages similar segments within the same raga to be closer in embedding space.
    """
    def __init__(self, hidden_size, temperature=0.07):
        super(ContrastiveSegmentModule, self).__init__()
        
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
    def forward(self, segment_features):
        # Project segment features to lower-dimensional space
        projected = self.projection_head(segment_features)
        
        # Normalize embeddings
        projected = F.normalize(projected, dim=-1)
        
        # Compute contrastive loss
        batch_size, num_segments, feat_dim = projected.shape
        
        # Reshape to treat each segment separately
        reshaped = projected.view(-1, feat_dim)  # (batch_size * num_segments, feat_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(reshaped, reshaped.T) / self.temperature
        
        # Create labels for contrastive learning
        # Segments from the same audio sample are positive pairs
        labels = torch.arange(batch_size).unsqueeze(1).expand(-1, num_segments).flatten().to(projected.device)
        labels = labels.unsqueeze(0).expand(len(labels), -1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Only consider positive pairs in the loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        contrastive_loss = -mean_log_prob_pos.mean()
        
        return contrastive_loss


class RagaclassificationHead(nn.Module):
    """
    Lightweight classification head for raga prediction.
    Takes segment-level representations and predicts the raga.
    """
    def __init__(self, hidden_size, num_classes, dropout_rate=0.1):
        super(RagaclassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, segment_representations):
        # Average across segments for global representation
        global_repr = torch.mean(segment_representations, dim=1)
        
        # Classify
        logits = self.classifier(global_repr)
        return logits


class SAMAUDIOModel(PreTrainedModel):
    """
    Complete SAM-Audio model for Carnatic raga classification.
    
    Architecture:
    - Audio encoder for feature extraction
    - Latent segmentation tokens for temporal modeling
    - Masked segment prediction for self-supervised learning
    - Contrastive segment learning
    - Classification head for raga prediction
    """
    config_class = None
    
    def __init__(self, encoder_config, num_classes, lora_config=None):
        super(SAMAUDIOModel, self).__init__(None)  # Pass None since we're not using traditional config
        
        self.audio_encoder = AudioEncoder(**encoder_config)
        encoder_hidden_size = encoder_config.get('hidden_dims', [64, 128, 256, 512])[-1]
        
        self.segment_predictor = SegmentTokenPredictor(
            hidden_size=encoder_hidden_size
        )
        
        self.contrastive_module = ContrastiveSegmentModule(
            hidden_size=encoder_hidden_size
        )
        
        self.raga_classifier = RagaclassificationHead(
            hidden_size=encoder_hidden_size,
            num_classes=num_classes
        )
        
        # Store configuration
        self.encoder_config = encoder_config
        self.num_classes = num_classes
        
        # Initialize LoRA if provided
        if lora_config:
            self.apply_lora(lora_config)
        
    def apply_lora(self, lora_config):
        """Apply LoRA to specific layers of the model."""
        if hasattr(self.audio_encoder, 'conv_layers'):
            # Apply LoRA to linear layers in the encoder
            for layer in self.audio_encoder.conv_layers:
                if isinstance(layer, nn.Linear):
                    # Convert linear layer to LoRA-augmented layer
                    peft_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        inference_mode=False,
                        r=lora_config.get('r', 8),
                        lora_alpha=lora_config.get('alpha', 32),
                        lora_dropout=lora_config.get('dropout', 0.1),
                        target_modules=["weight"]
                    )
                    self.audio_encoder = get_peft_model(self.audio_encoder, peft_config)
    
    def forward(self, input_audio, masks=None, labels=None, return_contrastive_loss=True):
        # Extract audio features
        audio_features = self.audio_encoder(input_audio)  # (batch, seq_len, hidden_size)
        
        # Get segment representations through attention
        segment_reps, seg_predictions, seg_targets, attn_weights = self.segment_predictor(
            audio_features, masks
        )
        
        # Compute masked segment prediction loss if in training mode
        masked_lm_loss = None
        if seg_predictions is not None and seg_targets is not None:
            masked_lm_loss = F.mse_loss(seg_predictions, seg_targets)
        
        # Compute contrastive loss
        contrastive_loss = None
        if return_contrastive_loss:
            contrastive_loss = self.contrastive_module(segment_reps)
        
        # Get raga classification logits
        raga_logits = self.raga_classifier(segment_reps)
        
        # Compute classification loss if labels provided
        classification_loss = None
        if labels is not None:
            classification_loss = F.cross_entropy(raga_logits, labels)
        
        return {
            'segment_representations': segment_reps,
            'masked_lm_loss': masked_lm_loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss,
            'raga_logits': raga_logits,
            'attention_weights': attn_weights
        }


def preprocess_audio(waveform, target_sr=16000, max_length=16000*5):  # 5 seconds max
    """
    Preprocess audio waveform for the model.
    """
    # Resample to target sampling rate if needed
    if waveform.shape[-1] != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=waveform.shape[-1], new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Pad or truncate to max length
    if waveform.shape[-1] > max_length:
        waveform = waveform[..., :max_length]
    else:
        pad_length = max_length - waveform.shape[-1]
        waveform = F.pad(waveform, (0, pad_length))
    
    return waveform


def collate_fn(batch, mask_ratio=0.15):
    """
    Collate function for DataLoader that prepares batches and generates mask indices.
    """
    waveforms = []
    labels = []
    
    for item in batch:
        # Get audio waveform (already loaded as tensor by HF dataset)
        audio = item['audio']['array']
        if len(audio.shape) > 1:
            audio = audio[0]  # Take first channel if stereo
        waveforms.append(torch.tensor(audio, dtype=torch.float32))
        labels.append(item['raga'])
    
    # Stack waveforms
    waveforms = torch.stack(waveforms)
    
    # Generate mask indices for masked segment prediction
    batch_size, seq_len = waveforms.shape
    mask_indices = []
    num_masked = int(seq_len * mask_ratio)
    
    for _ in range(batch_size):
        indices = torch.randperm(seq_len)[:num_masked]
        mask_indices.append(indices)
    
    return {
        'input_audio': waveforms,
        'labels': torch.tensor(labels, dtype=torch.long),
        'masks': mask_indices
    }


def train_epoch(model, dataloader, optimizer, criterion, epoch, config):
    """
    Single training epoch.
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        input_audio = batch['input_audio'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks']
        
        optimizer.zero_grad()
        
        outputs = model(input_audio=input_audio, masks=masks, labels=labels)
        
        # Combine losses
        loss = 0
        if outputs['classification_loss'] is not None:
            loss += outputs['classification_loss']
        
        if outputs['masked_lm_loss'] is not None:
            loss += config.get('mask_weight', 0.5) * outputs['masked_lm_loss']
        
        if outputs['contrastive_loss'] is not None:
            loss += config.get('contrastive_weight', 0.3) * outputs['contrastive_loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        preds = torch.argmax(outputs['raga_logits'], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(input_audio)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, config):
    """
    Validation loop.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_audio = batch['input_audio'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks']
            
            outputs = model(input_audio=input_audio, masks=masks, labels=labels)
            
            # Combine losses
            loss = 0
            if outputs['classification_loss'] is not None:
                loss += outputs['classification_loss']
            
            if outputs['masked_lm_loss'] is not None:
                loss += config.get('mask_weight', 0.5) * outputs['masked_lm_loss']
            
            if outputs['contrastive_loss'] is not None:
                loss += config.get('contrastive_weight', 0.3) * outputs['contrastive_loss']
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = torch.argmax(outputs['raga_logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def main(config):
    """
    Main training function.
    """
    # Setup wandb
    wandb.init(
        project=config.get('wandb_project', 'sam-audio-carnatic-raga'),
        config=config,
        name=config.get('run_name', 'sam-audio-rg')
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("sarayusapa/carnatic-ragas")
    
    # Get unique ragas for classification
    unique_ragas = sorted(list(set(dataset['train']['raga'])))
    num_classes = len(unique_ragas)
    print(f"Number of ragas: {num_classes}")
    print(f"Raga classes: {unique_ragas}")
    
    # Create mapping from raga name to index
    raga_to_id = {raga: idx for idx, raga in enumerate(unique_ragas)}
    
    # Add raga ID to dataset
    def add_raga_id(example):
        example['raga_id'] = raga_to_id[example['raga']]
        return example
    
    dataset = dataset.map(add_raga_id)
    
    # Split dataset into train/validation
    train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=config.get('random_seed', 42))
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, mask_ratio=config.get('mask_ratio', 0.15))
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, mask_ratio=config.get('mask_ratio', 0.15))
    )
    
    # Initialize model
    encoder_config = {
        'input_dim': 1,
        'hidden_dims': config.get('encoder_dims', [64, 128, 256, 512]),
        'kernel_size': config.get('kernel_size', 3),
        'stride': config.get('stride', 2),
        'dropout_rate': config.get('dropout_rate', 0.1)
    }
    
    lora_config = None
    if config.get('use_lora', False):
        lora_config = {
            'r': config.get('lora_r', 8),
            'alpha': config.get('lora_alpha', 32),
            'dropout': config.get('lora_dropout', 0.1)
        }
    
    model = SAMAUDIOModel(
        encoder_config=encoder_config,
        num_classes=num_classes,
        lora_config=lora_config
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 10),
        gamma=config.get('lr_gamma', 0.9)
    )
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(config.get('num_epochs', 50)):
        print(f'\nEpoch {epoch+1}/{config.get("num_epochs")}')
        print('-' * 10)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, None, epoch, config)
        
        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, None, config)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, config.get('best_model_path', 'best_model.pth'))
            print(f"New best model saved with accuracy: {val_acc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), config.get('final_model_path', 'final_model.pth'))
    
    # Push model to Hugging Face Hub if specified
    if config.get('push_to_hub', False):
        model.push_to_hub(config.get('model_name', 'sam-audio-carnatic-raga'))
    
    wandb.finish()
    print("Training completed!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SAM-Audio model for Carnatic raga classification')
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # Default config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'random_seed': args.seed,
        'encoder_dims': [64, 128, 256, 512],
        'kernel_size': 3,
        'stride': 2,
        'dropout_rate': 0.1,
        'mask_ratio': 0.15,
        'temperature': 0.07,
        'mask_weight': 0.5,
        'contrastive_weight': 0.3,
        'weight_decay': 0.01,
        'lr_step_size': 10,
        'lr_gamma': 0.9,
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'wandb_project': 'sam-audio-carnatic-raga',
        'run_name': 'sam-audio-rg',
        'best_model_path': 'best_model.pth',
        'final_model_path': 'final_model.pth',
        'push_to_hub': True,
        'model_name': 'sarayu/sam-carnatic'
    }
    
    # Override with YAML config if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
    
    main(config)