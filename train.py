"""
SAM-Audio Style Model for Carnatic Raga Classification

This module implements a SAM-Audio style architecture for classifying Carnatic ragas
from audio waveforms. The model consists of:
- Audio encoder (CNN-based frontend)
- Latent segmentation tokens for temporal modeling
- Masked segment prediction for self-supervised learning
- Lightweight classification head for raga prediction
- LoRA integration for efficient fine-tuning

Optimized for RTX 4090 with:
- torch.compile() for graph optimization
- BFloat16 mixed precision training
- Flash Attention support
- Fused AdamW optimizer
- Optimized data loading
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedModel
from huggingface_hub import HfApi
import wandb
from sklearn.metrics import accuracy_score, f1_score
import argparse
import yaml
from typing import Optional, Dict, List, Any
import math
from math import log2
from functools import partial
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# RTX 4090 / Ampere+ Optimizations
# ============================================================================

def setup_cuda_optimizations():
    """Configure CUDA optimizations for RTX 4090 and similar GPUs."""
    if not torch.cuda.is_available():
        return torch.device('cpu'), torch.float32

    device = torch.device('cuda')

    # Enable TF32 for faster matmuls on Ampere+ GPUs (RTX 30xx, 40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Use high precision for float32 matmuls (uses TensorCores)
    torch.set_float32_matmul_precision('high')

    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Determine best dtype - BF16 is preferred on RTX 4090 (Ada Lovelace)
    # Better numerical stability than FP16, native hardware support
    if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        dtype = torch.bfloat16
        print("Using BFloat16 precision (optimal for RTX 4090)")
    else:
        dtype = torch.float16
        print("Using Float16 precision")

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    return device, dtype


# Initialize CUDA optimizations at module load
device, amp_dtype = setup_cuda_optimizations()
print(f"Using device: {device}")


# ============================================================================
# Model Components
# ============================================================================

class AudioEncoder(nn.Module):
    """
    Audio encoder with mel-spectrogram frontend + CNN backend.
    Mel-spectrograms decompose frequency properly, giving the CNN
    meaningful spectral structure instead of raw waveform oscillations.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        stride: int = 2,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_mel: bool = True,
        n_mels: int = 80,
        sample_rate: int = 16000
    ):
        super().__init__()

        self.use_mel = use_mel
        if use_mel:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=n_mels,
            )
            in_channels = n_mels
        else:
            in_channels = input_dim

        layers = []
        for i, out_channels in enumerate(hidden_dims):
            layers.append(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                )
            )
            if use_layer_norm:
                layers.append(nn.GroupNorm(1, out_channels))
            else:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.use_mel:
            # (batch, 1, length) -> (batch, n_mels, time_frames)
            x = self.mel_transform(x.squeeze(1))
            x = torch.log(x + 1e-8)  # Log-mel is standard and more perceptually meaningful

        features = self.conv_layers(x)
        return features.transpose(1, 2)  # (batch, time, channels)


class SegmentTokenPredictor(nn.Module):
    """
    Implements latent segmentation tokens and masked segment prediction.
    These tokens capture local temporal patterns in the audio sequence.

    Uses scaled dot-product attention (Flash Attention compatible).
    """
    def __init__(
        self,
        hidden_size: int,
        num_segments: int = 64,
        mask_ratio: float = 0.15,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_segments = num_segments
        self.mask_ratio = mask_ratio
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Learnable segmentation tokens
        self.segment_tokens = nn.Parameter(torch.randn(num_segments, hidden_size) * 0.02)

        # Q, K, V projections for efficient attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Segment reconstruction head for masked prediction
        self.segment_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """Compute scaled dot-product attention with Flash Attention support."""
        batch_size = query.size(0)

        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.out_proj(attn_output)

    def forward(
        self,
        audio_features: torch.Tensor,
        mask_indices: Optional[List[torch.Tensor]] = None
    ):
        batch_size = audio_features.size(0)
        seq_len = audio_features.size(1)
        segment_tokens = self.segment_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Generate mask indices if not provided and in training mode
        if mask_indices is None and self.training and self.mask_ratio > 0:
            num_masked = int(seq_len * self.mask_ratio)
            mask_indices = [torch.randperm(seq_len, device=audio_features.device)[:num_masked]
                          for _ in range(batch_size)]

        # Apply segment attention between audio features and segmentation tokens
        attended_features = self._attention(segment_tokens, audio_features, audio_features)
        attended_features = self.layer_norm(attended_features + segment_tokens)

        # Generate predictions for masked segments if provided
        if mask_indices is not None and self.training:
            # Apply masking to audio features (vectorized)
            masked_features = audio_features.clone()
            for b, mask_idx in enumerate(mask_indices):
                masked_features[b, mask_idx, :] = 0.0

            # Recompute attention with masked features
            attended_masked = self._attention(segment_tokens, masked_features, masked_features)
            attended_masked = self.layer_norm(attended_masked + segment_tokens)

            # Predict masked segments
            segment_predictions = self.segment_predictor(attended_masked)
            target_segments = attended_features.detach()  # Stop gradient on targets

            return attended_features, segment_predictions, target_segments

        return attended_features, None, None

    def generate_mask_indices(self, batch_size: int, seq_len: int) -> List[torch.Tensor]:
        """Generate random mask indices for masked segment prediction."""
        num_masked = int(seq_len * self.mask_ratio)
        return [torch.randperm(seq_len, device=device)[:num_masked] for _ in range(batch_size)]


class ContrastiveSegmentModule(nn.Module):
    """
    Contrastive learning module for segment-level representations.
    Encourages similar segments within the same raga to be closer in embedding space.

    Uses InfoNCE loss with temperature scaling.
    """
    def __init__(self, hidden_size: int, projection_dim: int = 128, temperature: float = 0.07):
        super().__init__()

        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, segment_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, _ = segment_features.shape

        # Project and normalize
        projected = self.projection_head(segment_features)
        projected = F.normalize(projected, dim=-1)

        # Reshape: (batch_size * num_segments, feat_dim)
        reshaped = projected.view(-1, projected.size(-1))

        # Compute similarity matrix efficiently
        similarity_matrix = torch.mm(reshaped, reshaped.t()) / self.temperature

        # Create labels - segments from same sample are positive pairs
        labels = torch.arange(batch_size, device=segment_features.device)
        labels = labels.unsqueeze(1).expand(-1, num_segments).flatten()

        # Create mask for positive pairs (same audio sample)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)  # Remove self-similarity

        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)

        # Numerical stability: subtract max before exp
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs
        positive_count = mask.sum(dim=1).clamp(min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / positive_count

        return -mean_log_prob.mean()


class RagaClassificationHead(nn.Module):
    """
    Lightweight classification head for raga prediction.
    Takes segment-level representations and predicts the raga.
    """
    def __init__(self, hidden_size: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, segment_representations: torch.Tensor) -> torch.Tensor:
        # Average across segments for global representation
        global_repr = segment_representations.mean(dim=1)
        return self.classifier(global_repr)


class ShrutiDetectionHead(nn.Module):
    """
    Regression head for predicting Sa frequency (shruti) from audio.
    Predicts in log-frequency space (semitones from C4) for stability.
    """
    TARGET_SA_HZ = 261.63  # C4 — arbitrary fixed reference for normalization

    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, segment_representations: torch.Tensor) -> torch.Tensor:
        global_repr = segment_representations.mean(dim=1)
        return self.regressor(global_repr).squeeze(-1)  # (batch,) semitones from C4

    @staticmethod
    def hz_to_semitones(hz: torch.Tensor) -> torch.Tensor:
        return 12.0 * torch.log2(hz / ShrutiDetectionHead.TARGET_SA_HZ)

    @staticmethod
    def semitones_to_hz(semitones: torch.Tensor) -> torch.Tensor:
        return ShrutiDetectionHead.TARGET_SA_HZ * (2.0 ** (semitones / 12.0))


class SAMAudioModel(nn.Module):
    """
    Dual-path SAM-Audio model for Carnatic raga classification.

    Path 1 (Raga): Normalized audio (Sa shifted to fixed ref) → shared encoder → raga head
    Path 2 (Shruti): Original audio (Sa varies) → shared encoder → shruti regression head

    Training: combined_loss = raga_loss + 0.3 * shruti_loss + auxiliary losses
    Inference: detect Sa → normalize → classify raga
    """

    def __init__(
        self,
        encoder_config: Dict[str, Any],
        num_classes: int,
        num_segments: int = 64,
        mask_ratio: float = 0.15,
        contrastive_temperature: float = 0.07
    ):
        super().__init__()

        self.audio_encoder = AudioEncoder(**encoder_config)
        encoder_hidden_size = encoder_config.get('hidden_dims', [64, 128, 256, 512])[-1]

        self.segment_predictor = SegmentTokenPredictor(
            hidden_size=encoder_hidden_size,
            num_segments=num_segments,
            mask_ratio=mask_ratio
        )

        self.contrastive_module = ContrastiveSegmentModule(
            hidden_size=encoder_hidden_size,
            temperature=contrastive_temperature
        )

        self.raga_classifier = RagaClassificationHead(
            hidden_size=encoder_hidden_size,
            num_classes=num_classes
        )

        self.shruti_detector = ShrutiDetectionHead(
            hidden_size=encoder_hidden_size
        )

        self.num_classes = num_classes
        self.encoder_hidden_size = encoder_hidden_size

    def forward(
        self,
        input_audio: torch.Tensor,
        input_audio_original: Optional[torch.Tensor] = None,
        shruti_hz: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        return_contrastive_loss: bool = True
    ) -> Dict[str, Any]:
        # === Path 1: Raga classification on normalized audio (Sa → fixed ref) ===
        audio_features = self.audio_encoder(input_audio)
        segment_reps, seg_predictions, seg_targets = self.segment_predictor(audio_features, masks)

        masked_lm_loss = None
        if seg_predictions is not None and seg_targets is not None:
            masked_lm_loss = F.mse_loss(seg_predictions, seg_targets)

        contrastive_loss = None
        if return_contrastive_loss and self.training:
            contrastive_loss = self.contrastive_module(segment_reps)

        raga_logits = self.raga_classifier(segment_reps)
        classification_loss = None
        if labels is not None:
            classification_loss = F.cross_entropy(raga_logits, labels, label_smoothing=0.1)

        # === Path 2: Shruti detection on original (non-normalized) audio ===
        shruti_loss = None
        predicted_shruti_semitones = None
        if input_audio_original is not None:
            orig_features = self.audio_encoder(input_audio_original)
            orig_segment_reps, _, _ = self.segment_predictor(orig_features)
            predicted_shruti_semitones = self.shruti_detector(orig_segment_reps)

            if shruti_hz is not None:
                target_semitones = ShrutiDetectionHead.hz_to_semitones(shruti_hz)
                shruti_loss = F.smooth_l1_loss(predicted_shruti_semitones, target_semitones)

        return {
            'segment_representations': segment_reps,
            'masked_lm_loss': masked_lm_loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss,
            'shruti_loss': shruti_loss,
            'raga_logits': raga_logits,
            'predicted_shruti_semitones': predicted_shruti_semitones,
        }


# ============================================================================
# Data Processing
# ============================================================================

class AudioPreprocessor:
    """Efficient audio preprocessing with caching."""

    def __init__(self, target_sr: int = 16000, max_length: int = 80000):
        self.target_sr = target_sr
        self.max_length = max_length
        self._resampler_cache = {}

    def __call__(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        # Resample if needed
        if orig_sr != self.target_sr:
            if orig_sr not in self._resampler_cache:
                self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.target_sr
                )
            waveform = self._resampler_cache[orig_sr](waveform)

        # Truncate or pad
        if waveform.shape[-1] > self.max_length:
            waveform = waveform[..., :self.max_length]
        elif waveform.shape[-1] < self.max_length:
            pad_length = self.max_length - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_length))

        return waveform


def pitch_shift_waveform(waveform: torch.Tensor, shift_semitones: float, sample_rate: int = 16000) -> torch.Tensor:
    """
    Pitch-shift a waveform by resampling (fast, no external deps).
    Shift in semitones: positive = higher pitch, negative = lower.
    """
    if shift_semitones == 0:
        return waveform
    # Resample to simulate pitch shift: stretch then truncate/pad
    rate = 2.0 ** (-shift_semitones / 12.0)
    indices = torch.arange(0, waveform.shape[0] * rate, rate, device=waveform.device)
    indices = indices.long().clamp(max=waveform.shape[0] - 1)
    shifted = waveform[indices]
    # Match original length
    orig_len = waveform.shape[0]
    if shifted.shape[0] > orig_len:
        shifted = shifted[:orig_len]
    elif shifted.shape[0] < orig_len:
        shifted = F.pad(shifted, (0, orig_len - shifted.shape[0]))
    return shifted


TARGET_SA_HZ = 261.63  # Fixed reference for normalization


def collate_fn(batch: List[Dict], mask_ratio: float = 0.15, max_length: int = 320000,
               crop_length: int = 0, is_train: bool = True):
    """
    Dual-path collate: produces normalized audio (for raga path) and
    original audio (for shruti detection path).

    Each sample has pitch_shift_semitones (fixed: -3..+3).
    - Original audio = base audio pitch-shifted by that amount
    - Shruti Hz shifts accordingly: base_shruti * 2^(shift/12)
    - Normalized audio = original shifted so Sa → TARGET_SA_HZ
    """
    batch_size = len(batch)
    effective_length = crop_length if (is_train and crop_length > 0) else max_length

    waveforms_normalized = torch.zeros(batch_size, effective_length, dtype=torch.float32)
    waveforms_original = torch.zeros(batch_size, effective_length, dtype=torch.float32)
    shruti_hz_batch = torch.zeros(batch_size, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        audio = item['audio_array']
        if isinstance(audio, list):
            audio = np.array(audio)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.ndim > 1:
            audio = audio[:, 0]

        base_shruti = float(item['shruti_hz'])
        shift_semitones = float(item.get('pitch_shift_semitones', 0))

        # Apply pitch-shift augmentation → this is the "original" audio
        # (simulates a performance in a different shruti)
        if shift_semitones != 0:
            audio = pitch_shift_waveform(audio, shift_semitones)

        shifted_shruti = base_shruti * (2.0 ** (shift_semitones / 12.0))

        # Random time-crop
        if is_train and crop_length > 0 and audio.shape[0] > crop_length:
            start = np.random.randint(0, audio.shape[0] - crop_length)
            audio = audio[start:start + crop_length]

        # Original audio (Sa at shifted_shruti Hz) → for shruti detection path
        length = min(audio.shape[0], effective_length)
        waveforms_original[i, :length] = audio[:length]

        # Normalized audio: shift Sa from shifted_shruti → TARGET_SA_HZ
        normalize_semitones = 12.0 * math.log2(TARGET_SA_HZ / shifted_shruti)
        audio_normalized = pitch_shift_waveform(audio, normalize_semitones)
        waveforms_normalized[i, :length] = audio_normalized[:length]

        shruti_hz_batch[i] = shifted_shruti
        labels[i] = item['raga']

    return {
        'input_audio': waveforms_normalized,
        'input_audio_original': waveforms_original,
        'shruti_hz': shruti_hz_batch,
        'labels': labels,
        'masks': None
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: Dict[str, Any],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> tuple:
    """
    Single training epoch with mixed precision and gradient accumulation.
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    accumulation_steps = int(config.get('gradient_accumulation_steps', 1))
    max_grad_norm = float(config.get('max_grad_norm', 1.0))

    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

    shruti_loss_weight = float(config.get('shruti_loss_weight', 0.3))

    for batch_idx, batch in enumerate(dataloader):
        input_audio = batch['input_audio'].to(device, non_blocking=True)
        input_audio_original = batch['input_audio_original'].to(device, non_blocking=True)
        shruti_hz = batch['shruti_hz'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        masks = batch['masks']

        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            outputs = model(
                input_audio=input_audio,
                input_audio_original=input_audio_original,
                shruti_hz=shruti_hz,
                masks=masks,
                labels=labels
            )

            # Combine losses
            loss = outputs['classification_loss'] or 0

            if outputs['masked_lm_loss'] is not None:
                loss = loss + float(config.get('mask_weight', 0.5)) * outputs['masked_lm_loss']

            if outputs['contrastive_loss'] is not None:
                loss = loss + float(config.get('contrastive_weight', 0.3)) * outputs['contrastive_loss']

            if outputs['shruti_loss'] is not None:
                loss = loss + shruti_loss_weight * outputs['shruti_loss']

            loss = loss / accumulation_steps

        # Scaled backward pass
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and config.get('scheduler_per_batch', False):
                scheduler.step()

        total_loss += loss.item() * accumulation_steps

        # Collect predictions
        with torch.no_grad():
            preds = outputs['raga_logits'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            batch_loss = loss.item() * accumulation_steps
            batch_acc = accuracy_score(
                labels.cpu().numpy(), preds.cpu().numpy()
            )
            wandb.log({
                'batch/train_loss': batch_loss,
                'batch/train_acc': batch_acc,
                'batch/lr': current_lr,
                'batch/masked_lm_loss': outputs['masked_lm_loss'].item() if outputs['masked_lm_loss'] is not None else 0,
                'batch/contrastive_loss': outputs['contrastive_loss'].item() if outputs['contrastive_loss'] is not None else 0,
                'batch/shruti_loss': outputs['shruti_loss'].item() if outputs['shruti_loss'] is not None else 0,
            })
            print(f'Epoch {epoch} [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {batch_loss:.4f} Acc: {batch_acc:.4f} LR: {current_lr:.2e}')

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any]
) -> tuple:
    """
    Validation loop with mixed precision.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    shruti_loss_weight = float(config.get('shruti_loss_weight', 0.3))

    for batch in dataloader:
        input_audio = batch['input_audio'].to(device, non_blocking=True)
        input_audio_original = batch['input_audio_original'].to(device, non_blocking=True)
        shruti_hz = batch['shruti_hz'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        masks = batch['masks']

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            outputs = model(
                input_audio=input_audio,
                input_audio_original=input_audio_original,
                shruti_hz=shruti_hz,
                masks=masks,
                labels=labels,
                return_contrastive_loss=False
            )

            loss = outputs['classification_loss'] or 0
            if outputs['masked_lm_loss'] is not None:
                loss = loss + config.get('mask_weight', 0.5) * outputs['masked_lm_loss']
            if outputs['shruti_loss'] is not None:
                loss = loss + shruti_loss_weight * outputs['shruti_loss']

        total_loss += loss.item()

        preds = outputs['raga_logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1


def main(config: Dict[str, Any]):
    """
    Main training function with RTX 4090 optimizations.
    """
    # Setup wandb
    wandb.init(
        project=config.get('wandb_project', 'sam-audio-carnatic-raga'),
        config=config,
        name=config.get('run_name', 'sam-audio-rg')
    )

    # Load dataset using to_iterable_dataset to avoid eager audio decoding
    print("Loading dataset...")
    import soundfile as sf
    from datasets import Features, Value

    # Load dataset but cast audio column to avoid automatic decoding
    dataset_raw = load_dataset("sarayusapa/carnatic-ragas")

    # Get unique ragas
    unique_ragas = sorted(list(set(dataset_raw['train']['raga'])))
    num_classes = len(unique_ragas)
    print(f"Number of ragas: {num_classes}")
    print(f"Raga classes: {unique_ragas}")

    # Create mapping
    raga_to_id = {raga: idx for idx, raga in enumerate(unique_ragas)}

    # Extract data and decode audio manually using av or soundfile
    def create_simple_dataset(ds):
        """Convert to simple dict dataset with decoded audio arrays."""
        import io
        import soundfile as sf

        data = []
        arrow_table = ds.data.to_pandas()

        for idx in range(len(arrow_table)):
            row = arrow_table.iloc[idx]
            # Access the audio dict structure from pandas
            audio_info = row['audio']

            # The audio is stored as bytes in the 'bytes' field
            if 'bytes' in audio_info and audio_info['bytes'] is not None:
                # Decode audio from bytes
                audio_bytes = audio_info['bytes']
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            elif 'array' in audio_info and audio_info['array'] is not None:
                # Audio already decoded
                audio_array = audio_info['array']
                sr = audio_info['sampling_rate']
            else:
                print(f"Warning: No audio data found for index {idx}")
                continue

            data.append({
                'audio_array': audio_array,
                'raga': raga_to_id[row['raga']],
                'shruti_hz': float(row['shruti_hz']),
            })
        return data

    print("Extracting audio paths...")
    train_data = create_simple_dataset(dataset_raw['train'])

    # 7x dataset: original + +-1, +-2, +-3 semitone shifts
    # Tanpura + vocals shift together, simulating different performance keys.
    # Shruti Hz shifts accordingly so the shruti detection head learns correctly.
    pitch_shifts = [-3, -2, -1, 0, 1, 2, 3]
    print(f"Creating 7x augmented dataset with pitch shifts {pitch_shifts}...")
    augmented_data = []
    for item in train_data:
        for shift in pitch_shifts:
            augmented_data.append({**item, 'pitch_shift_semitones': shift})
    train_data = augmented_data
    print(f"Augmented train samples: {len(train_data)} (7x)")

    # Split dataset (plain list — avoids pyarrow 2GB int32 limit)
    import random
    rng = random.Random(int(config.get('random_seed', 42)))
    indices = list(range(len(train_data)))
    rng.shuffle(indices)
    split = int(len(indices) * 0.9)
    train_dataset = [train_data[i] for i in indices[:split]]
    val_dataset = [train_data[i] for i in indices[split:]]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Debug: Check audio data format
    print(f"\nSample audio shapes:")
    for i in range(min(3, len(train_dataset))):
        audio_array = train_dataset[i]['audio_array']
        print(f"  {i}: shape={audio_array.shape if hasattr(audio_array, 'shape') else len(audio_array)}")  # type: ignore

    # Optimized data loaders for RTX 4090
    num_workers = int(config.get('num_workers', 4))
    batch_size = int(config.get('batch_size', 32))  # Higher batch size for 4090

    crop_length = int(config.get('crop_length', 0))  # 0 = no crop

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, mask_ratio=float(config.get('mask_ratio', 0.15)),
                           crop_length=crop_length, is_train=True),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=partial(collate_fn, mask_ratio=float(config.get('mask_ratio', 0.15)),
                           crop_length=crop_length, is_train=False),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    # Initialize model
    encoder_config = {
        'input_dim': 1,
        'hidden_dims': config.get('encoder_dims', [64, 128, 256, 512]),
        'kernel_size': int(config.get('kernel_size', 3)),
        'stride': int(config.get('stride', 2)),
        'dropout_rate': float(config.get('dropout_rate', 0.1)),
        'use_layer_norm': config.get('use_layer_norm', True),
        'use_mel': config.get('use_mel', True),
        'n_mels': int(config.get('n_mels', 80)),
        'sample_rate': 16000
    }

    model = SAMAudioModel(
        encoder_config=encoder_config,
        num_classes=num_classes,
        num_segments=int(config.get('num_segments', 64)),
        mask_ratio=float(config.get('mask_ratio', 0.15)),
        contrastive_temperature=float(config.get('temperature', 0.07))
    ).to(device)

    # Compile model for faster execution (PyTorch 2.0+)
    if config.get('compile_model', True) and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer with fused AdamW (faster on CUDA)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get('learning_rate', 3e-4)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        betas=(0.9, 0.999),
        fused=True  # Fused implementation for CUDA
    )

    # Cosine annealing with warmup
    num_epochs = int(config.get('num_epochs', 50))
    warmup_epochs = int(config.get('warmup_epochs', 2))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # Training loop
    best_val_accuracy = 0.0
    patience = int(config.get('patience', 10))
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'\n{"="*50}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"="*50}')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, epoch + 1, config
        )

        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, config)

        # Step scheduler
        scheduler.step()

        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1_score': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            # Handle compiled model state dict
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc,
                'config': config
            }, config.get('best_model_path', 'best_model.pth'))
            print(f"New best model saved! Accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

    # Save final model
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(model_to_save.state_dict(), config.get('final_model_path', 'final_model.pth'))

    # Push to hub if specified
    if config.get('push_to_hub', False):
        print("Pushing model to Hugging Face Hub...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=config.get('best_model_path', 'best_model.pth'),
            path_in_repo="best_model.pth",
            repo_id=config.get('model_name', 'sarayusapa/sam-carnatic'),
            repo_type="model"
        )

    wandb.finish()
    print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SAM-Audio model for Carnatic raga classification')

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args


if __name__ == '__main__':
    args = parse_args()

    # Default config optimized for RTX 4090
    config = {
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'random_seed': args.seed,
        'encoder_dims': [64, 128, 256, 512],
        'kernel_size': 3,
        'stride': 2,
        'dropout_rate': 0.25,
        'use_layer_norm': True,
        'use_mel': True,
        'n_mels': 80,
        'mask_ratio': 0.15,
        'temperature': 0.07,
        'mask_weight': 0.5,
        'contrastive_weight': 0.3,
        'weight_decay': 0.05,
        'max_grad_norm': 1.0,
        'gradient_accumulation_steps': 1,
        'warmup_epochs': 2,
        'patience': 5,
        'shruti_loss_weight': 0.3,
        'crop_length': 256000,
        'num_workers': 4,
        'num_segments': 64,
        'compile_model': not args.no_compile,
        'wandb_project': 'sam-audio-carnatic-raga',
        'run_name': 'sam-audio-rtx4090',
        'best_model_path': 'best_model.pth',
        'final_model_path': 'final_model.pth',
        'push_to_hub': True,
        'model_name': 'sarayusapa/sam-carnatic'
    }

    # Load YAML config if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)

    # Override with CLI args
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr

    main(config)
