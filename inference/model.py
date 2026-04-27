"""
SAM-Audio Model — inference architecture.
Matches train.py from sarayusapa/sam-carnatic exactly.

Path 1: Normalized audio → shared encoder → raga classification
Path 2: Original audio → shared encoder → shruti detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class AudioEncoder(nn.Module):
    def __init__(self, encoder_config):
        super().__init__()
        self.use_mel = encoder_config.get("use_mel", True)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=encoder_config["sample_rate"],
            n_fft=1024,
            hop_length=256,
            n_mels=encoder_config["n_mels"],
        )

        in_channels = encoder_config["n_mels"]
        layers = []
        for out_channels in encoder_config["hidden_dims"]:
            layers.append(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=encoder_config["kernel_size"],
                          stride=encoder_config["stride"],
                          padding=encoder_config["kernel_size"] // 2)
            )
            if encoder_config.get("use_layer_norm", True):
                layers.append(nn.GroupNorm(1, out_channels))
            else:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(encoder_config["dropout_rate"]))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.use_mel:
            x = self.mel_transform(x.squeeze(1))       # (B, n_mels, T_mel)
            x = torch.log(x + 1e-8)                    # log-mel spectrogram

        features = self.conv_layers(x)                  # (B, C, T_enc)
        return features.transpose(1, 2)                 # (B, T_enc, C)


class SegmentTokenPredictor(nn.Module):
    def __init__(self, hidden_size, num_segments=64, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.segment_tokens = nn.Parameter(torch.randn(num_segments, hidden_size) * 0.02)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.segment_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, audio_features):
        batch_size = audio_features.size(0)
        segment_tokens = self.segment_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        q = self.q_proj(segment_tokens).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(audio_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(audio_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        attended_features = self.out_proj(attn_output)

        attended_features = self.layer_norm(attended_features + segment_tokens)

        return attended_features, None, None


class RagaClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, segment_representations):
        global_repr = segment_representations.mean(dim=1)
        return self.classifier(global_repr)


class ShrutiDetectionHead(nn.Module):
    """Predicts Sa frequency in semitones-from-C4 (log-freq space)."""
    TARGET_SA_HZ = 261.63

    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, segment_representations):
        global_repr = segment_representations.mean(dim=1)
        return self.regressor(global_repr).squeeze(-1)

    @staticmethod
    def semitones_to_hz(semitones):
        return ShrutiDetectionHead.TARGET_SA_HZ * (2.0 ** (semitones / 12.0))


class ContrastiveSegmentModule(nn.Module):
    def __init__(self, hidden_size, projection_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
        )


class SAMAudioModel(nn.Module):
    """
    Dual-path SAM-Audio model.
    Path 1: normalized audio → raga classification
    Path 2: original audio → shruti detection
    """
    def __init__(self, encoder_config, num_classes, num_segments=64,
                 mask_ratio=0.15, contrastive_temperature=0.07):
        super().__init__()
        self.audio_encoder = AudioEncoder(encoder_config)
        encoder_hidden_size = encoder_config.get("hidden_dims", [64, 128, 256, 512])[-1]

        self.segment_predictor = SegmentTokenPredictor(
            hidden_size=encoder_hidden_size,
            num_segments=num_segments,
        )
        self.contrastive_module = ContrastiveSegmentModule(
            hidden_size=encoder_hidden_size,
            temperature=contrastive_temperature,
        )
        self.raga_classifier = RagaClassificationHead(
            hidden_size=encoder_hidden_size,
            num_classes=num_classes,
        )
        self.shruti_detector = ShrutiDetectionHead(
            hidden_size=encoder_hidden_size,
        )

    def forward(self, input_audio, input_audio_original=None, **kwargs):
        result = {}

        if input_audio is not None:
            audio_features = self.audio_encoder(input_audio)
            segment_reps, _, _ = self.segment_predictor(audio_features)
            result["raga_logits"] = self.raga_classifier(segment_reps)

        if input_audio_original is not None:
            orig_features = self.audio_encoder(input_audio_original)
            orig_segment_reps, _, _ = self.segment_predictor(orig_features)
            result["predicted_shruti_semitones"] = self.shruti_detector(orig_segment_reps)

        return result
