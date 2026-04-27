"""
Carnatic Raga Classifier — Dual-Path Inference Script

Pipeline:
1. Original audio → Path 2 → Detect Sa frequency
2. Normalize audio so Sa → fixed reference (261.63 Hz)
3. Normalized audio → Path 1 → Raga prediction

Usage:
    python inference.py path/to/audio.mp4
    python inference.py path/to/audio.mp4 --top-k 3
"""

import argparse
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import SAMAudioModel, ShrutiDetectionHead

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 320000  # 20 seconds at 16 kHz
TARGET_SA_HZ = 261.63   # Fixed reference for normalization

HF_REPO = "sarayusapa/sam-carnatic"
MODEL_DIR = Path(__file__).parent


def pitch_shift_waveform(waveform: torch.Tensor, shift_semitones: float) -> torch.Tensor:
    """Pitch-shift a waveform by resampling."""
    if shift_semitones == 0:
        return waveform
    rate = 2.0 ** (-shift_semitones / 12.0)
    indices = torch.arange(0, waveform.shape[0] * rate, rate, device=waveform.device)
    indices = indices.long().clamp(max=waveform.shape[0] - 1)
    shifted = waveform[indices]
    orig_len = waveform.shape[0]
    if shifted.shape[0] > orig_len:
        shifted = shifted[:orig_len]
    elif shifted.shape[0] < orig_len:
        shifted = F.pad(shifted, (0, orig_len - shifted.shape[0]))
    return shifted


def load_model(device: torch.device, repo: str = HF_REPO):
    config_path = hf_hub_download(repo_id=repo, filename="config.json")
    config = json.loads(Path(config_path).read_text())
    model = SAMAudioModel(
        encoder_config=config["encoder"],
        num_classes=config["num_classes"],
        num_segments=config["num_segments"],
        contrastive_temperature=config.get("contrastive_temperature", 0.07),
    )
    local_pth = MODEL_DIR / "best_model.pth"
    try:
        weights_path = hf_hub_download(repo_id=repo, filename="best_model.pth")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
    except Exception:
        if local_pth.exists():
            checkpoint = torch.load(str(local_pth), map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"]
        else:
            weights_path = hf_hub_download(repo_id=repo, filename="model.safetensors")
            state_dict = load_file(weights_path, device=str(device))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config


def load_audio(path: str) -> np.ndarray:
    """Load any audio/video file and return mono waveform at 16 kHz."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def chunk_audio(waveform: np.ndarray) -> list[np.ndarray]:
    """Split waveform into 20-second chunks, padding the last one if needed."""
    if len(waveform) <= CHUNK_SAMPLES:
        padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        padded[: len(waveform)] = waveform
        return [padded]

    chunks = []
    for start in range(0, len(waveform), CHUNK_SAMPLES):
        chunk = waveform[start : start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded
        chunks.append(chunk)
    return chunks


@torch.no_grad()
def detect_shruti(model, waveform: np.ndarray, device: torch.device) -> float:
    """
    Path 2: Detect Sa frequency from original audio.
    Averages predictions across chunks and returns Hz.
    """
    chunks = chunk_audio(waveform)
    all_semitones = []

    for chunk in chunks:
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        out = model(input_audio=None, input_audio_original=tensor)
        all_semitones.append(out["predicted_shruti_semitones"])

    avg_semitones = torch.stack(all_semitones).mean().item()
    return ShrutiDetectionHead.semitones_to_hz(torch.tensor(avg_semitones)).item()


@torch.no_grad()
def predict(model, waveform: np.ndarray, detected_sa_hz: float,
            device: torch.device) -> torch.Tensor:
    """
    Full inference pipeline:
    1. Normalize audio (shift detected Sa → TARGET_SA_HZ)
    2. Path 1: classify raga on normalized audio
    """
    # Normalize: shift from detected Sa to reference
    normalize_semitones = 12.0 * math.log2(TARGET_SA_HZ / detected_sa_hz)
    waveform_tensor = torch.from_numpy(waveform).float()
    normalized = pitch_shift_waveform(waveform_tensor, normalize_semitones).numpy()

    chunks = chunk_audio(normalized)
    all_probs = []

    for chunk in chunks:
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        out = model(input_audio=tensor)
        probs = torch.softmax(out["raga_logits"], dim=-1)
        all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Carnatic Raga Classifier")
    parser.add_argument("audio", help="Path to audio/video file (mp4, mp3, wav, …)")
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top predictions to show (default: 3)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: cpu, cuda, mps, or auto (default: auto)",
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help="Cumulative prediction interval in seconds (default: 10). "
             "Set to 0 to disable cumulative output.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {HF_REPO}...")
    model, config = load_model(device)
    id2label = config["id2label"]

    # Load audio
    print(f"Loading audio: {audio_path}")
    waveform = load_audio(str(audio_path))
    duration = len(waveform) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s  |  Chunks: {len(chunk_audio(waveform))}")

    # Step 1: Detect Sa frequency
    print("Detecting shruti (Sa frequency)...")
    detected_sa = detect_shruti(model, waveform, device)
    print(f"Detected Sa: {detected_sa:.1f} Hz")

    # Step 2 & 3: Normalize and predict
    print("Running raga inference...")

    step_samples = args.step * SAMPLE_RATE if args.step > 0 else 0
    total_samples = len(waveform)

    if step_samples > 0 and total_samples > step_samples:
        checkpoints = list(range(step_samples, total_samples, step_samples))
        if checkpoints[-1] != total_samples:
            checkpoints.append(total_samples)

        top_k = min(args.top_k, len(id2label))
        header_label = "Time"
        header_cols = [f"#{i+1}" for i in range(top_k)]
        print(f"\n{'─'*70}")
        print(f"  {header_label:<8s}", end="")
        for c in header_cols:
            print(f"  {c:<26s}", end="")
        print(f"\n{'─'*70}")

        for end in checkpoints:
            sub = waveform[:end]
            secs = end / SAMPLE_RATE
            probs = predict(model, sub, detected_sa, device)
            sorted_idx = probs.argsort(descending=True)

            label_str = f"{secs:5.0f}s"
            print(f"  {label_str:<8s}", end="")
            for i in range(top_k):
                idx = sorted_idx[i].item()
                name = id2label[str(idx)]
                conf = probs[idx].item()
                print(f"  {name} ({conf:.0%}){'':<1s}", end="")
                used = len(f"{name} ({conf:.0%})") + 1
                print(" " * max(0, 26 - used), end="")
            print()

        print(f"{'─'*70}")

        final_probs = probs
        final_sorted = sorted_idx
    else:
        final_probs = predict(model, waveform, detected_sa, device)
        final_sorted = final_probs.argsort(descending=True)
        top_k = min(args.top_k, len(final_probs))

    # Final result
    print(f"\n{'='*40}")
    print(f"  Detected Sa:    {detected_sa:.1f} Hz")
    print(f"  Predicted Raga: {id2label[str(final_sorted[0].item())]}")
    print(f"  Confidence:     {final_probs[final_sorted[0]].item():.1%}")
    print(f"{'='*40}")

    if top_k > 1:
        print(f"\nTop-{top_k} predictions:")
        for i in range(top_k):
            idx = final_sorted[i].item()
            label = id2label[str(idx)]
            prob = final_probs[idx].item()
            print(f"  {i+1}. {label:20s} {prob:6.1%}")
    print()


if __name__ == "__main__":
    main()
