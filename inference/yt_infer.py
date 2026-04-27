"""
YouTube Raga Classifier

Downloads audio from any YouTube URL and runs raga inference.

Usage:
    python yt_infer.py https://www.youtube.com/watch?v=...
    python yt_infer.py https://www.youtube.com/watch?v=... --top-k 5
    python yt_infer.py https://www.youtube.com/watch?v=... --duration 120
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yt_dlp

sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model, load_audio, chunk_audio, detect_shruti, predict

import torch


def download_audio(url: str, out_path: str, duration: int | None = None) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    if duration:
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, duration)])
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", url)
        ext = info.get("ext", "webm")

    # rename to a predictable name so caller knows the path
    src = Path(out_path + f".{ext}")
    dst = Path(out_path + ".audio")
    src.rename(dst)

    return title, str(dst)


def main():
    parser = argparse.ArgumentParser(description="YouTube Raga Classifier")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions (default: 3)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Max seconds of audio to download (default: full video)")
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--step", type=int, default=10,
                        help="Cumulative prediction interval in seconds (default: 10)")
    args = parser.parse_args()

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
    print(f"Loading model...")
    model, config = load_model(device)
    id2label = config["id2label"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = str(Path(tmp_dir) / "audio")

        print(f"Downloading audio: {args.url}")
        if args.duration:
            print(f"  (first {args.duration}s only)")
        title, audio_path = download_audio(args.url, out_path, args.duration)
        print(f"  Title: {title}")

        if not Path(audio_path).exists():
            print("Error: audio download failed")
            sys.exit(1)

        waveform = load_audio(audio_path)

    import math
    SAMPLE_RATE = 16000
    duration = len(waveform) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s  |  Chunks: {len(chunk_audio(waveform))}")

    print("Detecting shruti (Sa frequency)...")
    detected_sa = detect_shruti(model, waveform, device)
    print(f"Detected Sa: {detected_sa:.1f} Hz")

    print("Running raga inference...")

    step_samples = args.step * SAMPLE_RATE if args.step > 0 else 0
    total_samples = len(waveform)

    if step_samples > 0 and total_samples > step_samples:
        checkpoints = list(range(step_samples, total_samples, step_samples))
        if checkpoints[-1] != total_samples:
            checkpoints.append(total_samples)

        top_k = min(args.top_k, len(id2label))
        print(f"\n{'─'*70}")
        print(f"  {'Time':<8s}", end="")
        for i in range(top_k):
            print(f"  {'#'+str(i+1):<26s}", end="")
        print(f"\n{'─'*70}")

        for end in checkpoints:
            sub = waveform[:end]
            secs = end / SAMPLE_RATE
            probs = predict(model, sub, detected_sa, device)
            sorted_idx = probs.argsort(descending=True)

            print(f"  {secs:5.0f}s  ", end="")
            for i in range(top_k):
                idx = sorted_idx[i].item()
                name = id2label[str(idx)]
                conf = probs[idx].item()
                col = f"{name} ({conf:.0%})"
                print(f"  {col:<26s}", end="")
            print()

        print(f"{'─'*70}")
        final_probs = probs
        final_sorted = sorted_idx
    else:
        final_probs = predict(model, waveform, detected_sa, device)
        final_sorted = final_probs.argsort(descending=True)
        top_k = min(args.top_k, len(final_probs))

    print(f"\n{'='*40}")
    print(f"  Title:          {title}")
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
