"""
Download first 60 seconds of YouTube videos and run raga inference on each.

Usage:
    python download_and_infer.py
"""

import subprocess
import sys
from pathlib import Path

VENV_PYTHON = str(Path(__file__).parent / "venv" / "bin" / "python3")
YT_DLP = str(Path(__file__).parent / "venv" / "bin" / "yt-dlp")
INFERENCE_SCRIPT = str(Path(__file__).parent / "inference.py")
OUT_DIR = Path(__file__).parent.parent / "samples" / "youtube"

VIDEOS = [
    ("jhnrib44RLM", "https://www.youtube.com/watch?v=jhnrib44RLM"),
    ("N-V6GjLUMd0", "https://www.youtube.com/watch?v=N-V6GjLUMd0"),
    ("xg3tN-TVkQI", "https://www.youtube.com/watch?v=xg3tN-TVkQI"),
    ("_N26NtypvnQ", "https://www.youtube.com/watch?v=_N26NtypvnQ"),
    ("0cnRGPxIpr4", "https://www.youtube.com/watch?v=0cnRGPxIpr4"),
    ("qgehU_o-HuI", "https://www.youtube.com/watch?v=qgehU_o-HuI"),
    ("8_5U97Hiwqg", "https://www.youtube.com/watch?v=8_5U97Hiwqg"),
    ("N3O10qr3Afk", "https://www.youtube.com/watch?v=N3O10qr3Afk"),
]


def download_video(video_id: str, url: str) -> Path:
    """Download first 60s of audio from a YouTube video."""
    out_path = OUT_DIR / f"{video_id}.wav"
    if out_path.exists():
        print(f"  Already downloaded: {out_path.name}")
        return out_path

    print(f"  Downloading {video_id}...")
    # Download audio only, use ffmpeg to extract first 60 seconds as wav
    subprocess.run(
        [
            YT_DLP,
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-t 60 -ac 1 -ar 16000",
            "-o", str(out_path),
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_path


def run_inference(audio_path: Path):
    """Run raga inference on an audio file."""
    result = subprocess.run(
        [VENV_PYTHON, INFERENCE_SCRIPT, str(audio_path), "--top-k", "3"],
        capture_output=True,
        text=True,
    )
    # Print stdout, skip stderr warnings
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Step 1: Downloading first 60s of each YouTube video")
    print("=" * 60)

    downloaded = []
    for video_id, url in VIDEOS:
        try:
            path = download_video(video_id, url)
            downloaded.append((video_id, path))
        except subprocess.CalledProcessError as e:
            print(f"  FAILED to download {video_id}: {e.stderr[-200:]}")

    print(f"\nDownloaded {len(downloaded)}/{len(VIDEOS)} videos.\n")

    print("=" * 60)
    print("  Step 2: Running raga inference")
    print("=" * 60)

    for video_id, path in downloaded:
        print(f"\n--- {video_id} ({path.name}) ---")
        run_inference(path)


if __name__ == "__main__":
    main()
