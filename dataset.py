"""
Dataset collection and preprocessing for Carnatic Raga Classification.

Downloads YouTube videos of Carnatic music performances, chunks them into
segments, and creates a Hugging Face dataset for training.

Improvements:
- Parallel processing for faster downloads
- Better error handling and retry logic
- Progress tracking
- Configurable parameters
"""

import os
import subprocess
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
import warnings

import librosa
import soundfile as sf
import numpy as np
from datasets import Dataset, Audio
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Note name to frequency mapping (Hz, octave 4)
NOTE_TO_FREQ = {
    "C":  261.63,
    "C#": 277.18,
    "D":  293.66,
    "D#": 311.13,
    "E":  329.63,
    "F":  349.23,
    "F#": 369.99,
    "G":  392.00,
    "G#": 415.30,
    "A":  440.00,
    "A#": 466.16,
    "B":  493.88,
}

# Target Sa frequency for normalization
TARGET_SA_HZ = 261.63

# Dataset configuration: (url, shruti_note)
RAGA_VIDEOS = {
    "Kalyani": [
        ("https://youtu.be/ZOoudH6olzM?si=dcekb3Ckm3YhvLe0", "G#"),
        ("https://youtu.be/bqjJbvUe0EI?si=AEpSKPm1t34QtOm9", "D"),
    ],
    "Kharaharapriya": [
        ("https://youtu.be/w1fJSK_LN68?si=0wTuXIwG1yzneh0m", "G#"),
        ("https://youtu.be/zIOYBCBjgh8?si=GhbB6GYmDdEm68Sb", "C#"),
    ],
    "Mayamalavagoulai": [
        ("https://youtu.be/CF7L0nlV4v8?si=o2uZc3OVpuEoPI5L", "G#"),
        ("https://youtu.be/zGrDSEDeCI0?si=iUqi-2FCA1Ny8tBd", "C#"),
    ],
    "Todi": [
        ("https://youtu.be/ORwF_WXFtL8?si=Ya59qKfW7P9TsmAz", "G"),
        ("https://youtu.be/x1pbtvX89Bk?si=bsCayPE7_YnnhxOz", "C#"),
    ],
    "Amritavarshini": [
        ("https://youtu.be/UtBIJPCX-ps?si=v-Um_x2oYJYEE7iQ", "C#"),
        ("https://youtu.be/s7RijjTfl5I?si=XzpzfiefcGXr7PoO", "E"),
    ],
    "Hamsanaadam": [
        ("https://youtu.be/jNacNu-ar0E?si=cm8hFuk8SchY8ugs", "G#"),
        ("https://youtu.be/FnSRfFdiDwY?si=vcw09xAdN_bR5sfu", "E"),
    ],
    "Varali": [
        ("https://youtu.be/Lr02Fe93vcM?si=TZbYfJ2AjdlwXkt8", "G#"),
        ("https://youtu.be/3M3-5E6xx2o?si=dhPjeTX6SjoOOnEx", "D"),
    ],
    "Sindhubhairavi": [
        ("https://youtu.be/wFnshTKK_DI?si=UG-B1b_EkOcMKrxG", "E"),
        ("https://youtu.be/x1aLwDLwn_M?si=dV4eTKKXCcLA6ugs", "F"),
        ("https://youtu.be/H2Gj1ZY6tP0?si=Ub-uLpKQobtg59_Q", "D"),
    ],
}

# Processing parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 20  # seconds
MAX_RETRIES = 3
WORKDIR = Path("data")
AUDIO_DIR = WORKDIR / "audio"


def ensure_directories():
    """Create necessary directories."""
    WORKDIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)


def download_audio(youtube_url: str, output_path: Path, retries: int = MAX_RETRIES) -> bool:
    """
    Download audio from YouTube URL with retry logic.

    Args:
        youtube_url: YouTube video URL
        output_path: Path to save the audio file
        retries: Number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-o", str(output_path),
        youtube_url
    ]

    for attempt in range(retries):
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            return True
        except subprocess.TimeoutExpired:
            print(f"  Timeout downloading {youtube_url}, attempt {attempt + 1}/{retries}")
        except subprocess.CalledProcessError as e:
            print(f"  Error downloading {youtube_url}: {e.stderr.decode()[:100]}")

        if attempt < retries - 1:
            print(f"  Retrying...")

    return False


def chunk_audio(
    wav_path: Path,
    raga: str,
    shruti_hz: float,
    chunk_duration: int = CHUNK_DURATION,
    sample_rate: int = SAMPLE_RATE,
    overlap: float = 0.0
) -> List[Dict]:
    """
    Split audio file into fixed-length chunks.

    Args:
        wav_path: Path to the WAV file
        raga: Raga label for the audio
        shruti_hz: Sa frequency in Hz for this video
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Target sample rate
        overlap: Overlap ratio between chunks (0.0 to 0.5)

    Returns:
        List of dictionaries with audio paths, labels, and shruti frequency
    """
    try:
        y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"  Error loading {wav_path}: {e}")
        return []

    chunk_samples = chunk_duration * sample_rate
    hop_samples = int(chunk_samples * (1 - overlap))

    examples = []

    for i in range(0, len(y) - chunk_samples + 1, hop_samples):
        chunk = y[i:i + chunk_samples]

        # Skip chunks that are mostly silence
        if np.abs(chunk).max() < 0.01:
            continue

        # Normalize amplitude (not pitch)
        chunk = chunk / (np.abs(chunk).max() + 1e-8)

        chunk_id = uuid.uuid4().hex[:12]
        chunk_path = AUDIO_DIR / f"{raga}_{chunk_id}.wav"

        sf.write(chunk_path, chunk, sample_rate)

        examples.append({
            "audio": str(chunk_path),
            "raga": raga,
            "shruti_hz": shruti_hz,
        })

    return examples


def process_video(args: tuple) -> List[Dict]:
    """
    Process a single video: download and chunk.

    Args:
        args: Tuple of (raga, url, shruti_note)

    Returns:
        List of processed examples
    """
    raga, url, shruti_note = args
    shruti_hz = NOTE_TO_FREQ[shruti_note]
    video_id = uuid.uuid4().hex[:8]
    wav_path = WORKDIR / f"temp_{video_id}.wav"

    try:
        print(f"  Downloading: {raga} ({shruti_note}={shruti_hz:.1f}Hz) - {url[:50]}...")
        if not download_audio(url, wav_path):
            print(f"  Failed to download {url}")
            return []

        # Handle yt-dlp output naming (adds extension)
        actual_path = wav_path
        if not actual_path.exists():
            # yt-dlp might add .wav extension
            actual_path = Path(str(wav_path) + ".wav")

        if not actual_path.exists():
            # Try finding the file
            possible_files = list(WORKDIR.glob(f"temp_{video_id}*"))
            if possible_files:
                actual_path = possible_files[0]
            else:
                print(f"  Could not find downloaded file for {url}")
                return []

        print(f"  Chunking: {raga}...")
        examples = chunk_audio(actual_path, raga, shruti_hz, overlap=0.5)
        print(f"  Created {len(examples)} chunks for {raga}")

        return examples

    finally:
        # Cleanup temporary files
        for temp_file in WORKDIR.glob(f"temp_{video_id}*"):
            try:
                temp_file.unlink()
            except Exception:
                pass


def build_dataset(
    max_workers: int = 4,
    push_to_hub: bool = True,
    hub_repo: str = "sarayusapa/carnatic-ragas"
) -> Dataset:
    """
    Build the complete dataset from all raga videos.

    Args:
        max_workers: Number of parallel download workers
        push_to_hub: Whether to push to Hugging Face Hub
        hub_repo: Hugging Face repository name

    Returns:
        Hugging Face Dataset object
    """
    ensure_directories()

    # Prepare all video tasks
    tasks = []
    for raga, videos in RAGA_VIDEOS.items():
        for url, shruti_note in videos:
            tasks.append((raga, url, shruti_note))

    print(f"Processing {len(tasks)} videos across {len(RAGA_VIDEOS)} ragas...")

    # Process videos in parallel
    all_examples = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            task = futures[future]
            try:
                examples = future.result()
                all_examples.extend(examples)
            except Exception as e:
                print(f"Error processing {task}: {e}")

    print(f"\nTotal examples: {len(all_examples)}")

    # Print distribution
    raga_counts = {}
    for ex in all_examples:
        raga_counts[ex['raga']] = raga_counts.get(ex['raga'], 0) + 1

    print("\nRaga distribution:")
    for raga, count in sorted(raga_counts.items()):
        print(f"  {raga}: {count}")

    # Create dataset
    dataset = Dataset.from_list(all_examples)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Save locally
    local_path = WORKDIR / "hf_raga_dataset"
    print(f"\nSaving dataset to {local_path}...")
    dataset.save_to_disk(str(local_path))

    # Push to Hub
    if push_to_hub:
        print(f"Pushing to Hugging Face Hub: {hub_repo}...")
        try:
            dataset.push_to_hub(hub_repo)
            print("Successfully pushed to Hub!")
        except Exception as e:
            print(f"Failed to push to Hub: {e}")

    return dataset


def cleanup_temp_files():
    """Remove temporary files from the work directory."""
    for temp_file in WORKDIR.glob("temp_*"):
        try:
            temp_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Carnatic Raga dataset")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HF Hub")
    parser.add_argument("--repo", type=str, default="sarayusapa/carnatic-ragas", help="HF Hub repo")

    args = parser.parse_args()

    try:
        ds = build_dataset(
            max_workers=args.workers,
            push_to_hub=not args.no_push,
            hub_repo=args.repo
        )
        print(f"\nDataset info:\n{ds}")
    finally:
        cleanup_temp_files()
