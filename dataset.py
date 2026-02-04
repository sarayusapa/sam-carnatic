import os
import subprocess
import librosa
import soundfile as sf
from datasets import Dataset, Audio
from tqdm import tqdm
import uuid

RAGA_VIDEOS = {
    "Kalyani": [
        "https://youtu.be/ZOoudH6olzM?si=dcekb3Ckm3YhvLe0", 
        "https://youtu.be/bqjJbvUe0EI?si=AEpSKPm1t34QtOm9", 

    ],
    "Kharaharapriya": [
        "https://youtu.be/w1fJSK_LN68?si=0wTuXIwG1yzneh0m",
        "https://youtu.be/zIOYBCBjgh8?si=GhbB6GYmDdEm68Sb",

    ],
    "Mayamalavagoulai": [
        "https://youtu.be/CF7L0nlV4v8?si=o2uZc3OVpuEoPI5L",
        "https://youtu.be/zGrDSEDeCI0?si=iUqi-2FCA1Ny8tBd",

    ],
    "Todi": [
        "https://youtu.be/ORwF_WXFtL8?si=Ya59qKfW7P9TsmAz",
        "https://youtu.be/x1pbtvX89Bk?si=bsCayPE7_YnnhxOz",

    ],
    "Amritavarshini": [
        "https://youtu.be/UtBIJPCX-ps?si=v-Um_x2oYJYEE7iQ",
        "https://youtu.be/s7RijjTfl5I?si=XzpzfiefcGXr7PoO",
    ],
    "Hamsanaadam": [
        "https://youtu.be/jNacNu-ar0E?si=cm8hFuk8SchY8ugs",
        "https://youtu.be/FnSRfFdiDwY?si=vcw09xAdN_bR5sfu",

    ],
    "Varali": [
        "https://youtu.be/Lr02Fe93vcM?si=TZbYfJ2AjdlwXkt8",
        "https://youtu.be/WknDE3b7Jjo?si=L1yyoHnXp1Hn0KNV",

    ],
    "Sindhubhairavi": [
        "https://youtu.be/wFnshTKK_DI?si=UG-B1b_EkOcMKrxG",
        "https://youtu.be/x1aLwDLwn_M?si=dV4eTKKXCcLA6ugs",
        "https://youtu.be/H2Gj1ZY6tP0?si=Ub-uLpKQobtg59_Q",

    ],
}

SAMPLE_RATE = 16000
CHUNK_DURATION = 20  # seconds
WORKDIR = "data"
AUDIO_DIR = os.path.join(WORKDIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio(youtube_url, output_path):
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "-o", output_path,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

def chunk_audio(wav_path, raga):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    chunk_samples = CHUNK_DURATION * sr

    examples = []

    for i in range(0, len(y) - chunk_samples, chunk_samples):
        chunk = y[i:i + chunk_samples]
        chunk_id = uuid.uuid4().hex
        chunk_path = os.path.join(AUDIO_DIR, f"{chunk_id}.wav")

        sf.write(chunk_path, chunk, sr)

        examples.append({
            "audio": chunk_path,
            "raga": raga
        })

    return examples

def build_dataset():
    data = []

    for raga, urls in RAGA_VIDEOS.items():
        for url in urls:
            video_id = uuid.uuid4().hex
            wav_path = os.path.join(WORKDIR, f"{video_id}.wav")

            print(f"Downloading {url} ({raga})")
            download_audio(url, wav_path)

            print(f"Chunking audio for raga: {raga}")
            data.extend(chunk_audio(wav_path, raga))

            os.remove(wav_path)

    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return dataset

if __name__ == "__main__":
    ds = build_dataset()
    print(ds)

    # Optional: save locally
    ds.save_to_disk("hf_raga_dataset")

    # Optional: push to HF Hub
    ds.push_to_hub("sarayusapa/carnatic-ragas")