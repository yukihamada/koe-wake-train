"""Download Piper TTS voice packs used by the training pipeline."""
import subprocess
import sys

VOICES = [
    "en_US-libritts-high",
    "ja_JP-takumi-medium",
]

for voice in VOICES:
    try:
        script = f"from piper import PiperVoice; PiperVoice.load('{voice}', download=True)"
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False, timeout=180, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"downloaded {voice}")
        else:
            print(f"skip {voice}: {result.stderr[:200]}")
    except Exception as e:
        print(f"skip {voice}: {e}")
