"""
generate_samples.py — compatibility shim for openWakeWord training pipeline.

openWakeWord's train.py does:
    sys.path.insert(0, piper_sample_generator_path)
    from generate_samples import generate_samples

TTS priority:
  1. edge-tts (Microsoft, no model files needed, works for Japanese & English)
  2. piper CLI (if voice model file exists in PIPER_VOICES_DIR)
  3. piper Python API
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", "/opt/piper-voices")

# edge-tts voice map: lang prefix → Azure voice name
_EDGE_VOICES = {
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "en": ["en-US-JennyNeural", "en-US-GuyNeural"],
}

try:
    import edge_tts
    _HAS_EDGE = True
except ImportError:
    _HAS_EDGE = False

try:
    from piper import PiperVoice
    _HAS_PIPER_API = True
except ImportError:
    _HAS_PIPER_API = False

try:
    import soundfile as sf
    import numpy as np
    _HAS_SF = True
except ImportError:
    _HAS_SF = False


def _detect_lang(text: str) -> str:
    """Guess language from text characters."""
    for ch in text:
        if '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff':
            return "ja"
    return "en"


def _edge_voices_for(voice_hint: str) -> list[str]:
    """Given a piper-style voice name or lang, return edge-tts voice names."""
    if voice_hint.startswith("ja"):
        return _EDGE_VOICES["ja"]
    if voice_hint.startswith("en"):
        return _EDGE_VOICES["en"]
    return _EDGE_VOICES["en"]


def generate_samples(
    text,
    max_samples: int = 100,
    output_dir: str = None,
    voices=None,
    **kwargs,
):
    """Generate WAV audio samples for each phrase in `text`.

    Args:
        text:        list of strings or a single string
        max_samples: max number of .wav files to generate
        output_dir:  directory to write .wav files
        voices:      list of piper voice names (used to infer language)
        **kwargs:    ignored

    Returns:
        list of paths to generated .wav files
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="oww_samples_")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    phrases = [text] if isinstance(text, str) else list(text)
    if not voices:
        voices = ["ja_JP-takumi-medium"]

    generated = []
    sample_idx = 0

    while sample_idx < max_samples:
        prev = sample_idx
        for phrase in phrases:
            if sample_idx >= max_samples:
                break
            for voice in voices:
                if sample_idx >= max_samples:
                    break
                out_file = out_path / f"sample_{sample_idx:05d}.wav"
                ok = _synth(phrase, voice, str(out_file))
                if ok:
                    generated.append(str(out_file))
                    sample_idx += 1
                    print(f"[generate_samples] {sample_idx}/{max_samples} — {out_file.name}", flush=True)

        if sample_idx == prev:
            print("[generate_samples] ERROR: could not generate any samples", file=sys.stderr)
            break

    return generated


def _synth(text: str, voice: str, out_path: str) -> bool:
    """Try TTS methods in order: edge-tts → piper CLI → piper API."""
    # 1. edge-tts (primary — no model files, reliable)
    if _HAS_EDGE:
        for ev in _edge_voices_for(voice):
            if _synth_edge(text, ev, out_path):
                return True

    # 2. piper CLI
    if _synth_piper_cli(text, voice, out_path):
        return True

    # 3. piper Python API
    if _HAS_PIPER_API and _synth_piper_api(text, voice, out_path):
        return True

    print(f"[generate_samples] WARN: all TTS methods failed for voice={voice}", file=sys.stderr)
    return False


def _synth_edge(text: str, edge_voice: str, out_path: str) -> bool:
    """Synthesise via Microsoft edge-tts (requires internet)."""
    try:
        # edge-tts outputs MP3/OGG; we need WAV 16kHz mono
        tmp_mp3 = out_path + ".tmp.mp3"

        async def _run():
            comm = edge_tts.Communicate(text, edge_voice)
            await comm.save(tmp_mp3)

        asyncio.run(_run())

        if not os.path.exists(tmp_mp3):
            return False

        # Convert to WAV 16kHz mono using soundfile+ffmpeg
        if _convert_to_wav16k(tmp_mp3, out_path):
            os.remove(tmp_mp3)
            return True

        # Fallback: ffmpeg directly
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True, timeout=30
        )
        try:
            os.remove(tmp_mp3)
        except Exception:
            pass
        return r.returncode == 0 and os.path.exists(out_path)

    except Exception as e:
        print(f"[generate_samples] edge-tts error ({edge_voice}): {e}", file=sys.stderr)
        return False


def _convert_to_wav16k(src: str, dst: str) -> bool:
    """Convert audio file to 16kHz mono WAV via soundfile+resampy."""
    try:
        import numpy as np
        data, sr = sf.read(src)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            try:
                import resampy
                data = resampy.resample(data, sr, 16000)
            except ImportError:
                # Basic linear resampling fallback
                ratio = 16000 / sr
                n_out = int(len(data) * ratio)
                x_old = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, n_out)
                data = np.interp(x_new, x_old, data)
        sf.write(dst, data.astype(np.float32), 16000, subtype="PCM_16")
        return os.path.exists(dst)
    except Exception as e:
        print(f"[generate_samples] soundfile convert error: {e}", file=sys.stderr)
        return False


def _model_path(voice: str) -> str | None:
    p = Path(PIPER_VOICES_DIR) / f"{voice}.onnx"
    if p.exists():
        return str(p)
    for base in [Path.home() / ".local/share/piper-tts",
                 Path("/root/.local/share/piper-tts"),
                 Path("/root/.cache/piper-tts")]:
        p = base / f"{voice}.onnx"
        if p.exists():
            return str(p)
    return None


def _synth_piper_cli(text: str, voice: str, out_path: str) -> bool:
    model = _model_path(voice)
    if model is None:
        return False
    try:
        r = subprocess.run(
            ["piper", "--model", model, "--output_file", out_path],
            input=text, text=True, capture_output=True, timeout=30
        )
        return r.returncode == 0 and os.path.exists(out_path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _synth_piper_api(text: str, voice: str, out_path: str) -> bool:
    model = _model_path(voice)
    if model is None:
        return False
    try:
        v = PiperVoice.load(model)
        with wave.open(out_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(v.config.sample_rate)
            v.synthesize(text, wf)
        return os.path.exists(out_path)
    except Exception as e:
        print(f"[generate_samples] piper API error: {e}", file=sys.stderr)
        return False
