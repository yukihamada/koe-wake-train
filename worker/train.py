"""
koe-wake-train worker.

Takes a text prompt and produces a trained openWakeWord `.onnx` model.

Pipeline (matches what `python -m openwakeword.train` expects upstream):

  1. Write a training YAML config pointing at a cloned piper-sample-generator
  2. Run `python -m openwakeword.train --training_config <yaml> --generate_clips
     --augment_clips --train_model`
  3. openwakeword writes `<output_dir>/<model_name>/<model_name>.onnx`
  4. Move that file to the final location and return the path

This file is a scaffold: the actual YAML structure, Piper voice selection, and
number of training clips still need tuning. Upstream's reference config is at
https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

Required environment:
  - torch, torchinfo, torchmetrics
  - openwakeword (with `download_models()` already called once in the image)
  - piper-sample-generator cloned to /opt/piper-sample-generator
  - Piper voice files under /opt/piper-voices/

Usage:
  python worker/train.py --text "ヘイこえ" --model-name hey_koe --lang ja \\
      --output-dir /tmp/jobs/<job_id>/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PIPER_PATH = os.environ.get("PIPER_SAMPLE_GENERATOR_PATH", "/opt/piper-sample-generator")

VOICE_PACKS = {
    "ja": [
        "ja_JP-takumi-medium",
        "ja_JP-tsukuyomi-medium",
    ],
    "en": [
        "en_US-libritts-high",
        "en_GB-alan-low",
    ],
}


def write_training_config(text: str, model_name: str, lang: str, out_dir: Path) -> Path:
    """Write a minimal training YAML for openwakeword.train."""
    voices = VOICE_PACKS.get(lang, VOICE_PACKS["en"])
    config = {
        "piper_sample_generator_path": PIPER_PATH,
        "output_dir": str(out_dir),
        "model_name": model_name,
        "target_phrase": [text],
        "n_samples": 500,         # positive samples to generate
        "n_samples_val": 100,
        "tts_voices": voices,
        "augmentation_rounds": 2,
        "augmentation_batch_size": 8,
        "feature_data_files": [],   # use auto-downloaded ones
        "background_paths": [],     # use defaults
        "false_positive_validation_data_path": None,
        "steps": 10000,
        "max_negative_weight": 1500,
        "layer_size": 32,
        "seconds_per_example": 1.5,
        "target_false_positives_per_hour": 0.5,
        "target_accuracy": 0.7,
        "target_recall": 0.25,
    }
    yaml_path = out_dir / "training_config.yaml"
    # Minimal hand-rolled YAML to avoid adding PyYAML dep here
    lines = []
    for k, v in config.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item!r}" if isinstance(item, str) else f"  - {item}")
        elif isinstance(v, str):
            lines.append(f"{k}: {v!r}")
        elif v is None:
            lines.append(f"{k}: null")
        else:
            lines.append(f"{k}: {v}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def run_training(yaml_path: Path, progress_cb=None) -> subprocess.CompletedProcess:
    """Run openwakeword.train with the 3-stage flags. Blocks until done."""
    cmd = [
        sys.executable, "-m", "openwakeword.train",
        "--training_config", str(yaml_path),
        "--generate_clips",
        "--augment_clips",
        "--train_model",
    ]
    if progress_cb:
        progress_cb(f"running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def emit(status: str, **extra):
    """Write one-line JSON status to stdout so the Rust server can tail it."""
    obj = {"status": status, **extra}
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--lang", default="en", choices=list(VOICE_PACKS.keys()))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emit("running", progress="writing training config")
    yaml_path = write_training_config(args.text, args.model_name, args.lang, out_dir)

    emit("running", progress="training (this takes 5-15 min)")
    result = run_training(yaml_path, progress_cb=lambda m: emit("running", progress=m))

    onnx_path = out_dir / args.model_name / f"{args.model_name}.onnx"
    if result.returncode != 0 or not onnx_path.exists():
        emit(
            "failed",
            progress=f"exit={result.returncode}",
            stderr_tail=(result.stderr or "")[-800:],
        )
        sys.exit(1)

    emit("done", onnx_path=str(onnx_path))


if __name__ == "__main__":
    main()
