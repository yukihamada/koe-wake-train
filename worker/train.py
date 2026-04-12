"""
koe-wake-train worker.

Pipeline:
  1. Write training YAML config
  2. Run --generate_clips  (TTS sample generation)
  3. Inject user voice samples into clips dir (if --user-samples-dir provided)
  4. Run --augment_clips --train_model
  5. Return .onnx path

Args:
  --text              Wake word phrase (e.g. "ヘイKoe")
  --model-name        Model identifier, [a-z0-9_] only
  --lang              Language: "ja" or "en"
  --output-dir        Job output directory
  --user-samples-dir  Directory of user WAV files to inject as positive examples
  --negatives-dir     Directory of WAV files to use as negative examples (false positives)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PIPER_PATH = os.environ.get("PIPER_SAMPLE_GENERATOR_PATH", "/opt/piper-sample-generator")
PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", "/opt/piper-voices")

VOICE_PACKS = {
    "ja": [
        "ja_JP-takumi-medium",
        "ja_JP-tsukuyomi-medium",
    ],
    "en": [
        "en_US-libritts_r-high",
        "en_GB-alan-low",
    ],
}


def write_training_config(
    text: str,
    model_name: str,
    lang: str,
    out_dir: Path,
    has_user_samples: bool,
    false_positive_path: str | None = None,
) -> Path:
    voices = VOICE_PACKS.get(lang, VOICE_PACKS["en"])
    # Reduce TTS samples when user provides their own voice (they compensate for diversity)
    n_samples = 200 if has_user_samples else 500

    # English negative phrases used to generate adversarial TTS clips.
    # generate_adversarial_texts() is English-only and returns empty for Japanese text,
    # so we must supply explicit negatives; otherwise the negative clips dir stays empty,
    # compute_features_from_generator writes an all-zero memmap, and "Trimming empty rows"
    # loops out of bounds with an IndexError.
    custom_negative_phrases = [
        "hey go", "hey ko", "hey joe", "hey slow", "hey though",
        "okay", "hey", "go", "code", "cold", "cake",
        "wake", "word", "key", "password",
    ]

    config = {
        "piper_sample_generator_path": PIPER_PATH,
        "output_dir": str(out_dir),
        "model_name": model_name,
        "target_phrase": [text],
        "n_samples": n_samples,
        "n_samples_val": 50 if has_user_samples else 100,
        "tts_voices": voices,
        "tts_batch_size": 4,
        "augmentation_rounds": 2,
        "augmentation_batch_size": 8,
        "feature_data_files": {},   # dict (not list) — openwakeword calls .keys() on it
        "background_paths": [],
        "background_paths_duplication_rate": [],
        "rir_paths": [],
        "custom_negative_phrases": custom_negative_phrases,
        "false_positive_validation_data_path": false_positive_path,
        "model_type": "dnn",
        "steps": 10000,
        "batch_n_per_class": {},    # dict (not int) — mmap_batch_generator expects dict; {} = auto-calculate
        "max_negative_weight": 1500,
        "layer_size": 32,
        "seconds_per_example": 1.5,
        "target_false_positives_per_hour": 0.5,
        "target_accuracy": 0.7,
        "target_recall": 0.25,
    }
    yaml_path = out_dir / "training_config.yaml"
    lines = []
    for k, v in config.items():
        if isinstance(v, dict):
            if not v:
                lines.append(f"{k}: {{}}")
            else:
                lines.append(f"{k}:")
                for dk, dv in v.items():
                    lines.append(f"  {dk!r}: {dv!r}")
        elif isinstance(v, list):
            if not v:
                lines.append(f"{k}: []")
            else:
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


def run_stage(yaml_path: Path, flags: list[str], progress_cb=None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "openwakeword.train", "--training_config", str(yaml_path)] + flags
    if progress_cb:
        progress_cb(f"running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def inject_user_samples(clips_dir: Path, user_samples_dir: Path, repeat: int = 5):
    """Copy user WAV files into clips dir, repeating them to boost their weight."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for wav in sorted(user_samples_dir.glob("*.wav")):
        for i in range(repeat):
            dest = clips_dir / f"user_{wav.stem}_{i:02d}.wav"
            shutil.copy(wav, dest)
            copied += 1
    return copied


def inject_negatives(negatives_dir: Path, job_dir: Path, model_name: str):
    """Copy false-positive audio into the negative clips dir for training."""
    neg_clips = job_dir / model_name / "clips_neg"
    neg_clips.mkdir(parents=True, exist_ok=True)
    copied = 0
    for wav in sorted(negatives_dir.glob("*.wav"))[:100]:  # cap at 100
        shutil.copy(wav, neg_clips / wav.name)
        copied += 1
    return copied


def emit(status: str, **extra):
    obj = {"status": status, **extra}
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--lang", default="en", choices=list(VOICE_PACKS.keys()))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--user-samples-dir", default=None,
                    help="Directory containing user WAV files to inject as positive samples")
    ap.add_argument("--negatives-dir", default=None,
                    help="Directory of false-positive WAV files to inject as negatives")
    args = ap.parse_args()

    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_samples_dir = Path(args.user_samples_dir) if args.user_samples_dir else None
    negatives_dir    = Path(args.negatives_dir)    if args.negatives_dir    else None
    has_user_samples = user_samples_dir is not None and any(user_samples_dir.glob("*.wav"))

    emit("running", progress="writing training config")
    yaml_path = write_training_config(
        args.text, args.model_name, args.lang, out_dir, has_user_samples,
        false_positive_path=None,   # will be updated after --augment_clips
    )

    # ── Stage 1: generate TTS clips ──────────────────────────────────────────
    emit("running", progress="generating TTS clips (stage 1/3)")
    r1 = run_stage(yaml_path, ["--generate_clips"],
                   progress_cb=lambda m: emit("running", progress=m))
    if r1.returncode != 0:
        emit("failed", progress="generate_clips failed",
             stderr_tail=(r1.stderr or "")[-800:])
        sys.exit(1)

    # ── Stage 2: inject user voice samples ───────────────────────────────────
    clips_dir = out_dir / args.model_name / "clips"
    if has_user_samples:
        emit("running", progress="injecting user voice samples (stage 2/3)")
        n = inject_user_samples(clips_dir, user_samples_dir)
        emit("running", progress=f"injected {n} user sample clips")

    if negatives_dir and negatives_dir.exists():
        n_neg = inject_negatives(negatives_dir, out_dir, args.model_name)
        emit("running", progress=f"injected {n_neg} negative (false-positive) clips")

    # ── Stage 3a: augment clips (compute audio features) ────────────────────
    emit("running", progress="augmenting clips and computing features (stage 3a/3, ~5 min)")
    r_aug = run_stage(yaml_path, ["--augment_clips"],
                      progress_cb=lambda m: emit("running", progress=m))
    if r_aug.returncode != 0:
        emit("failed",
             progress=f"augment_clips exit={r_aug.returncode}",
             stderr_tail=(r_aug.stderr or "")[-800:])
        sys.exit(1)

    # ── Update YAML: point false_positive_validation_data_path at generated npy ──
    # openwakeword unconditionally calls np.load(path) on this, so null crashes.
    # The code then does: arr[i:i+input_shape[0]] expecting a 2D array (M, 96).
    # negative_features_test.npy is 3D (N, time_steps, 96) — reshape to 2D first.
    neg_test_npy = out_dir / args.model_name / "negative_features_test.npy"
    fp_path = None
    if neg_test_npy.exists():
        try:
            import numpy as _np
            arr = _np.load(str(neg_test_npy))            # (N, T, 96)
            fp_flat = arr.reshape(-1, arr.shape[-1])     # (N*T, 96) — 2D as expected
            fp_flat_path = out_dir / args.model_name / "fp_validation_flat.npy"
            _np.save(str(fp_flat_path), fp_flat)
            fp_path = str(fp_flat_path)
            emit("running", progress=f"created fp_validation_flat.npy shape={fp_flat.shape}")
        except Exception as e:
            emit("running", progress=f"WARN: could not reshape fp validation: {e}")
    if fp_path:
        yaml_path = write_training_config(
            args.text, args.model_name, args.lang, out_dir, has_user_samples,
            false_positive_path=fp_path,
        )
    else:
        emit("running", progress="WARN: fp validation not available, proceeding without it")

    # ── Stage 3b: train model ────────────────────────────────────────────────
    emit("running", progress="training model (stage 3b/3, 5-10 min)")
    r2 = run_stage(yaml_path, ["--train_model"],
                   progress_cb=lambda m: emit("running", progress=m))

    # ONNX is exported to output_dir/model_name.onnx (flat, not nested in model_name/).
    # TFLite conversion (convert_onnx_to_tflite) requires onnx_tf which is not installed;
    # it crashes AFTER ONNX is written. Accept non-zero exit if the ONNX file exists.
    onnx_path = out_dir / f"{args.model_name}.onnx"
    if not onnx_path.exists():
        emit("failed",
             progress=f"train_model exit={r2.returncode}, onnx not found",
             stderr_tail=(r2.stderr or "")[-800:])
        sys.exit(1)
    if r2.returncode != 0:
        emit("running", progress=f"train_model exited {r2.returncode} (likely TFLite step failed, ONNX is present — continuing)")

    emit("done", onnx_path=str(onnx_path))


if __name__ == "__main__":
    main()
