# syntax=docker/dockerfile:1.6
# ── Stage 1: Rust binary build ───────────────────────────────────────────────
FROM rust:1.88-bookworm AS chef
RUN cargo install cargo-chef --locked --version 0.1.71
WORKDIR /app

FROM chef AS planner
COPY server/Cargo.toml server/Cargo.lock* ./
COPY server/src ./src
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY server/Cargo.toml server/Cargo.lock* ./
COPY server/src ./src
RUN cargo build --release --bin koe-wake-train

# ── Stage 2: Python runtime ─────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps: torch CPU + openwakeword + audio libs
RUN pip install --no-cache-dir \
      "torch==2.2.2" \
      "torchaudio==2.2.2" \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
      "numpy<2.0" \
      "scipy<1.15" \
      "openwakeword>=0.6.0" \
      torchinfo \
      "torchmetrics<1" \
      audiomentations \
      "torch-audiomentations<1" \
      "speechbrain<1" \
      "mutagen<2" \
      "acoustics<1" \
      "datasets<3" \
      "deep-phonemizer==0.0.19" \
      pronouncing \
      pyyaml \
      inflect \
      huggingface_hub \
      edge-tts \
      pydub \
      soundfile

# Pre-download openWakeWord base models (melspectrogram.onnx etc.)
RUN python3 -c "from openwakeword.utils import download_models; download_models()"

# Clone piper-sample-generator for TTS-based sample generation
RUN git clone --depth=1 https://github.com/rhasspy/piper-sample-generator /opt/piper-sample-generator \
    && pip install --no-cache-dir -r /opt/piper-sample-generator/requirements.txt || true

# Install our generate_samples.py shim
COPY server/scripts/generate_samples.py /opt/piper-sample-generator/generate_samples.py

# Install Piper TTS for voice synthesis
RUN pip install --no-cache-dir piper-tts || true

# Download Piper voice models directly from HuggingFace
# Japanese (takumi-medium) and English (libritts_r-high)
RUN mkdir -p /opt/piper-voices && python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

voices = [
    # (repo, remote_path, local_name)
    ("rhasspy/piper-voices", "ja/ja_JP/takumi/medium/ja_JP-takumi-medium.onnx",      "ja_JP-takumi-medium.onnx"),
    ("rhasspy/piper-voices", "ja/ja_JP/takumi/medium/ja_JP-takumi-medium.onnx.json", "ja_JP-takumi-medium.onnx.json"),
    ("rhasspy/piper-voices", "en/en_US/libritts_r/high/en_US-libritts_r-high.onnx",      "en_US-libritts_r-high.onnx"),
    ("rhasspy/piper-voices", "en/en_US/libritts_r/high/en_US-libritts_r-high.onnx.json", "en_US-libritts_r-high.onnx.json"),
]

for repo, path, name in voices:
    dest = f"/opt/piper-voices/{name}"
    if os.path.exists(dest):
        print(f"  skip (exists): {name}")
        continue
    try:
        p = hf_hub_download(repo_id=repo, filename=path)
        shutil.copy(p, dest)
        print(f"  downloaded: {name}")
    except Exception as e:
        print(f"  WARN {name}: {e}")
PYEOF

WORKDIR /app

# Copy Rust binary
COPY --from=builder /app/target/release/koe-wake-train /usr/local/bin/koe-wake-train

# Copy Python worker
COPY worker/ ./worker/

EXPOSE 8080
ENV RUST_LOG=info
ENV PIPER_SAMPLE_GENERATOR_PATH=/opt/piper-sample-generator
ENV PIPER_VOICES_DIR=/opt/piper-voices
ENV JOBS_DIR=/tmp/koe-jobs
# BASE_URL: set via fly.toml [env] or fly secrets

CMD ["koe-wake-train"]
