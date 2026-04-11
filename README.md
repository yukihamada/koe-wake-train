# koe-wake-train

Cloud wake word training service for Koe.

**Why this exists**: on-device training of custom openWakeWord models requires
`torch` + `torchinfo` + `torchmetrics` + `piper-sample-generator` (not on PyPI,
must be cloned separately) + 2-5 GB of model downloads + 5-30 min per training
run. That is too heavy for end-user Macs.

This service moves the training off-device: the client sends a text prompt
(e.g. `ヘイこえ`), the server runs the full Piper TTS + openWakeWord training
pipeline, and returns a `.onnx` model the client can drop into its wake-word
detector.

## Components

- `server/` — Rust (axum) HTTP frontend: validates requests, manages job queue,
  serves finished `.onnx` files with signed URLs.
- `worker/` — Python process that actually runs the training pipeline. Picks
  jobs off the queue, writes progress back to shared state, uploads the
  generated `.onnx` to object storage.
- `docs/wake-train-protocol.md` — HTTP API contract (what the Koe macOS client
  expects).

## Deploy target

Fly.io `koe-wake-train` (planned). CPU-only Dockerfile, scale-to-zero, 4 GB RAM,
Tokyo region (`nrt`). First training run downloads the base openWakeWord models
(~13 MB) and the Piper TTS voices (~150 MB); subsequent runs are warm.

## Status

🚧 Scaffold only. See `docs/wake-train-protocol.md` for the API we'll implement,
and `worker/train.py` for the pipeline outline.
