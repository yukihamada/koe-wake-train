# koe-wake-train HTTP API

Client: `Koe-swift/Sources/Koe/CloudWakeTrainer.swift`
Server: Rust (axum), this repo under `server/`

## POST /v1/wake/train

Submit a custom wake-word training job.

```http
POST /v1/wake/train
Content-Type: application/json

{
  "text": "ヘイこえ",
  "model_name": "hey_koe",
  "lang": "ja"
}
```

**Response — 202 Accepted**

```json
{"job_id": "01H8PBRXF7..."}
```

**Validation**
- `text`: 2-32 chars, UTF-8
- `model_name`: `[a-z0-9_]{1,32}`, unique per API key
- `lang`: `ja` | `en`. Affects which Piper voice pack is used for sample
  generation.

**Rate limit**: 3 concurrent jobs per API key, 30 submissions/day.

## GET /v1/wake/train/{job_id}

Poll a training job. Client polls every 15 s.

**Response — 200 OK**

```json
{
  "status": "pending | running | done | failed",
  "progress": "generating_samples(320/1000)",
  "onnx_url": "https://koe-wake-train.fly.dev/v1/wake/models/<sig>/hey_koe.onnx"
}
```

- `progress`: free-form status string for UI display
- `onnx_url`: **only present when** `status == "done"`. Signed URL valid for
  30 min. Client downloads the file and stores it under
  `~/Library/Application Support/Koe/models/{model_name}.onnx`.

## GET /v1/wake/models/{sig}/{model}.onnx

Returns the trained model as `application/octet-stream`. Content-Length header
is required so Swift `URLSession.downloadTask` can show progress.

Sig is a short HMAC over `(job_id, model_name, expiry)` to prevent link
sharing.

## Error format

All non-2xx responses:

```json
{"error": "short_code", "message": "human readable"}
```

| Code | Meaning |
|---|---|
| `invalid_input`   | 400 — text/model_name failed validation |
| `rate_limited`    | 429 — too many submissions |
| `server_error`    | 500 — worker crash, Piper failure, etc. |
| `not_found`       | 404 — job_id doesn't exist or expired |
| `link_expired`    | 410 — signed URL expired |

## Lifecycle

- Jobs: retained 7 days, then purged.
- Generated `.onnx` files: retained 30 days, then purged. Client is expected to
  cache the file locally after first download.

## Why not SSE / websocket?

Poll is simpler to implement in Swift `URLSession` + worker-side state is just
a shared dict. SSE would add timeout complexity for a 5-15 min operation. The
cost of a GET every 15 s for 15 min is 60 requests — trivial.
