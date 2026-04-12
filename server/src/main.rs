//! koe-wake-train HTTP server.
//!
//! POST /v1/wake/train            → spawn Python worker, return job_id
//! GET  /v1/wake/train/:id        → poll job status
//! GET  /v1/wake/models/:id/:file → download trained .onnx
//! POST /v1/wake/samples          → upload user voice samples (base64 WAV), return session_id
//! POST /v1/wake/feedback         → report false positive / miss for future retraining

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{Json, Response},
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose, Engine as _};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};
use ulid::Ulid;

// ── State ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    jobs:     Arc<DashMap<String, JobRow>>,
    /// voice_session_id → directory of uploaded WAV files
    samples:  Arc<DashMap<String, PathBuf>>,
    jobs_dir: PathBuf,
    base_url: String,
}

#[derive(Clone, Serialize)]
struct JobRow {
    status:    String,
    progress:  String,
    onnx_url:  Option<String>,
    #[serde(skip)]
    onnx_path: Option<String>,
}

// ── Request / response types ───────────────────────────────────────────────

#[derive(Deserialize)]
struct TrainReq {
    text:             String,
    model_name:       String,
    lang:             Option<String>,
    /// optional: session_id from POST /v1/wake/samples
    voice_session_id: Option<String>,
}

#[derive(Serialize)]
struct SubmitResp {
    job_id: String,
}

/// Upload user voice samples (base64-encoded WAV).
#[derive(Deserialize)]
struct UploadSamplesReq {
    /// base64-encoded WAV files (up to 50)
    samples: Vec<String>,
}

#[derive(Serialize)]
struct UploadSamplesResp {
    session_id: String,
    count:      usize,
}

/// False-positive / miss feedback from the client.
#[derive(Deserialize)]
struct FeedbackReq {
    /// Job / model that fired
    model_name: String,
    /// "false_positive" or "miss"
    kind:       String,
    /// base64-encoded WAV of the audio around the event (optional)
    audio_b64:  Option<String>,
}

/// One JSON line emitted by worker/train.py to stdout.
#[derive(Deserialize)]
struct WorkerLine {
    status:      String,
    progress:    Option<String>,
    onnx_path:   Option<String>,
    stderr_tail: Option<String>,
}

// ── main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let jobs_dir = PathBuf::from(
        std::env::var("JOBS_DIR").unwrap_or_else(|_| "/tmp/koe-jobs".into()),
    );
    let base_url = std::env::var("BASE_URL")
        .unwrap_or_else(|_| "http://localhost:8080".into());

    std::fs::create_dir_all(&jobs_dir).expect("create jobs dir");

    let state = AppState {
        jobs:     Arc::new(DashMap::new()),
        samples:  Arc::new(DashMap::new()),
        jobs_dir,
        base_url,
    };

    let app = Router::new()
        .route("/healthz",                           get(|| async { "ok" }))
        .route("/v1/wake/train",                     post(submit))
        .route("/v1/wake/train/:job_id",             get(poll_status))
        .route("/v1/wake/models/:job_id/:filename",  get(serve_model))
        .route("/v1/wake/samples",                   post(upload_samples))
        .route("/v1/wake/feedback",                  post(feedback))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Handlers ───────────────────────────────────────────────────────────────

async fn submit(
    State(state): State<AppState>,
    Json(req): Json<TrainReq>,
) -> Result<(StatusCode, Json<SubmitResp>), (StatusCode, Json<serde_json::Value>)> {
    if req.text.chars().count() < 2 || req.text.chars().count() > 32 {
        return Err(err(StatusCode::BAD_REQUEST, "invalid_input", "text must be 2-32 chars"));
    }
    if req.model_name.is_empty()
        || req.model_name.len() > 32
        || !req.model_name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "invalid_input",
            "model_name must match [a-z0-9_]{1,32}",
        ));
    }

    let job_id  = Ulid::new().to_string();
    let lang    = req.lang.unwrap_or_else(|| "en".into());
    let job_dir = state.jobs_dir.join(&job_id);

    // Resolve optional user-voice sample directory
    let user_samples_dir: Option<PathBuf> = req.voice_session_id.as_ref().and_then(|sid| {
        state.samples.get(sid).map(|p| p.clone())
    });

    state.jobs.insert(job_id.clone(), JobRow {
        status:    "pending".into(),
        progress:  "queued".into(),
        onnx_url:  None,
        onnx_path: None,
    });

    tracing::info!(%job_id, text=%req.text, model=%req.model_name, %lang,
        has_user_samples=%user_samples_dir.is_some(), "job queued");

    let jobs     = state.jobs.clone();
    let base_url = state.base_url.clone();
    let text       = req.text.clone();
    let model_name = req.model_name.clone();
    let jid        = job_id.clone();

    tokio::spawn(async move {
        run_worker(jobs, jid, job_dir, text, model_name, lang, base_url, user_samples_dir).await;
    });

    Ok((StatusCode::ACCEPTED, Json(SubmitResp { job_id })))
}

async fn poll_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobRow>, (StatusCode, Json<serde_json::Value>)> {
    state
        .jobs
        .get(&job_id)
        .map(|r| Json(r.clone()))
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "not_found", "no such job"))
}

async fn serve_model(
    State(state): State<AppState>,
    Path((job_id, _filename)): Path<(String, String)>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let onnx_path = {
        let job = state.jobs.get(&job_id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "not_found", "no such job"))?;
        if job.status != "done" {
            return Err(err(StatusCode::NOT_FOUND, "not_ready", "job not done yet"));
        }
        job.onnx_path.clone()
            .ok_or_else(|| err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "missing onnx path"))?
    };

    let data = tokio::fs::read(&onnx_path).await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &e.to_string()))?;

    let response = Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, data.len())
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"",
                PathBuf::from(&onnx_path).file_name()
                    .and_then(|n| n.to_str()).unwrap_or("model.onnx")),
        )
        .body(Body::from(data))
        .unwrap();

    Ok(response)
}

/// POST /v1/wake/samples — store base64 WAV files, return session_id.
async fn upload_samples(
    State(state): State<AppState>,
    Json(req): Json<UploadSamplesReq>,
) -> Result<(StatusCode, Json<UploadSamplesResp>), (StatusCode, Json<serde_json::Value>)> {
    if req.samples.is_empty() || req.samples.len() > 50 {
        return Err(err(StatusCode::BAD_REQUEST, "invalid", "provide 1-50 samples"));
    }

    let session_id  = Ulid::new().to_string();
    let session_dir = state.jobs_dir.join("samples").join(&session_id);
    tokio::fs::create_dir_all(&session_dir).await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "io", &e.to_string()))?;

    let mut count = 0usize;
    for (i, b64) in req.samples.iter().enumerate() {
        let bytes = general_purpose::STANDARD.decode(b64)
            .map_err(|_| err(StatusCode::BAD_REQUEST, "invalid", "bad base64 in sample"))?;
        let path = session_dir.join(format!("user_{i:03}.wav"));
        tokio::fs::write(&path, bytes).await
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "io", &e.to_string()))?;
        count += 1;
    }

    state.samples.insert(session_id.clone(), session_dir);
    tracing::info!(%session_id, count, "voice samples uploaded");

    Ok((StatusCode::OK, Json(UploadSamplesResp { session_id, count })))
}

/// POST /v1/wake/feedback — store false-positive/miss audio for future retraining.
async fn feedback(
    State(state): State<AppState>,
    Json(req): Json<FeedbackReq>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let valid_kinds = ["false_positive", "miss"];
    if !valid_kinds.contains(&req.kind.as_str()) {
        return Err(err(StatusCode::BAD_REQUEST, "invalid", "kind must be false_positive or miss"));
    }

    let fb_dir = state.jobs_dir
        .join("feedback")
        .join(&req.model_name)
        .join(&req.kind);
    tokio::fs::create_dir_all(&fb_dir).await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "io", &e.to_string()))?;

    if let Some(b64) = req.audio_b64 {
        let bytes = general_purpose::STANDARD.decode(b64)
            .map_err(|_| err(StatusCode::BAD_REQUEST, "invalid", "bad base64 audio"))?;
        let fname = format!("{}.wav", Ulid::new());
        tokio::fs::write(fb_dir.join(&fname), bytes).await
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "io", &e.to_string()))?;
        tracing::info!(model=%req.model_name, kind=%req.kind, %fname, "feedback stored");
    } else {
        tracing::info!(model=%req.model_name, kind=%req.kind, "feedback event (no audio)");
    }

    Ok(StatusCode::NO_CONTENT)
}

// ── Worker ─────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn run_worker(
    jobs:            Arc<DashMap<String, JobRow>>,
    job_id:          String,
    job_dir:         PathBuf,
    text:            String,
    model_name:      String,
    lang:            String,
    base_url:        String,
    user_samples_dir: Option<PathBuf>,
) {
    set_progress(&jobs, &job_id, "running", "starting worker");

    if let Err(e) = tokio::fs::create_dir_all(&job_dir).await {
        set_failed(&jobs, &job_id, &format!("mkdir: {e}"));
        return;
    }

    let python = find_python();
    tracing::info!(%job_id, %python, "spawning worker");

    let mut args = vec![
        "worker/train.py".to_string(),
        "--text".into(),       text,
        "--model-name".into(), model_name.clone(),
        "--lang".into(),       lang,
        "--output-dir".into(), job_dir.to_string_lossy().into_owned(),
    ];

    if let Some(ref dir) = user_samples_dir {
        args.push("--user-samples-dir".into());
        args.push(dir.to_string_lossy().into_owned());

        // Also pass feedback negatives if they exist
        let neg_dir = job_dir.parent()
            .unwrap_or(&job_dir)
            .join("feedback")
            .join(&model_name)
            .join("false_positive");
        if neg_dir.exists() {
            args.push("--negatives-dir".into());
            args.push(neg_dir.to_string_lossy().into_owned());
        }
    }

    let mut child = match Command::new(&python)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c)  => c,
        Err(e) => { set_failed(&jobs, &job_id, &format!("spawn failed: {e}")); return; }
    };

    // Drain stderr in background
    let stderr = child.stderr.take();
    tokio::spawn(async move {
        if let Some(se) = stderr {
            let mut reader = BufReader::new(se).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::debug!("worker stderr: {line}");
            }
        }
    });

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None    => { set_failed(&jobs, &job_id, "no stdout pipe"); return; }
    };
    let mut reader = BufReader::new(stdout).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        tracing::debug!(%job_id, "worker: {line}");
        let Ok(w) = serde_json::from_str::<WorkerLine>(&line) else { continue };

        let mut job = match jobs.get_mut(&job_id) {
            Some(j) => j,
            None    => break,
        };

        match w.status.as_str() {
            "done" => {
                if let Some(ref path) = w.onnx_path {
                    let url = format!("{base_url}/v1/wake/models/{job_id}/{model_name}.onnx");
                    job.status    = "done".into();
                    job.progress  = "done".into();
                    job.onnx_path = Some(path.clone());
                    job.onnx_url  = Some(url);
                    tracing::info!(%job_id, %path, "training done");
                } else {
                    job.status   = "failed".into();
                    job.progress = "done without onnx_path".into();
                }
            }
            "failed" => {
                let detail = w.stderr_tail
                    .or(w.progress)
                    .unwrap_or_else(|| "unknown".into());
                job.status   = "failed".into();
                job.progress = detail.chars().take(2000).collect();
                tracing::warn!(%job_id, "training failed");
            }
            _ => {
                job.status = w.status.clone();
                if let Some(p) = w.progress { job.progress = p; }
            }
        }
    }

    match child.wait().await {
        Ok(code) if !code.success() => {
            if let Some(mut job) = jobs.get_mut(&job_id) {
                if job.status != "done" && job.status != "failed" {
                    job.status   = "failed".into();
                    job.progress = format!("exit {}", code.code().unwrap_or(-1));
                }
            }
        }
        _ => {}
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn find_python() -> String {
    for candidate in &[
        "/opt/venv/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ] {
        if std::path::Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }
    "python3".into()
}

fn set_progress(jobs: &DashMap<String, JobRow>, job_id: &str, status: &str, msg: &str) {
    if let Some(mut job) = jobs.get_mut(job_id) {
        job.status   = status.into();
        job.progress = msg.into();
    }
}

fn set_failed(jobs: &DashMap<String, JobRow>, job_id: &str, msg: &str) {
    tracing::warn!(%job_id, "failed: {msg}");
    if let Some(mut job) = jobs.get_mut(job_id) {
        job.status   = "failed".into();
        job.progress = msg.chars().take(400).collect();
    }
}

fn err(code: StatusCode, error: &str, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (code, Json(serde_json::json!({"error": error, "message": message})))
}
