//! koe-wake-train HTTP frontend.
//!
//! Scaffold — serves the API contract in docs/wake-train-protocol.md but does
//! not actually spawn the Python worker yet. Job status is held in an in-memory
//! DashMap which is fine for a single instance; scaling past 1 instance will
//! need a real queue.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use ulid::Ulid;

#[derive(Clone)]
struct AppState {
    jobs: Arc<DashMap<String, JobRow>>,
}

#[derive(Clone, Serialize)]
struct JobRow {
    status: String,     // pending | running | done | failed
    progress: String,
    onnx_url: Option<String>,
}

#[derive(Deserialize)]
struct TrainReq {
    text: String,
    model_name: String,
    lang: Option<String>,
}

#[derive(Serialize)]
struct SubmitResp {
    job_id: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = AppState {
        jobs: Arc::new(DashMap::new()),
    };

    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/wake/train", post(submit))
        .route("/v1/wake/train/:job_id", get(status))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
    tracing::info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn submit(
    State(state): State<AppState>,
    Json(req): Json<TrainReq>,
) -> Result<(StatusCode, Json<SubmitResp>), (StatusCode, Json<serde_json::Value>)> {
    if req.text.chars().count() < 2 || req.text.chars().count() > 32 {
        return Err(err(StatusCode::BAD_REQUEST, "invalid_input", "text must be 2-32 chars"));
    }
    if !req
        .model_name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        || req.model_name.is_empty()
        || req.model_name.len() > 32
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "invalid_input",
            "model_name must match [a-z0-9_]{1,32}",
        ));
    }

    let job_id = Ulid::new().to_string();
    state.jobs.insert(
        job_id.clone(),
        JobRow {
            status: "pending".into(),
            progress: "queued".into(),
            onnx_url: None,
        },
    );

    // TODO: actually spawn the Python worker. For now the job stays "pending"
    // so the contract is testable end-to-end.
    tracing::info!(%job_id, text = %req.text, model = %req.model_name, lang = ?req.lang, "job submitted");

    Ok((StatusCode::ACCEPTED, Json(SubmitResp { job_id })))
}

async fn status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobRow>, (StatusCode, Json<serde_json::Value>)> {
    state
        .jobs
        .get(&job_id)
        .map(|r| Json(r.clone()))
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "not_found", "no such job"))
}

fn err(code: StatusCode, error: &str, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        code,
        Json(serde_json::json!({"error": error, "message": message})),
    )
}
