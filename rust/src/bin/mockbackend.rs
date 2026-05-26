//! Fast mock backend: instant static SSE response.
//!
//! Exists so the throughput benchmark isn't bottlenecked by the slower Python
//! `uvicorn` mock (which caps ~200 req/s). This responds in microseconds, so the
//! gateway becomes the variable under test. Not a prefix-cache simulator — for
//! routing/cache correctness use the Python mock; for raw throughput use this.
//!
//! Usage: `mockbackend [bind_addr]`  (default 127.0.0.1:9101)

use std::env;

use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;

const SSE: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n",
    "data: [DONE]\n\n",
);

async fn chat() -> impl IntoResponse {
    (
        [("content-type", "text/event-stream"), ("x-prefix-cache-hit", "false")],
        SSE,
    )
}

async fn metrics() -> impl IntoResponse {
    ([("content-type", "application/json")], "{\"kv_usage\":0.0,\"running\":0}")
}

#[tokio::main]
async fn main() {
    let addr = env::args()
        .nth(1)
        .or_else(|| env::var("MOCK_ADDR").ok())
        .unwrap_or_else(|| "127.0.0.1:9101".to_string());
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat));
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    println!("mock backend listening on {addr}");
    axum::serve(listener, app).await.expect("serve");
}
