//! OpenAI-compatible async reverse proxy (the Rust data plane).
//!
//! Hot path: parse -> route (radix-tree prefix match + est-TTFT, under a short
//! sync lock) -> stream the upstream SSE response straight back. The routing
//! lock is never held across an `.await`. An in-flight `Drop` guard decrements
//! load even if the client disconnects mid-stream.
//!
//! Deferred to later phases (present in the Python version): circuit breaker,
//! tenancy/rate limiting, Prometheus metrics, OpenTelemetry tracing.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router as AxumRouter;
use futures_util::{Stream, StreamExt};

use crate::config::{Config, Strategy};
use crate::load::LoadTracker;
use crate::radix_tree::RadixTree;
use crate::router::Router;

pub struct Backend {
    pub id: String,
    pub url: String,
}

struct Inner {
    tree: RadixTree,
    load: LoadTracker,
    router: Router,
}

pub struct AppState {
    inner: Mutex<Inner>,
    backends: Vec<Backend>,
    strategy: Strategy,
    client: reqwest::Client,
}

pub type SharedState = Arc<AppState>;

pub fn build_state(cfg: Config, backends: Vec<Backend>) -> SharedState {
    let strategy = cfg.strategy;
    let inner = Inner {
        tree: RadixTree::new(cfg.backend_cache_blocks),
        load: LoadTracker::new(),
        router: Router::new(cfg),
    };
    Arc::new(AppState {
        inner: Mutex::new(inner),
        backends,
        strategy,
        client: reqwest::Client::new(),
    })
}

pub fn app(state: SharedState) -> AxumRouter {
    AxumRouter::new()
        .route("/healthz", get(healthz))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

fn extract_prompt(body: &serde_json::Value) -> String {
    let mut s = String::new();
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for m in msgs {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(role);
            s.push_str(": ");
            s.push_str(content);
        }
    }
    s
}

/// Decrements in-flight when the response stream ends OR is dropped (client
/// disconnect), so load accounting can't leak.
struct InflightGuard {
    state: SharedState,
    backend: String,
    tokens: u64,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.state.inner.lock() {
            inner.load.on_complete(&self.backend, self.tokens);
        }
    }
}

/// Wraps the upstream byte stream and carries the in-flight guard so the guard
/// drops exactly when the response stream finishes or is cancelled.
struct GuardedStream<S> {
    inner: S,
    _guard: InflightGuard,
}

impl<S: Stream + Unpin> Stream for GuardedStream<S> {
    type Item = S::Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }
}

async fn chat_completions(State(state): State<SharedState>, body_bytes: Bytes) -> Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid JSON").into_response(),
    };
    let prompt = extract_prompt(&parsed);

    let backend_ids: Vec<String> = state.backends.iter().map(|b| b.id.clone()).collect();
    if backend_ids.is_empty() {
        return (StatusCode::SERVICE_UNAVAILABLE, "no backends").into_response();
    }

    // Routing decision + state update under a SHORT sync lock (no await held).
    let (backend_id, tokens, match_blocks) = {
        let mut guard = state.inner.lock().unwrap();
        let inner = &mut *guard;
        let res = inner
            .router
            .choose(&prompt, &backend_ids, &inner.tree, &inner.load, state.strategy, 0);
        inner.tree.insert(&res.hashes, &res.backend_id);
        inner.load.on_dispatch(&res.backend_id, res.tokens as u64);
        (res.backend_id, res.tokens as u64, res.match_blocks)
    };

    let url = match state.backends.iter().find(|b| b.id == backend_id) {
        Some(b) => format!("{}/v1/chat/completions", b.url),
        None => return (StatusCode::INTERNAL_SERVER_ERROR, "backend lookup failed").into_response(),
    };

    // Proxy + stream the upstream SSE back (the lock is already released).
    let upstream = state
        .client
        .post(&url)
        .header("content-type", "application/json")
        .body(body_bytes.clone())
        .send()
        .await;

    let resp = match upstream {
        Ok(r) => r,
        Err(_) => {
            // pre-first-byte failure: undo the dispatch and report 502
            state.inner.lock().unwrap().load.on_complete(&backend_id, tokens);
            return (StatusCode::BAD_GATEWAY, "backend unreachable").into_response();
        }
    };

    let status = resp.status().as_u16();
    let cache_hit = resp
        .headers()
        .get("x-prefix-cache-hit")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let guard = InflightGuard { state: state.clone(), backend: backend_id.clone(), tokens };
    let guarded = GuardedStream { inner: resp.bytes_stream().boxed(), _guard: guard };

    let mut builder = Response::builder()
        .status(status)
        .header("x-gw-backend", backend_id)
        .header("x-gw-match-blocks", match_blocks.to_string());
    if let Some(ch) = cache_hit {
        builder = builder.header("x-prefix-cache-hit", ch);
    }
    builder.body(Body::from_stream(guarded)).unwrap()
}
