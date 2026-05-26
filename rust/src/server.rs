//! OpenAI-compatible async reverse proxy (the Rust data plane).
//!
//! Hot path: parse -> filter candidates by circuit -> route under a short sync
//! lock (never held across `.await`) -> stream the upstream SSE back. Pre-first-
//! byte failover re-routes optimally on the shrinking candidate set, with the
//! breaker recording success/failure. An in-flight `Drop` guard decrements load
//! even on client disconnect. Per-request Prometheus metrics are emitted along
//! the way; `/metrics` exposes them.
//!
//! Still deferred (present in Python): tenancy/rate-limiting, OpenTelemetry
//! tracing, structured JSON logging.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router as AxumRouter;
use futures_util::{Stream, StreamExt};

use crate::circuit::CircuitBreaker;
use crate::config::{Config, Strategy};
use crate::load::LoadTracker;
use crate::metrics::{MetricsCollector, BLOCK_BUCKETS, LATENCY_BUCKETS};
use crate::radix_tree::RadixTree;
use crate::router::{RouteResult, Router};

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
    metrics: MetricsCollector,
    breaker: Mutex<CircuitBreaker>,
    cfg: Arc<Config>,
    clock: Instant,
}

pub type SharedState = Arc<AppState>;

pub fn build_state(cfg: Config, backends: Vec<Backend>) -> SharedState {
    let strategy = cfg.strategy;
    let cache_blocks = cfg.backend_cache_blocks;
    let fail_threshold = cfg.circuit_fail_threshold;
    let cooldown_s = cfg.circuit_cooldown_s;
    let inner = Inner {
        tree: RadixTree::new(cache_blocks),
        load: LoadTracker::new(),
        router: Router::new(cfg.clone()),
    };
    Arc::new(AppState {
        inner: Mutex::new(inner),
        backends,
        strategy,
        client: reqwest::Client::new(),
        metrics: MetricsCollector::new(),
        breaker: Mutex::new(CircuitBreaker::new(fail_threshold, cooldown_s)),
        cfg: Arc::new(cfg),
        clock: Instant::now(),
    })
}

pub fn app(state: SharedState) -> AxumRouter {
    AxumRouter::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

fn strategy_name(s: Strategy) -> &'static str {
    match s {
        Strategy::RoundRobin => "round_robin",
        Strategy::ConsistentHash => "consistent_hash",
        Strategy::PrefixTree => "prefix_tree",
    }
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

// ---------- /metrics ----------

async fn metrics_endpoint(State(state): State<SharedState>) -> impl IntoResponse {
    // refresh point-in-time gauges from current state (pull model)
    let now = state.clock.elapsed().as_secs_f64();
    {
        let inner = state.inner.lock().unwrap();
        let breaker = state.breaker.lock().unwrap();
        for b in &state.backends {
            state.metrics.set_gauge(
                "gateway_inflight",
                inner.load.inflight(&b.id) as f64,
                "In-flight requests per backend",
                &[("backend", &b.id)],
            );
            state.metrics.set_gauge(
                "gateway_kv_usage",
                inner.load.kv_usage(&b.id),
                "Reconciled KV-cache usage (0..1)",
                &[("backend", &b.id)],
            );
            state.metrics.set_gauge(
                "gateway_circuit_state",
                breaker.state_code(&b.id, now) as f64,
                "Circuit state (0 closed, 1 half_open, 2 open)",
                &[("backend", &b.id)],
            );
            state.metrics.set_gauge(
                "gateway_backend_up",
                1.0, // Rust gateway has no scrape loop yet -> assume up
                "1 if backend is healthy else 0",
                &[("backend", &b.id)],
            );
        }
    }
    (
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
}

// ---------- POST /v1/chat/completions ----------

async fn chat_completions(State(state): State<SharedState>, body_bytes: Bytes) -> Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid JSON").into_response(),
    };
    let prompt = extract_prompt(&parsed);

    let backend_ids: Vec<String> = state.backends.iter().map(|b| b.id.clone()).collect();
    if backend_ids.is_empty() {
        state.metrics.inc_counter(
            "gateway_errors_total",
            1.0,
            "Gateway-side errors",
            &[("code", "503")],
        );
        return (StatusCode::SERVICE_UNAVAILABLE, "no backends").into_response();
    }

    // Exclude backends with an open circuit; degrade to all if every one is open.
    let now = state.clock.elapsed().as_secs_f64();
    let mut remaining: Vec<String> = {
        let breaker = state.breaker.lock().unwrap();
        backend_ids
            .iter()
            .filter(|b| breaker.allow(b, now))
            .cloned()
            .collect()
    };
    if remaining.is_empty() {
        remaining = backend_ids.clone();
    }

    let mut chosen: Option<RouteResult> = None;
    let mut upstream: Option<reqwest::Response> = None;
    let mut routing_recorded = false;

    // Failover loop. Safe only before the first byte; once streaming starts,
    // retrying would duplicate tokens, so a mid-stream failure propagates.
    for _ in 0..=state.cfg.max_retries {
        if remaining.is_empty() {
            break;
        }
        let t0 = Instant::now();
        let res = {
            let mut guard = state.inner.lock().unwrap();
            let inner = &mut *guard;
            let r = inner.router.choose(
                &prompt,
                &remaining,
                &inner.tree,
                &inner.load,
                state.strategy,
                0,
            );
            inner.tree.insert(&r.hashes, &r.backend_id);
            inner.load.on_dispatch(&r.backend_id, r.tokens as u64);
            r
        };
        if !routing_recorded {
            state.metrics.observe(
                "gateway_routing_seconds",
                t0.elapsed().as_secs_f64(),
                LATENCY_BUCKETS,
                "Time spent in the routing decision (gateway added latency)",
                &[],
            );
            routing_recorded = true;
        }

        let url = state
            .backends
            .iter()
            .find(|b| b.id == res.backend_id)
            .map(|b| format!("{}/v1/chat/completions", b.url));
        let url = match url {
            Some(u) => u,
            None => {
                state
                    .inner
                    .lock()
                    .unwrap()
                    .load
                    .on_complete(&res.backend_id, res.tokens as u64);
                continue;
            }
        };

        match state
            .client
            .post(&url)
            .header("content-type", "application/json")
            .body(body_bytes.clone())
            .send()
            .await
        {
            Ok(resp) => {
                chosen = Some(res);
                upstream = Some(resp);
                break;
            }
            Err(_) => {
                let now2 = state.clock.elapsed().as_secs_f64();
                let bid = res.backend_id.clone();
                let toks = res.tokens as u64;
                state.inner.lock().unwrap().load.on_complete(&bid, toks);
                state.breaker.lock().unwrap().record_failure(&bid, now2);
                remaining.retain(|x| x != &bid);
                state.metrics.inc_counter(
                    "gateway_retries_total",
                    1.0,
                    "Failover attempts after a backend connect failure",
                    &[],
                );
            }
        }
    }

    let (chosen, upstream) = match (chosen, upstream) {
        (Some(c), Some(u)) => (c, u),
        _ => {
            state.metrics.inc_counter(
                "gateway_errors_total",
                1.0,
                "Gateway-side errors",
                &[("code", "502")],
            );
            return (StatusCode::BAD_GATEWAY, "all candidate backends unreachable")
                .into_response();
        }
    };

    state.breaker.lock().unwrap().record_success(&chosen.backend_id);
    let strat = strategy_name(state.strategy);
    state.metrics.inc_counter(
        "gateway_requests_total",
        1.0,
        "Total routed requests",
        &[("strategy", strat), ("backend", &chosen.backend_id)],
    );
    state.metrics.observe(
        "gateway_prefix_match_blocks",
        chosen.match_blocks as f64,
        BLOCK_BUCKETS,
        "Prefix blocks reused (cache affinity) per request",
        &[],
    );

    let status = upstream.status().as_u16();
    let cache_hit = upstream
        .headers()
        .get("x-prefix-cache-hit")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    if let Some(ch) = &cache_hit {
        let name = if ch == "true" {
            "gateway_cache_hits_total"
        } else {
            "gateway_cache_misses_total"
        };
        state.metrics.inc_counter(
            name,
            1.0,
            "Backend prefix-cache outcomes",
            &[("backend", &chosen.backend_id)],
        );
    }

    let bid = chosen.backend_id.clone();
    let toks = chosen.tokens as u64;
    let guard = InflightGuard { state: state.clone(), backend: bid.clone(), tokens: toks };
    let guarded = GuardedStream { inner: upstream.bytes_stream().boxed(), _guard: guard };

    let mut builder = Response::builder()
        .status(status)
        .header("x-gw-backend", bid)
        .header("x-gw-match-blocks", chosen.match_blocks.to_string());
    if let Some(ch) = cache_hit {
        builder = builder.header("x-prefix-cache-hit", ch);
    }
    builder.body(Body::from_stream(guarded)).unwrap()
}
