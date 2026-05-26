//! OpenAI-compatible async reverse proxy (the Rust data plane).
//!
//! Hot path: parse -> identify tenant -> admit (rate limit) -> filter
//! candidates by circuit -> route under a short sync lock (never held across
//! `.await`) -> stream the upstream SSE back. Pre-first-byte failover re-routes
//! optimally on the shrinking candidate set, with the breaker recording
//! success/failure. The in-flight `Drop` guard decrements load + releases the
//! tenant in-flight slot + emits the per-request structured log line on stream
//! end or client disconnect.
//!
//! Still deferred (present in Python): OpenTelemetry tracing,
//! scrape-loop driven backend health.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router as AxumRouter;
use futures_util::{Stream, StreamExt};

use crate::circuit::CircuitBreaker;
use crate::config::{Config, Strategy};
use crate::load::LoadTracker;
use crate::logging::{info_enabled, new_request_id, RequestLog};
use crate::metrics::{MetricsCollector, BLOCK_BUCKETS, LATENCY_BUCKETS};
use crate::radix_tree::RadixTree;
use crate::router::{RouteResult, Router};
use crate::tenancy::{tenant_seed, Admission, RateLimiter, Tenant, TenantRegistry};

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
    tenants: TenantRegistry,
    limiter: Mutex<RateLimiter>,
    cfg: Arc<Config>,
    clock: Instant,
}

pub type SharedState = Arc<AppState>;

pub fn build_state(cfg: Config, backends: Vec<Backend>) -> SharedState {
    let strategy = cfg.strategy;
    let cache_blocks = cfg.backend_cache_blocks;
    let fail_threshold = cfg.circuit_fail_threshold;
    let cooldown_s = cfg.circuit_cooldown_s;
    let tenants_spec = cfg.tenants.clone();
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
        tenants: TenantRegistry::new(&tenants_spec),
        limiter: Mutex::new(RateLimiter::new()),
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

fn parse_strategy(s: &str) -> Option<Strategy> {
    match s {
        "round_robin" => Some(Strategy::RoundRobin),
        "consistent_hash" => Some(Strategy::ConsistentHash),
        "prefix_tree" => Some(Strategy::PrefixTree),
        _ => None,
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

struct LogCtx {
    request_id: String,
    tenant: String,
    model: String,
    strategy: &'static str,
    match_blocks: usize,
    retries: u32,
    status: u16,
    cache_hit: Option<bool>,
    t_request: Instant,
}

/// Decrements per-backend in-flight + tenant in-flight + emits the per-request
/// log line on stream end or drop, so accounting can't leak even on client
/// disconnect.
struct InflightGuard {
    state: SharedState,
    backend: String,
    tokens: u64,
    /// Some((tenant_id, reserved_output)) when rate-limit enabled; else None.
    tenant_release: Option<(String, f64)>,
    /// Some(...) when log_level is INFO; else None (logging suppressed).
    log: Option<LogCtx>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.state.inner.lock() {
            inner.load.on_complete(&self.backend, self.tokens);
        }
        if let Some((tid, reserved)) = &self.tenant_release {
            if let Ok(mut lim) = self.state.limiter.lock() {
                lim.release(tid, *reserved, *reserved);
            }
        }
        if let Some(c) = self.log.take() {
            let dur = c.t_request.elapsed().as_secs_f64() * 1000.0;
            RequestLog {
                level: "INFO",
                request_id: &c.request_id,
                tenant: Some(&c.tenant),
                model: Some(&c.model),
                strategy: Some(c.strategy),
                backend: Some(&self.backend),
                match_blocks: Some(c.match_blocks),
                cache_hit: c.cache_hit,
                retries: Some(c.retries),
                status: c.status,
                reason: None,
                duration_ms: dur,
                output_tokens: None,
            }
            .emit();
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

fn ratelimit_response(t: &Tenant, adm: &Admission, request_id: &str) -> Response {
    let retry_after = if adm.retry_after.is_finite() {
        (adm.retry_after.ceil() as u64).max(1)
    } else {
        3600
    };
    let body = serde_json::json!({
        "error": {
            "message": format!("rate limit exceeded ({})", adm.reason.unwrap_or("")),
            "type": "rate_limit_exceeded"
        }
    });
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("content-type", "application/json")
        .header("Retry-After", retry_after)
        .header("X-RateLimit-Limit-Requests", t.rps as u64)
        .header("X-RateLimit-Limit-Tokens", t.tps as u64)
        .header("X-RateLimit-Remaining-Requests", adm.remaining_rps.max(0.0) as u64)
        .header("X-RateLimit-Remaining-Tokens", adm.remaining_tps.max(0.0) as u64)
        .header("x-request-id", request_id)
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn plain_response(status: StatusCode, msg: &str, request_id: &str) -> Response {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain")
        .header("x-request-id", request_id)
        .body(Body::from(msg.to_string()))
        .unwrap()
}

// ---------- /metrics ----------

async fn metrics_endpoint(State(state): State<SharedState>) -> impl IntoResponse {
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
                1.0,
                "1 if backend is healthy else 0",
                &[("backend", &b.id)],
            );
        }
    }
    {
        let lim = state.limiter.lock().unwrap();
        for (tid, &n) in lim.inflight.iter() {
            state.metrics.set_gauge(
                "gateway_tenant_inflight",
                n as f64,
                "In-flight requests per tenant",
                &[("tenant", tid)],
            );
        }
    }
    (
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
}

// ---------- POST /v1/chat/completions ----------

async fn chat_completions(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    let t_request = Instant::now();
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(new_request_id);
    let log_enabled = info_enabled(&state.cfg.log_level);

    let mut parsed: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return plain_response(StatusCode::BAD_REQUEST, "invalid JSON", &request_id),
    };
    let prompt = extract_prompt(&parsed);
    let model_str = parsed
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let strategy = headers
        .get("x-routing-strategy")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_strategy)
        .unwrap_or(state.strategy);
    let strat_name = strategy_name(strategy);

    // tenant + admission
    let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
    let tenant = state.tenants.resolve(auth_header);
    state.metrics.inc_counter(
        "gateway_tenant_requests_total",
        1.0,
        "Requests per tenant",
        &[("tenant", &tenant.id)],
    );

    let input_tokens =
        ((prompt.len() / state.cfg.chars_per_token.max(1) as usize) as f64).max(1.0);
    let max_tokens = parsed
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(state.cfg.default_output_tokens as u64);
    let reserved_output = max_tokens.min(state.cfg.max_output_tokens as u64) as f64;
    let est_cost = input_tokens + reserved_output;

    let mut admitted = false;
    if state.cfg.rate_limit_enabled {
        let now_lim = state.clock.elapsed().as_secs_f64();
        let adm = state.limiter.lock().unwrap().admit(&tenant, est_cost, now_lim);
        if !adm.allowed {
            state.metrics.inc_counter(
                "gateway_tenant_throttled_total",
                1.0,
                "Rate-limited requests per tenant",
                &[("tenant", &tenant.id), ("reason", adm.reason.unwrap_or(""))],
            );
            if log_enabled {
                let dur = t_request.elapsed().as_secs_f64() * 1000.0;
                RequestLog {
                    level: "INFO",
                    request_id: &request_id,
                    tenant: Some(&tenant.id),
                    model: Some(&model_str),
                    strategy: Some(strat_name),
                    backend: None,
                    match_blocks: None,
                    cache_hit: None,
                    retries: None,
                    status: 429,
                    reason: adm.reason,
                    duration_ms: dur,
                    output_tokens: None,
                }
                .emit();
            }
            return ratelimit_response(&tenant, &adm, &request_id);
        }
        admitted = true;
    }

    let backend_ids: Vec<String> = state.backends.iter().map(|b| b.id.clone()).collect();
    if backend_ids.is_empty() {
        if admitted {
            state.limiter.lock().unwrap().release(&tenant.id, reserved_output, 0.0);
        }
        state.metrics.inc_counter(
            "gateway_errors_total",
            1.0,
            "Gateway-side errors",
            &[("code", "503")],
        );
        if log_enabled {
            let dur = t_request.elapsed().as_secs_f64() * 1000.0;
            RequestLog {
                level: "INFO",
                request_id: &request_id,
                tenant: Some(&tenant.id),
                model: Some(&model_str),
                strategy: Some(strat_name),
                backend: None,
                match_blocks: None,
                cache_hit: None,
                retries: None,
                status: 503,
                reason: Some("no_backend"),
                duration_ms: dur,
                output_tokens: None,
            }
            .emit();
        }
        return plain_response(StatusCode::SERVICE_UNAVAILABLE, "no backends", &request_id);
    }

    // Per-tenant prefix isolation
    let mut seed: u64 = 0;
    if state.cfg.prefix_isolation == "tenant" {
        seed = tenant_seed(&tenant.id);
        parsed["cache_salt"] = serde_json::Value::String(tenant.id.clone());
    }
    let body_to_forward: Bytes = match serde_json::to_vec(&parsed) {
        Ok(v) => v.into(),
        Err(_) => body_bytes.clone(),
    };

    // Circuit filter
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
    let mut retries: u32 = 0;

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
                strategy,
                seed,
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
                state.inner.lock().unwrap().load.on_complete(&res.backend_id, res.tokens as u64);
                continue;
            }
        };

        match state
            .client
            .post(&url)
            .header("content-type", "application/json")
            .body(body_to_forward.clone())
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
                retries += 1;
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
            if admitted {
                state.limiter.lock().unwrap().release(&tenant.id, reserved_output, 0.0);
            }
            state.metrics.inc_counter(
                "gateway_errors_total",
                1.0,
                "Gateway-side errors",
                &[("code", "502")],
            );
            if log_enabled {
                let dur = t_request.elapsed().as_secs_f64() * 1000.0;
                RequestLog {
                    level: "INFO",
                    request_id: &request_id,
                    tenant: Some(&tenant.id),
                    model: Some(&model_str),
                    strategy: Some(strat_name),
                    backend: None,
                    match_blocks: None,
                    cache_hit: None,
                    retries: Some(retries),
                    status: 502,
                    reason: Some("unreachable"),
                    duration_ms: dur,
                    output_tokens: None,
                }
                .emit();
            }
            return plain_response(
                StatusCode::BAD_GATEWAY,
                "all candidate backends unreachable",
                &request_id,
            );
        }
    };

    state.breaker.lock().unwrap().record_success(&chosen.backend_id);
    state.metrics.inc_counter(
        "gateway_requests_total",
        1.0,
        "Total routed requests",
        &[("strategy", strat_name), ("backend", &chosen.backend_id)],
    );
    state.metrics.inc_counter(
        "gateway_tenant_tokens_total",
        input_tokens,
        "Input tokens accounted per tenant",
        &[("tenant", &tenant.id)],
    );
    state.metrics.observe(
        "gateway_prefix_match_blocks",
        chosen.match_blocks as f64,
        BLOCK_BUCKETS,
        "Prefix blocks reused (cache affinity) per request",
        &[],
    );

    let status = upstream.status().as_u16();
    let cache_hit_str = upstream
        .headers()
        .get("x-prefix-cache-hit")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let cache_hit_bool = cache_hit_str.as_deref().map(|s| s == "true");
    if let Some(ch) = &cache_hit_str {
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
    let log_ctx = if log_enabled {
        Some(LogCtx {
            request_id: request_id.clone(),
            tenant: tenant.id.clone(),
            model: model_str,
            strategy: strat_name,
            match_blocks: chosen.match_blocks,
            retries,
            status,
            cache_hit: cache_hit_bool,
            t_request,
        })
    } else {
        None
    };
    let guard = InflightGuard {
        state: state.clone(),
        backend: bid.clone(),
        tokens: toks,
        tenant_release: if admitted { Some((tenant.id.clone(), reserved_output)) } else { None },
        log: log_ctx,
    };
    let guarded = GuardedStream { inner: upstream.bytes_stream().boxed(), _guard: guard };

    let mut builder = Response::builder()
        .status(status)
        .header("x-gw-backend", bid)
        .header("x-gw-match-blocks", chosen.match_blocks.to_string())
        .header("x-request-id", &request_id);
    if let Some(ch) = cache_hit_str {
        builder = builder.header("x-prefix-cache-hit", ch);
    }
    builder.body(Body::from_stream(guarded)).unwrap()
}
