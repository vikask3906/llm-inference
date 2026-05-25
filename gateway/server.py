from __future__ import annotations

"""OpenAI-compatible reverse proxy (the data plane).

Hot path: parse -> filter by model -> prefix-affinity + load route -> stream the
upstream SSE response back without buffering. A background loop scrapes backend
/metrics (the control-plane job) and reconciles load + health off the hot path.

This is the Python MVP. Phase 2 rewrites this hot path in Rust (Tokio/hyper) for
predictable sub-2ms added latency; the control loop stays in Python.
"""

import asyncio
import contextlib
import math
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from opentelemetry.trace import Status, StatusCode

from .backends import BackendRegistry
from .circuit import CircuitBreaker
from .config import Config
from .load_tracker import LoadTracker
from .logging_setup import configure_logging, log_event
from .metrics import BLOCK_BUCKETS, MetricsCollector
from .radix_tree import RadixTree
from .router import Router
from .tenancy import RateLimiter, TenantRegistry, tenant_seed
from .tracing import get_tracer, setup_tracing

cfg = Config.from_env()
registry = BackendRegistry(cfg)
tree = RadixTree(cfg.backend_cache_blocks)
load = LoadTracker()
router = Router(cfg, tree, load)
metrics = MetricsCollector()
breaker = CircuitBreaker(cfg.circuit_fail_threshold, cfg.circuit_cooldown_s)
tenants = TenantRegistry(cfg.tenants)
limiter = RateLimiter()
log = configure_logging(cfg.log_level)


def _trace_id(span) -> str | None:
    tid = span.get_span_context().trace_id
    return format(tid, "032x") if tid else None

# Tracing is a no-op unless an exporter is configured (OTLP endpoint, or the
# in-memory exporter in tests), so it costs nothing in an unconfigured deploy.
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        setup_tracing(OTLPSpanExporter())
    except Exception:
        pass
tracer = get_tracer()


def extract_prompt(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt) // cfg.chars_per_token)


def ratelimit_headers(tenant, adm) -> dict[str, str]:
    h = {
        "X-RateLimit-Limit-Requests": str(int(tenant.rps)),
        "X-RateLimit-Limit-Tokens": str(int(tenant.tps)),
        "X-RateLimit-Remaining-Requests": str(max(0, int(adm.remaining_rps))),
        "X-RateLimit-Remaining-Tokens": str(max(0, int(adm.remaining_tps))),
    }
    if adm.retry_after and adm.retry_after != float("inf"):
        h["Retry-After"] = str(max(1, math.ceil(adm.retry_after)))
    return h


async def scrape_loop(client: httpx.AsyncClient) -> None:
    while True:
        for b in registry.all():
            try:
                resp = await client.get(f"{b.url}/metrics", timeout=1.0)
                data = resp.json()
                load.update_scraped(b.id, float(data.get("kv_usage", 0.0)))
                registry.set_health(b.id, True)
                breaker.record_success(b.id)     # health probe drives recovery
            except Exception:
                registry.set_health(b.id, False)
                breaker.record_failure(b.id)
                tree.remove_backend(b.id)        # membership eviction
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    task = asyncio.create_task(scrape_loop(app.state.client))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"backends": {b.id: b.healthy for b in registry.all()},
            "inflight": dict(load.inflight)}


@app.get("/metrics")
async def metrics_endpoint():
    # refresh point-in-time gauges from current state (pull model)
    for b in registry.all():
        metrics.set_gauge("gateway_backend_up", 1.0 if b.healthy else 0.0,
                          help="1 if backend is healthy else 0", backend=b.id)
        metrics.set_gauge("gateway_inflight", load.inflight.get(b.id, 0),
                          help="In-flight requests per backend", backend=b.id)
        metrics.set_gauge("gateway_kv_usage", load.kv_usage.get(b.id, 0.0),
                          help="Reconciled KV-cache usage (0..1)", backend=b.id)
        metrics.set_gauge("gateway_circuit_state", breaker.state_code(b.id),
                          help="Circuit state (0 closed, 1 half_open, 2 open)", backend=b.id)
    for tid, n in list(limiter.inflight.items()):
        metrics.set_gauge("gateway_tenant_inflight", n,
                          help="In-flight requests per tenant", tenant=tid)
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    prompt = extract_prompt(body.get("messages", []))
    strategy = request.headers.get("x-routing-strategy")

    eff_strategy = strategy or cfg.strategy

    # --- tenant identification + admission control (before routing) ---
    tenant = tenants.resolve(request.headers)
    metrics.inc_counter("gateway_tenant_requests_total", help="Requests per tenant",
                        tenant=tenant.id)

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    t_request = time.perf_counter()
    span = tracer.start_span("chat.completion")
    span.set_attribute("request.id", request_id)
    span.set_attribute("llm.model", str(model))
    span.set_attribute("routing.strategy", eff_strategy)
    span.set_attribute("tenant.id", tenant.id)

    input_tokens = estimate_prompt_tokens(prompt)
    reserved_output = min(int(body.get("max_tokens") or cfg.default_output_tokens),
                          cfg.max_output_tokens)
    est_cost = input_tokens + reserved_output

    if cfg.rate_limit_enabled:
        adm = limiter.admit(tenant, est_cost)
        if not adm.allowed:
            metrics.inc_counter("gateway_tenant_throttled_total",
                                help="Rate-limited requests per tenant",
                                tenant=tenant.id, reason=adm.reason)
            span.set_attribute("http.status_code", 429)
            span.set_attribute("ratelimit.reason", adm.reason or "")
            span.set_status(Status(StatusCode.ERROR, "rate_limited"))
            span.end()
            rl_headers = ratelimit_headers(tenant, adm)
            rl_headers["x-request-id"] = request_id
            log_event(log, "request", request_id=request_id, trace_id=_trace_id(span),
                      tenant=tenant.id, model=str(model), status=429, reason=adm.reason,
                      duration_ms=round((time.perf_counter() - t_request) * 1000, 2))
            return JSONResponse(
                {"error": {"message": f"rate limit exceeded ({adm.reason})",
                           "type": "rate_limit_exceeded"}},
                status_code=429, headers=rl_headers)

    def release(actual_output: int) -> None:
        if cfg.rate_limit_enabled:
            limiter.release(tenant, reserved_output, actual_output)

    ids = registry.ids_for(model)
    if not ids:
        release(0)
        metrics.inc_counter("gateway_errors_total", help="Gateway-side errors", code="503")
        span.set_attribute("http.status_code", 503)
        span.set_status(Status(StatusCode.ERROR, "no_backend"))
        span.end()
        log_event(log, "request", request_id=request_id, trace_id=_trace_id(span),
                  tenant=tenant.id, model=str(model), status=503,
                  duration_ms=round((time.perf_counter() - t_request) * 1000, 2))
        return JSONResponse({"error": f"no healthy backend for model {model!r}"},
                            status_code=503, headers={"x-request-id": request_id})

    # Per-tenant prefix isolation: seed the gateway's routing hashes AND tell the
    # backend to salt its own KV cache (vLLM `cache_salt`), so tenants neither
    # share nor leak (via TTFT) each other's cache. "global" keeps caches shared.
    seed = 0
    if cfg.prefix_isolation == "tenant":
        seed = tenant_seed(tenant.id)
        body["cache_salt"] = tenant.id

    # Exclude backends with an open circuit. If every circuit is open, degrade to
    # trying all of them (better to attempt than to hard-fail).
    remaining = [b for b in ids if breaker.allow(b)] or list(ids)

    client: httpx.AsyncClient = request.app.state.client
    r = cm = upstream = None
    routing_recorded = False
    retries = 0

    # Failover loop: re-route optimally on the shrinking candidate set. This is
    # safe ONLY before the first byte; once streaming starts, retrying would
    # duplicate tokens, so a mid-stream failure propagates instead.
    for _ in range(cfg.max_retries + 1):
        if not remaining:
            break
        t0 = time.perf_counter()
        r = router.choose(prompt, remaining, strategy=strategy, seed=seed)
        if not routing_recorded:
            metrics.observe("gateway_routing_seconds", time.perf_counter() - t0,
                            help="Time spent in the routing decision (gateway added latency)")
            routing_recorded = True
        # Reflect the decision in shared state BEFORE connecting, so concurrent
        # requests for the same prefix converge instead of duplicating cache.
        tree.insert(r.hashes, r.backend_id)
        load.on_dispatch(r.backend_id, r.tokens)
        cm = client.stream("POST", f"{registry.url(r.backend_id)}/v1/chat/completions", json=body)
        try:
            upstream = await cm.__aenter__()
            break
        except Exception:
            load.on_complete(r.backend_id, r.tokens)   # undo dispatch; tree belief is harmless
            breaker.record_failure(r.backend_id)
            registry.set_health(r.backend_id, False)
            remaining.remove(r.backend_id)
            retries += 1
            metrics.inc_counter("gateway_retries_total",
                                help="Failover attempts after a backend connect failure")
            upstream = None

    if upstream is None:
        release(0)
        metrics.inc_counter("gateway_errors_total", help="Gateway-side errors", code="502")
        span.set_attribute("http.status_code", 502)
        span.set_attribute("routing.retries", retries)
        span.set_status(Status(StatusCode.ERROR, "unreachable"))
        span.end()
        log_event(log, "request", request_id=request_id, trace_id=_trace_id(span),
                  tenant=tenant.id, model=str(model), status=502, retries=retries,
                  duration_ms=round((time.perf_counter() - t_request) * 1000, 2))
        return JSONResponse({"error": "all candidate backends unreachable"},
                            status_code=502, headers={"x-request-id": request_id})

    breaker.record_success(r.backend_id)
    metrics.inc_counter("gateway_requests_total", help="Total routed requests",
                        strategy=eff_strategy, backend=r.backend_id)
    metrics.inc_counter("gateway_tenant_tokens_total", value=input_tokens,
                        help="Input tokens accounted per tenant", tenant=tenant.id)
    metrics.observe("gateway_prefix_match_blocks", r.match_blocks, buckets=BLOCK_BUCKETS,
                    help="Prefix blocks reused (cache affinity) per request")

    headers = {"x-gw-backend": r.backend_id, "x-gw-match-blocks": str(r.match_blocks),
               "x-request-id": request_id}
    cache_hit = upstream.headers.get("x-prefix-cache-hit")
    if cache_hit is not None:
        headers["x-prefix-cache-hit"] = cache_hit
        metric = "gateway_cache_hits_total" if cache_hit == "true" else "gateway_cache_misses_total"
        metrics.inc_counter(metric, help="Backend prefix-cache outcomes", backend=r.backend_id)

    span.set_attribute("routing.backend", r.backend_id)
    span.set_attribute("routing.match_blocks", r.match_blocks)
    span.set_attribute("routing.retries", retries)
    span.set_attribute("http.status_code", upstream.status_code)
    if cache_hit is not None:
        span.set_attribute("cache.hit", cache_hit == "true")

    final_r, final_cm = r, cm

    async def body_iter():
        out_events = 0
        try:
            async for chunk in upstream.aiter_raw():
                out_events += chunk.count(b"data:")
                yield chunk
        finally:
            await final_cm.__aexit__(None, None, None)
            load.on_complete(final_r.backend_id, final_r.tokens)
            # reconcile TPS: actual streamed tokens vs the reserved estimate
            actual_output = max(0, out_events - 1)     # minus the [DONE] event
            release(actual_output)
            span.set_attribute("output.tokens", actual_output)
            span.set_status(Status(StatusCode.OK))
            span.end()
            log_event(log, "request", request_id=request_id, trace_id=_trace_id(span),
                      tenant=tenant.id, model=str(model), strategy=eff_strategy,
                      backend=final_r.backend_id, match_blocks=final_r.match_blocks,
                      cache_hit=(cache_hit == "true") if cache_hit is not None else None,
                      retries=retries, output_tokens=actual_output, status=200,
                      duration_ms=round((time.perf_counter() - t_request) * 1000, 2))

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )
