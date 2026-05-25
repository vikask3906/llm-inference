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
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .backends import BackendRegistry
from .circuit import CircuitBreaker
from .config import Config
from .load_tracker import LoadTracker
from .metrics import BLOCK_BUCKETS, MetricsCollector
from .radix_tree import RadixTree
from .router import Router

cfg = Config.from_env()
registry = BackendRegistry(cfg)
tree = RadixTree(cfg.backend_cache_blocks)
load = LoadTracker()
router = Router(cfg, tree, load)
metrics = MetricsCollector()
breaker = CircuitBreaker(cfg.circuit_fail_threshold, cfg.circuit_cooldown_s)


def extract_prompt(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


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
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    prompt = extract_prompt(body.get("messages", []))
    strategy = request.headers.get("x-routing-strategy")

    eff_strategy = strategy or cfg.strategy
    ids = registry.ids_for(model)
    if not ids:
        metrics.inc_counter("gateway_errors_total", help="Gateway-side errors", code="503")
        return JSONResponse({"error": f"no healthy backend for model {model!r}"},
                            status_code=503)

    # Exclude backends with an open circuit. If every circuit is open, degrade to
    # trying all of them (better to attempt than to hard-fail).
    remaining = [b for b in ids if breaker.allow(b)] or list(ids)

    client: httpx.AsyncClient = request.app.state.client
    r = cm = upstream = None
    routing_recorded = False

    # Failover loop: re-route optimally on the shrinking candidate set. This is
    # safe ONLY before the first byte; once streaming starts, retrying would
    # duplicate tokens, so a mid-stream failure propagates instead.
    for _ in range(cfg.max_retries + 1):
        if not remaining:
            break
        t0 = time.perf_counter()
        r = router.choose(prompt, remaining, strategy=strategy)
        if not routing_recorded:
            metrics.observe("gateway_routing_seconds", time.perf_counter() - t0,
                            help="Time spent in the routing decision (gateway added latency)")
            routing_recorded = True
        cm = client.stream("POST", f"{registry.url(r.backend_id)}/v1/chat/completions", json=body)
        try:
            upstream = await cm.__aenter__()
            break
        except Exception:
            breaker.record_failure(r.backend_id)
            registry.set_health(r.backend_id, False)
            remaining.remove(r.backend_id)
            metrics.inc_counter("gateway_retries_total",
                                help="Failover attempts after a backend connect failure")
            upstream = None

    if upstream is None:
        metrics.inc_counter("gateway_errors_total", help="Gateway-side errors", code="502")
        return JSONResponse({"error": "all candidate backends unreachable"}, status_code=502)

    breaker.record_success(r.backend_id)
    metrics.inc_counter("gateway_requests_total", help="Total routed requests",
                        strategy=eff_strategy, backend=r.backend_id)
    metrics.observe("gateway_prefix_match_blocks", r.match_blocks, buckets=BLOCK_BUCKETS,
                    help="Prefix blocks reused (cache affinity) per request")
    tree.insert(r.hashes, r.backend_id)          # commit belief
    load.on_dispatch(r.backend_id, r.tokens)

    headers = {"x-gw-backend": r.backend_id, "x-gw-match-blocks": str(r.match_blocks)}
    cache_hit = upstream.headers.get("x-prefix-cache-hit")
    if cache_hit is not None:
        headers["x-prefix-cache-hit"] = cache_hit
        metric = "gateway_cache_hits_total" if cache_hit == "true" else "gateway_cache_misses_total"
        metrics.inc_counter(metric, help="Backend prefix-cache outcomes", backend=r.backend_id)

    final_r, final_cm = r, cm

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await final_cm.__aexit__(None, None, None)
            load.on_complete(final_r.backend_id, final_r.tokens)

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )
