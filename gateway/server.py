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
import hmac
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
from .extensions.bpe_hashing import count_tokens_with_fallback
from .extensions.lora import apply_adapter_config, lora_filter, parse_model_spec
from .extensions.semantic_cache import SemanticCache, default_embedder
from .extensions.speculative import race, top_k_backends
from .extensions.ttft_predictor import Observation, ObservationLogger, TTFTPredictor
from .load_tracker import LoadTracker
from .logging_setup import configure_logging, log_event
from .metrics import BLOCK_BUCKETS, MetricsCollector
from .auth import Authenticator, parse_api_keys
from .cluster import ClusterConfig, ClusterCoordinator, make_bus, make_store
from .radix_tree import RadixTree
from .hashing import block_hashes as _block_hashes
from .router import RouteResult, Router
from .tenancy import RateLimiter, TenantRegistry, tenant_seed
from .tracing import get_tracer, setup_tracing

# Opt-in standalone routers wired behind their own flags (default OFF), so the
# benchmarked text hot path is untouched unless explicitly enabled.
from .admission.config import AdmissionConfig
from .admission.policy import ADMIT, AdmissionRequest, FleetState
from .admission.policy import decide as admission_decide
from .autoscale.config import AutoscaleConfig
from .autoscale.planner import ScalingSignal, plan as autoscale_plan
from .dag.config import DagSchedConfig
from .dag.graph import DagError, DagNode, RequestDag
from .dag.scheduler import schedule as dag_schedule
from .disagg.config import DisaggConfig
from .disagg.pools import PoolRegistry
from .disagg.router import choose_disaggregated
from .multimodal.config import MultiModalConfig
from .multimodal.index import MediaAffinityIndex
from .multimodal.request import parse_multimodal_request
from .multimodal.router import choose_multimodal_backend, parse_capabilities
from .rag.config import RagConfig
from .rag.index import ChunkAffinityIndex
from .rag.router import choose_rag_backend
from .rag.structure import (
    build_messages as rag_build_messages,
    canonicalize_chunks,
    chunk_ids as rag_chunk_ids,
    parse_rag_request,
)

cfg = Config.from_env()
registry = BackendRegistry(cfg)
tree = RadixTree(cfg.backend_cache_blocks)
load = LoadTracker()
# Predictive TTFT: load a trained model if one is configured (router blends it
# with the static formula); log per-request (features, observed TTFT) for
# offline training when an observations path is set. Both no-op when unset.
predictor = TTFTPredictor(cfg.ttft_model_path) if cfg.ttft_model_path else None
obs_logger = ObservationLogger(cfg.ttft_observations_path) if cfg.ttft_observations_path else None
router = Router(cfg, tree, load, predictor=predictor)
metrics = MetricsCollector()
breaker = CircuitBreaker(cfg.circuit_fail_threshold, cfg.circuit_cooldown_s)
tenants = TenantRegistry(cfg.tenants)
limiter = RateLimiter()
# API-key auth (opt-in): valid set = configured tenant keys + extra GW_API_KEYS.
# require=False by default, so check() is a no-op and the gateway stays open.
authenticator = Authenticator(tenants.keys() | parse_api_keys(cfg.api_keys),
                              cfg.require_auth, parse_api_keys(cfg.api_key_hashes))
log = configure_logging(cfg.log_level)

# LoRA-aware routing: attach declared adapters to the backend objects so the
# request-time filter can match "base:adapter" specs. No-op when unconfigured.
if cfg.backend_adapters:
    apply_adapter_config(registry, cfg.backend_adapters)

# Semantic cache sits before routing: a paraphrase hit returns a cached
# response without touching a backend. Built lazily; an embedder import/load
# failure leaves it disabled rather than failing the gateway.
sem_cache: SemanticCache | None = None
if cfg.semantic_cache_enabled:
    try:
        sem_cache = SemanticCache(
            default_embedder(cfg.semantic_cache_model),
            cfg.semantic_cache_threshold,
            cfg.semantic_cache_max_entries_per_tenant,
        )
    except Exception:
        sem_cache = None

# RAG-aware routing (opt-in): when a request carries a structured RAG payload,
# canonicalize its retrieved chunks into a cacheable system prefix and route to
# the backend already holding the most of those chunks. No-op when disabled.
rag_cfg = RagConfig.from_env()
rag_index = ChunkAffinityIndex(rag_cfg.cache_capacity_chunks)

# SLO-aware admission control (opt-in): before dispatch, fail fast on
# deadline-infeasible requests and shed/queue low-priority traffic under fleet
# pressure (gold protected). No-op when disabled.
adm_cfg = AdmissionConfig.from_env()

# Disaggregated prefill/decode routing (opt-in): when pool roles are configured,
# split prefill and decode phases across specialised backends. No-op when
# pools is empty.
disagg_cfg = DisaggConfig.from_env()
disagg_pools = PoolRegistry.from_spec(
    [b.id for b in registry.all()], disagg_cfg.pools) if disagg_cfg.pools else None

# Multi-modal capability + affinity routing (opt-in): filter to backends that
# serve the request's modalities, route by media-affinity cache. No-op when
# disabled.
mm_cfg = MultiModalConfig.from_env()
mm_capabilities: dict[str, set[str]] = (
    parse_capabilities(mm_cfg.capabilities) if mm_cfg.enabled else {})
mm_index = MediaAffinityIndex(mm_cfg.cache_capacity_media)

# DAG scheduler (opt-in): exposes /v1/dag/schedule for cache-locality-aware
# multi-step workflow planning. No-op when disabled.
dag_cfg = DagSchedConfig.from_env()

# Autoscale planner (opt-in): runs in the scrape loop, emits scaling
# recommendations as metrics + a /autoscale GET endpoint. No-op when disabled.
autoscale_cfg = AutoscaleConfig.from_env()
_autoscale_request_count = 0          # simple counter, snapshot each scrape tick
_autoscale_last_snapshot = (0.0, 0)   # (time, count) for RPS calc
_autoscale_last_decision = None       # most recent ScalingDecision
_autoscale_last_scale_up = float("-inf")
_autoscale_last_scale_down = float("-inf")

# Multi-replica prefix-state replication (opt-in): when enabled, each replica
# publishes its radix-tree mutations to a shared bus and applies peers' mutations
# in a background loop, so prefix-aware routing works across a horizontally
# scaled gateway fleet. No-op (single-replica behaviour) when disabled.
cluster_cfg = ClusterConfig.from_env()
cluster = None
if cluster_cfg.enabled:
    try:
        cluster = ClusterCoordinator(tree, make_bus(cluster_cfg), cluster_cfg.replica_id,
                                     registry=registry, store=make_store(cluster_cfg))
        router.fleet = cluster.fleet   # router scores by fleet-wide (local+peer) load
        cluster.warm_start()           # seed drain state from the durable snapshot
    except Exception:
        cluster = None            # a bus init failure must not break the gateway

# Latency histogram buckets (seconds) for per-route SLO metrics -- wider than the
# default LATENCY_BUCKETS so real TTFT / total-latency tails land in a bucket.
SLO_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


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
    # Accurate token count for quota/admission when enabled (falls back to the
    # char heuristic if the tokenizer can't load); the cheap heuristic otherwise.
    n, _ = count_tokens_with_fallback(prompt, cfg.chars_per_token,
                                      cfg.tokenizer_model,
                                      cfg.token_accurate_accounting)
    return n


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
    global _autoscale_last_snapshot, _autoscale_last_decision
    global _autoscale_last_scale_up, _autoscale_last_scale_down
    while True:
        for b in registry.all():
            try:
                resp = await client.get(f"{b.url}/health", timeout=3.0)
                if resp.status_code == 200:
                    registry.set_health(b.id, True)
                    breaker.record_success(b.id)
                else:
                    registry.set_health(b.id, False)
                    breaker.record_failure(b.id)
            except Exception:
                registry.set_health(b.id, False)
                breaker.record_failure(b.id)
                tree.remove_backend(b.id)
                if cluster is not None:
                    cluster.publish_remove(b.id)        # tell peers it's gone

        # Autoscale planner: compute offered RPS and emit a scaling recommendation.
        if autoscale_cfg.enabled:
            now = time.monotonic()
            prev_t, prev_c = _autoscale_last_snapshot
            cur_c = _autoscale_request_count
            elapsed = now - prev_t if prev_t > 0 else 0.0
            rps = (cur_c - prev_c) / elapsed if elapsed > 0.5 else 0.0
            _autoscale_last_snapshot = (now, cur_c)
            n_replicas = len([b for b in registry.all() if b.healthy])
            signal = ScalingSignal(
                offered_rps=rps, current_replicas=max(1, n_replicas),
                now_s=now, last_scale_up_s=_autoscale_last_scale_up,
                last_scale_down_s=_autoscale_last_scale_down)
            decision = autoscale_plan(signal, autoscale_cfg)
            _autoscale_last_decision = decision
            if decision.direction == "up":
                _autoscale_last_scale_up = now
            elif decision.direction == "down":
                _autoscale_last_scale_down = now
            metrics.set_gauge("gateway_autoscale_desired_replicas",
                              decision.desired_replicas,
                              help="Autoscaler recommended replica count")
            metrics.inc_counter("gateway_autoscale_decisions_total",
                                help="Autoscaler decisions",
                                direction=decision.direction)

        await asyncio.sleep(2.0)


async def cluster_sync_loop() -> None:
    """Drain peers' mutations into local state, off the hot path. Also re-asserts
    this replica's load (every tick) and full drain map (anti-entropy, every
    DIGEST_EVERY ticks) so late-joining / restarted replicas converge."""
    interval = max(0.02, cluster_cfg.sync_interval_ms / 1000.0)
    DIGEST_EVERY = 8                       # ~2s at the 250ms default
    tick = 0
    while True:
        try:
            # Publish this replica's load + locally-shed backends, then drain peers'.
            unhealthy = [b.id for b in registry.all()
                         if breaker.state_code(b.id) == 2 or not b.healthy]
            cluster.publish_load(dict(load.inflight), unhealthy)
            if tick % DIGEST_EVERY == 0:
                cluster.publish_drain_digest()      # anti-entropy
            applied = cluster.sync()
            metrics.set_gauge("gateway_cluster_peers", cluster.fleet.peers(),
                              help="Number of peer gateway replicas seen")
            if applied:
                metrics.inc_counter("gateway_cluster_events_applied_total",
                                    help="Remote mutations applied from peers",
                                    value=applied)
        except Exception:
            pass            # a transient bus error must never stop routing
        tick += 1
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    tasks = [asyncio.create_task(scrape_loop(app.state.client))]
    if cluster is not None:
        tasks.append(asyncio.create_task(cluster_sync_loop()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if cluster is not None:
            cluster.close()
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


@app.get("/autoscale")
async def autoscale_endpoint():
    """Current autoscale recommendation (no-op when autoscale is disabled)."""
    if not autoscale_cfg.enabled:
        return JSONResponse({"enabled": False}, status_code=200)
    d = _autoscale_last_decision
    if d is None:
        return JSONResponse({"enabled": True, "status": "warming_up"}, status_code=200)
    return JSONResponse({
        "enabled": True,
        "current_replicas": d.current_replicas,
        "desired_replicas": d.desired_replicas,
        "direction": d.direction,
        "reason": d.reason,
        "est_wait_ms": round(d.est_wait_ms, 2),
        "blocked_by_cooldown": d.blocked_by_cooldown,
    })


@app.post("/cluster/gossip")
async def cluster_gossip(request: Request):
    """Receive a batch of replication events pushed by a peer (gossip transport).
    Network-internal: restrict to the replica subnet in deployment."""
    if cluster is None:
        return JSONResponse({"error": "cluster disabled"}, status_code=404)
    if cluster_cfg.secret and not hmac.compare_digest(
            request.headers.get("x-cluster-secret", ""), cluster_cfg.secret):
        return JSONResponse({"error": "invalid cluster secret"}, status_code=403)
    body = await request.json()
    n = cluster.receive_gossip(body.get("events") or [])
    return {"accepted": n}


# --- Control-plane admin API ------------------------------------------------
# Disabled until GW_ADMIN_TOKEN is set; then every /admin/* call must present it
# as `Authorization: Bearer <token>` or `X-Admin-Token: <token>`.

def _admin_authorized(request: Request) -> bool:
    token = cfg.admin_token
    if not token:
        return False                          # admin surface not enabled
    presented = request.headers.get("x-admin-token") or ""
    if not presented:
        from .auth import extract_bearer
        presented = extract_bearer(request.headers) or ""
    return bool(presented) and hmac.compare_digest(presented, token)


def _admin_guard(request: Request):
    """Return a 403 JSONResponse if not authorized, else None."""
    if _admin_authorized(request):
        return None
    reason = "admin disabled (set GW_ADMIN_TOKEN)" if not cfg.admin_token \
        else "invalid or missing admin token"
    return JSONResponse({"error": {"message": reason, "type": "forbidden"}},
                        status_code=403)


def _backend_view(b) -> dict:
    return {
        "id": b.id,
        "url": b.url,
        "healthy": b.healthy,
        "draining": b.draining,
        "inflight": load.inflight.get(b.id, 0),
        "kv_usage": round(load.kv_usage.get(b.id, 0.0), 4),
        "circuit": breaker.state(b.id),
        "held_blocks": tree.held_blocks(b.id),
    }


@app.get("/admin/backends")
async def admin_list_backends(request: Request):
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    return {"backends": [_backend_view(b) for b in registry.all()]}


@app.post("/admin/backends/{backend_id}/drain")
async def admin_drain(backend_id: str, request: Request):
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    if not registry.set_draining(backend_id, True):
        return JSONResponse({"error": f"unknown backend {backend_id!r}"}, status_code=404)
    if cluster is not None:
        cluster.publish_drain(backend_id, True)        # propagate fleet-wide
    metrics.inc_counter("gateway_admin_drain_total", help="Admin drain actions",
                        backend=backend_id, action="drain")
    log_event(log, "admin", action="drain", backend=backend_id)
    return {"backend": backend_id, "draining": True}


@app.post("/admin/backends/{backend_id}/undrain")
async def admin_undrain(backend_id: str, request: Request):
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    if not registry.set_draining(backend_id, False):
        return JSONResponse({"error": f"unknown backend {backend_id!r}"}, status_code=404)
    if cluster is not None:
        cluster.publish_drain(backend_id, False)       # propagate fleet-wide
    metrics.inc_counter("gateway_admin_drain_total", help="Admin drain actions",
                        backend=backend_id, action="undrain")
    log_event(log, "admin", action="undrain", backend=backend_id)
    return {"backend": backend_id, "draining": False}


@app.get("/admin/state")
async def admin_state(request: Request):
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    state = {
        "strategy": cfg.strategy,
        "backends": [_backend_view(b) for b in registry.all()],
        "routable": registry.ids_for(cfg.default_model),
    }
    if cluster is not None:
        state["cluster"] = {"replica_id": cluster.replica_id,
                            "peers": cluster.fleet.peers(),
                            "published": cluster.published,
                            "applied_remote": cluster.applied_remote}
    return state


@app.post("/v1/dag/schedule")
async def dag_schedule_endpoint(request: Request):
    """Plan a cache-locality-aware DAG schedule (no-op when disabled)."""
    if not dag_cfg.enabled:
        return JSONResponse({"error": "DAG scheduler is disabled (set GW_DAG_ENABLED=true)"},
                            status_code=400)
    body = await request.json()
    nodes_raw = body.get("nodes", [])
    strategy = body.get("strategy", "locality")
    try:
        dag_nodes = [DagNode(id=n["id"], prompt=n.get("prompt", ""),
                             parents=tuple(n.get("parents", ())))
                     for n in nodes_raw]
        dag = RequestDag(dag_nodes)
        backend_ids = [b.id for b in registry.all() if b.healthy] or [b.id for b in registry.all()]
        result = dag_schedule(dag=dag, backends=backend_ids, cfg=dag_cfg,
                              load=load, strategy=strategy)
    except (DagError, KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if result is None:
        return JSONResponse({"error": "no backends available"}, status_code=503)
    metrics.inc_counter("gateway_dag_schedules_total",
                        help="DAG schedules computed", strategy=strategy)
    return JSONResponse({
        "strategy": result.strategy,
        "est_makespan_ms": round(result.est_makespan_ms, 2),
        "cache_hit_rate": round(result.cache_hit_rate, 4),
        "cache_hit_blocks": result.cache_hit_blocks,
        "total_blocks": result.total_blocks,
        "placements": [
            {"node_id": p.node_id, "backend_id": p.backend_id, "layer": p.layer,
             "match_blocks": p.match_blocks, "total_blocks": p.total_blocks,
             "est_ms": round(p.est_ms, 2)}
            for p in result.placements
        ],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    prompt = extract_prompt(body.get("messages", []))
    strategy = request.headers.get("x-routing-strategy")

    eff_strategy = strategy or cfg.strategy

    # RAG structuring (opt-in): rewrite a structured RAG payload into canonical
    # messages -- chunks deduped+sorted into the (cacheable) system prefix, the
    # variable query as the tail -- so identical chunk SETS share a prefix. Done
    # before token accounting and hashing so they see the real prompt. The chunk
    # ids drive chunk-affinity backend selection further down.
    rag_chunks: list[int] | None = None
    rag_overlap = 0
    if rag_cfg.enabled:
        rag = parse_rag_request(body, rag_cfg.request_field)
        if rag is not None:
            ordered = canonicalize_chunks(rag.chunks) if rag_cfg.canonicalize else rag.chunks
            body["messages"] = rag_build_messages(rag, ordered)
            prompt = extract_prompt(body["messages"])
            rag_chunks = rag_chunk_ids(ordered)

    # Count requests for the autoscale RPS estimator (before any early return).
    global _autoscale_request_count
    _autoscale_request_count += 1

    # --- API-key authentication (before any tenant/routing work) ---
    auth_ok, auth_reason = authenticator.check(request.headers)
    if not auth_ok:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        metrics.inc_counter("gateway_auth_rejected_total",
                            help="Requests rejected by API-key auth", reason=auth_reason)
        return JSONResponse(
            {"error": {"message": "missing or invalid API key",
                       "type": "invalid_request_error", "code": "unauthorized"}},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer", "x-request-id": request_id})

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

    # LoRA-aware candidate selection: "base:adapter" routes only to backends
    # that have `adapter` loaded (falling back to the base pool per config).
    # When no adapters are configured this is exactly registry.ids_for(model).
    if cfg.backend_adapters:
        base, adapter = parse_model_spec(model)
        ids = lora_filter(registry, base, adapter, cfg.lora_fallback_to_base)
    else:
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

    # Semantic cache: a paraphrase of a recently-served prompt returns the
    # cached response without touching a backend. Per-tenant, so no cross-tenant
    # leak. The reserved output quota is released since no backend is used.
    if sem_cache is not None:
        cached = sem_cache.lookup(tenant.id, prompt)
        if cached is not None:
            cached_body, sim = cached
            release(0)
            metrics.inc_counter("gateway_semantic_cache_hits_total",
                                help="Semantic cache hits (paraphrase reuse)", tenant=tenant.id)
            span.set_attribute("semantic_cache.hit", True)
            span.set_attribute("semantic_cache.similarity", sim)
            span.set_attribute("http.status_code", 200)
            span.set_status(Status(StatusCode.OK))
            span.end()
            log_event(log, "request", request_id=request_id, trace_id=_trace_id(span),
                      tenant=tenant.id, model=str(model), semantic_cache="hit",
                      similarity=round(sim, 4), status=200,
                      duration_ms=round((time.perf_counter() - t_request) * 1000, 2))
            return StreamingResponse(
                iter([cached_body.encode("utf-8")]),
                media_type="text/event-stream",
                headers={"x-gw-semantic-cache": "hit", "x-gw-semantic-sim": f"{sim:.4f}",
                         "x-request-id": request_id})
        metrics.inc_counter("gateway_semantic_cache_misses_total",
                            help="Semantic cache misses", tenant=tenant.id)

    # Exclude backends with an open circuit. If every circuit is open, degrade to
    # trying all of them (better to attempt than to hard-fail).
    remaining = [b for b in ids if breaker.allow(b)] or list(ids)

    # RAG chunk-affinity selection (opt-in): among healthy candidates, pin to the
    # backend already holding the most of this request's chunks -- the set-based
    # analogue of prefix routing. The normal dispatch loop then routes the pinned
    # singleton, reusing all the streaming/metrics/observation machinery below.
    if rag_chunks is not None and remaining:
        rr = choose_rag_backend(chunk_ids=rag_chunks, backends=list(remaining),
                                index=rag_index, load=load, cfg=rag_cfg)
        if rr is not None:
            remaining = [rr.backend_id]
            rag_overlap = rr.overlap
            metrics.inc_counter("gateway_rag_requests_total",
                                help="RAG requests routed by chunk affinity",
                                backend=rr.backend_id)

    # SLO-aware admission control (opt-in): fail fast on deadline-infeasible
    # requests and shed/queue low-priority traffic under fleet pressure before
    # spending any prefill compute. Gold tiers are protected; sheds carry a
    # Retry-After. The probe route is side-effect-free (it does not touch the
    # tree or load), so it adds work only when admission is enabled.
    if adm_cfg.enabled and remaining:
        probe = router.choose(prompt, remaining, strategy=strategy, seed=seed)
        uncached = max(0, probe.tokens - probe.match_blocks * cfg.block_tokens)
        est_ttft_ms = cfg.prefill_ms_per_token * uncached
        fleet = FleetState(
            backend_inflight={b: load.inflight.get(b, 0) for b in remaining},
            backend_kv_usage={b: load.kv_usage.get(b, 0.0) for b in remaining})
        decision = admission_decide(
            AdmissionRequest(est_ttft_ms=est_ttft_ms, tier=tenant.tier,
                             est_output_tokens=reserved_output),
            fleet, adm_cfg)
        metrics.inc_counter("gateway_admission_total", help="Admission decisions",
                            action=decision.action, tier=tenant.tier)
        if decision.action != ADMIT:
            release(0)
            retry_s = max(1, math.ceil(decision.retry_after_ms / 1000.0))
            span.set_attribute("http.status_code", 503)
            span.set_attribute("admission.action", decision.action)
            span.set_attribute("admission.pressure", decision.pressure)
            span.set_status(Status(StatusCode.ERROR, "admission_" + decision.action))
            span.end()
            log_event(log, "request", request_id=request_id, trace_id=_trace_id(span),
                      tenant=tenant.id, model=str(model), status=503,
                      admission=decision.action, pressure=round(decision.pressure, 3),
                      reason=decision.reason,
                      duration_ms=round((time.perf_counter() - t_request) * 1000, 2))
            return JSONResponse(
                {"error": {"message": f"admission {decision.action}: {decision.reason}",
                           "type": "service_unavailable"}},
                status_code=503,
                headers={"x-request-id": request_id, "Retry-After": str(retry_s),
                         "x-gw-admission": decision.action,
                         "x-gw-admission-pressure": f"{decision.pressure:.3f}"})

    # Disaggregated prefill/decode routing (opt-in): when pool roles are set,
    # decide whether to split this request's phases across specialised backends.
    # The decision is informational (headers + metrics); the actual prefill
    # backend becomes the routing target (decode handoff is a backend concern).
    disagg_decision = None
    if disagg_pools is not None and remaining:
        _dh = _block_hashes(prompt, cfg.block_chars, cfg.hash_cutoff_blocks, seed=seed)
        _dm = tree.match(_dh)
        prefill_match = {b: _dm.get(b, 0) for b in remaining}
        disagg_decision = choose_disaggregated(
            prompt_tokens=input_tokens, pools=disagg_pools, load=load,
            cfg=disagg_cfg, output_tokens=reserved_output,
            prefill_match=prefill_match)
        if disagg_decision is not None:
            remaining = [disagg_decision.prefill_backend]
            metrics.inc_counter("gateway_disagg_requests_total",
                                help="Disaggregated routing decisions",
                                disaggregated=str(disagg_decision.disaggregated).lower(),
                                prefill=disagg_decision.prefill_backend,
                                decode=disagg_decision.decode_backend)

    # Multi-modal capability + affinity routing (opt-in): if the request carries
    # media (images/audio), filter to capable backends and route by media cache.
    mm_result = None
    if mm_cfg.enabled and remaining:
        modal_req = parse_multimodal_request(body, mm_cfg)
        if modal_req is not None:
            mm_result = choose_multimodal_backend(
                req=modal_req, backends=list(remaining),
                capabilities=mm_capabilities, index=mm_index,
                load=load, cfg=mm_cfg)
            if mm_result is not None:
                remaining = [mm_result.backend_id]
                metrics.inc_counter("gateway_multimodal_requests_total",
                                    help="Multi-modal requests routed by capability + affinity",
                                    backend=mm_result.backend_id)
            else:
                # No capable backend for the request's modalities -> 415
                release(0)
                span.set_attribute("http.status_code", 415)
                span.set_status(Status(StatusCode.ERROR, "unsupported_media"))
                span.end()
                log_event(log, "request", request_id=request_id,
                          trace_id=_trace_id(span), tenant=tenant.id,
                          model=str(model), status=415,
                          duration_ms=round((time.perf_counter() - t_request) * 1000, 2))
                return JSONResponse(
                    {"error": {"message": "no backend supports the request's modalities",
                               "type": "unsupported_media_type"}},
                    status_code=415, headers={"x-request-id": request_id})

    client: httpx.AsyncClient = request.app.state.client
    r = cm = upstream = None
    routing_recorded = False
    retries = 0
    t_dispatch = time.perf_counter()   # start of the dispatch section, for observed TTFT

    if eff_strategy == "speculative" and len(remaining) > 1:
        # Speculative routing: dispatch to the top-K candidates in parallel and
        # take whichever responds first; the losers are cancelled before decode.
        # Trades K-multiplicative prefill cost for a tail latency that follows
        # the fastest backend rather than the slowest.
        t0 = time.perf_counter()
        topk = top_k_backends(router, prompt, remaining, cfg.speculative_k, seed=seed)
        metrics.observe("gateway_routing_seconds", time.perf_counter() - t0,
                        help="Time spent in the routing decision (gateway added latency)")
        routing_recorded = True
        candidates = [(bid, registry.url(bid)) for bid in topk.backend_ids]
        outcome = await race(client, candidates, body)
        if outcome is not None:
            r = RouteResult(outcome.winner_id, topk.hashes, topk.tokens,
                            topk.match_blocks.get(outcome.winner_id, 0), topk.hash_mode)
            cm = outcome.winner_cm
            upstream = outcome.winner_response
            tree.insert(r.hashes, r.backend_id)
            if cluster is not None:
                cluster.publish_insert(r.hashes, r.backend_id)   # fan out to peers
            load.on_dispatch(r.backend_id, r.tokens)
            metrics.inc_counter("gateway_speculative_races_total",
                                help="Speculative races dispatched", k=str(len(candidates)))
    else:
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
            if cluster is not None:
                cluster.publish_insert(r.hashes, r.backend_id)   # fan out to peers
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
    if rag_chunks is not None:
        headers["x-gw-rag"] = "true"
        headers["x-gw-rag-overlap"] = str(rag_overlap)
        headers["x-gw-rag-chunks"] = str(len(rag_chunks))
    if disagg_decision is not None:
        headers["x-gw-disagg"] = "true" if disagg_decision.disaggregated else "false"
        headers["x-gw-disagg-prefill"] = disagg_decision.prefill_backend
        headers["x-gw-disagg-decode"] = disagg_decision.decode_backend
    if mm_result is not None:
        headers["x-gw-multimodal"] = "true"
        headers["x-gw-multimodal-media"] = str(mm_result.total_media)
        headers["x-gw-multimodal-overlap"] = str(mm_result.media_overlap)
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
    # Snapshot the load the routing decision saw, to label the observation.
    obs_inflight = load.inflight[final_r.backend_id]
    obs_kv = load.kv_usage[final_r.backend_id]

    async def body_iter():
        out_events = 0
        buf = bytearray() if sem_cache is not None else None
        first_byte_ms = None
        try:
            async for chunk in upstream.aiter_raw():
                if first_byte_ms is None:
                    first_byte_ms = (time.perf_counter() - t_dispatch) * 1000.0
                out_events += chunk.count(b"data:")
                if buf is not None:
                    buf += chunk
                yield chunk
        finally:
            await final_cm.__aexit__(None, None, None)
            load.on_complete(final_r.backend_id, final_r.tokens)
            # Record (features, observed TTFT) for offline predictor training.
            if obs_logger is not None and first_byte_ms is not None:
                obs_logger.log(Observation(
                    ts=time.time(), backend_id=final_r.backend_id,
                    prompt_tokens=final_r.tokens, match_blocks=final_r.match_blocks,
                    uncached_tokens=max(0, final_r.tokens - final_r.match_blocks * cfg.block_tokens),
                    inflight=obs_inflight, kv_usage=obs_kv,
                    observed_ttft_ms=first_byte_ms, hash_mode=final_r.hash_mode))
            # Populate the semantic cache with the full response so a later
            # paraphrase of this prompt can be served without a backend.
            if buf is not None and upstream.status_code == 200:
                try:
                    sem_cache.store(tenant.id, prompt, buf.decode("utf-8", "replace"))
                except Exception:
                    pass
            # Mark this request's chunks as cached on the serving backend, so a
            # later request over the same chunk SET prefers it (warm chunk KV).
            if rag_chunks is not None and upstream.status_code == 200:
                rag_index.record(final_r.backend_id, rag_chunks)
            if mm_result is not None and upstream.status_code == 200:
                modal_req_final = parse_multimodal_request(body, mm_cfg)
                if modal_req_final is not None:
                    mm_index.record(final_r.backend_id, modal_req_final.media_ids())
            # reconcile TPS: actual streamed tokens vs the reserved estimate
            actual_output = max(0, out_events - 1)     # minus the [DONE] event
            release(actual_output)
            # Per-route SLO metrics: TTFT + total-latency distributions per model
            # (Prometheus histograms -> p50/p95/p99 per route via histogram_quantile).
            model_label = str(model)
            if first_byte_ms is not None:
                metrics.observe("gateway_ttft_seconds", first_byte_ms / 1000.0,
                                buckets=SLO_BUCKETS,
                                help="Time-to-first-token (s) per model", model=model_label)
                if cfg.slo_ttft_ms > 0 and first_byte_ms > cfg.slo_ttft_ms:
                    metrics.inc_counter("gateway_slo_violations_total",
                                        help="Requests exceeding the TTFT SLO",
                                        model=model_label)
            metrics.observe("gateway_request_duration_seconds",
                            time.perf_counter() - t_request, buckets=SLO_BUCKETS,
                            help="Total request duration (s) per model", model=model_label)
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
