"""End-to-end wiring tests for the opt-in extensions in server.py.

These exercise the *hot path* (gateway/server.py), not the extension modules in
isolation -- they prove the LoRA filter, semantic cache, and speculative race
are actually invoked by the request handler when their config flags are set,
and that the default path is untouched when they are not.
"""

import asyncio
import json

import httpx
import numpy as np

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.extensions.lora import apply_adapter_config
from gateway.extensions.semantic_cache import SemanticCache
from gateway.extensions.ttft_predictor import ObservationLogger
from gateway.load_tracker import LoadTracker
from gateway.metrics import MetricsCollector
from gateway.radix_tree import RadixTree
from gateway.router import Router
from gateway.tenancy import RateLimiter, TenantRegistry
from mock_backend.app import create_app


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def _wire(backends_str, transport):
    gw.cfg.backends = backends_str
    gw.cfg.rate_limit_enabled = False
    gw.cfg.tenants = ""
    gw.cfg.prefix_isolation = "tenant"
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry(gw.cfg.tenants)
    gw.limiter = RateLimiter()
    gw.app.state.client = httpx.AsyncClient(transport=transport)


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


def _msgs():
    return [
        {"role": "system", "content": "S" * 128 + "D" * 1000},
        {"role": "user", "content": "question"},
    ]


def _two_backends():
    a = create_app("b0", 600)
    b = create_app("b1", 600)
    _wire("b0=http://b0:9000,b1=http://b1:9000",
          MultiHostTransport({"b0": httpx.ASGITransport(app=a),
                              "b1": httpx.ASGITransport(app=b)}))


async def _post(client, model="mock-model", strategy="prefix_tree", headers=None):
    h = {"x-routing-strategy": strategy}
    if headers:
        h.update(headers)
    async with client.stream("POST", "/v1/chat/completions",
                             json={"model": model, "messages": _msgs()},
                             headers=h) as resp:
        body = b""
        async for chunk in resp.aiter_raw():
            body += chunk
        return resp, dict(resp.headers), body


# --- LoRA-aware routing: "base:adapter" pins to the adapter-capable backend ---

def test_lora_routes_to_adapter_backend():
    async def run():
        _two_backends()
        gw.cfg.backend_adapters = "b1:summary-v2"
        apply_adapter_config(gw.registry, gw.cfg.backend_adapters)
        try:
            async with _gwclient() as c:
                # Only b1 has the adapter -> every request must land on b1.
                for _ in range(5):
                    resp, hdrs, _ = await _post(c, model="mock-model:summary-v2")
                    assert resp.status_code == 200
                    assert hdrs.get("x-gw-backend") == "b1"
        finally:
            gw.cfg.backend_adapters = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_lora_unknown_adapter_falls_back_to_base_pool():
    async def run():
        _two_backends()
        gw.cfg.backend_adapters = "b1:summary-v2"
        gw.cfg.lora_fallback_to_base = True
        apply_adapter_config(gw.registry, gw.cfg.backend_adapters)
        try:
            async with _gwclient() as c:
                # No backend has "ghost" -> fallback serves from the base pool.
                resp, hdrs, body = await _post(c, model="mock-model:ghost")
                assert resp.status_code == 200
                assert hdrs.get("x-gw-backend") in ("b0", "b1")
                assert b"[DONE]" in body
        finally:
            gw.cfg.backend_adapters = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- Speculative routing: race the top-K, return the winner ---

def test_speculative_returns_a_winner_and_records_metric():
    async def run():
        _two_backends()
        gw.cfg.speculative_k = 2
        try:
            async with _gwclient() as c:
                resp, hdrs, body = await _post(c, strategy="speculative")
                assert resp.status_code == 200
                assert hdrs.get("x-gw-backend") in ("b0", "b1")
                assert b"[DONE]" in body
            assert "gateway_speculative_races_total" in gw.metrics.render()
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- Semantic cache: a repeat prompt is served from cache, no backend touched ---

def _fake_embedder(text: str):
    # Deterministic per-text unit vector: identical text -> identical vector
    # (cosine 1.0 -> hit); different text -> ~orthogonal (miss).
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(16).astype("float32")
    return v / (np.linalg.norm(v) + 1e-9)


def test_semantic_cache_hit_bypasses_backend():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        gw.sem_cache = SemanticCache(_fake_embedder, similarity_threshold=0.9,
                                     max_entries_per_tenant=128)
        try:
            async with _gwclient() as c:
                # First call: miss -> routed to b0, response stored on completion.
                r1, h1, b1 = await _post(c)
                assert r1.status_code == 200
                assert h1.get("x-gw-backend") == "b0"
                assert h1.get("x-gw-semantic-cache") is None

                # Second identical call: hit -> served from cache, no backend.
                r2, h2, b2 = await _post(c)
                assert r2.status_code == 200
                assert h2.get("x-gw-semantic-cache") == "hit"
                assert h2.get("x-gw-backend") is None
                assert b2 == b1
            assert gw.sem_cache.stats()["hits"] >= 1
        finally:
            gw.sem_cache = None
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- Predictive TTFT: the gateway logs (features, observed TTFT) per request ---

def test_observation_logger_records_ttft(tmp_path):
    log_path = tmp_path / "obs.jsonl"

    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        gw.obs_logger = ObservationLogger(str(log_path))
        try:
            async with _gwclient() as c:
                resp, _, _ = await _post(c)
                assert resp.status_code == 200
        finally:
            gw.obs_logger = None
            await gw.app.state.client.aclose()
    asyncio.run(run())

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["backend_id"] == "b0"
    assert rec["observed_ttft_ms"] >= 0
    assert {"prompt_tokens", "match_blocks", "uncached_tokens",
            "inflight", "kv_usage", "hash_mode"} <= rec.keys()


def test_semantic_cache_miss_on_distinct_prompt():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        gw.sem_cache = SemanticCache(_fake_embedder, similarity_threshold=0.9,
                                     max_entries_per_tenant=128)
        try:
            async with _gwclient() as c:
                await _post(c)  # warm one entry
                # A different prompt embeds ~orthogonally -> miss -> hits backend.
                async with c.stream("POST", "/v1/chat/completions",
                                    json={"model": "mock-model",
                                          "messages": [{"role": "user",
                                                        "content": "totally unrelated query"}]},
                                    headers={"x-routing-strategy": "prefix_tree"}) as resp:
                    async for _ in resp.aiter_raw():
                        pass
                    assert resp.headers.get("x-gw-semantic-cache") is None
                    assert resp.headers.get("x-gw-backend") == "b0"
        finally:
            gw.sem_cache = None
            await gw.app.state.client.aclose()
    asyncio.run(run())
