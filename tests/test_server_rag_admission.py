"""End-to-end wiring tests for RAG-aware routing and SLO-aware admission
control in server.py.

These exercise the hot path (gateway/server.py), proving the standalone RAG and
admission packages are actually invoked by the request handler when their flags
are set -- and that the default path is untouched when they are not.
"""

import asyncio

import httpx

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.load_tracker import LoadTracker
from gateway.metrics import MetricsCollector
from gateway.radix_tree import RadixTree
from gateway.rag.index import ChunkAffinityIndex
from gateway.router import Router
from gateway.tenancy import RateLimiter, TenantRegistry
from mock_backend.app import create_app


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def _wire(backends_str, transport, tenants=""):
    gw.cfg.backends = backends_str
    gw.cfg.rate_limit_enabled = False
    gw.cfg.tenants = tenants
    gw.cfg.prefix_isolation = "global"          # keep hashing un-salted for clarity
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry(tenants)
    gw.limiter = RateLimiter()
    gw.rag_index = ChunkAffinityIndex(gw.rag_cfg.cache_capacity_chunks)
    gw.app.state.client = httpx.AsyncClient(transport=transport)


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


def _two_backends(tenants=""):
    a = create_app("b0", 600)
    b = create_app("b1", 600)
    _wire("b0=http://b0:9000,b1=http://b1:9000",
          MultiHostTransport({"b0": httpx.ASGITransport(app=a),
                              "b1": httpx.ASGITransport(app=b)}), tenants=tenants)


async def _post(client, body, headers=None):
    async with client.stream("POST", "/v1/chat/completions", json=body,
                             headers=headers or {}) as resp:
        raw = b""
        async for chunk in resp.aiter_raw():
            raw += chunk
        return resp, dict(resp.headers), raw


def _rag_body(chunks, query="what about this?"):
    return {"model": "mock-model",
            "messages": [],
            "rag": {"system": "You are a helpful assistant.",
                    "chunks": chunks, "query": query}}


# --- RAG: structuring + chunk-affinity routing -------------------------------

def test_rag_structures_request_and_emits_headers():
    async def run():
        _two_backends()
        gw.rag_cfg.enabled = True
        try:
            async with _gwclient() as c:
                chunks = ["doc about cats", "doc about dogs", "doc about birds"]
                resp, hdrs, raw = await _post(c, _rag_body(chunks))
                assert resp.status_code == 200
                assert hdrs.get("x-gw-rag") == "true"
                assert hdrs.get("x-gw-rag-chunks") == "3"
                assert b"[DONE]" in raw
            assert "gateway_rag_requests_total" in gw.metrics.render()
        finally:
            gw.rag_cfg.enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_rag_chunk_affinity_reuse_routes_to_warm_backend():
    async def run():
        _two_backends()
        gw.rag_cfg.enabled = True
        try:
            async with _gwclient() as c:
                chunks = ["alpha chunk", "beta chunk", "gamma chunk"]
                # First request: cold, no backend holds these chunks (overlap 0).
                r1, h1, _ = await _post(c, _rag_body(chunks))
                assert r1.status_code == 200
                assert h1.get("x-gw-rag-overlap") == "0"
                first_backend = h1.get("x-gw-backend")

                # Same chunk SET, different ORDER: canonicalization makes it an
                # identical prefix, so it must route back to the warm backend
                # with full overlap.
                r2, h2, _ = await _post(c, _rag_body(list(reversed(chunks))))
                assert r2.status_code == 200
                assert h2.get("x-gw-backend") == first_backend
                assert h2.get("x-gw-rag-overlap") == "3"
        finally:
            gw.rag_cfg.enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_rag_disabled_ignores_payload():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        # rag_cfg.enabled defaults False; a body carrying a rag payload but real
        # messages must route normally with no RAG headers.
        body = _rag_body(["x", "y"])
        body["messages"] = [{"role": "user", "content": "hello"}]
        try:
            async with _gwclient() as c:
                resp, hdrs, raw = await _post(c, body)
                assert resp.status_code == 200
                assert hdrs.get("x-gw-rag") is None
                assert b"[DONE]" in raw
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- Admission control: fail-fast + priority shedding ------------------------

def test_admission_fail_fast_on_infeasible_ttft():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        gw.adm_cfg.enabled = True
        gw.adm_cfg.ttft_slo_ms = 0.0          # any uncached prefill blows the deadline
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "D" * 2000}]}
                resp, hdrs, _ = await _post(c, body)
                assert resp.status_code == 503
                assert hdrs.get("x-gw-admission") == "reject"
                assert hdrs.get("retry-after") is not None
            assert "gateway_admission_total" in gw.metrics.render()
        finally:
            gw.adm_cfg.enabled = False
            gw.adm_cfg.ttft_slo_ms = 500.0
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_admission_sheds_low_tier_protects_gold_under_pressure():
    async def run():
        _two_backends(tenants="goldkey=acme:gold,bronzekey=joe:bronze")
        gw.adm_cfg.enabled = True
        # Saturate every backend so fleet pressure == 1.0 -> admit floor = gold.
        for b in ("b0", "b1"):
            gw.load.inflight[b] = gw.adm_cfg.max_inflight_per_backend
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "short"}]}
                # bronze: below the gold floor -> shed (queued) with a Retry-After.
                rb, hb, _ = await _post(c, body, {"authorization": "Bearer bronzekey"})
                assert rb.status_code == 503
                assert hb.get("x-gw-admission") in ("queue", "reject")
                assert hb.get("retry-after") is not None
                # gold: protected -> admitted and served.
                rg, hg, raw = await _post(c, body, {"authorization": "Bearer goldkey"})
                assert rg.status_code == 200
                assert hg.get("x-gw-backend") in ("b0", "b1")
                assert b"[DONE]" in raw
        finally:
            gw.adm_cfg.enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_admission_disabled_admits_everything():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        # enabled defaults False: even an "infeasible" SLO is irrelevant.
        gw.adm_cfg.ttft_slo_ms = 0.0
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "D" * 2000}]}
                resp, hdrs, _ = await _post(c, body)
                assert resp.status_code == 200
                assert hdrs.get("x-gw-admission") is None
        finally:
            gw.adm_cfg.ttft_slo_ms = 500.0
            await gw.app.state.client.aclose()
    asyncio.run(run())
