"""End-to-end wiring tests for disaggregated, multimodal, DAG, and autoscale
packages in server.py.

These exercise the hot path (gateway/server.py), proving the standalone packages
are actually invoked when their flags are set -- and that the default path is
untouched when they are not.
"""

import asyncio
import time

import httpx

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.disagg.pools import PoolRegistry
from gateway.load_tracker import LoadTracker
from gateway.metrics import MetricsCollector
from gateway.multimodal.index import MediaAffinityIndex
from gateway.multimodal.router import parse_capabilities
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
    gw.cfg.prefix_isolation = "global"
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry(tenants)
    gw.limiter = RateLimiter()
    gw.rag_index = ChunkAffinityIndex(gw.rag_cfg.cache_capacity_chunks)
    gw.mm_index = MediaAffinityIndex(gw.mm_cfg.cache_capacity_media)
    gw.app.state.client = httpx.AsyncClient(transport=transport)


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


def _two_backends(tenants=""):
    a = create_app("b0", 600)
    b = create_app("b1", 600)
    _wire("b0=http://b0:9000,b1=http://b1:9000",
          MultiHostTransport({"b0": httpx.ASGITransport(app=a),
                              "b1": httpx.ASGITransport(app=b)}), tenants=tenants)


def _single_backend():
    a = create_app("b0", 600)
    _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))


async def _post(client, body, headers=None):
    async with client.stream("POST", "/v1/chat/completions", json=body,
                             headers=headers or {}) as resp:
        raw = b""
        async for chunk in resp.aiter_raw():
            raw += chunk
        return resp, dict(resp.headers), raw


# --- Disaggregated routing ---------------------------------------------------

def test_disagg_emits_headers_when_pools_configured():
    async def run():
        _two_backends()
        gw.disagg_cfg.pools = "b0:prefill;b1:decode"
        gw.disagg_pools = PoolRegistry.from_spec(["b0", "b1"], gw.disagg_cfg.pools)
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "hello world"}]}
                resp, hdrs, raw = await _post(c, body)
                assert resp.status_code == 200
                assert hdrs.get("x-gw-disagg") in ("true", "false")
                assert hdrs.get("x-gw-disagg-prefill") is not None
                assert hdrs.get("x-gw-disagg-decode") is not None
                assert b"[DONE]" in raw
            assert "gateway_disagg_requests_total" in gw.metrics.render()
        finally:
            gw.disagg_cfg.pools = ""
            gw.disagg_pools = None
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_disagg_picks_correct_phase_backends():
    async def run():
        _two_backends()
        # Pure split: b0=prefill only, b1=decode only
        gw.disagg_cfg.pools = "b0:prefill;b1:decode"
        gw.disagg_pools = PoolRegistry.from_spec(["b0", "b1"], gw.disagg_cfg.pools)
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "X" * 1000}]}
                resp, hdrs, _ = await _post(c, body)
                assert resp.status_code == 200
                # With disjoint pools, prefill must be b0 and decode must be b1
                assert hdrs.get("x-gw-disagg-prefill") == "b0"
                assert hdrs.get("x-gw-disagg-decode") == "b1"
                assert hdrs.get("x-gw-disagg") == "true"
        finally:
            gw.disagg_cfg.pools = ""
            gw.disagg_pools = None
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_disagg_disabled_by_default():
    async def run():
        _single_backend()
        # disagg_pools is None by default -> no disagg headers
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "hello"}]}
                resp, hdrs, raw = await _post(c, body)
                assert resp.status_code == 200
                assert hdrs.get("x-gw-disagg") is None
                assert b"[DONE]" in raw
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- Multimodal routing -------------------------------------------------------

def test_multimodal_routes_image_request():
    async def run():
        _two_backends()
        gw.mm_cfg.enabled = True
        gw.mm_capabilities = parse_capabilities("b0:text,image;b1:text")
        gw.mm_index = MediaAffinityIndex(gw.mm_cfg.cache_capacity_media)
        try:
            async with _gwclient() as c:
                # OpenAI vision-style multi-part content
                body = {"model": "mock-model", "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "describe this image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                    ]},
                ]}
                resp, hdrs, raw = await _post(c, body)
                assert resp.status_code == 200
                # b0 is the only vision-capable backend
                assert hdrs.get("x-gw-backend") == "b0"
                assert hdrs.get("x-gw-multimodal") == "true"
                assert hdrs.get("x-gw-multimodal-media") == "1"
                assert b"[DONE]" in raw
            assert "gateway_multimodal_requests_total" in gw.metrics.render()
        finally:
            gw.mm_cfg.enabled = False
            gw.mm_capabilities = {}
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_multimodal_rejects_unsupported_modality():
    async def run():
        _two_backends()
        gw.mm_cfg.enabled = True
        # Neither backend supports audio
        gw.mm_capabilities = parse_capabilities("b0:text,image;b1:text")
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model", "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "transcribe this"},
                        {"type": "input_audio", "input_audio": {"url": "https://example.com/speech.mp3", "seconds": 5.0}},
                    ]},
                ]}
                resp, hdrs, _ = await _post(c, body)
                assert resp.status_code == 415
        finally:
            gw.mm_cfg.enabled = False
            gw.mm_capabilities = {}
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_multimodal_media_affinity_reuse():
    async def run():
        _two_backends()
        gw.mm_cfg.enabled = True
        gw.mm_capabilities = parse_capabilities("b0:text,image;b1:text,image")
        gw.mm_index = MediaAffinityIndex(gw.mm_cfg.cache_capacity_media)
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model", "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}},
                    ]},
                ]}
                # First request: cold, overlap=0
                r1, h1, _ = await _post(c, body)
                assert r1.status_code == 200
                first_backend = h1.get("x-gw-backend")
                assert h1.get("x-gw-multimodal-overlap") == "0"

                # Same image again -> should route to the same backend with overlap=1
                r2, h2, _ = await _post(c, body)
                assert r2.status_code == 200
                assert h2.get("x-gw-backend") == first_backend
                assert h2.get("x-gw-multimodal-overlap") == "1"
        finally:
            gw.mm_cfg.enabled = False
            gw.mm_capabilities = {}
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_multimodal_disabled_ignores_media():
    async def run():
        _single_backend()
        # mm_cfg.enabled defaults to False; a body with media should route normally
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model", "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}},
                    ]},
                ]}
                resp, hdrs, raw = await _post(c, body)
                assert resp.status_code == 200
                assert hdrs.get("x-gw-multimodal") is None
                assert b"[DONE]" in raw
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- DAG scheduler endpoint ---------------------------------------------------

def test_dag_schedule_returns_placements():
    async def run():
        _two_backends()
        gw.dag_cfg.enabled = True
        try:
            async with _gwclient() as c:
                dag_body = {"nodes": [
                    {"id": "summarize_a", "prompt": "Summarize document A. " * 50},
                    {"id": "summarize_b", "prompt": "Summarize document B. " * 50},
                    {"id": "combine", "prompt": "Combine the summaries.",
                     "parents": ["summarize_a", "summarize_b"]},
                ], "strategy": "locality"}
                resp = await c.post("/v1/dag/schedule", json=dag_body)
                assert resp.status_code == 200
                data = resp.json()
                assert data["strategy"] == "locality"
                assert len(data["placements"]) == 3
                assert data["total_blocks"] > 0
                assert data["est_makespan_ms"] >= 0
                # The combine node must be in a later layer than the summaries
                layers = {p["node_id"]: p["layer"] for p in data["placements"]}
                assert layers["combine"] > layers["summarize_a"]
                assert layers["combine"] > layers["summarize_b"]
            assert "gateway_dag_schedules_total" in gw.metrics.render()
        finally:
            gw.dag_cfg.enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_dag_schedule_rejects_cycle():
    async def run():
        _single_backend()
        gw.dag_cfg.enabled = True
        try:
            async with _gwclient() as c:
                dag_body = {"nodes": [
                    {"id": "a", "prompt": "X", "parents": ["b"]},
                    {"id": "b", "prompt": "Y", "parents": ["a"]},
                ]}
                resp = await c.post("/v1/dag/schedule", json=dag_body)
                assert resp.status_code == 400
                assert "cycle" in resp.json()["error"].lower()
        finally:
            gw.dag_cfg.enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_dag_disabled_returns_400():
    async def run():
        _single_backend()
        # dag_cfg.enabled defaults False
        try:
            async with _gwclient() as c:
                resp = await c.post("/v1/dag/schedule", json={"nodes": []})
                assert resp.status_code == 400
                assert "disabled" in resp.json()["error"].lower()
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- Autoscale endpoint -------------------------------------------------------

def test_autoscale_disabled_returns_json():
    async def run():
        _single_backend()
        # autoscale_cfg.enabled defaults False
        try:
            async with _gwclient() as c:
                resp = await c.get("/autoscale")
                assert resp.status_code == 200
                assert resp.json()["enabled"] is False
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_autoscale_enabled_returns_recommendation():
    async def run():
        _single_backend()
        gw.autoscale_cfg.enabled = True
        # Simulate a scrape tick that produces a decision
        from gateway.autoscale.planner import ScalingSignal, plan as autoscale_plan
        signal = ScalingSignal(offered_rps=10.0, current_replicas=1,
                               now_s=time.monotonic())
        gw._autoscale_last_decision = autoscale_plan(signal, gw.autoscale_cfg)
        try:
            async with _gwclient() as c:
                resp = await c.get("/autoscale")
                assert resp.status_code == 200
                data = resp.json()
                assert data["enabled"] is True
                assert "desired_replicas" in data
                assert data["direction"] in ("up", "down", "hold")
                assert "reason" in data
        finally:
            gw.autoscale_cfg.enabled = False
            gw._autoscale_last_decision = None
            await gw.app.state.client.aclose()
    asyncio.run(run())
