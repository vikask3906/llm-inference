"""Per-route SLO metrics: per-model TTFT + total-latency histograms, and an
optional TTFT-budget violation counter."""

import asyncio

import httpx

import gateway.server as gw
from gateway.auth import Authenticator
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
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


def _wire(slo_ttft_ms=0.0):
    gw.cfg.backends = "b0=http://b0:9000"
    gw.cfg.rate_limit_enabled = False
    gw.cfg.require_auth = False
    gw.cfg.prefix_isolation = "global"
    gw.cfg.slo_ttft_ms = slo_ttft_ms
    a = create_app("b0", 600)
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry("")
    gw.limiter = RateLimiter()
    gw.authenticator = Authenticator(set(), False)
    gw.cluster = None
    gw.app.state.client = httpx.AsyncClient(
        transport=MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


async def _chat(c):
    body = {"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}]}
    async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
        async for _ in resp.aiter_raw():
            pass
        return resp.status_code


def test_per_model_latency_histograms_emitted():
    async def run():
        _wire()
        try:
            async with _gwclient() as c:
                assert await _chat(c) == 200
                m = (await c.get("/metrics")).text
            # both histograms present, labeled by model
            assert 'gateway_ttft_seconds_bucket{' in m and 'model="mock-model"' in m
            assert "gateway_ttft_seconds_count" in m
            assert "gateway_request_duration_seconds_count" in m
        finally:
            gw.cfg.slo_ttft_ms = 0.0
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_slo_violation_counter_fires_when_budget_tiny():
    async def run():
        _wire(slo_ttft_ms=0.0001)        # 0.1us -> any real TTFT violates
        try:
            async with _gwclient() as c:
                assert await _chat(c) == 200
                m = (await c.get("/metrics")).text
            assert "gateway_slo_violations_total" in m
            assert 'model="mock-model"' in m
        finally:
            gw.cfg.slo_ttft_ms = 0.0
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_no_violation_counter_when_budget_generous():
    async def run():
        _wire(slo_ttft_ms=60000.0)       # 60s budget -> never violated
        try:
            async with _gwclient() as c:
                assert await _chat(c) == 200
                m = (await c.get("/metrics")).text
            assert "gateway_slo_violations_total" not in m   # counter never created
        finally:
            gw.cfg.slo_ttft_ms = 0.0
            await gw.app.state.client.aclose()
    asyncio.run(run())
