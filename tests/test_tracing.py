import asyncio

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.load_tracker import LoadTracker
from gateway.metrics import MetricsCollector
from gateway.radix_tree import RadixTree
from gateway.router import Router
from gateway.tenancy import RateLimiter, TenantRegistry
from gateway.tracing import setup_tracing
from mock_backend.app import create_app

# Install an in-memory span exporter on the global provider (once).
_EXPORTER = InMemorySpanExporter()
setup_tracing(_EXPORTER)


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def _wire():
    gw.cfg.backends = "b0=http://b0:9000"
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
    mock = create_app("b0", 600)
    gw.app.state.client = httpx.AsyncClient(
        transport=MultiHostTransport({"b0": httpx.ASGITransport(app=mock)}))


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


def _msgs():
    return [{"role": "system", "content": "S" * 128 + "D" * 1000},
            {"role": "user", "content": "q"}]


def _completion_spans():
    return [s for s in _EXPORTER.get_finished_spans() if s.name == "chat.completion"]


def test_success_emits_span_with_attributes():
    async def run():
        _EXPORTER.clear()
        _wire()
        async with _gwclient() as c:
            async with c.stream("POST", "/v1/chat/completions",
                                json={"model": "mock-model", "messages": _msgs()},
                                headers={"x-routing-strategy": "prefix_tree"}) as resp:
                async for _ in resp.aiter_raw():
                    pass
        await gw.app.state.client.aclose()
        spans = _completion_spans()
        assert spans
        attrs = spans[-1].attributes
        assert attrs["routing.backend"] == "b0"
        assert attrs["http.status_code"] == 200
        assert attrs["routing.strategy"] == "prefix_tree"
        assert "cache.hit" in attrs
        assert spans[-1].status.status_code.name == "OK"
    asyncio.run(run())


def test_rate_limited_request_emits_error_span():
    async def run():
        _EXPORTER.clear()
        _wire()
        gw.cfg.rate_limit_enabled = True
        gw.cfg.tenants = "sk-x=x:bronze"          # bronze rps=5
        gw.tenants = TenantRegistry(gw.cfg.tenants)
        gw.limiter = RateLimiter()
        async with _gwclient() as c:
            hdr = {"authorization": "Bearer sk-x"}
            for _ in range(12):
                await c.post("/v1/chat/completions",
                             json={"model": "mock-model", "messages": _msgs(), "max_tokens": 8},
                             headers=hdr)
        await gw.app.state.client.aclose()
        spans = _completion_spans()
        codes = [s.attributes.get("http.status_code") for s in spans]
        assert 429 in codes
        throttled = next(s for s in spans if s.attributes.get("http.status_code") == 429)
        assert throttled.attributes.get("ratelimit.reason") in ("rps", "tps", "inflight")
        assert throttled.status.status_code.name == "ERROR"
    asyncio.run(run())
