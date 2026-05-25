import asyncio
import json
import logging

import httpx

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.load_tracker import LoadTracker
from gateway.logging_setup import JsonFormatter, log_event
from gateway.metrics import MetricsCollector
from gateway.radix_tree import RadixTree
from gateway.router import Router
from gateway.tenancy import RateLimiter, TenantRegistry
from mock_backend.app import create_app


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


# --- unit: formatter + log_event ---

def test_json_formatter_includes_fields():
    rec = logging.LogRecord("gateway", logging.INFO, __file__, 1, "request", None, None)
    rec.fields = {"request_id": "abc123", "status": 200, "backend": "b0"}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "request"
    assert out["level"] == "INFO"
    assert out["request_id"] == "abc123"
    assert out["status"] == 200


def test_log_event_drops_none_fields():
    logger = logging.getLogger("test.gateway.logging")
    cap = _Capture()
    logger.addHandler(cap)
    logger.setLevel("INFO")
    log_event(logger, "x", a=1, b=None)
    logger.removeHandler(cap)
    assert cap.records[0].fields == {"a": 1}


# --- integration: a request emits a correlated structured log ---

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


def test_request_emits_correlated_structured_log():
    async def run():
        cap = _Capture()
        logging.getLogger("gateway").addHandler(cap)
        _wire()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app),
                                   base_url="http://gw")
        async with client:
            async with client.stream("POST", "/v1/chat/completions",
                                     json={"model": "mock-model",
                                           "messages": [{"role": "user", "content": "hi"}]},
                                     headers={"x-routing-strategy": "prefix_tree"}) as resp:
                rid = resp.headers.get("x-request-id")
                async for _ in resp.aiter_raw():
                    pass
        await gw.app.state.client.aclose()
        logging.getLogger("gateway").removeHandler(cap)

        recs = [r for r in cap.records if getattr(r, "fields", {}).get("status") == 200]
        assert recs
        f = recs[-1].fields
        assert rid and f["request_id"] == rid     # response header correlates with the log
        assert f["backend"] == "b0"
        assert "duration_ms" in f
    asyncio.run(run())
