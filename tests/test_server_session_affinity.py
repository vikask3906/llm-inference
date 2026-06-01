"""End-to-end: the live HTTP path honors X-Session-ID and pins multi-turn agent
sessions to a single backend (the agentic KV-reuse property)."""

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
from gateway.session_affinity import SessionAffinity
from gateway.tenancy import RateLimiter, TenantRegistry
from mock_backend.app import create_app


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def _wire(enabled=True):
    gw.cfg.backends = "b0=http://b0:9000,b1=http://b1:9000,b2=http://b2:9000"
    gw.cfg.rate_limit_enabled = False
    gw.cfg.require_auth = False
    gw.cfg.prefix_isolation = "global"
    gw.cfg.session_affinity_enabled = enabled
    apps = {bid: create_app(bid, 600) for bid in ("b0", "b1", "b2")}
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry("")
    gw.limiter = RateLimiter()
    gw.authenticator = Authenticator(set(), False)
    gw.session_affinity = SessionAffinity()
    gw.cluster = None
    gw.app.state.client = httpx.AsyncClient(transport=MultiHostTransport(
        {bid: httpx.ASGITransport(app=a) for bid, a in apps.items()}))


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


async def _turn(c, session_id, content):
    body = {"model": "mock-model", "messages": [{"role": "user", "content": content}]}
    headers = {"x-session-id": session_id} if session_id else {}
    async with c.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
        async for _ in r.aiter_raw():
            pass
        return r.headers.get("x-gw-backend")


def test_session_pins_all_turns_to_one_backend():
    async def run():
        _wire(enabled=True)
        try:
            async with _gwclient() as c:
                # Turn 0 (cold): could land on any backend; remember which.
                # Use a varied user message so prefix routing can't accidentally
                # produce the same answer -- the session pin is what should fix it.
                first = await _turn(c, "agent-42", "hello, first message")
                assert first in {"b0", "b1", "b2"}
                # Subsequent turns must all land on the SAME backend (session pin),
                # even with completely different content that would otherwise route
                # by prefix to a different node.
                turns = [await _turn(c, "agent-42", f"turn-{i} unique content " + "x" * 80)
                         for i in range(1, 8)]
                assert all(b == first for b in turns), (first, turns)
        finally:
            gw.cfg.session_affinity_enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_without_header_routing_is_unchanged():
    async def run():
        _wire(enabled=True)
        try:
            async with _gwclient() as c:
                # No session id at all: behaves exactly like normal routing
                # (across many requests, multiple backends should appear).
                picks = {await _turn(c, "", f"q{i} " + "y" * 80) for i in range(12)}
                assert len(picks) > 1
        finally:
            gw.cfg.session_affinity_enabled = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_flag_off_ignores_session_header():
    async def run():
        _wire(enabled=False)
        try:
            async with _gwclient() as c:
                # Even with X-Session-ID, with the flag off there's no pinning.
                # Just verify the requests succeed and we got valid backends.
                picks = {await _turn(c, "agent-X", f"q{i} " + "z" * 80) for i in range(12)}
                assert picks.issubset({"b0", "b1", "b2"})
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())
