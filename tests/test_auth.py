"""API-key authentication (gateway/auth + server enforcement)."""

import asyncio

import httpx

import gateway.server as gw
from gateway.auth import INVALID, MISSING, Authenticator, extract_bearer, parse_api_keys
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.load_tracker import LoadTracker
from gateway.metrics import MetricsCollector
from gateway.radix_tree import RadixTree
from gateway.router import Router
from gateway.tenancy import RateLimiter, TenantRegistry
from mock_backend.app import create_app


# --- unit: Authenticator ----------------------------------------------------

def test_disabled_always_authorizes():
    a = Authenticator(set(), require=False)
    assert a.check({}) == (True, None)                       # no key, still ok
    assert a.check({"authorization": "Bearer whatever"}) == (True, None)


def test_missing_key_rejected_when_required():
    a = Authenticator({"sk-1"}, require=True)
    assert a.check({}) == (False, MISSING)
    assert a.check({"authorization": "Token sk-1"}) == (False, MISSING)   # wrong scheme


def test_invalid_key_rejected_and_valid_accepted():
    a = Authenticator({"sk-good"}, require=True)
    assert a.check({"authorization": "Bearer sk-bad"}) == (False, INVALID)
    assert a.check({"authorization": "Bearer sk-good"}) == (True, None)


def test_extract_bearer_and_parse_keys():
    assert extract_bearer({"authorization": "Bearer  sk-x "}) == "sk-x"
    assert extract_bearer({"authorization": "bearer sk-y"}) == "sk-y"   # case-insensitive
    assert extract_bearer({}) is None
    assert parse_api_keys("a, b ,,c") == {"a", "b", "c"}


# --- end-to-end through the server ------------------------------------------

class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def _wire(valid_keys, require, tenants_spec=""):
    gw.cfg.backends = "b0=http://b0:9000"
    gw.cfg.rate_limit_enabled = False
    gw.cfg.require_auth = require
    gw.cfg.tenants = tenants_spec
    gw.cfg.prefix_isolation = "global"
    a = create_app("b0", 600)
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry(tenants_spec)
    gw.limiter = RateLimiter()
    gw.authenticator = Authenticator(set(valid_keys) | gw.tenants.keys(), require)
    gw.app.state.client = httpx.AsyncClient(
        transport=MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


async def _post(client, headers=None):
    body = {"model": "mock-model", "messages": [{"role": "user", "content": "hello"}]}
    async with client.stream("POST", "/v1/chat/completions", json=body,
                             headers=headers or {}) as resp:
        raw = b""
        async for chunk in resp.aiter_raw():
            raw += chunk
        return resp, dict(resp.headers), raw


def test_server_401_without_key_when_required():
    async def run():
        _wire({"sk-prod"}, require=True)
        try:
            async with _gwclient() as c:
                resp, hdrs, _ = await _post(c)                       # no Authorization
                assert resp.status_code == 401
                assert hdrs.get("www-authenticate") == "Bearer"
                assert hdrs.get("x-request-id")
            assert "gateway_auth_rejected_total" in gw.metrics.render()
        finally:
            gw.cfg.require_auth = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_server_401_with_bad_key_200_with_good_key():
    async def run():
        _wire({"sk-prod"}, require=True)
        try:
            async with _gwclient() as c:
                bad, _, _ = await _post(c, {"authorization": "Bearer nope"})
                assert bad.status_code == 401
                good, _, raw = await _post(c, {"authorization": "Bearer sk-prod"})
                assert good.status_code == 200
                assert b"[DONE]" in raw
        finally:
            gw.cfg.require_auth = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_tenant_key_counts_as_valid_when_auth_required():
    async def run():
        # A configured tenant key should authenticate AND resolve to its tenant.
        _wire(set(), require=True, tenants_spec="sk-gold=acme:gold")
        try:
            async with _gwclient() as c:
                resp, _, raw = await _post(c, {"authorization": "Bearer sk-gold"})
                assert resp.status_code == 200
                assert b"[DONE]" in raw
        finally:
            gw.cfg.require_auth = False
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_auth_disabled_allows_anonymous():
    async def run():
        _wire(set(), require=False)
        try:
            async with _gwclient() as c:
                resp, _, raw = await _post(c)                        # no key, auth off
                assert resp.status_code == 200
                assert b"[DONE]" in raw
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())
