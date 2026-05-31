"""Control-plane admin API: backend draining + state inspection, token-guarded."""

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

TOKEN = "admin-sekret"


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def _wire2(admin_token=TOKEN):
    gw.cfg.backends = "b0=http://b0:9000,b1=http://b1:9000"
    gw.cfg.rate_limit_enabled = False
    gw.cfg.require_auth = False
    gw.cfg.admin_token = admin_token
    gw.cfg.prefix_isolation = "global"
    a, b = create_app("b0", 600), create_app("b1", 600)
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
    # b2's transport is wired but it's NOT in the registry -> a runtime add can
    # bring it into rotation and serve real traffic.
    spare = create_app("b2", 600)
    gw.app.state.client = httpx.AsyncClient(transport=MultiHostTransport(
        {"b0": httpx.ASGITransport(app=a), "b1": httpx.ASGITransport(app=b),
         "b2": httpx.ASGITransport(app=spare)}))


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


async def _chat_backend(client):
    body = {"model": "mock-model", "messages": [{"role": "user", "content": "hi there"}]}
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        async for _ in resp.aiter_raw():
            pass
        return resp.headers.get("x-gw-backend")


def _admin(tok=TOKEN):
    return {"x-admin-token": tok} if tok else {}


# --- auth on the admin surface ----------------------------------------------

def test_admin_disabled_without_token_config():
    async def run():
        _wire2(admin_token="")                     # not configured -> surface off
        try:
            async with _gwclient() as c:
                r = await c.get("/admin/backends")
                assert r.status_code == 403
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_admin_requires_correct_token():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                assert (await c.get("/admin/backends")).status_code == 403          # none
                assert (await c.get("/admin/backends",
                                    headers=_admin("wrong"))).status_code == 403     # bad
                ok = await c.get("/admin/backends", headers=_admin())
                assert ok.status_code == 200
                ids = {b["id"] for b in ok.json()["backends"]}
                assert ids == {"b0", "b1"}
                # bearer form also accepted
                assert (await c.get("/admin/backends",
                                    headers={"authorization": f"Bearer {TOKEN}"})).status_code == 200
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


# --- draining changes routing -----------------------------------------------

def test_drain_excludes_backend_then_undrain_restores():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                # drain b0 -> all new traffic must land on b1
                d = await c.post("/admin/backends/b0/drain", headers=_admin())
                assert d.status_code == 200 and d.json()["draining"] is True
                picks = {await _chat_backend(c) for _ in range(8)}
                assert picks == {"b1"}

                # undrain -> b0 eligible again (both appear over enough requests)
                u = await c.post("/admin/backends/b0/undrain", headers=_admin())
                assert u.status_code == 200 and u.json()["draining"] is False
                picks2 = {await _chat_backend(c) for _ in range(12)}
                assert "b0" in picks2
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_add_backend_at_runtime_then_serves_traffic():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                # add b2 (wired in transport, absent from registry)
                r = await c.post("/admin/backends", headers=_admin(),
                                 json={"id": "b2", "url": "http://b2:9000"})
                assert r.status_code == 200 and r.json()["created"] is True
                assert "b2" in {b["id"] for b in
                                (await c.get("/admin/backends", headers=_admin())).json()["backends"]}
                # starts unhealthy; simulate the scrape confirming it, then funnel
                # all traffic to it by draining the originals.
                gw.registry.set_health("b2", True)
                await c.post("/admin/backends/b0/drain", headers=_admin())
                await c.post("/admin/backends/b1/drain", headers=_admin())
                picks = {await _chat_backend(c) for _ in range(6)}
                assert picks == {"b2"}                  # the runtime-added backend serves
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_remove_backend_at_runtime_excludes_from_routing():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                r = await c.delete("/admin/backends/b0", headers=_admin())
                assert r.status_code == 200 and r.json()["removed"] is True
                s = (await c.get("/admin/state", headers=_admin())).json()
                assert "b0" not in {b["id"] for b in s["backends"]}   # gone entirely
                assert set(s["routable"]) == {"b1"}
                picks = {await _chat_backend(c) for _ in range(6)}
                assert picks == {"b1"}
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_add_requires_id_and_url():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                r = await c.post("/admin/backends", headers=_admin(), json={"id": "x"})
                assert r.status_code == 400
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_remove_unknown_backend_404():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                r = await c.delete("/admin/backends/ghost", headers=_admin())
                assert r.status_code == 404
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_drain_unknown_backend_404():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                r = await c.post("/admin/backends/ghost/drain", headers=_admin())
                assert r.status_code == 404
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_admin_state_shape():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                r = await c.get("/admin/state", headers=_admin())
                assert r.status_code == 200
                s = r.json()
                assert s["strategy"] == gw.cfg.strategy
                assert set(s["routable"]) == {"b0", "b1"}
                assert {b["id"] for b in s["backends"]} == {"b0", "b1"}
                assert all("inflight" in b and "circuit" in b for b in s["backends"])
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_drained_backend_still_served_in_state_but_not_routable():
    async def run():
        _wire2()
        try:
            async with _gwclient() as c:
                await c.post("/admin/backends/b0/drain", headers=_admin())
                s = (await c.get("/admin/state", headers=_admin())).json()
                assert "b0" not in s["routable"]               # excluded from routing
                b0 = next(b for b in s["backends"] if b["id"] == "b0")
                assert b0["draining"] is True and b0["healthy"] is True   # alive, just drained
        finally:
            gw.cfg.admin_token = ""
            await gw.app.state.client.aclose()
    asyncio.run(run())
