"""End-to-end: the live HTTP path publishes radix-tree mutations to the cluster
bus, and a peer replica converges on the routing state.

Proves the gateway/cluster package is actually invoked by the request handler
when GW_CLUSTER_ENABLED is set -- a prefix dispatched through replica A becomes
known to replica B after it drains the bus.
"""

import asyncio

import httpx

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.circuit import CircuitBreaker
from gateway.cluster import ClusterCoordinator, InMemoryBroker, InMemoryBus
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
    gw.cfg.prefix_isolation = "global"
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.metrics = MetricsCollector()
    gw.breaker = CircuitBreaker(gw.cfg.circuit_fail_threshold, gw.cfg.circuit_cooldown_s)
    gw.tenants = TenantRegistry("")
    gw.limiter = RateLimiter()
    gw.app.state.client = httpx.AsyncClient(transport=transport)


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


async def _post(client, body):
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        raw = b""
        async for chunk in resp.aiter_raw():
            raw += chunk
        return resp, dict(resp.headers), raw


def test_dispatch_publishes_and_peer_converges():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))

        # Replica A = the live gateway, wrapped around its tree on a shared broker.
        broker = InMemoryBroker()
        gw.cluster = ClusterCoordinator(gw.tree, InMemoryBus(broker, "A"), "A")
        # Replica B = a peer with its own tree on the same broker.
        peer_tree = RadixTree(gw.cfg.backend_cache_blocks)
        peer = ClusterCoordinator(peer_tree, InMemoryBus(broker, "B"), "B")
        saved = gw.cluster
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "X" * 4000}]}
                resp, hdrs, raw = await _post(c, body)
                assert resp.status_code == 200
                assert b"[DONE]" in raw

            # The hot path published the dispatched prefix...
            assert gw.cluster.published > 0
            # ...and replica B, before syncing, knows nothing.
            assert peer_tree.held_blocks("b0") == 0
            # After draining the bus, B's tree holds b0's prefix -> converged.
            applied = peer.sync()
            assert applied > 0
            assert peer_tree.held_blocks("b0") > 0
        finally:
            gw.cluster = None          # reset global so other tests are unaffected
            saved.close()
            await gw.app.state.client.aclose()
    asyncio.run(run())


def test_cluster_disabled_by_default_no_publish():
    async def run():
        a = create_app("b0", 600)
        _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=a)}))
        gw.cluster = None              # default/disabled
        try:
            async with _gwclient() as c:
                body = {"model": "mock-model",
                        "messages": [{"role": "user", "content": "hello world"}]}
                resp, _, raw = await _post(c, body)
                assert resp.status_code == 200
                assert b"[DONE]" in raw     # routes normally with cluster off
        finally:
            await gw.app.state.client.aclose()
    asyncio.run(run())
