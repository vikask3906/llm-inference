import asyncio

import httpx

import gateway.server as gw
from gateway.backends import BackendRegistry
from gateway.load_tracker import LoadTracker
from gateway.radix_tree import RadixTree
from gateway.router import Router
from mock_backend.app import create_app


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


class DeadTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise httpx.ConnectError("backend down", request=request)


def _wire(backends_str, transport):
    gw.cfg.backends = backends_str
    gw.registry = BackendRegistry(gw.cfg)
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)
    gw.app.state.client = httpx.AsyncClient(transport=transport)


def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


def _msgs():
    return [
        {"role": "system", "content": "S" * 128 + "D" * 1000},
        {"role": "user", "content": "question"},
    ]


def _single_backend():
    mock = create_app("b0", 600)
    _wire("b0=http://b0:9000", MultiHostTransport({"b0": httpx.ASGITransport(app=mock)}))


async def _post_hit(client):
    async with client.stream("POST", "/v1/chat/completions",
                             json={"model": "mock-model", "messages": _msgs()},
                             headers={"x-routing-strategy": "prefix_tree"}) as resp:
        hit = resp.headers.get("x-prefix-cache-hit")
        body = b""
        async for chunk in resp.aiter_raw():
            body += chunk
        return resp, hit, body


# --- streaming + header propagation ---

def test_stream_and_headers():
    async def run():
        _single_backend()
        async with _gwclient() as c:
            resp, hit, body = await _post_hit(c)
            assert resp.status_code == 200
            assert resp.headers.get("x-gw-backend") == "b0"
            assert hit in ("true", "false")
            assert b"[DONE]" in body
        await gw.app.state.client.aclose()
    asyncio.run(run())


# --- end-to-end caching: repeat of identical prompt becomes a hit ---

def test_warm_cache_on_repeat():
    async def run():
        _single_backend()
        async with _gwclient() as c:
            _, hit1, _ = await _post_hit(c)
            _, hit2, _ = await _post_hit(c)
        assert hit1 == "false"
        assert hit2 == "true"
        await gw.app.state.client.aclose()
    asyncio.run(run())


# --- no backend serves the requested model -> 503 ---

def test_no_backend_for_model_returns_503():
    async def run():
        _single_backend()
        async with _gwclient() as c:
            resp = await c.post("/v1/chat/completions",
                                json={"model": "nonexistent", "messages": _msgs()})
            assert resp.status_code == 503
        await gw.app.state.client.aclose()
    asyncio.run(run())


# --- fault injection: backend unreachable -> 502 ---

def test_unreachable_backend_returns_502():
    async def run():
        _wire("dead=http://dead:9000", DeadTransport())
        async with _gwclient() as c:
            resp = await c.post("/v1/chat/completions",
                                json={"model": "mock-model", "messages": _msgs()})
            assert resp.status_code == 502
        await gw.app.state.client.aclose()
    asyncio.run(run())
