from __future__ import annotations

"""In-process end-to-end test of the HTTP layer (no network, no Docker).

Wires the real gateway ASGI app to 3 isolated mock-backend ASGI apps via a
custom httpx transport that dispatches by host. This exercises the full data
plane: JSON parse -> model filter -> prefix/load routing -> tree.insert ->
in-flight accounting -> upstream SSE stream -> header propagation.
"""

import asyncio
import logging
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GW_BACKENDS"] = "b0=http://b0:9000,b1=http://b1:9000,b2=http://b2:9000"

import httpx  # noqa: E402

from gateway.load_tracker import LoadTracker  # noqa: E402
from gateway.radix_tree import RadixTree      # noqa: E402
from gateway.router import Router             # noqa: E402
import gateway.server as gw                   # noqa: E402
from mock_backend.app import create_app       # noqa: E402

logging.getLogger("gateway").setLevel(logging.WARNING)   # quiet per-request logs in the demo

DOC_CHARS, N_DOCS = 6144, 15


class MultiHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, by_host: dict[str, httpx.ASGITransport]):
        self.by_host = by_host

    async def handle_async_request(self, request):
        return await self.by_host[request.url.host].handle_async_request(request)


def make_requests(n: int, seed: int):
    rng = random.Random(seed)
    system = "S" * 128
    docs = [(f"doc{i:04d}-" * (DOC_CHARS // 9 + 1))[:DOC_CHARS] for i in range(N_DOCS)]
    out = []
    for k in range(n):
        doc = docs[rng.randrange(N_DOCS)]
        out.append([
            {"role": "system", "content": system + doc},
            {"role": "user", "content": f"q{k:06d}-unique-question"},
        ])
    return out


async def run(strategy: str, reqs, concurrency: int = 32):
    # fresh gateway state + fresh isolated mocks each run
    gw.cfg.service_ms_per_request = 1.0
    gw.cfg.max_inflight = 24
    gw.cfg.hysteresis_ms = 2.0
    gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
    gw.load = LoadTracker()
    gw.router = Router(gw.cfg, gw.tree, gw.load)

    mocks = {h: create_app(h, 600) for h in ("b0", "b1", "b2")}
    gw.app.state.client = httpx.AsyncClient(
        transport=MultiHostTransport({h: httpx.ASGITransport(app=a) for h, a in mocks.items()})
    )

    hits, dist, done = Counter(), Counter(), Counter()
    sem = asyncio.Semaphore(concurrency)
    gwclient = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    async def one(messages):
        async with sem:
            body = {"model": "mock-model", "messages": messages, "stream": True}
            headers = {"x-routing-strategy": strategy}
            async with gwclient.stream("POST", "/v1/chat/completions",
                                       json=body, headers=headers) as resp:
                hits["hit" if resp.headers.get("x-prefix-cache-hit") == "true" else "miss"] += 1
                dist[resp.headers.get("x-gw-backend", "?")] += 1
                tail = b""
                async for chunk in resp.aiter_raw():
                    tail = chunk
                if b"[DONE]" in tail:
                    done["ok"] += 1

    await asyncio.gather(*(one(m) for m in reqs))
    await gwclient.aclose()
    await gw.app.state.client.aclose()

    total = hits["hit"] + hits["miss"]
    print(f"{strategy:<16} hit={hits['hit']/total*100:5.1f}%  "
          f"streamed_ok={done['ok']}/{total}  dist={dict(dist)}")


async def main():
    reqs = make_requests(900, seed=1)
    print("In-process E2E (gateway ASGI -> 3 mock ASGI backends)")
    for strat in ("round_robin", "consistent_hash", "prefix_tree"):
        await run(strat, reqs)


if __name__ == "__main__":
    asyncio.run(main())
