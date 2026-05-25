from __future__ import annotations

"""End-to-end load test against a running gateway.

Generates shared-system-prompt + document + unique-question traffic, fires it at
the gateway with bounded concurrency, and tallies the prefix-cache hit rate
(from the backend's x-prefix-cache-hit header) plus the per-backend distribution
(from the gateway's x-gw-backend header).

Usage:
  python bench/loadtest.py --strategy prefix_tree --n 1500 --concurrency 32
  python bench/loadtest.py --strategy round_robin
"""

import argparse
import asyncio
import random
from collections import Counter

import httpx

DOC_CHARS = 6144
N_DOCS = 15


def make_requests(n: int, seed: int) -> list[list[dict]]:
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


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--strategy", default="prefix_tree")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    reqs = make_requests(args.n, seed=1)
    sem = asyncio.Semaphore(args.concurrency)
    hits = Counter()
    dist = Counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def one(messages):
            async with sem:
                body = {"model": "mock-model", "messages": messages, "stream": True}
                headers = {"x-routing-strategy": args.strategy}
                async with client.stream("POST", f"{args.url}/v1/chat/completions",
                                         json=body, headers=headers) as resp:
                    hits["hit" if resp.headers.get("x-prefix-cache-hit") == "true" else "miss"] += 1
                    dist[resp.headers.get("x-gw-backend", "?")] += 1
                    async for _ in resp.aiter_raw():
                        pass

        await asyncio.gather(*(one(m) for m in reqs))

    total = hits["hit"] + hits["miss"]
    print(f"strategy={args.strategy}  requests={total}")
    print(f"  request cache-hit rate: {hits['hit'] / total * 100:.1f}%")
    print(f"  backend distribution:   {dict(dist)}")


if __name__ == "__main__":
    asyncio.run(main())
