from __future__ import annotations

"""Latency / throughput benchmark against a running gateway (or a backend directly).

Measures the gateway's own overhead by hitting the same endpoint under load and
reporting throughput + latency percentiles. Run it against the direct backend
(baseline), the Python gateway, and the Rust gateway to compare; the added
latency = gateway latency - direct baseline at the same concurrency.

Usage:
  python bench/latency_bench.py --url http://127.0.0.1:8000 --n 3000 --concurrency 64 --label rust
"""

import argparse
import asyncio
import statistics
import time

import httpx

SYSTEM = "You are a helpful assistant. " + "context " * 40  # ~5 shared blocks


def percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
    return sorted_xs[i]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL of gateway or backend")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--docs", type=int, default=12, help="distinct prompts (spreads routing)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    docs = [f"doc-{i} " + ("body " * 60) for i in range(args.docs)]
    sem = asyncio.Semaphore(args.concurrency)
    lat: list[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def one(k: int):
            nonlocal errors
            body = {
                "model": "mock-model",
                "messages": [
                    {"role": "system", "content": SYSTEM + docs[k % args.docs]},
                    {"role": "user", "content": f"q{k}"},
                ],
                "stream": True,
            }
            async with sem:
                t0 = time.perf_counter()
                try:
                    async with client.stream("POST", endpoint, json=body) as r:
                        async for _ in r.aiter_raw():
                            pass
                        if r.status_code != 200:
                            errors += 1
                            return
                except Exception:
                    errors += 1
                    return
                lat.append((time.perf_counter() - t0) * 1000.0)

        t_start = time.perf_counter()
        await asyncio.gather(*(one(k) for k in range(args.n)))
        wall = time.perf_counter() - t_start

    lat.sort()
    thru = len(lat) / wall if wall > 0 else 0.0
    label = args.label or endpoint
    print(f"{label:<10} n={len(lat):<5} conc={args.concurrency:<4} errors={errors:<3} "
          f"throughput={thru:7.0f} req/s  "
          f"p50={percentile(lat, 0.50):6.2f}  p95={percentile(lat, 0.95):6.2f}  "
          f"p99={percentile(lat, 0.99):6.2f}  mean={statistics.mean(lat) if lat else 0:6.2f}  (ms)")


if __name__ == "__main__":
    asyncio.run(main())
