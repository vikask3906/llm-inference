from __future__ import annotations

"""Time-to-first-token benchmark — the headline TTFT-reduction number.

For each request, measures the wall-clock time from sending until the first SSE
chunk arrives (not the full response). Fires N requests at concurrency C,
reports TTFT p50/p95/p99/mean and the average gateway-reported prefix-match
blocks (the routing-affinity signal).

Compare strategies on the same workload: `round_robin` vs `prefix_tree`. The
delta is the real-world impact of prefix-aware routing on first-token latency.

Usage:
  python bench/ttft_bench.py --url http://127.0.0.1:8000 \
      --strategy prefix_tree --n 200 --concurrency 8 \
      --model Qwen/Qwen2.5-1.5B-Instruct --label "prefix c8"
"""

import argparse
import asyncio
import statistics
import time

import httpx

SYSTEM = "You are a helpful assistant. " + ("Read the following carefully. " * 40)


def percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
    return sorted_xs[i]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="gateway base URL (no trailing /v1/...)")
    ap.add_argument("--strategy", default="prefix_tree",
                    choices=["round_robin", "consistent_hash", "prefix_tree"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--docs", type=int, default=8, help="distinct prompts to cycle")
    ap.add_argument("--model", default="mock-model",
                    help="model name the backend serves; matches vLLM's "
                         "--served-model-name (default in run_vllm_benchmark.sh)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    docs = [f"doc-{i} " + ("body " * 60) for i in range(args.docs)]
    sem = asyncio.Semaphore(args.concurrency)
    ttft: list[float] = []
    match_blocks: list[int] = []
    errors = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def one(k: int):
            nonlocal errors
            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM + docs[k % args.docs]},
                    {"role": "user", "content": f"q{k}"},
                ],
                "stream": True,
                "max_tokens": 32,        # keep responses short; we only care about TTFT
            }
            async with sem:
                t0 = time.perf_counter()
                try:
                    async with client.stream("POST", endpoint, json=body,
                                             headers={"x-routing-strategy": args.strategy}) as r:
                        if r.status_code != 200:
                            errors += 1
                            return
                        mb = r.headers.get("x-gw-match-blocks")
                        if mb is not None:
                            try:
                                match_blocks.append(int(mb))
                            except ValueError:
                                pass
                        first = True
                        async for _ in r.aiter_raw():
                            if first:
                                ttft.append((time.perf_counter() - t0) * 1000.0)
                                first = False
                            # drain the rest to release the connection cleanly
                except Exception:
                    errors += 1

        await asyncio.gather(*(one(k) for k in range(args.n)))

    ttft.sort()
    n = len(ttft)
    label = args.label or f"{args.strategy}@c{args.concurrency}"
    avg_match = (sum(match_blocks) / len(match_blocks)) if match_blocks else 0.0
    print(
        f"{label:<22} n={n:<4} errors={errors:<3} "
        f"avg_match_blocks={avg_match:5.1f}  "
        f"TTFT p50={percentile(ttft, 0.50):7.1f}  "
        f"p95={percentile(ttft, 0.95):7.1f}  "
        f"p99={percentile(ttft, 0.99):7.1f}  "
        f"mean={(statistics.mean(ttft) if ttft else 0):7.1f}  (ms)"
    )


if __name__ == "__main__":
    asyncio.run(main())
