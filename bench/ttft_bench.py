from __future__ import annotations

"""Time-to-first-token benchmark -- the headline TTFT-reduction number.

For each request, measures the wall-clock time from sending until the first SSE
chunk arrives (not the full response). Fires N requests at concurrency C,
reports TTFT p50/p95/p99/mean and the average gateway-reported prefix-match
blocks (the routing-affinity signal).

Compare strategies on the same workload: `round_robin` vs `prefix_tree`. The
delta is the real-world impact of prefix-aware routing on first-token latency.

WORKLOAD (must create cache pressure or nothing discriminates):
  Each request = a small shared system prefix + ONE of `--n-docs` large
  documents (the cacheable prefix) + a unique question (the tail). The doc is
  big (`--doc-chars`, ~4 chars/token) so:
    * a cache HIT skips a real prefill -> low TTFT
    * a cache MISS pays the full doc prefill -> high TTFT
  and the working set (`n_docs * doc_chars/4` tokens) must EXCEED the backend's
  KV cache, so round_robin (which scatters each doc across backends) thrashes
  while prefix_tree (which pins each doc to one backend) keeps its half resident.
  The shared system prefix is kept SMALL so the variable doc dominates the
  hit-rate signal -- a large shared prefix makes both strategies look ~99%
  (the trap the prior GPU run fell into).

Usage:
  python bench/ttft_bench.py --url http://127.0.0.1:8000 \
      --strategy prefix_tree --n 600 --concurrency 8 \
      --n-docs 50 --doc-chars 24000 --model mock-model --label "prefix c8"
"""

import argparse
import asyncio
import random
import statistics
import time

import httpx


def percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
    return sorted_xs[i]


def make_docs(n_docs: int, doc_chars: int) -> list[str]:
    # Each doc is distinct (prefixed with its index) and `doc_chars` long, so it
    # becomes a large, unique cacheable prefix once the system prompt is prepended.
    return [(f"doc{i:04d}-" * (doc_chars // 9 + 1))[:doc_chars] for i in range(n_docs)]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="gateway base URL (no trailing /v1/...)")
    ap.add_argument("--strategy", default="prefix_tree",
                    choices=["round_robin", "consistent_hash", "prefix_tree"])
    ap.add_argument("--n", type=int, default=600,
                    help="total requests (keep > n_docs so docs get revisited)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=50,
                    help="distinct large documents (the working set); size so "
                         "n_docs/2 docs ~fit the KV cache but n_docs don't")
    ap.add_argument("--doc-chars", type=int, default=24000,
                    help="chars per document (~4 chars/token); big => a cache miss "
                         "costs a real prefill, so TTFT moves")
    ap.add_argument("--system-chars", type=int, default=512,
                    help="shared system-prefix length (kept small so the variable "
                         "doc dominates the hit-rate signal)")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="output cap; we only measure TTFT")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="mock-model",
                    help="model name the backend serves; matches vLLM's "
                         "--served-model-name (default in run_vllm_benchmark.sh)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    system = "S" * args.system_chars
    docs = make_docs(args.n_docs, args.doc_chars)
    rng = random.Random(args.seed)
    # Pre-pick the doc per request so all strategies see the IDENTICAL sequence.
    doc_idx = [rng.randrange(args.n_docs) for _ in range(args.n)]

    sem = asyncio.Semaphore(args.concurrency)
    ttft: list[float] = []
    match_blocks: list[int] = []
    errors = 0

    async with httpx.AsyncClient(timeout=180.0) as client:
        async def one(k: int):
            nonlocal errors
            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system + docs[doc_idx[k]]},
                    {"role": "user", "content": f"q{k:06d}-unique-question"},
                ],
                "stream": True,
                "max_tokens": args.max_tokens,
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
