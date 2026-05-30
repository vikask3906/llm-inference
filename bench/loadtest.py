from __future__ import annotations

"""End-to-end load test against a running gateway.

Generates shared-system-prompt + document + unique-question traffic, fires it at
the gateway with bounded concurrency, and tallies the prefix-cache hit rate
(from the backend's x-prefix-cache-hit header) plus the per-backend distribution
(from the gateway's x-gw-backend header).

The workload size (--n-docs, --doc-chars) is tunable so you can dial up the
working set until it EXCEEDS the backend's KV cache -- the only regime where
prefix routing's advantage shows up. Against a huge GPU KV cache a small working
set fits entirely, so every strategy hits ~99% and the benchmark can't
discriminate (see docs/GPU_RUNBOOK.md and the §D lessons in docs/BENCHMARKS.md).

NOTE on real vLLM: vLLM does NOT emit the `x-prefix-cache-hit` response header,
so the "request cache-hit rate" line reads 0% against real vLLM -- that column
is only meaningful against the mock backend. On real vLLM use the backend
distribution here plus `bench/vllm_cache_stats.py` (vllm:gpu_prefix_cache_hit_rate)
and `bench/ttft_bench.py` (TTFT) for the discriminating signals.

Usage:
  python bench/loadtest.py --strategy prefix_tree --n 1500 --concurrency 32
  # GPU cache-pressure workload (large working set):
  python bench/loadtest.py --strategy prefix_tree --n 600 --concurrency 16 \
      --n-docs 200 --doc-chars 24000 --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import asyncio
import random
import time
from collections import Counter

import httpx

DEFAULT_DOC_CHARS = 8000
DEFAULT_N_DOCS = 50


def make_requests(n: int, seed: int, n_docs: int = DEFAULT_N_DOCS,
                  doc_chars: int = DEFAULT_DOC_CHARS,
                  system_chars: int = 128) -> list[list[dict]]:
    rng = random.Random(seed)
    system = "S" * system_chars
    docs = [(f"doc{i:04d}-" * (doc_chars // 9 + 1))[:doc_chars] for i in range(n_docs)]
    out = []
    for k in range(n):
        doc = docs[rng.randrange(n_docs)]
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
    ap.add_argument("--n-docs", type=int, default=DEFAULT_N_DOCS,
                    help="distinct documents in the working set (raise to exceed KV cache)")
    ap.add_argument("--doc-chars", type=int, default=DEFAULT_DOC_CHARS,
                    help="characters per document (~4 chars/token; raise for cache pressure)")
    ap.add_argument("--system-chars", type=int, default=128,
                    help="shared system-prefix length (chars)")
    ap.add_argument("--model", default="mock-model",
                    help="model name the backend serves (use vLLM's --served-model-name)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap output tokens (0 = let the backend decide; set small on real vLLM)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to keep re-firing the batch (0 = a single pass of --n); "
                         "use a positive value to sustain traffic for a live Grafana demo")
    args = ap.parse_args()

    reqs = make_requests(args.n, seed=1, n_docs=args.n_docs,
                         doc_chars=args.doc_chars, system_chars=args.system_chars)
    sem = asyncio.Semaphore(args.concurrency)
    hits = Counter()
    dist = Counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def one(messages):
            async with sem:
                body = {"model": args.model, "messages": messages, "stream": True}
                if args.max_tokens > 0:
                    body["max_tokens"] = args.max_tokens
                headers = {"x-routing-strategy": args.strategy}
                async with client.stream("POST", f"{args.url}/v1/chat/completions",
                                         json=body, headers=headers) as resp:
                    hits["hit" if resp.headers.get("x-prefix-cache-hit") == "true" else "miss"] += 1
                    dist[resp.headers.get("x-gw-backend", "?")] += 1
                    async for _ in resp.aiter_raw():
                        pass

        async def batch():
            await asyncio.gather(*(one(m) for m in reqs))

        if args.duration > 0:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                await batch()
        else:
            await batch()

    total = hits["hit"] + hits["miss"]
    print(f"strategy={args.strategy}  requests={total}  "
          f"workload={args.n_docs} docs x {args.doc_chars} chars")
    print(f"  request cache-hit rate: {hits['hit'] / total * 100:.1f}%  "
          f"(mock-backend header only; reads 0% on real vLLM)")
    print(f"  backend distribution:   {dict(dist)}")


if __name__ == "__main__":
    asyncio.run(main())
