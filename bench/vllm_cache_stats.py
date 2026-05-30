#!/usr/bin/env python3
"""Snapshot vLLM's prefix-cache hit rate across the fleet (version-robust).

vLLM does NOT emit `x-prefix-cache-hit` response headers (lesson from the prior
GPU run -- bench/loadtest.py's hit-rate column reads 0% against real vLLM), and
it does NOT expose a TTFT metric we depend on (TTFT is measured client-side in
bench/ttft_bench.py). The ONE vLLM metric we read is the prefix-cache hit rate,
whose name changed across engine versions:

  * V0 engine (vLLM 0.6-0.7.x): gauge  vllm:gpu_prefix_cache_hit_rate
  * V1 engine (vLLM 0.8+):      counters vllm:prefix_cache_hits
                                         vllm:prefix_cache_queries
                                (hit rate = hits / queries; also seen with a
                                 `gpu_` prefix and/or `_total` suffix)

This script tries the gauge first, then computes hits/queries from whatever
counters are present, and dumps the raw `prefix_cache` lines so you can see
exactly what your build exposes.

Counters are CUMULATIVE since vLLM start, so to attribute a hit rate to ONE
strategy, restart vLLM before that strategy (run_vllm_benchmark.sh does this) --
then the reading after the run is that strategy's own rate. (For the gauge, the
same restart-for-clean-read rule applies.)

Usage:
  python bench/vllm_cache_stats.py http://127.0.0.1:9001 http://127.0.0.1:9002
  python bench/vllm_cache_stats.py --label "after prefix_tree" \
      http://127.0.0.1:9001 http://127.0.0.1:9002
  python bench/vllm_cache_stats.py --raw http://127.0.0.1:9001   # dump all cache lines
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

GAUGE = "vllm:gpu_prefix_cache_hit_rate"
# Accept the V1 counter names with or without a `gpu_` infix / `_total` suffix.
HIT_BASES = ("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits")
QUERY_BASES = ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries")


def _fetch_text(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] {url}: {exc}", file=sys.stderr)
        return None


def _parse(text: str) -> tuple[dict[str, float], list[str]]:
    """Return ({normalized_metric_name: summed_value}, [raw prefix_cache lines])."""
    sums: dict[str, float] = {}
    raw: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "prefix_cache" in line:
            raw.append(line)
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0].strip()
        # normalize: drop a trailing _total so counters match regardless of suffix
        base = name[:-6] if name.endswith("_total") else name
        try:
            sums[base] = sums.get(base, 0.0) + float(value)
        except ValueError:
            continue
    return sums, raw


def hit_rate_for(url: str) -> tuple[float | None, str, list[str]]:
    """Return (hit_rate or None, source_description, raw_lines)."""
    text = _fetch_text(url)
    if text is None:
        return None, "unreachable", []
    sums, raw = _parse(text)

    if GAUGE in sums:
        return sums[GAUGE], f"gauge {GAUGE}", raw

    hits = next((sums[b] for b in HIT_BASES if b in sums), None)
    queries = next((sums[b] for b in QUERY_BASES if b in sums), None)
    if hits is not None and queries is not None:
        rate = (hits / queries) if queries > 0 else 0.0
        return rate, f"counters hits/queries = {hits:.0f}/{queries:.0f}", raw

    return None, "no recognized prefix-cache metric", raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="one or more vLLM backend base URLs")
    ap.add_argument("--label", default="", help="snapshot label (printed in output)")
    ap.add_argument("--raw", action="store_true",
                    help="dump every metrics line containing 'prefix_cache'")
    args = ap.parse_args()

    label = f"[{args.label}] " if args.label else ""
    print(f"{label}prefix-cache hit rate:")
    values: list[float] = []
    for url in args.urls:
        rate, source, raw = hit_rate_for(url)
        if rate is None:
            print(f"  {url}  (no reading -- {source})")
            if raw:
                print(f"      raw prefix_cache lines from {url}:")
                for ln in raw:
                    print(f"        {ln}")
        else:
            print(f"  {url}  {rate:.4f}   [{source}]")
            values.append(rate)
        if args.raw and raw:
            for ln in raw:
                print(f"      {ln}")
    if values:
        print(f"  fleet mean: {sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
