#!/usr/bin/env python3
"""Snapshot vLLM's prefix-cache hit-rate gauges across the fleet.

vLLM does NOT emit `x-prefix-cache-hit` response headers (lesson from the prior
GPU run -- bench/loadtest.py's hit-rate column reads 0% against real vLLM).
The discriminating signal is `vllm:gpu_prefix_cache_hit_rate`, a per-engine
gauge exposed on /metrics in Prometheus text format.

This script pulls the gauge from each backend's /metrics and prints a one-line
snapshot. Run it BEFORE and AFTER each strategy so the delta is attributable
to that strategy alone -- the gauge is CUMULATIVE, so a single absolute reading
after both strategies have run is meaningless.

A clean A/B requires restarting vLLM between strategies (so each strategy reads
from a zero baseline); run_vllm_benchmark.sh now does that automatically.

Usage:
  python bench/vllm_cache_stats.py http://127.0.0.1:9001 http://127.0.0.1:9002
  python bench/vllm_cache_stats.py --label "after prefix_tree" \
      http://127.0.0.1:9001 http://127.0.0.1:9002
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

METRIC = "vllm:gpu_prefix_cache_hit_rate"


def fetch_metric(url: str, metric: str = METRIC, timeout: float = 5.0) -> float | None:
    """Pull /metrics, parse Prometheus text, return the metric's value or None."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] {url}: {exc}", file=sys.stderr)
        return None
    for line in text.splitlines():
        # Lines look like: vllm:gpu_prefix_cache_hit_rate{model_name="..."} 0.987
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0]
        if name == metric:
            try:
                return float(value)
            except ValueError:
                continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="one or more vLLM backend base URLs")
    ap.add_argument("--label", default="", help="snapshot label (printed in output)")
    ap.add_argument("--metric", default=METRIC, help="Prometheus metric name to pull")
    args = ap.parse_args()

    label = f"[{args.label}] " if args.label else ""
    values: list[float] = []
    print(f"{label}{args.metric}:")
    for url in args.urls:
        v = fetch_metric(url, args.metric)
        if v is None:
            print(f"  {url}  (no reading)")
        else:
            print(f"  {url}  {v:.4f}")
            values.append(v)
    if values:
        print(f"  fleet mean: {sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
