#!/usr/bin/env python
"""Worked example of the disaggregated prefill/decode router.

Prints, for a fixed 3-node fleet (one prefill+decode node, one decode-only
node), how the split-vs-co-locate decision changes as prompt length and the
co-located node's load vary. Demonstrates the core Splitwise/DistServe
tradeoff with no GPU: disaggregation wins under load until the KV-handoff
cost (which grows with prompt length) overtakes the decode-queue saving.

Usage:
    python scripts/disagg_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway.disagg import DisaggConfig, PoolRegistry, choose_disaggregated
from gateway.load_tracker import LoadTracker


def main() -> None:
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill,decode;b1:decode")
    cfg = DisaggConfig()

    print(f"fleet: b0=prefill+decode  b1=decode-only   "
          f"handoff={cfg.kv_bytes_per_token/1e3:.0f}KB/tok over {cfg.link_gbps:.0f}GB/s")
    print(f"{'prompt':>8} {'b0 load':>8} {'decision':>12} {'prefill':>8} {'decode':>7} "
          f"{'handoff':>8} {'total':>8}  reason")
    print("-" * 100)

    for prompt_tokens in (256, 2000, 8000, 100_000):
        for b0_inflight in (0, 4, 12):
            load = LoadTracker()
            load.inflight["b0"] = b0_inflight
            load.inflight["b1"] = 0
            d = choose_disaggregated(prompt_tokens=prompt_tokens, pools=reg,
                                     load=load, cfg=cfg)
            tag = "SPLIT" if d.disaggregated else "co-locate"
            route = f"{d.prefill_backend}->{d.decode_backend}"
            print(f"{prompt_tokens:>8} {b0_inflight:>8} {tag:>7} {route:>4} "
                  f"{d.est_prefill_ms:>8.1f} {d.est_decode_ms:>7.1f} "
                  f"{d.est_handoff_ms:>8.1f} {d.est_total_ms:>8.1f}  {d.reason}")
        print()


if __name__ == "__main__":
    main()
