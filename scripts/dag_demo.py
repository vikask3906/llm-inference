#!/usr/bin/env python
"""Worked example of the cache-locality-aware DAG scheduler.

Builds a small agentic workload -- three independent retrieval+summarize
chains, each sharing its own long context prefix -- and schedules it across a
2-backend fleet two ways:

  * round_robin : cache-blind cyclic assignment (the baseline).
  * locality    : co-locate cache-sharing family members via longest-prefix
                  match + parent affinity.

Prints each placement and the aggregate prefix-cache hit rate / estimated
makespan, showing the locality scheduler reuse the chains' shared context that
round-robin scatters. No GPU required.

Usage:
    python scripts/dag_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway.dag import DagNode, RequestDag, schedule
from gateway.dag.config import DagSchedConfig

BLOCK = DagSchedConfig().block_chars


def _ctx(letter: str, blocks: int) -> str:
    """Stand-in for a shared retrieved-context prefix of `blocks` full blocks."""
    return letter * (BLOCK * blocks)


def build_workload() -> RequestDag:
    # Three chains A/B/C; each "summarize" child continues its chain's context.
    return RequestDag([
        DagNode("A.retrieve", _ctx("A", 8)),
        DagNode("B.retrieve", _ctx("B", 8)),
        DagNode("C.retrieve", _ctx("C", 8)),
        DagNode("A.summarize", _ctx("A", 8), parents=("A.retrieve",)),
        DagNode("B.summarize", _ctx("B", 8), parents=("B.retrieve",)),
        DagNode("C.summarize", _ctx("C", 8), parents=("C.retrieve",)),
    ])


def show(strategy: str) -> None:
    sched = schedule(dag=build_workload(), backends=["b0", "b1"],
                     cfg=DagSchedConfig(), strategy=strategy)
    print(f"== strategy: {strategy} ==")
    print(f"  {'node':>14} {'layer':>5} {'backend':>8} {'match':>6} "
          f"{'blocks':>6} {'est_ms':>7}")
    for p in sched.placements:
        print(f"  {p.node_id:>14} {p.layer:>5} {p.backend_id:>8} "
              f"{p.match_blocks:>6} {p.total_blocks:>6} {p.est_ms:>7.1f}")
    print(f"  -> cache hit rate {sched.cache_hit_rate:6.1%}   "
          f"({sched.cache_hit_blocks}/{sched.total_blocks} blocks)   "
          f"makespan {sched.est_makespan_ms:.1f} ms\n")


def main() -> None:
    print("workload: 3 retrieve->summarize chains, each sharing 8 context "
          "blocks, on a 2-backend fleet\n")
    show("round_robin")
    show("locality")


if __name__ == "__main__":
    main()
