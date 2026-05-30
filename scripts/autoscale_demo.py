#!/usr/bin/env python
"""Worked example of the SLO-driven autoscaler.

Sweeps offered load against a fixed fleet config and prints the controller's
decision: how many replicas the Erlang-C model + utilization cap want, what
that does to expected queue wait, and how clamping / step limits / cooldowns
temper the move. No GPU required.

Usage:
    python scripts/autoscale_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway.autoscale import AutoscaleConfig, ScalingSignal, plan


def main() -> None:
    cfg = AutoscaleConfig(
        service_rate_per_replica_rps=5.0,   # each replica clears ~5 req/s
        target_wait_ms=100.0,               # SLO: <=100ms expected queue wait
        target_utilization=0.70,
        min_replicas=1, max_replicas=20, max_scale_step=4,
    )
    print(f"fleet: mu={cfg.service_rate_per_replica_rps:.0f} req/s/replica   "
          f"SLO wait<={cfg.target_wait_ms:.0f}ms   "
          f"util<={cfg.target_utilization:.0%}   step<= {cfg.max_scale_step}   "
          f"bounds [{cfg.min_replicas},{cfg.max_replicas}]")
    print(f"{'offered rps':>11} {'current':>7} {'slo_n':>6} {'util_n':>6} "
          f"{'decision':>9} {'desired':>7} {'wait@desired':>13}  reason")
    print("-" * 104)

    current = 1
    now = 0.0
    for rps in (0, 5, 15, 30, 60, 90, 140):
        # fresh cooldown window each tick so the demo shows the unsuppressed move
        sig = ScalingSignal(offered_rps=float(rps), current_replicas=current,
                            now_s=now, last_scale_up_s=now - 999,
                            last_scale_down_s=now - 999)
        d = plan(sig, cfg)
        wait = "inf" if d.est_wait_ms == float("inf") else f"{d.est_wait_ms:.1f}ms"
        print(f"{rps:>11} {d.current_replicas:>7} {d.slo_replicas:>6} "
              f"{d.util_replicas:>6} {d.direction:>9} {d.desired_replicas:>7} "
              f"{wait:>13}  {d.reason}")
        current = d.desired_replicas       # carry the actuated count forward
        now += 60.0


if __name__ == "__main__":
    main()
