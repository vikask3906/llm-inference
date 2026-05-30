#!/usr/bin/env python
"""Worked example of SLO-aware admission control + priority load shedding.

Shows two things with no GPU:

  1. the TTFT-budget gate rejecting a request whose best-case TTFT already
     blows the deadline (fail fast), and
  2. how the admit decision for each tenant tier changes as fleet pressure
     climbs -- bronze sheds first, then silver, with gold protected longest.

Usage:
    python scripts/admission_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway.admission import (
    AdmissionConfig,
    AdmissionRequest,
    FleetState,
    decide,
)


def main() -> None:
    cfg = AdmissionConfig(ttft_slo_ms=500.0, shed_pressure=0.75,
                          queue_enabled=True, max_queue_depth=256)

    print(f"SLO: TTFT <= {cfg.ttft_slo_ms:.0f}ms   shed above pressure "
          f"{cfg.shed_pressure:.2f}   tiers: gold > silver > bronze\n")

    print("== 1. TTFT-budget gate (fleet idle) ==")
    for ttft in (120.0, 480.0, 800.0):
        d = decide(AdmissionRequest(est_ttft_ms=ttft, tier="gold"),
                   FleetState(), cfg)
        print(f"  est TTFT {ttft:>5.0f}ms -> {d.action:>6}   {d.reason}")

    print("\n== 2. priority shedding as fleet pressure climbs ==")
    print(f"  {'pressure':>8} {'floor':>6}  {'bronze':>8} {'silver':>8} {'gold':>8}")
    print("  " + "-" * 46)
    for pressure in (0.50, 0.75, 0.85, 0.95, 1.00):
        fleet = FleetState(backend_kv_usage={"b0": pressure})
        row = {}
        floor = 0.0
        for tier in ("bronze", "silver", "gold"):
            d = decide(AdmissionRequest(est_ttft_ms=100.0, tier=tier), fleet, cfg)
            row[tier] = d.action
            floor = d.admit_floor
        print(f"  {pressure:>8.2f} {floor:>6.2f}  "
              f"{row['bronze']:>8} {row['silver']:>8} {row['gold']:>8}")


if __name__ == "__main__":
    main()
