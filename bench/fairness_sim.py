from __future__ import annotations

"""Fairness-under-contention benchmark (stdlib only).

Two tenants offer load for a fixed window: a GREEDY tenant floods well past its
quota, a POLITE tenant sends a modest, within-quota rate. We compare two limiter
designs with the SAME total budget:

  - per-tenant : each tenant gets its own token bucket (this gateway's design)
  - shared     : one global bucket for everyone

With per-tenant buckets the polite tenant is fully served regardless of the
greedy flood; with a shared bucket the greedy tenant crowds it out (starvation).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.tenancy import TokenBucket  # noqa: E402

WINDOW_S = 10.0
STEP = 0.1
PER_TENANT_RPS = 10
OFFERED = {"greedy": 100, "polite": 5}      # requests/sec each tenant attempts


def _simulate(shared: bool, seed: int = 0):
    rng = random.Random(seed)
    if shared:
        glob = TokenBucket(PER_TENANT_RPS * len(OFFERED), PER_TENANT_RPS * len(OFFERED))
        bucket_of = {t: glob for t in OFFERED}
    else:
        bucket_of = {t: TokenBucket(PER_TENANT_RPS, PER_TENANT_RPS) for t in OFFERED}

    offered = {t: 0 for t in OFFERED}
    admitted = {t: 0 for t in OFFERED}
    throttled = {t: 0 for t in OFFERED}
    pending = {t: 0.0 for t in OFFERED}
    now = 0.0
    while now < WINDOW_S - 1e-9:
        attempts = []
        for t, rate in OFFERED.items():
            pending[t] += rate * STEP
            while pending[t] >= 1:
                pending[t] -= 1
                attempts.append(t)
        rng.shuffle(attempts)                # fair interleaving (no arrival-order bias)
        for t in attempts:
            offered[t] += 1
            if bucket_of[t].try_consume(1, now=now):
                admitted[t] += 1
            else:
                throttled[t] += 1
        now += STEP
    return offered, admitted, throttled


def run_per_tenant(seed: int = 0):
    return _simulate(shared=False, seed=seed)


def run_shared(seed: int = 0):
    return _simulate(shared=True, seed=seed)


def _report(mode: str, offered, admitted, throttled) -> None:
    for t in OFFERED:
        o, a = offered[t], admitted[t]
        pct = (a / o * 100) if o else 0.0
        print(f"{mode:<12}{t:<9}{o:<9}{a:<10}{throttled[t]:<11}{pct:5.1f}%")


def main() -> None:
    print(f"per-tenant limit = {PER_TENANT_RPS} rps  window = {WINDOW_S:.0f}s  "
          f"offered: greedy={OFFERED['greedy']} rps, polite={OFFERED['polite']} rps")
    print(f"{'mode':<12}{'tenant':<9}{'offered':<9}{'admitted':<10}{'throttled':<11}admit%")
    _report("shared", *run_shared(seed=1))
    _report("per-tenant", *run_per_tenant(seed=1))

    _, a_pt, _ = run_per_tenant(seed=1)
    _, a_sh, _ = run_shared(seed=1)
    o, _, _ = run_per_tenant(seed=1)
    pt = a_pt["polite"] / o["polite"] * 100
    sh = a_sh["polite"] / o["polite"] * 100
    print(f"\nPolite tenant served: {pt:.0f}% (per-tenant) vs {sh:.0f}% (shared) "
          f"-> isolation prevents the greedy tenant from starving it.")


if __name__ == "__main__":
    main()
