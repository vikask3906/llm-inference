from __future__ import annotations

"""Admission-control benchmark: SLO protection under pressure.

Drives a mixed gold/silver/bronze workload against a 3-backend fleet at
increasing offered load, two ways:

  * baseline    -- no admission control; every request is dispatched. Above
                   capacity, queue depth blows up and the per-request TTFT
                   degrades across ALL tiers (the head-of-line problem).
  * admission   -- TTFT-budget fail-fast + fleet-pressure priority shedding.
                   The admit floor rises with pressure; bronze is shed first,
                   so gold's TTFT stays under SLO while bronze gets backpressured.

The metric users care about: gold-tier SLO compliance (% of admitted gold
requests that meet `ttft_slo_ms`). Show that the controller holds it near 100%
while baseline collapses; report bronze's served-rate as the explicit cost.

Standalone, no network, no GPU. Same caveat as bench/matrix.py, bench/rag_bench.py,
bench/dag_bench.py: this measures policy behaviour, not absolute latency.

Usage:
    python bench/admission_bench.py                  # CSV + markdown + chart
    python bench/admission_bench.py --no-charts      # CSV + markdown only
    python bench/admission_bench.py --requests 1000  # faster smoke run
"""

import argparse
import csv
import os
import random
import statistics
import sys
from collections import Counter, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.admission.config import AdmissionConfig             # noqa: E402
from gateway.admission.policy import (                            # noqa: E402
    ADMIT, AdmissionRequest, FleetState, decide as admission_decide,
)

BACKENDS = ["b0", "b1", "b2"]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

# Cost model (same proxy used by the live admission gate)
PREFILL_MS_PER_TOKEN = 0.05
TOKENS_PER_REQUEST_BASE = 800          # ~RAG-sized prompt
SERVICE_WINDOW = 30                    # ticks a request stays inflight
MAX_INFLIGHT = 16                      # per-backend cap (tighter than the default 64
                                       # so saturation is reachable in a bench run)
TTFT_SLO_MS = 500.0
SHED_PRESSURE = 0.75

# Tier mix in the offered traffic (sums to 1.0)
TIER_MIX = {"gold": 0.20, "silver": 0.30, "bronze": 0.50}

# Offered-load multiplier: 1.0 = matched to baseline capacity (about
# 3 * MAX_INFLIGHT / SERVICE_WINDOW req/tick), > 1.0 = overload.
SWEEP_LOAD = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
BASE_OFFER_PER_TICK = (3 * MAX_INFLIGHT) // SERVICE_WINDOW or 1


def make_workload(n_requests: int, load_mult: float, seed: int = 1) -> list[dict]:
    """A stream of requests with tier mix and a per-request TTFT estimate.

    A few requests carry a long prompt so their estimated TTFT exceeds the SLO
    -- the controller should fail-fast these. The rest land near the SLO so
    queue pressure decides who runs.
    """
    rng = random.Random(seed)
    tiers = []
    for t, p in TIER_MIX.items():
        tiers += [t] * int(round(p * 1000))
    rng.shuffle(tiers)
    out = []
    for i in range(n_requests):
        tier = tiers[i % len(tiers)]
        # 5% of requests carry a "huge" prompt -> doomed by TTFT budget.
        huge = rng.random() < 0.05
        prompt_tokens = (TOKENS_PER_REQUEST_BASE * 4) if huge else TOKENS_PER_REQUEST_BASE
        out.append({
            "tier": tier,
            "prompt_tokens": prompt_tokens,
            # All requests assumed uncached for this benchmark (worst case).
            "est_ttft_ms": PREFILL_MS_PER_TOKEN * prompt_tokens,
            "load_mult": load_mult,
            "huge": huge,
        })
    return out


def evaluate(controller: str, n_requests: int, load_mult: float) -> dict:
    """controller: "baseline" (no admission) or "admission" (gate enabled).

    Tick model: each tick offers `BASE_OFFER_PER_TICK * load_mult` requests.
    A request stays inflight for SERVICE_WINDOW ticks before completing. At
    load_mult > 1.0 the arrival rate exceeds drain rate, inflight builds up,
    and fleet pressure crosses the shed threshold.
    """
    cfg = AdmissionConfig()
    cfg.enabled = True
    cfg.ttft_slo_ms = TTFT_SLO_MS
    cfg.shed_pressure = SHED_PRESSURE
    cfg.max_inflight_per_backend = MAX_INFLIGHT
    workload = make_workload(n_requests, load_mult)
    offers_per_tick = max(1, int(round(BASE_OFFER_PER_TICK * load_mult)))

    backend_inflight: dict[str, int] = {b: 0 for b in BACKENDS}
    queue_depth = 0
    # Each entry: (completion_tick, backend) -- so a single per-tick drain pass
    # handles arbitrary offers_per_tick batching.
    inflight_q: deque = deque()

    counts = Counter()
    admitted_ttfts: dict[str, list[float]] = {t: [] for t in TIER_MIX}
    served_by_tier: dict[str, int] = {t: 0 for t in TIER_MIX}
    offered_by_tier: dict[str, int] = {t: 0 for t in TIER_MIX}

    req_iter = iter(workload)
    tick = 0
    done = False
    while not done:
        # Drain completed requests for this tick.
        while inflight_q and inflight_q[0][0] <= tick:
            _, b = inflight_q.popleft()
            backend_inflight[b] = max(0, backend_inflight[b] - 1)

        # Offer this tick's batch of requests.
        for _ in range(offers_per_tick):
            try:
                req = next(req_iter)
            except StopIteration:
                done = True
                break
            offered_by_tier[req["tier"]] += 1
            target = min(BACKENDS, key=lambda b: backend_inflight[b])

            if controller == "admission":
                # kv_usage tracks inflight saturation -- in production it's the
                # GPU KV cache fill fraction, which scales with concurrent
                # active prefills. Using the ratio gives the controller a clean
                # [0,1] saturation signal.
                kv = {b: backend_inflight[b] / MAX_INFLIGHT for b in BACKENDS}
                fleet = FleetState(
                    backend_inflight=dict(backend_inflight),
                    backend_kv_usage=kv,
                    queue_depth=queue_depth)
                decision = admission_decide(
                    AdmissionRequest(est_ttft_ms=req["est_ttft_ms"], tier=req["tier"]),
                    fleet, cfg)
                counts[decision.action] += 1
                if decision.action != ADMIT:
                    continue
            else:
                counts[ADMIT] += 1

            # Observed TTFT: prefill cost + queue penalty proportional to
            # inflight already on the target backend.
            observed_ttft = (req["est_ttft_ms"]
                             + backend_inflight[target] * 25.0)
            admitted_ttfts[req["tier"]].append(observed_ttft)
            served_by_tier[req["tier"]] += 1

            backend_inflight[target] += 1
            inflight_q.append((tick + SERVICE_WINDOW, target))
        tick += 1

    slo_ok: dict[str, float] = {}
    p99: dict[str, float] = {}
    mean_ttft: dict[str, float] = {}
    for t in TIER_MIX:
        ts = admitted_ttfts[t]
        if not ts:
            slo_ok[t] = float("nan")
            p99[t] = float("nan")
            mean_ttft[t] = float("nan")
            continue
        slo_ok[t] = sum(1 for v in ts if v <= TTFT_SLO_MS) / len(ts)
        p99[t] = sorted(ts)[max(0, int(0.99 * len(ts)) - 1)]
        mean_ttft[t] = statistics.mean(ts)

    return {
        "controller": controller,
        "load_mult": load_mult,
        "n_requests": n_requests,
        "admitted": counts[ADMIT],
        "queued": counts.get("queue", 0),
        "rejected": counts.get("reject", 0),
        "gold_served_rate": served_by_tier["gold"] / max(1, offered_by_tier["gold"]),
        "silver_served_rate": served_by_tier["silver"] / max(1, offered_by_tier["silver"]),
        "bronze_served_rate": served_by_tier["bronze"] / max(1, offered_by_tier["bronze"]),
        "gold_slo_ok": slo_ok["gold"],
        "silver_slo_ok": slo_ok["silver"],
        "bronze_slo_ok": slo_ok["bronze"],
        "gold_p99_ms": p99["gold"],
        "gold_mean_ms": mean_ttft["gold"],
        "bronze_mean_ms": mean_ttft["bronze"],
    }


def run_sweep(n_requests: int) -> list[dict]:
    rows: list[dict] = []
    for load in SWEEP_LOAD:
        for ctrl in ("baseline", "admission"):
            rows.append(evaluate(ctrl, n_requests, load))
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    fields = ["controller", "load_mult", "n_requests", "admitted", "queued",
              "rejected",
              "gold_served_rate", "silver_served_rate", "bronze_served_rate",
              "gold_slo_ok", "silver_slo_ok", "bronze_slo_ok",
              "gold_p99_ms", "gold_mean_ms", "bronze_mean_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(rows: list[dict], path: str) -> None:
    lines = [
        "# Admission-Control Benchmark: SLO Protection Under Pressure",
        "",
        "Mixed gold/silver/bronze workload (mix: "
        f"{int(TIER_MIX['gold']*100)}% gold / "
        f"{int(TIER_MIX['silver']*100)}% silver / "
        f"{int(TIER_MIX['bronze']*100)}% bronze) at increasing offered load,",
        f"compared with and without admission control. SLO = {TTFT_SLO_MS:.0f}ms TTFT;",
        f"shed pressure = {SHED_PRESSURE}; fleet = {len(BACKENDS)} backends "
        f"@ {MAX_INFLIGHT} max inflight each.",
        "",
        "## Gold-tier SLO compliance",
        "",
        "| load_mult | baseline gold OK | admission gold OK | "
        "baseline bronze served | admission bronze served |",
        "|---|---|---|---|---|",
    ]
    by_load: dict[float, dict[str, dict]] = {}
    for r in rows:
        by_load.setdefault(r["load_mult"], {})[r["controller"]] = r
    for load in sorted(by_load):
        b = by_load[load]["baseline"]
        a = by_load[load]["admission"]
        lines.append(
            f"| {load:.2f}x | "
            f"{b['gold_slo_ok']*100:.1f}% | "
            f"{a['gold_slo_ok']*100:.1f}% | "
            f"{b['bronze_served_rate']*100:.1f}% | "
            f"{a['bronze_served_rate']*100:.1f}% |")
    lines += [
        "",
        "**Reading the table**: at light load (< 1x) baseline and admission are",
        "identical -- nothing to protect against. As offered load climbs past",
        f"capacity, baseline's gold-SLO compliance collapses (every tier shares the",
        "queue); admission's stays high because it sheds bronze first. The cost",
        "is visible too: bronze's served-rate drops under admission, by design.",
        "",
        "## Full per-tier metrics",
        "",
        "| load_mult | controller | gold served | silver served | bronze served | "
        "gold mean TTFT (ms) | bronze mean TTFT (ms) | rejected | queued |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for load in sorted(by_load):
        for ctrl in ("baseline", "admission"):
            r = by_load[load][ctrl]
            lines.append(
                f"| {load:.2f}x | {ctrl} | "
                f"{r['gold_served_rate']*100:.1f}% | "
                f"{r['silver_served_rate']*100:.1f}% | "
                f"{r['bronze_served_rate']*100:.1f}% | "
                f"{r['gold_mean_ms']:.1f} | "
                f"{r['bronze_mean_ms']:.1f} | "
                f"{r['rejected']} | "
                f"{r['queued']} |")
    lines += [
        "",
        "## Caveat",
        "",
        "TTFT here is the same prefill + queue-penalty proxy the admission gate's",
        "cost function uses; not wall-clock. The benchmark measures the controller's",
        "policy behaviour (who gets admitted at what fleet pressure), which is what",
        "it actually decides. Absolute ms validation is the GPU job; same caveat",
        "as `bench/matrix.py`, `bench/rag_bench.py`, `bench/dag_bench.py`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_charts(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[charts skipped: matplotlib unavailable -- {exc}]")
        return

    by_load: dict[float, dict[str, dict]] = {}
    for r in rows:
        by_load.setdefault(r["load_mult"], {})[r["controller"]] = r
    loads = sorted(by_load)

    base_gold = [by_load[l]["baseline"]["gold_slo_ok"] * 100 for l in loads]
    adm_gold = [by_load[l]["admission"]["gold_slo_ok"] * 100 for l in loads]
    base_bronze_srv = [by_load[l]["baseline"]["bronze_served_rate"] * 100 for l in loads]
    adm_bronze_srv = [by_load[l]["admission"]["bronze_served_rate"] * 100 for l in loads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loads, base_gold, marker="o", label="baseline (no admission)", color="#888")
    ax1.plot(loads, adm_gold, marker="o", label="admission", color="#1f77b4")
    ax1.set_xlabel("offered load multiplier")
    ax1.set_ylabel("gold-tier SLO compliance (%)")
    ax1.set_title("Gold-tier SLO under pressure")
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(loads, base_bronze_srv, marker="o", label="baseline (no admission)", color="#888")
    ax2.plot(loads, adm_bronze_srv, marker="o", label="admission", color="#1f77b4")
    ax2.set_xlabel("offered load multiplier")
    ax2.set_ylabel("bronze-tier served rate (%)")
    ax2.set_title("Bronze sheds first (the cost)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Admission control: gold protected, bronze shed")
    fig.tight_layout()
    out = os.path.join(out_dir, "admission_sweep.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def print_summary(rows: list[dict]) -> None:
    by_load: dict[float, dict[str, dict]] = {}
    for r in rows:
        by_load.setdefault(r["load_mult"], {})[r["controller"]] = r
    print(f"backends={len(BACKENDS)}  ttft_slo={TTFT_SLO_MS:.0f}ms  "
          f"shed_pressure={SHED_PRESSURE}  mix={TIER_MIX}")
    print(f"{'load':>5} {'controller':>10}  "
          f"{'gold_OK':>8} {'bronze_OK':>10} "
          f"{'gold_srv':>9} {'bronze_srv':>11}")
    for load in sorted(by_load):
        for ctrl in ("baseline", "admission"):
            r = by_load[load][ctrl]
            print(f"{load:>5.2f} {ctrl:>10}  "
                  f"{r['gold_slo_ok']*100:>7.1f}% "
                  f"{r['bronze_slo_ok']*100:>9.1f}% "
                  f"{r['gold_served_rate']*100:>8.1f}% "
                  f"{r['bronze_served_rate']*100:>10.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000,
                    help="requests per sweep point")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = run_sweep(args.requests)
    csv_path = os.path.join(OUT_DIR, "admission_sweep.csv")
    md_path = os.path.join(OUT_DIR, "ADMISSION_RESULTS.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print_summary(rows)
    print()
    if not args.no_charts:
        render_charts(rows, OUT_DIR)
    print(f"wrote {os.path.relpath(csv_path, _ROOT)}, "
          f"{os.path.relpath(md_path, _ROOT)}")


if __name__ == "__main__":
    main()
