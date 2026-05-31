from __future__ import annotations

"""Weighted-fair-queuing benchmark: WFQ vs FIFO under a greedy tenant.

Scenario: a scarce dispatch slot is contended by three tiers. A **greedy bronze**
tenant floods the gateway (70% of arrivals) while gold sends only 10%. Capacity
serves half the arrivals — whose requests get served?

Metric = per-tier **completion rate** (served ÷ that tier's own arrivals):

  * FIFO  — serve in arrival order. Bronze's flood pushes everyone's completion
            down equally, so gold (a paying tier sending little) is dragged to
            ~50% by a noisy neighbour.
  * WFQ   — Start-time Fair Queuing (gateway/fairqueue). Serves the well-behaved
            tiers in full and makes the greedy bronze absorb the shortfall, so
            gold + silver stay near 100% completion regardless of bronze's flood.

Pure simulation, no network/GPU.

Usage:
    python bench/fairqueue_bench.py
    python bench/fairqueue_bench.py --no-charts
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.fairqueue import WeightedFairQueue   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

WEIGHTS = {"gold": 3.0, "silver": 2.0, "bronze": 1.0}     # entitlement
ARRIVAL_MIX = {"gold": 0.10, "silver": 0.20, "bronze": 0.70}   # greedy bronze floods
N_ARRIVALS = 6000
SERVE_FRACTION = 0.5                                       # capacity serves half


def make_arrivals(n: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    tiers = list(ARRIVAL_MIX)
    w = [ARRIVAL_MIX[t] for t in tiers]
    return rng.choices(tiers, weights=w, k=n)


def run(n: int, serve_fraction: float) -> dict:
    arrivals = make_arrivals(n)
    serve = int(n * serve_fraction)

    # FIFO: arrival order.
    fifo = Counter(arrivals[:serve])

    # WFQ: enqueue everything (backlogged), serve `serve` in fair order.
    q = WeightedFairQueue()
    for cls in arrivals:
        q.enqueue(cls, cls, WEIGHTS[cls])
    wfq = Counter(q.dequeue() for _ in range(serve))

    arr = Counter(arrivals)
    return {"arrivals": arr, "fifo": fifo, "wfq": wfq, "serve": serve}


def _completion(served: Counter, arrivals: Counter, t: str) -> float:
    a = arrivals.get(t, 0)
    return served.get(t, 0) / a if a else 0.0


def write_csv(res: dict, path: str) -> None:
    arr = res["arrivals"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tier", "weight", "arrivals", "fifo_completion", "wfq_completion"])
        for t in WEIGHTS:
            w.writerow([t, WEIGHTS[t], arr.get(t, 0),
                        f"{_completion(res['fifo'], arr, t):.3f}",
                        f"{_completion(res['wfq'], arr, t):.3f}"])


def write_markdown(res: dict, charts: bool, path: str) -> None:
    arr = res["arrivals"]
    lines = [
        "# Weighted-Fair-Queuing Benchmark: WFQ vs FIFO under a greedy tenant",
        "",
        f"Tiers gold:silver:bronze = weights {WEIGHTS['gold']:.0f}:{WEIGHTS['silver']:.0f}:"
        f"{WEIGHTS['bronze']:.0f}. A **greedy bronze** sends "
        f"{ARRIVAL_MIX['bronze']*100:.0f}% of arrivals; gold only "
        f"{ARRIVAL_MIX['gold']*100:.0f}%. Capacity serves {int(SERVE_FRACTION*100)}% of "
        "arrivals. Metric: **completion rate** = served ÷ that tier's own arrivals.",
        "",
    ]
    if charts:
        lines += ["![fairqueue](fairqueue.png)", ""]
    lines += ["| tier | weight | arrivals | FIFO completion | WFQ completion |",
              "|---|---|---|---|---|"]
    for t in WEIGHTS:
        lines.append(
            f"| {t} | {WEIGHTS[t]:.0f} | {arr.get(t,0)} | "
            f"{_completion(res['fifo'], arr, t)*100:.0f}% | "
            f"{_completion(res['wfq'], arr, t)*100:.0f}% |")
    g_fifo = _completion(res["fifo"], arr, "gold")
    g_wfq = _completion(res["wfq"], arr, "gold")
    b_wfq = _completion(res["wfq"], arr, "bronze")
    lines += [
        "",
        f"**FIFO punishes everyone for bronze's flood** — gold, a paying tier "
        f"sending only {ARRIVAL_MIX['gold']*100:.0f}% of traffic, is dragged down to "
        f"~{g_fifo*100:.0f}% completion by the noisy neighbour. **WFQ protects the "
        f"well-behaved tiers**: gold + silver stay near **{g_wfq*100:.0f}%/100%** "
        f"completion and the greedy bronze absorbs the shortfall (~{b_wfq*100:.0f}%) — "
        "isolation by tier weight, independent of how much bronze floods.",
        "",
        "## Caveat",
        "",
        "Pure scheduling simulation of `gateway/fairqueue.WeightedFairQueue` "
        "(Start-time Fair Queuing). It proves the fairness/isolation property; "
        "wiring it as an async admission queue under a live concurrency limit is "
        "the integration step. Same GPU-independent framing as the other benches.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_chart(res: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = res["arrivals"]
    tiers = list(WEIGHTS)
    fifo = [_completion(res["fifo"], arr, t) * 100 for t in tiers]
    wfq = [_completion(res["wfq"], arr, t) * 100 for t in tiers]
    import numpy as np
    x = np.arange(len(tiers))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, fifo, 0.4, label="FIFO", color="#d62728")
    ax.bar(x + 0.2, wfq, 0.4, label="WFQ", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(w={WEIGHTS[t]:.0f}, arr={ARRIVAL_MIX[t]*100:.0f}%)"
                        for t in tiers])
    ax.set_ylabel("completion rate (% of that tier's requests served)")
    ax.set_title("WFQ protects well-behaved tiers from a greedy bronze;\n"
                 "FIFO drags everyone down equally")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "fairqueue.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def print_summary(res: dict) -> None:
    arr = res["arrivals"]
    print(f"arrivals={N_ARRIVALS}  serve={res['serve']}  greedy bronze "
          f"({ARRIVAL_MIX['bronze']*100:.0f}% of arrivals)   metric=completion rate")
    print(f"{'tier':<8}{'weight':>7}{'arrivals':>10}{'FIFO%':>8}{'WFQ%':>8}")
    for t in WEIGHTS:
        print(f"{t:<8}{WEIGHTS[t]:>7.0f}{arr.get(t,0):>10}"
              f"{_completion(res['fifo'],arr,t)*100:>7.0f}%{_completion(res['wfq'],arr,t)*100:>7.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    res = run(N_ARRIVALS, SERVE_FRACTION)
    print_summary(res)
    print()
    write_csv(res, os.path.join(OUT_DIR, "fairqueue.csv"))
    charts_ok = False
    if not args.no_charts:
        try:
            render_chart(res, OUT_DIR)
            charts_ok = True
        except Exception as e:
            print(f"[charts skipped: {e}]")
    write_markdown(res, charts_ok, os.path.join(OUT_DIR, "FAIRQUEUE_RESULTS.md"))
    rel = os.path.relpath(OUT_DIR, _ROOT)
    print(f"wrote {rel}/fairqueue.csv, {rel}/FAIRQUEUE_RESULTS.md"
          + (", fairqueue.png" if charts_ok else ""))


if __name__ == "__main__":
    main()
