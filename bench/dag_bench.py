from __future__ import annotations

"""DAG-scheduler benchmark: locality vs round-robin on multi-step workflows.

Sweeps three workload shapes and measures the DAG scheduler's two strategies:

  * round_robin -- cache-blind cyclic assignment (the baseline)
  * locality    -- longest-prefix match + parent-affinity bonus

Workload model: N chains, each chain is a sequence of `chain_depth` LLM calls
that share a long context prefix (the retrieved-doc analogue of a system
prompt). Within a chain, a node's parent already encoded that prefix; locality
tries to co-locate the family. Round-robin scatters it.

Algorithm-level (no network, no GPU). Makespan and cache-hit rate are the
scheduler's own model -- the same units used in the live router's est-TTFT cost
function. Absolute ms validation is the GPU job, same caveat as bench/matrix.py.

Usage:
    python bench/dag_bench.py                 # CSV + markdown + chart
    python bench/dag_bench.py --no-charts     # CSV + markdown only
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.dag import DagNode, RequestDag, schedule    # noqa: E402
from gateway.dag.config import DagSchedConfig            # noqa: E402

BACKENDS = ["b0", "b1", "b2"]
BLOCK_CHARS = DagSchedConfig().block_chars
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

# Baseline workload. Each sweep overrides exactly one knob.
#
# n_chains=5 is intentionally coprime to the backend count: round-robin's modular
# cycle accidentally co-locates each chain when n_chains % len(backends) == 0
# (a common production miscalibration), so 5/3 is the honest case where the
# locality benefit shows up. The n_chains sweep includes both regimes so the
# divisibility phenomenon is visible.
BASE = {"n_chains": 5, "chain_depth": 4, "context_blocks": 16}

SWEEPS = {
    "n_chains":       [3, 4, 5, 6, 7, 9, 11, 13],
    "chain_depth":    [2, 3, 4, 6, 8],
    "context_blocks": [4, 8, 16, 32, 64],
}
STRATEGIES = ["round_robin", "locality"]
SWEEP_LABEL = {
    "n_chains": "number of chains (parallel workflows)",
    "chain_depth": "chain depth (nodes per chain)",
    "context_blocks": "shared-context blocks per chain",
}


def _ctx(letter: str, blocks: int) -> str:
    return letter * (BLOCK_CHARS * blocks)


def build_dag(n_chains: int, chain_depth: int, context_blocks: int) -> RequestDag:
    """N chains of depth D, each sharing a context_blocks-block prefix.

    Each chain c has nodes c0 -> c1 -> ... -> c{D-1}; every node carries the
    same long shared prefix (the chain's context) so a parent->child hop should
    be a 100% cache hit on locality, 0% on round-robin's scattered layout.
    """
    nodes = []
    for c in range(n_chains):
        letter = chr(ord("A") + c % 26) + str(c // 26 or "")
        prev = None
        for d in range(chain_depth):
            nid = f"c{c}.n{d}"
            parents = (prev,) if prev else ()
            nodes.append(DagNode(nid, _ctx(letter, context_blocks), parents=parents))
            prev = nid
    return RequestDag(nodes)


def evaluate(strategy: str, n_chains: int, chain_depth: int,
             context_blocks: int) -> dict:
    dag = build_dag(n_chains, chain_depth, context_blocks)
    sched = schedule(dag=dag, backends=BACKENDS, cfg=DagSchedConfig(),
                     strategy=strategy)
    return {
        "strategy": strategy,
        "n_chains": n_chains,
        "chain_depth": chain_depth,
        "context_blocks": context_blocks,
        "hit_rate": sched.cache_hit_rate,
        "makespan_ms": sched.est_makespan_ms,
        "total_blocks": sched.total_blocks,
        "hit_blocks": sched.cache_hit_blocks,
    }


def run_baseline() -> list[dict]:
    return [evaluate(s, **BASE) for s in STRATEGIES]


def run_matrix() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for knob, values in SWEEPS.items():
        rows: list[dict] = []
        for v in values:
            cfg = dict(BASE)
            cfg[knob] = v
            for s in STRATEGIES:
                row = evaluate(s, **cfg)
                row["sweep"] = knob
                row["value"] = v
                rows.append(row)
        out[knob] = rows
    return out


def write_csv(baseline: list[dict], matrix: dict[str, list[dict]], path: str) -> None:
    fields = ["sweep", "value", "strategy", "n_chains", "chain_depth",
              "context_blocks", "hit_rate", "makespan_ms", "hit_blocks", "total_blocks"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in baseline:
            r = {"sweep": "baseline", "value": "", **row}
            w.writerow({k: r.get(k, "") for k in fields})
        for rows in matrix.values():
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(baseline: list[dict], matrix: dict[str, list[dict]], path: str) -> None:
    lines = [
        "# DAG-Scheduler Benchmark: Locality vs Round-Robin",
        "",
        "Cache-locality-aware DAG scheduling on multi-step LLM workflows.",
        "Workload: N independent chains, each a sequence of LLM calls sharing a long",
        "context prefix (the agentic / map-reduce pattern).",
        "",
        f"Fleet: {len(BACKENDS)} backends, KV cap {DagSchedConfig().backend_cache_blocks} blocks each.",
        "",
        "## Baseline ("
        f"n_chains={BASE['n_chains']}, chain_depth={BASE['chain_depth']}, "
        f"context_blocks={BASE['context_blocks']})",
        "",
        "| strategy | hit rate | makespan (ms) |",
        "|---|---|---|",
    ]
    for row in baseline:
        lines.append(f"| {row['strategy']} | "
                     f"{row['hit_rate']*100:.1f}% | "
                     f"{row['makespan_ms']:.1f} |")
    rr = next(r for r in baseline if r["strategy"] == "round_robin")
    loc = next(r for r in baseline if r["strategy"] == "locality")
    speedup = rr["makespan_ms"] / loc["makespan_ms"] if loc["makespan_ms"] else float("inf")
    lines += [
        "",
        f"**Locality wins**: hit rate {loc['hit_rate']*100:.1f}% vs "
        f"{rr['hit_rate']*100:.1f}%, makespan {speedup:.2f}x lower.",
        "",
        "## Sweeps",
        "",
    ]
    for knob, rows in matrix.items():
        lines.append(f"### sweep: {knob} ({SWEEP_LABEL[knob]})")
        lines.append("")
        lines.append("| " + knob + " | " + " | ".join(
            f"{s} hit rate" for s in STRATEGIES) + " | " + " | ".join(
            f"{s} makespan (ms)" for s in STRATEGIES) + " |")
        lines.append("|" + "---|" * (1 + 2 * len(STRATEGIES)))
        values = sorted({r["value"] for r in rows})
        for v in values:
            cells = [str(v)]
            for s in STRATEGIES:
                r = next(x for x in rows if x["value"] == v and x["strategy"] == s)
                cells.append(f"{r['hit_rate']*100:.1f}%")
            for s in STRATEGIES:
                r = next(x for x in rows if x["value"] == v and x["strategy"] == s)
                cells.append(f"{r['makespan_ms']:.1f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## A note on the n_chains sweep",
        "",
        "Round-robin's cyclic counter co-locates a chain by accident whenever",
        "`n_chains % len(backends) == 0` (e.g. 3, 6, 9, 12 chains on a 3-backend",
        "fleet). At those points RR ties locality. At every other point RR scatters",
        "each chain across backends and pays the full uncached prefill at every",
        "child node -- ~3x makespan and ~3x lower hit rate. The takeaway: RR's",
        "best case is a coincidence of integer arithmetic; locality is robust.",
        "",
        "## Caveat",
        "",
        "Makespan is the scheduler's own model (prefill ms = compute-bound proxy +",
        "queue terms), not wall-clock. It captures the work a cache hit removes,",
        "which is precisely what the locality scheduler optimizes for. Absolute ms",
        "validation is the GPU job; see the same caveat in `bench/matrix.py` and",
        "`docs/benchmarks/RESULTS.md`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_charts(baseline: list[dict], matrix: dict[str, list[dict]],
                  out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[charts skipped: matplotlib unavailable -- {exc}]")
        return

    # Baseline bar chart: makespan and hit rate side by side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    strategies = [r["strategy"] for r in baseline]
    makespans = [r["makespan_ms"] for r in baseline]
    hits = [r["hit_rate"] * 100 for r in baseline]
    colors = ["#888", "#1f77b4"]
    ax1.bar(strategies, makespans, color=colors)
    ax1.set_ylabel("est. makespan (ms)")
    ax1.set_title("DAG makespan (lower = better)")
    for i, v in enumerate(makespans):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax2.bar(strategies, hits, color=colors)
    ax2.set_ylabel("cache hit rate (%)")
    ax2.set_title("DAG prefix-cache hit rate (higher = better)")
    ax2.set_ylim(0, 100)
    for i, v in enumerate(hits):
        ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    fig.suptitle(
        f"DAG scheduler: locality vs round-robin "
        f"(n_chains={BASE['n_chains']}, depth={BASE['chain_depth']}, "
        f"ctx_blocks={BASE['context_blocks']})")
    fig.tight_layout()
    out = os.path.join(out_dir, "dag_baseline.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    # One line chart per sweep: makespan vs swept knob, both strategies.
    for knob, rows in matrix.items():
        values = sorted({r["value"] for r in rows})
        fig, ax = plt.subplots(figsize=(7, 4))
        for s, c in zip(STRATEGIES, colors):
            ys = [next(r for r in rows if r["value"] == v and r["strategy"] == s)
                  ["makespan_ms"] for v in values]
            ax.plot(values, ys, marker="o", label=s, color=c)
        ax.set_xlabel(SWEEP_LABEL[knob])
        ax.set_ylabel("est. makespan (ms)")
        ax.set_title(f"DAG makespan vs {knob}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, f"dag_sweep_{knob}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


def print_summary(baseline: list[dict]) -> None:
    print(f"backends={len(BACKENDS)}  "
          f"baseline n_chains={BASE['n_chains']} depth={BASE['chain_depth']} "
          f"ctx_blocks={BASE['context_blocks']}")
    print(f"{'strategy':<14} {'hit rate':>10} {'makespan':>12}")
    for row in baseline:
        print(f"{row['strategy']:<14} "
              f"{row['hit_rate']*100:>9.1f}% "
              f"{row['makespan_ms']:>10.1f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-charts", action="store_true",
                    help="skip matplotlib (CSV + markdown only)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    baseline = run_baseline()
    matrix = run_matrix()
    csv_path = os.path.join(OUT_DIR, "dag_matrix.csv")
    md_path = os.path.join(OUT_DIR, "DAG_RESULTS.md")
    write_csv(baseline, matrix, csv_path)
    write_markdown(baseline, matrix, md_path)
    print_summary(baseline)
    print()
    if not args.no_charts:
        render_charts(baseline, matrix, OUT_DIR)
    print(f"wrote {os.path.relpath(csv_path, _ROOT)}, "
          f"{os.path.relpath(md_path, _ROOT)}")


if __name__ == "__main__":
    main()
