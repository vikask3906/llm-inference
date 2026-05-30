from __future__ import annotations

"""Benchmark matrix: sweep workload dimensions across the three routing
strategies and chart how prefix-aware routing holds up vs round-robin and
consistent-hash.

Algorithm-level (no network, no GPU). The cache hit rate is measured against
independent per-backend KV models (SimBackend), so it is the backend's OWN
truth, not the gateway's belief. The TTFT figure is a *prefill proxy*
(PREFILL_MS_PER_TOKEN x uncached tokens): it captures the compute a cache hit
removes, not absolute wall-clock latency. Absolute ms, tokenizer fidelity and
KV-memory limits are the job of the GPU validation run -- see the caveat written
into docs/benchmarks/RESULTS.md.

Four one-dimensional sweeps (each varies ONE knob, holding the rest at BASE):
  n_docs        -- working-set size vs fleet KV capacity
  skew_alpha    -- document popularity (Zipf exponent); hot docs => more reuse
  concurrency   -- in-flight depth vs the max_inflight saturation cutoff
  system_blocks -- shared-prefix length; this is what collapses consistent-hash

Usage:
    python bench/matrix.py                 # charts -> docs/benchmarks/, CSV, md
    python bench/matrix.py --no-charts     # CSV + markdown only (no matplotlib)
    python bench/matrix.py --requests 500  # faster smoke run
"""

import argparse
import csv
import os
import random
import statistics
import sys
from collections import Counter, OrderedDict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.config import Config            # noqa: E402
from gateway.load_tracker import LoadTracker  # noqa: E402
from gateway.radix_tree import RadixTree      # noqa: E402
from gateway.router import Router             # noqa: E402

BACKENDS = ["b0", "b1", "b2"]
CAP_BLOCKS = 600                 # per-backend KV capacity (blocks); fleet = 1800
BLOCK_CHARS = 64
PREFILL_MS_PER_TOKEN = 0.05      # documented TTFT-proxy constant (compute-bound prefill)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

# Baseline workload point. Each sweep overrides exactly one of these.
BASE = {"n_docs": 15, "doc_blocks": 96, "system_blocks": 2,
        "skew_alpha": 1.0, "concurrency": 48}

SWEEPS = {
    "n_docs":        [5, 10, 15, 20, 30, 50],
    "skew_alpha":    [0.0, 0.5, 1.0, 1.5, 2.0],
    "concurrency":   [12, 24, 48, 96, 192],
    "system_blocks": [0, 1, 2, 8, 32],
}
STRATEGIES = ["round_robin", "consistent_hash", "prefix_tree"]
SWEEP_LABEL = {
    "n_docs": "working set (documents)",
    "skew_alpha": "popularity skew (Zipf alpha)",
    "concurrency": "in-flight concurrency",
    "system_blocks": "shared-prefix blocks",
}


class SimBackend:
    """A backend's real KV prefix cache: contiguous-prefix LRU over blocks."""

    def __init__(self, cap_blocks: int) -> None:
        self.cap = cap_blocks
        self.cache: "OrderedDict[int, bool]" = OrderedDict()

    def process(self, hashes: list[int]) -> int:
        hit = 0
        for h in hashes:
            if h in self.cache:
                hit += 1
            else:
                break
        for h in hashes:
            if h in self.cache:
                self.cache.move_to_end(h)
            else:
                self.cache[h] = True
                if len(self.cache) > self.cap:
                    self.cache.popitem(last=False)
        return hit


def make_workload(n_requests: int, n_docs: int, doc_blocks: int,
                  system_blocks: int, skew_alpha: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    system = "S" * (system_blocks * BLOCK_CHARS)        # shared by every request
    doc_chars = doc_blocks * BLOCK_CHARS
    docs = []
    for i in range(n_docs):
        base = f"doc{i:04d}-"
        docs.append((base * (doc_chars // len(base) + 1))[:doc_chars])   # distinct blocks
    weights = [1.0 / (r + 1) ** skew_alpha for r in range(n_docs)]       # alpha=0 -> uniform
    reqs = []
    for k in range(n_requests):
        di = rng.choices(range(n_docs), weights=weights, k=1)[0]
        q = f"q{k:06d}-unique-question"                  # < 64 chars -> partial block, dropped
        reqs.append(system + docs[di] + q)
    return reqs


def _cfg() -> Config:
    cfg = Config()
    cfg.backend_cache_blocks = CAP_BLOCKS
    # Same calibration as bench/sim.py: a gentle soft load tiebreaker so cached
    # prefixes stay pinned, with the HARD max_inflight cutoff providing balance.
    cfg.service_ms_per_request = 1.0
    cfg.max_inflight = 24
    cfg.hysteresis_ms = 2.0
    return cfg


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, k))]


def evaluate(strategy: str, requests: list[str], concurrency: int, cfg: Config) -> dict:
    tree = RadixTree(CAP_BLOCKS)
    load = LoadTracker()
    router = Router(cfg, tree, load)
    sim = {b: SimBackend(CAP_BLOCKS) for b in BACKENDS}
    window: deque = deque()
    total_blocks = hit_blocks = 0
    dist: Counter = Counter()
    ttfts: list[float] = []

    for prompt in requests:
        r = router.choose(prompt, BACKENDS, strategy=strategy)
        hit = sim[r.backend_id].process(r.hashes)            # backend ground truth
        uncached_blocks = len(r.hashes) - hit
        ttfts.append(PREFILL_MS_PER_TOKEN * uncached_blocks * cfg.block_tokens)
        hit_blocks += hit
        tree.insert(r.hashes, r.backend_id)                  # gateway updates its belief
        load.on_dispatch(r.backend_id, r.tokens)
        window.append((r.backend_id, r.tokens))
        if len(window) > concurrency:
            ob, ot = window.popleft()
            load.on_complete(ob, ot)
        total_blocks += len(r.hashes)
        dist[r.backend_id] += 1

    counts = [dist.get(b, 0) for b in BACKENDS]
    mean_count = sum(counts) / len(counts)
    cov = (statistics.pstdev(counts) / mean_count) if mean_count else 0.0  # 0 = balanced
    # The per-request proxy is bimodal (0 on a hit, full doc-prefill on a cold miss),
    # so p99 is the same cold-miss cost for every strategy; the MEAN is what tracks
    # routing quality (it is proportional to 1 - hit_rate). Lead with the mean.
    return {
        "hit_rate": hit_blocks / total_blocks if total_blocks else 0.0,
        "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_p99": _pct(sorted(ttfts), 99),
        "cov": cov,
        "dist": counts,
    }


def run_matrix(n_requests: int) -> list[dict]:
    cfg = _cfg()
    rows: list[dict] = []
    for dim, values in SWEEPS.items():
        for v in values:
            params = dict(BASE)
            params[dim] = v
            requests = make_workload(n_requests, params["n_docs"], params["doc_blocks"],
                                     params["system_blocks"], params["skew_alpha"], seed=1)
            for strat in STRATEGIES:
                m = evaluate(strat, requests, params["concurrency"], cfg)
                rows.append({"sweep": dim, "x": v, "strategy": strat,
                             "hit_rate": m["hit_rate"], "ttft_mean": m["ttft_mean"],
                             "ttft_p99": m["ttft_p99"], "cov": m["cov"]})
    return rows


def run_baseline(n_requests: int) -> dict[str, dict]:
    cfg = _cfg()
    requests = make_workload(n_requests, BASE["n_docs"], BASE["doc_blocks"],
                             BASE["system_blocks"], BASE["skew_alpha"], seed=1)
    return {s: evaluate(s, requests, BASE["concurrency"], cfg) for s in STRATEGIES}


# --- outputs -----------------------------------------------------------------

def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "x", "strategy", "hit_rate",
                                          "ttft_mean", "ttft_p99", "cov"])
        w.writeheader()
        for row in rows:
            w.writerow({**row, "hit_rate": f"{row['hit_rate']:.4f}",
                        "ttft_mean": f"{row['ttft_mean']:.2f}",
                        "ttft_p99": f"{row['ttft_p99']:.2f}",
                        "cov": f"{row['cov']:.4f}"})


def write_markdown(baseline: dict[str, dict], rows: list[dict], charts: bool, path: str) -> None:
    lines = ["# Routing benchmark matrix", "",
             f"Fleet: {len(BACKENDS)} backends x {CAP_BLOCKS} blocks. "
             f"Baseline workload: {BASE['n_docs']} docs, {BASE['doc_blocks']} blocks/doc, "
             f"{BASE['system_blocks']} shared-prefix blocks, Zipf alpha={BASE['skew_alpha']}, "
             f"concurrency={BASE['concurrency']}.", "",
             "## What these numbers are (and are not)", "",
             "The **block hit rate** is measured against an independent per-backend KV "
             "cache model, so it is the backend's own truth, not the gateway's belief. "
             "The **TTFT** column is a *prefill proxy* "
             f"({PREFILL_MS_PER_TOKEN} ms/token x uncached tokens): it isolates the "
             "compute a cache hit removes. The proxy is bimodal per request (~0 on a "
             "hit, the full document prefill on a cold miss), so the **mean** is the "
             "metric that tracks routing quality; the **p99** is the same cold-miss "
             "cost for every strategy -- prefix routing does not make a cold miss "
             "cheaper, it makes one *rarer*. These quantities are GPU-independent -- "
             "they measure *routing quality*. Absolute wall-clock latency, tokenizer "
             "fidelity and KV-memory limits require the vLLM GPU validation run and are "
             "out of scope here. **cov** is the coefficient of variation of the request "
             "distribution (0 = perfectly balanced).", "",
             "## Baseline", "",
             "| strategy | block hit rate | TTFT mean (ms) | TTFT p99 (ms) | load cov |",
             "|---|---|---|---|---|"]
    for s in STRATEGIES:
        m = baseline[s]
        lines.append(f"| {s} | {m['hit_rate']*100:.1f}% | {m['ttft_mean']:.1f} | "
                     f"{m['ttft_p99']:.1f} | {m['cov']:.2f} |")
    lines.append("")

    by = {}
    for row in rows:
        by.setdefault(row["sweep"], []).append(row)
    for dim, values in SWEEPS.items():
        lines += [f"## Sweep: {SWEEP_LABEL[dim]}", ""]
        if charts:
            lines += [f"![{dim}](sweep_{dim}.png)", ""]
        lines.append("| " + SWEEP_LABEL[dim] + " | " +
                     " | ".join(f"{s} hit%" for s in STRATEGIES) + " |")
        lines.append("|" + "---|" * (len(STRATEGIES) + 1))
        for v in values:
            cells = []
            for s in STRATEGIES:
                m = next(r for r in by[dim] if r["x"] == v and r["strategy"] == s)
                cells.append(f"{m['hit_rate']*100:.1f}")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def render_charts(rows: list[dict], baseline: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {}
    for row in rows:
        by.setdefault(row["sweep"], []).append(row)

    for dim, values in SWEEPS.items():
        fig, (ax_hit, ax_ttft) = plt.subplots(1, 2, figsize=(11, 4))
        for s in STRATEGIES:
            pts = sorted((r for r in by[dim] if r["strategy"] == s), key=lambda r: values.index(r["x"]))
            xs = [str(p["x"]) for p in pts]
            ax_hit.plot(xs, [p["hit_rate"] * 100 for p in pts], marker="o", label=s)
            ax_ttft.plot(xs, [p["ttft_mean"] for p in pts], marker="o", label=s)
        ax_hit.set(title="block hit rate", xlabel=SWEEP_LABEL[dim], ylabel="hit rate (%)")
        ax_ttft.set(title="TTFT mean (prefill proxy)", xlabel=SWEEP_LABEL[dim], ylabel="ms")
        ax_hit.grid(True, alpha=0.3)
        ax_ttft.grid(True, alpha=0.3)
        ax_hit.legend()
        fig.suptitle(f"Sweep: {SWEEP_LABEL[dim]}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"sweep_{dim}.png"), dpi=110)
        plt.close(fig)

    # baseline bars
    fig, (ax_hit, ax_ttft) = plt.subplots(1, 2, figsize=(9, 4))
    ax_hit.bar(STRATEGIES, [baseline[s]["hit_rate"] * 100 for s in STRATEGIES],
               color=["#bbb", "#f4a", "#4a8"])
    ax_hit.set(title="block hit rate", ylabel="hit rate (%)")
    ax_ttft.bar(STRATEGIES, [baseline[s]["ttft_mean"] for s in STRATEGIES],
                color=["#bbb", "#f4a", "#4a8"])
    ax_ttft.set(title="TTFT mean (prefill proxy)", ylabel="ms")
    for ax in (ax_hit, ax_ttft):
        ax.grid(True, axis="y", alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(15)
    fig.suptitle("Baseline workload")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "baseline.png"), dpi=110)
    plt.close(fig)


def print_summary(baseline: dict[str, dict]) -> None:
    print(f"\nbackends={len(BACKENDS)} cap={CAP_BLOCKS}  baseline n_docs={BASE['n_docs']} "
          f"skew={BASE['skew_alpha']} concurrency={BASE['concurrency']}")
    print(f"{'strategy':<18}{'hit rate':<12}{'TTFT mean':<13}{'TTFT p99':<12}{'load cov'}")
    for s in STRATEGIES:
        m = baseline[s]
        print(f"{s:<18}{m['hit_rate']*100:>6.1f}%     {m['ttft_mean']:>7.1f}ms     "
              f"{m['ttft_p99']:>7.1f}ms    {m['cov']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000, help="requests per sweep point")
    ap.add_argument("--no-charts", action="store_true", help="skip PNG charts (no matplotlib)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    baseline = run_baseline(args.requests)
    rows = run_matrix(args.requests)
    print_summary(baseline)

    write_csv(rows, os.path.join(OUT_DIR, "matrix.csv"))
    charts_ok = False
    if not args.no_charts:
        try:
            render_charts(rows, baseline)
            charts_ok = True
        except Exception as e:                # matplotlib missing / headless issue
            print(f"\n[charts skipped: {e}]")
    write_markdown(baseline, rows, charts_ok, os.path.join(OUT_DIR, "RESULTS.md"))

    rel = os.path.relpath(OUT_DIR, _ROOT)
    print(f"\nwrote {rel}/matrix.csv, {rel}/RESULTS.md"
          + (f", and {len(SWEEPS) + 1} PNG charts" if charts_ok else " (no charts)"))


if __name__ == "__main__":
    main()
