from __future__ import annotations

"""RAG chunk-affinity routing benchmark: affinity vs round-robin vs consistent-hash.

Workload: a pool of N chunks (a corpus); each request retrieves K chunks
according to a Zipf popularity distribution (some chunks much more popular than
others -- the canonical RAG access pattern). We then route each request to a
backend and measure two things:

  * chunk-cache hit rate -- (overlap with backend's cached chunks) / (chunks in
    the request), aggregated across the run. The chunk-set analogue of the
    prefix-cache hit rate that bench/matrix.py reports for text.
  * mean per-request est-ms -- the same prefill proxy the router optimizes:
    PREFILL_MS_PER_TOKEN x uncached_chunks x tokens_per_chunk. Captures the
    compute a cache hit removes; not wall-clock TTFT (the GPU caveat).

Three strategies are compared:
  * round_robin   -- cycle through backends regardless of cache (baseline).
  * consistent_hash -- hash the first chunk id (the cheap chunk-stable router).
  * chunk_affinity -- the real router (gateway.rag.router): pick the backend
                      already holding the most of the request's chunks.

Standalone, no network, no GPU. Same caveat as bench/matrix.py and
bench/dag_bench.py: this measures routing quality, not absolute latency.

Usage:
    python bench/rag_bench.py                 # CSV + markdown + chart
    python bench/rag_bench.py --no-charts     # CSV + markdown only
    python bench/rag_bench.py --requests 500  # faster smoke run
"""

import argparse
import csv
import os
import random
import statistics
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.load_tracker import LoadTracker             # noqa: E402
from gateway.rag.config import RagConfig                  # noqa: E402
from gateway.rag.index import ChunkAffinityIndex          # noqa: E402
from gateway.rag.router import choose_rag_backend         # noqa: E402

BACKENDS = ["b0", "b1", "b2"]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

# Per-backend chunk-cache capacity used in the bench. Must be SMALLER than
# n_chunks_pool * len(BACKENDS) so eviction is binding -- otherwise every
# backend eventually holds the whole corpus and routing is moot. This is the
# realistic regime (a real RAG corpus is much larger than per-GPU KV).
CACHE_CAP_CHUNKS = 64

# Concurrency model: a request stays "in flight" for SERVICE_WINDOW subsequent
# requests, then completes. This gives the affinity router a real load-balance
# signal (without it, affinity degenerates to "always pick b0" in steady state,
# duplicating the popular set across backends instead of partitioning it).
SERVICE_WINDOW = 24

# Baseline workload. Each sweep overrides exactly one knob.
#
# n_topics > 1 models multi-tenant / multi-domain RAG: each request belongs to
# one topic and draws chunks ONLY from that topic's sub-corpus. The topics have
# disjoint chunk ranges. This is the realistic case where the chunk-affinity
# router gets to specialize each backend on a topic; round-robin scatters every
# topic across every backend (cache duplication), so each backend's cap only
# holds a fraction of each topic's hot set.
BASE = {
    "n_chunks_pool": 600,    # total corpus (split across n_topics)
    "k_per_request": 8,      # chunks retrieved per query
    "skew_alpha": 1.2,       # Zipf popularity (within a topic)
    "n_topics": 3,           # disjoint sub-corpora
    "n_requests": 2000,
}

SWEEPS = {
    "n_chunks_pool": [150, 300, 600, 1200, 2400],
    "k_per_request": [3, 5, 8, 12, 20],
    "skew_alpha":    [0.0, 0.5, 1.0, 1.5, 2.0],
    "n_topics":      [1, 2, 3, 5, 9],
}
STRATEGIES = ["round_robin", "consistent_hash", "chunk_affinity"]
SWEEP_LABEL = {
    "n_chunks_pool": "corpus size (chunks in the pool)",
    "k_per_request": "chunks retrieved per request",
    "skew_alpha": "Zipf skew (chunk popularity)",
    "n_topics": "disjoint sub-corpora (multi-tenant)",
}


def make_workload(n_chunks_pool: int, k_per_request: int, skew_alpha: float,
                  n_topics: int, n_requests: int, seed: int = 1) -> list[list[int]]:
    """Generate a sequence of requests; each request is a list of K chunk ids.

    The pool is split into `n_topics` disjoint sub-corpora (chunk-id ranges).
    Each request picks one topic uniformly at random and draws its K chunks
    ONLY from that topic, weighted by Zipf within the topic. This models
    multi-tenant or multi-domain RAG: topic-A queries never overlap with
    topic-B's chunks, so a router that pins each topic to a backend gets a
    clean partitioning win. n_topics=1 collapses to the single-pool case.
    """
    rng = random.Random(seed)
    # Shuffle chunk ids before slicing into topics, so a topic is NOT a
    # contiguous id range. This breaks the cheap "consistent-hash on first chunk
    # id" trick (which only works when topic boundaries happen to align with id
    # mod n_backends) and forces a router to actually look at the multi-chunk
    # set to partition correctly. Stable across runs (seeded).
    all_chunks = list(range(n_chunks_pool))
    rng.shuffle(all_chunks)
    topic_size = max(1, n_chunks_pool // n_topics)
    topics = [all_chunks[t * topic_size:(t + 1) * topic_size]
              for t in range(n_topics)]
    topic_weights = [[1.0 / ((i + 1) ** skew_alpha) for i in range(len(topic))]
                     for topic in topics]
    out = []
    for _ in range(n_requests):
        t = rng.randrange(n_topics)
        pool = topics[t]
        weights = topic_weights[t]
        chosen: set[int] = set()
        while len(chosen) < min(k_per_request, len(pool)):
            chosen.add(rng.choices(pool, weights=weights, k=1)[0])
        out.append(sorted(chosen))
    return out


def evaluate(strategy: str, n_chunks_pool: int, k_per_request: int,
             skew_alpha: float, n_topics: int, n_requests: int) -> dict:
    cfg = RagConfig()
    cfg.cache_capacity_chunks = CACHE_CAP_CHUNKS    # constrain so eviction binds
    # Realistic chunk size: retrieved RAG docs are typically 400-800 tokens.
    cfg.tokens_per_chunk = 512
    # Well-pipelined queue: one more inflight request adds ~50ms of waiting,
    # not 200ms. The 200ms default treats inflight as fully serial -- only true
    # for a single-stream backend; production vLLM batches.
    cfg.service_ms_per_request = 50.0
    index = ChunkAffinityIndex(cfg.cache_capacity_chunks)
    load = LoadTracker()
    workload = make_workload(n_chunks_pool, k_per_request, skew_alpha,
                             n_topics, n_requests)

    total_chunks = 0
    hit_chunks = 0
    ms_samples: list[float] = []
    rr_counter = 0
    # Sliding inflight window: each request increments its backend's inflight on
    # dispatch; the request "completes" SERVICE_WINDOW iterations later. This
    # gives the affinity router a load-balance signal that matches what it sees
    # in production (gateway.load_tracker.LoadTracker).
    inflight_window: deque = deque()

    for idx, chunks in enumerate(workload):
        # Drain completed requests (those dispatched >= SERVICE_WINDOW iters ago).
        while inflight_window and inflight_window[0][0] <= idx - SERVICE_WINDOW:
            _, b_done = inflight_window.popleft()
            load.inflight[b_done] = max(0, load.inflight[b_done] - 1)

        total_chunks += len(chunks)

        if strategy == "round_robin":
            backend = BACKENDS[rr_counter % len(BACKENDS)]
            rr_counter += 1
        elif strategy == "consistent_hash":
            key = chunks[0] if chunks else 0
            backend = BACKENDS[key % len(BACKENDS)]
        elif strategy == "chunk_affinity":
            result = choose_rag_backend(chunk_ids=chunks, backends=BACKENDS,
                                        index=index, load=load, cfg=cfg)
            backend = result.backend_id
        else:
            raise ValueError(strategy)

        overlap = index.overlap(backend, chunks)
        hit_chunks += overlap
        uncached = len(chunks) - overlap
        est_ms = cfg.prefill_ms_per_token * uncached * cfg.tokens_per_chunk
        ms_samples.append(est_ms)

        # Update affinity index AFTER the routing decision (the live wiring also
        # records on response completion -- same monotonic LRU semantics).
        index.record(backend, chunks)
        # Reflect dispatch in shared load state, like the live server.
        load.inflight[backend] = load.inflight.get(backend, 0) + 1
        inflight_window.append((idx, backend))

    return {
        "strategy": strategy,
        "n_chunks_pool": n_chunks_pool,
        "k_per_request": k_per_request,
        "skew_alpha": skew_alpha,
        "n_topics": n_topics,
        "n_requests": n_requests,
        "hit_rate": hit_chunks / total_chunks if total_chunks else 0.0,
        "est_ms_mean": statistics.mean(ms_samples) if ms_samples else 0.0,
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
    fields = ["sweep", "value", "strategy", "n_chunks_pool", "k_per_request",
              "skew_alpha", "n_topics", "n_requests", "hit_rate", "est_ms_mean"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in baseline:
            w.writerow({"sweep": "baseline", "value": "", **row})
        for rows in matrix.values():
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(baseline: list[dict], matrix: dict[str, list[dict]], path: str) -> None:
    rr = next(r for r in baseline if r["strategy"] == "round_robin")
    ch = next(r for r in baseline if r["strategy"] == "consistent_hash")
    af = next(r for r in baseline if r["strategy"] == "chunk_affinity")
    lines = [
        "# RAG-Routing Benchmark: Chunk-Affinity vs Cache-Blind",
        "",
        "RAG-style workload: a corpus of `n_chunks_pool` chunks; each request",
        "retrieves `k_per_request` of them via a Zipf-weighted popularity",
        "distribution (the canonical RAG access pattern where a few chunks",
        "dominate traffic). The chunk-affinity router routes to the backend",
        "already holding the most of the request's chunks; round-robin and",
        "consistent-hash ignore chunk state.",
        "",
        f"Fleet: {len(BACKENDS)} backends, per-backend chunk-cache capacity "
        f"{CACHE_CAP_CHUNKS} (constrained so eviction binds -- the realistic",
        "case where the corpus exceeds per-GPU KV).",
        "",
        "## Baseline ("
        f"pool={BASE['n_chunks_pool']}, k={BASE['k_per_request']}, "
        f"alpha={BASE['skew_alpha']}, topics={BASE['n_topics']}, "
        f"n_requests={BASE['n_requests']})",
        "",
        "| strategy | chunk hit rate | mean est-ms |",
        "|---|---|---|",
    ]
    for row in baseline:
        lines.append(f"| {row['strategy']} | "
                     f"{row['hit_rate']*100:.1f}% | "
                     f"{row['est_ms_mean']:.2f} |")
    lift = af["hit_rate"] / rr["hit_rate"] if rr["hit_rate"] > 0 else float("inf")
    ms_red = rr["est_ms_mean"] / af["est_ms_mean"] if af["est_ms_mean"] > 0 else float("inf")
    lines += [
        "",
        f"**Chunk-affinity wins**: {af['hit_rate']*100:.1f}% hit rate vs "
        f"{rr['hit_rate']*100:.1f}% (RR) and {ch['hit_rate']*100:.1f}% "
        f"(consistent-hash) -- a {lift:.1f}x lift, and a "
        f"{ms_red:.1f}x reduction in mean per-request prefill cost.",
        "",
        "## Sweeps",
        "",
    ]
    for knob, rows in matrix.items():
        lines.append(f"### sweep: {knob} ({SWEEP_LABEL[knob]})")
        lines.append("")
        lines.append("| " + knob + " | "
                     + " | ".join(f"{s} hit rate" for s in STRATEGIES) + " |")
        lines.append("|" + "---|" * (1 + len(STRATEGIES)))
        values = sorted({r["value"] for r in rows})
        for v in values:
            cells = [str(v)]
            for s in STRATEGIES:
                r = next(x for x in rows if x["value"] == v and x["strategy"] == s)
                cells.append(f"{r['hit_rate']*100:.1f}%")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Caveat",
        "",
        "Hit rate is measured against the same `ChunkAffinityIndex` the live",
        "router uses (LRU set per backend) -- the backend's *own* truth, not a",
        "gateway belief. est-ms is the prefill proxy (uncached tokens x",
        "ms/token), not wall-clock. Absolute ms validation is the GPU job; same",
        "caveat as `bench/matrix.py` and `bench/dag_bench.py`.",
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

    # Baseline bar chart.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    strategies = [r["strategy"] for r in baseline]
    hits = [r["hit_rate"] * 100 for r in baseline]
    mss = [r["est_ms_mean"] for r in baseline]
    colors = ["#888", "#aaa", "#1f77b4"]
    ax1.bar(strategies, hits, color=colors)
    ax1.set_ylabel("chunk-cache hit rate (%)")
    ax1.set_title("RAG hit rate (higher = better)")
    ax1.set_ylim(0, 100)
    for i, v in enumerate(hits):
        ax1.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    ax2.bar(strategies, mss, color=colors)
    ax2.set_ylabel("mean est-ms / request")
    ax2.set_title("RAG prefill cost (lower = better)")
    for i, v in enumerate(mss):
        ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.suptitle(
        f"RAG chunk-affinity routing "
        f"(pool={BASE['n_chunks_pool']}, k={BASE['k_per_request']}, "
        f"alpha={BASE['skew_alpha']}, {BASE['n_requests']} requests)")
    fig.tight_layout()
    out = os.path.join(out_dir, "rag_baseline.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    for knob, rows in matrix.items():
        values = sorted({r["value"] for r in rows})
        fig, ax = plt.subplots(figsize=(7, 4))
        for s, c in zip(STRATEGIES, colors):
            ys = [next(r for r in rows if r["value"] == v and r["strategy"] == s)
                  ["hit_rate"] * 100 for v in values]
            ax.plot(values, ys, marker="o", label=s, color=c)
        ax.set_xlabel(SWEEP_LABEL[knob])
        ax.set_ylabel("chunk-cache hit rate (%)")
        ax.set_title(f"RAG hit rate vs {knob}")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, f"rag_sweep_{knob}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


def print_summary(baseline: list[dict]) -> None:
    print(f"backends={len(BACKENDS)}  cache_cap={CACHE_CAP_CHUNKS}  "
          f"baseline pool={BASE['n_chunks_pool']} k={BASE['k_per_request']} "
          f"alpha={BASE['skew_alpha']} topics={BASE['n_topics']} "
          f"n={BASE['n_requests']}")
    print(f"{'strategy':<18} {'hit rate':>10} {'est-ms mean':>14}")
    for row in baseline:
        print(f"{row['strategy']:<18} "
              f"{row['hit_rate']*100:>9.1f}% "
              f"{row['est_ms_mean']:>12.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=None,
                    help="override n_requests at every sweep point (faster smoke run)")
    ap.add_argument("--no-charts", action="store_true",
                    help="skip matplotlib (CSV + markdown only)")
    args = ap.parse_args()

    if args.requests:
        BASE["n_requests"] = args.requests

    os.makedirs(OUT_DIR, exist_ok=True)
    baseline = run_baseline()
    matrix = run_matrix()
    csv_path = os.path.join(OUT_DIR, "rag_matrix.csv")
    md_path = os.path.join(OUT_DIR, "RAG_RESULTS.md")
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
