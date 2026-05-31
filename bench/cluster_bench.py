from __future__ import annotations

"""Cluster-scaling benchmark: why shared prefix state matters as you add replicas.

A single gateway holds its radix tree in-process. Put N replicas behind a
round-robin load balancer and, WITHOUT shared state, each replica only knows the
prefixes IT has handled (~1/N of traffic). So:
  * each document is cold-missed once PER REPLICA (≈ N cold misses, not 1), and
  * replicas independently pick different backends for the same doc -> the doc
    is duplicated across backends -> cache fragmentation -> more eviction.

WITH shared state (the gateway/cluster replication bus), every replica converges
on the same prefix→backend map, so a doc is cold-missed once fleet-wide and
pinned to one backend. This benchmark sweeps replica count and reports, shared
vs unshared:
  * fleet block-hit rate (measured against independent per-backend KV models)
  * cache duplication factor (avg distinct backends a doc lands on; 1.0 = ideal)

Algorithm-level, no network/GPU -- same caveat as bench/matrix.py. The point is
routing quality vs replica count, not absolute latency.

Usage:
    python bench/cluster_bench.py                 # CSV + markdown + chart
    python bench/cluster_bench.py --no-charts
"""

import argparse
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # for `matrix`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix import BACKENDS, CAP_BLOCKS, SimBackend, _cfg, make_workload  # noqa: E402

from gateway.load_tracker import LoadTracker      # noqa: E402
from gateway.radix_tree import RadixTree          # noqa: E402
from gateway.router import Router                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

REPLICAS = [1, 2, 3, 4, 8, 16]
WINDOW_PER_REPLICA = 8                # in-flight depth modeled per replica
WORKLOAD = {"n_docs": 20, "doc_blocks": 96, "system_blocks": 2, "skew_alpha": 1.0}


def evaluate(n_replicas: int, shared: bool, requests: list[str], cfg) -> dict:
    trees = [RadixTree(CAP_BLOCKS) for _ in range(n_replicas)]
    loads = [LoadTracker() for _ in range(n_replicas)]
    routers = [Router(cfg, trees[i], loads[i]) for i in range(n_replicas)]
    windows = [deque() for _ in range(n_replicas)]
    sim = {b: SimBackend(CAP_BLOCKS) for b in BACKENDS}   # shared backend fleet (ground truth)
    doc_backends: dict[int, set] = defaultdict(set)
    total_blocks = hit_blocks = 0
    dist: Counter = Counter()

    for k, prompt in enumerate(requests):
        i = k % n_replicas                              # round-robin LB
        r = routers[i].choose(prompt, BACKENDS, strategy="prefix_tree")
        hit_blocks += sim[r.backend_id].process(r.hashes)   # backend's own truth
        total_blocks += len(r.hashes)
        dist[r.backend_id] += 1

        # commit on the handling replica
        trees[i].insert(r.hashes, r.backend_id)
        loads[i].on_dispatch(r.backend_id, r.tokens)
        windows[i].append((r.backend_id, r.tokens))
        if len(windows[i]) > WINDOW_PER_REPLICA:
            ob, ot = windows[i].popleft()
            loads[i].on_complete(ob, ot)

        # shared state: the cluster bus converges every replica's tree
        if shared:
            for j in range(n_replicas):
                if j != i:
                    trees[j].insert(r.hashes, r.backend_id)

        if r.hashes:                                    # last full block ~ doc identity
            doc_backends[r.hashes[-1]].add(r.backend_id)

    dup = statistics.mean(len(s) for s in doc_backends.values()) if doc_backends else 0.0
    counts = [dist.get(b, 0) for b in BACKENDS]
    mean_c = sum(counts) / len(counts)
    cov = (statistics.pstdev(counts) / mean_c) if mean_c else 0.0
    return {"hit_rate": hit_blocks / total_blocks if total_blocks else 0.0,
            "dup": dup, "cov": cov}


def run(n_requests: int) -> list[dict]:
    cfg = _cfg()
    requests = make_workload(n_requests, WORKLOAD["n_docs"], WORKLOAD["doc_blocks"],
                             WORKLOAD["system_blocks"], WORKLOAD["skew_alpha"], seed=1)
    rows = []
    for n in REPLICAS:
        for shared in (False, True):
            m = evaluate(n, shared, requests, cfg)
            rows.append({"replicas": n, "shared": shared, **m})
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["replicas", "shared", "hit_rate", "dup", "cov"])
        w.writeheader()
        for r in rows:
            w.writerow({"replicas": r["replicas"], "shared": r["shared"],
                        "hit_rate": f"{r['hit_rate']:.4f}", "dup": f"{r['dup']:.3f}",
                        "cov": f"{r['cov']:.4f}"})


def write_markdown(rows: list[dict], charts: bool, path: str) -> None:
    def get(n, sh):
        return next(r for r in rows if r["replicas"] == n and r["shared"] is sh)
    lines = [
        "# Cluster-Scaling Benchmark: shared vs unshared prefix state",
        "",
        f"Fleet: {len(BACKENDS)} backends x {CAP_BLOCKS} blocks. Workload: "
        f"{WORKLOAD['n_docs']} docs x {WORKLOAD['doc_blocks']} blocks, Zipf "
        f"alpha={WORKLOAD['skew_alpha']}, round-robin load balancer across N replicas.",
        "",
        "Each replica keeps its own radix tree. **unshared** = replicas route from "
        "only their own history; **shared** = the cluster bus converges all trees "
        "(what `GW_CLUSTER_ENABLED` does). Hit rate is measured against independent "
        "per-backend KV models. **dup** = avg distinct backends a doc lands on; it's "
        ">1 even when shared because the router intentionally replicates *hot* docs "
        "across backends for load balance -- so read the unshared-minus-shared gap "
        "as the *uncoordinated* fragmentation that shared state removes, not dup itself.",
        "",
    ]
    if charts:
        lines += ["![cluster scaling](cluster_scaling.png)", ""]
    lines += ["| replicas | unshared hit% | shared hit% | unshared dup | shared dup |",
              "|---|---|---|---|---|"]
    for n in REPLICAS:
        u, s = get(n, False), get(n, True)
        lines.append(f"| {n} | {u['hit_rate']*100:.1f}% | {s['hit_rate']*100:.1f}% | "
                     f"{u['dup']:.2f} | {s['dup']:.2f} |")
    u1 = get(REPLICAS[-1], False); s1 = get(REPLICAS[-1], True)
    lines += [
        "",
        f"**Headline**: shared prefix state holds **~{s1['hit_rate']*100:.0f}%** hit rate "
        f"at every replica count, while unshared collapses from {get(1, False)['hit_rate']*100:.0f}% "
        f"(1 replica) to **{u1['hit_rate']*100:.0f}%** at {REPLICAS[-1]} replicas -- each "
        "replica only learns ~1/N of the prefix map, so most requests land on a replica "
        "that's never seen the doc and routes blind. At 1 replica the two are identical "
        "(nothing to share); the widening gap is exactly the value of replicated prefix "
        f"state (dup also rises {s1['dup']:.1f}→{u1['dup']:.1f} from uncoordinated fragmentation).",
        "",
        "## Caveat",
        "",
        "Algorithm-level: hit rate is the backend KV model's own truth, not the "
        "gateway's belief; no network/GPU. It measures how routing quality scales "
        "with replica count, not absolute latency. Same caveat as `bench/matrix.py`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_chart(rows: list[dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(sh, key):
        return [next(r for r in rows if r["replicas"] == n and r["shared"] is sh)[key]
                for n in REPLICAS]

    xs = [str(n) for n in REPLICAS]
    fig, (ax_hit, ax_dup) = plt.subplots(1, 2, figsize=(11, 4))
    ax_hit.plot(xs, [v * 100 for v in series(True, "hit_rate")], marker="o",
                label="shared (cluster)", color="#1f77b4", linewidth=2.5)
    ax_hit.plot(xs, [v * 100 for v in series(False, "hit_rate")], marker="o",
                label="unshared", color="#d62728")
    ax_hit.set(title="Fleet prefix-cache hit rate", xlabel="gateway replicas",
               ylabel="hit rate (%)")
    ax_hit.legend(); ax_hit.grid(True, alpha=0.3)
    ax_dup.plot(xs, series(True, "dup"), marker="o", label="shared", color="#1f77b4",
                linewidth=2.5)
    ax_dup.plot(xs, series(False, "dup"), marker="o", label="unshared", color="#d62728")
    ax_dup.set(title="Cache duplication (backends per doc)", xlabel="gateway replicas",
               ylabel="distinct backends / doc (1 = ideal)")
    ax_dup.legend(); ax_dup.grid(True, alpha=0.3)
    fig.suptitle("Shared prefix state vs replica count")
    fig.tight_layout()
    out = os.path.join(out_dir, "cluster_scaling.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def print_summary(rows: list[dict]) -> None:
    print(f"backends={len(BACKENDS)} cap={CAP_BLOCKS}  "
          f"workload={WORKLOAD['n_docs']} docs alpha={WORKLOAD['skew_alpha']}")
    print(f"{'replicas':>8} {'unshared hit':>14} {'shared hit':>12} "
          f"{'unshared dup':>14} {'shared dup':>11}")
    for n in REPLICAS:
        u = next(r for r in rows if r["replicas"] == n and r["shared"] is False)
        s = next(r for r in rows if r["replicas"] == n and r["shared"] is True)
        print(f"{n:>8} {u['hit_rate']*100:>13.1f}% {s['hit_rate']*100:>11.1f}% "
              f"{u['dup']:>14.2f} {s['dup']:>11.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = run(args.requests)
    print_summary(rows)
    print()
    csv_path = os.path.join(OUT_DIR, "cluster_scaling.csv")
    md_path = os.path.join(OUT_DIR, "CLUSTER_RESULTS.md")
    write_csv(rows, csv_path)
    charts_ok = False
    if not args.no_charts:
        try:
            render_chart(rows, OUT_DIR)
            charts_ok = True
        except Exception as e:
            print(f"[charts skipped: {e}]")
    write_markdown(rows, charts_ok, md_path)
    print(f"wrote {os.path.relpath(csv_path, _ROOT)}, {os.path.relpath(md_path, _ROOT)}")


if __name__ == "__main__":
    main()
