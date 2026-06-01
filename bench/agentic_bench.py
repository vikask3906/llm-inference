from __future__ import annotations

"""Agentic benchmark: session-affinity vs round-robin on multi-turn agent loops.

Model: M parallel agent sessions, each runs T turns. Every turn extends the
session's prompt by ~`turn_blocks` new blocks (tool call + result), so by turn T
the prompt is a long shared prefix of T*turn_blocks blocks. Round-robin scatters
each turn across backends -- so the FULL accumulated context is re-prefilled on
a different node every time. Session-affinity pins every turn of one session to
the backend that served turn 0 -- so each new turn only prefills its small
delta and reuses ~all prior blocks.

Reports per-strategy block hit rate + the prefill the gateway saved, swept over
turn depth. Same algorithm-level framing as bench/matrix.py.

Usage:
    python bench/agentic_bench.py
    python bench/agentic_bench.py --no-charts
"""

import argparse
import csv
import os
import statistics
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # for `matrix`

from matrix import BACKENDS, CAP_BLOCKS, SimBackend                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "docs", "benchmarks")

SYSTEM_BLOCKS = 2          # small shared system prefix
TURN_BLOCKS = 8            # blocks added per agent turn (tool call + result)
N_SESSIONS = 24            # parallel agent sessions
TURN_DEPTHS = [1, 2, 3, 5, 8, 13]   # how many turns each session runs
PREFILL_MS_PER_BLOCK = 0.05 * 16    # ms/block (~vLLM 0.05 ms/token x 16 tok/block)


def _hashes_for_turn(session_id: int, turn: int) -> list[int]:
    """Block hashes for a session's prompt at turn `turn`. Each turn extends the
    session's prefix by `turn_blocks` new (deterministic) blocks. Session ids
    are namespaced so different sessions diverge on their first session-specific
    block; system blocks are shared across all sessions."""
    out = [10_000 + b for b in range(SYSTEM_BLOCKS)]                          # shared system
    for t in range(turn + 1):                                                  # turns 0..turn
        # session-namespaced blocks; mixing session_id keeps blocks distinct
        out.extend(1_000_000 + session_id * 10_000 + t * TURN_BLOCKS + b
                   for b in range(TURN_BLOCKS))
    return out


def run_strategy(strategy: str, n_sessions: int, depth: int) -> dict:
    """Simulate `n_sessions` agents each running `depth+1` turns under strategy.

      session_affinity: pin every turn to the backend the session was first
                        routed to (round-robin'd at turn 0).
      round_robin:      every turn round-robins across backends.
    """
    sims = {b: SimBackend(CAP_BLOCKS) for b in BACKENDS}
    pinned: dict[int, str] = {}
    total = hit = 0
    rr = 0
    for s in range(n_sessions):
        for t in range(depth + 1):
            hashes = _hashes_for_turn(s, t)
            if strategy == "session_affinity":
                if s not in pinned:
                    pinned[s] = BACKENDS[rr % len(BACKENDS)]; rr += 1
                b = pinned[s]
            else:                                                              # round_robin
                b = BACKENDS[rr % len(BACKENDS)]; rr += 1
            hit += sims[b].process(hashes)
            total += len(hashes)
    return {"hit_rate": hit / total if total else 0.0,
            "uncached_blocks": total - hit}


def run(depths: list[int], n_sessions: int) -> list[dict]:
    rows = []
    for d in depths:
        sa = run_strategy("session_affinity", n_sessions, d)
        rr = run_strategy("round_robin", n_sessions, d)
        # estimated prefill saved per turn (ms) -- the cost a hit removes.
        saved_blocks = rr["uncached_blocks"] - sa["uncached_blocks"]
        total_turns = n_sessions * (d + 1)
        saved_ms_per_turn = (saved_blocks * PREFILL_MS_PER_BLOCK) / total_turns
        rows.append({"turns": d + 1, "depth": d,
                     "sa_hit_rate": sa["hit_rate"], "rr_hit_rate": rr["hit_rate"],
                     "saved_ms_per_turn": saved_ms_per_turn})
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["turns", "sa_hit_rate", "rr_hit_rate",
                                          "saved_ms_per_turn"])
        w.writeheader()
        for r in rows:
            w.writerow({"turns": r["turns"],
                        "sa_hit_rate": f"{r['sa_hit_rate']:.4f}",
                        "rr_hit_rate": f"{r['rr_hit_rate']:.4f}",
                        "saved_ms_per_turn": f"{r['saved_ms_per_turn']:.2f}"})


def write_markdown(rows: list[dict], charts: bool, path: str) -> None:
    lines = [
        "# Agentic Benchmark: session-affinity vs round-robin on multi-turn loops",
        "",
        f"Workload: {N_SESSIONS} parallel agent sessions, each running T turns; "
        f"every turn extends the session's prompt by {TURN_BLOCKS} blocks "
        f"(tool call + result) on top of a {SYSTEM_BLOCKS}-block shared system "
        f"prefix. Hit rate is measured against independent per-backend KV models "
        f"(the backend's own truth). Fleet: {len(BACKENDS)} backends.",
        "",
    ]
    if charts:
        lines += ["![agentic](agentic_scaling.png)", ""]
    lines += ["| turns / session | session-affinity hit% | round-robin hit% | prefill saved per turn |",
              "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['turns']} | {r['sa_hit_rate']*100:.1f}% | "
                     f"{r['rr_hit_rate']*100:.1f}% | "
                     f"{r['saved_ms_per_turn']:.1f} ms |")
    last = rows[-1]
    lines += [
        "",
        f"**Headline at {last['turns']} turns**: session-affinity keeps "
        f"**{last['sa_hit_rate']*100:.0f}%** of blocks cached (each turn only "
        "prefills its small delta on top of the previous turn's warm KV); "
        f"round-robin scatters each turn across backends, dragging hit rate to "
        f"**{last['rr_hit_rate']*100:.0f}%** -- every turn re-prefills the full "
        f"growing context on whichever node the LB happened to pick. The win "
        f"COMPOUNDS with turn count -- the natural shape of an agent loop.",
        "",
        "## Caveat",
        "",
        "Algorithm-level. Hit rate is the backend KV model's own truth, not the "
        "gateway's belief; no network/GPU. It measures the cross-turn KV reuse "
        "session-affinity unlocks, which is exactly what frontier 'agentic "
        "inference' workloads (the SGLang/DeepLearning.AI slide) call out as the "
        "hard part: 'KV cache management across turns'. Same GPU-independent "
        "framing as the other benches.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_chart(rows: list[dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [str(r["turns"]) for r in rows]
    fig, (ax_hit, ax_ms) = plt.subplots(1, 2, figsize=(11, 4))
    ax_hit.plot(xs, [r["sa_hit_rate"] * 100 for r in rows], marker="o",
                label="session-affinity", color="#1f77b4", linewidth=2.5)
    ax_hit.plot(xs, [r["rr_hit_rate"] * 100 for r in rows], marker="o",
                label="round-robin", color="#d62728")
    ax_hit.set(title="Cross-turn prefix-cache hit rate", xlabel="turns / session",
               ylabel="hit rate (%)")
    ax_hit.legend(); ax_hit.grid(True, alpha=0.3)

    ax_ms.plot(xs, [r["saved_ms_per_turn"] for r in rows], marker="o", color="#1f77b4")
    ax_ms.set(title="Prefill saved per turn vs round-robin", xlabel="turns / session",
              ylabel="ms / turn")
    ax_ms.grid(True, alpha=0.3)
    fig.suptitle("Session-affinity for agentic / multi-turn workloads "
                 "(KV reuse compounds with turn count)")
    fig.tight_layout()
    out = os.path.join(out_dir, "agentic_scaling.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def print_summary(rows: list[dict]) -> None:
    print(f"sessions={N_SESSIONS}  turn_blocks={TURN_BLOCKS}  "
          f"system_blocks={SYSTEM_BLOCKS}  backends={len(BACKENDS)}")
    print(f"{'turns':>6}  {'session-affinity':>17}  {'round-robin':>13}  {'saved/turn':>10}")
    for r in rows:
        print(f"{r['turns']:>6}  {r['sa_hit_rate']*100:>16.1f}%  "
              f"{r['rr_hit_rate']*100:>12.1f}%  {r['saved_ms_per_turn']:>8.1f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = run(TURN_DEPTHS, N_SESSIONS)
    print_summary(rows)
    print()
    write_csv(rows, os.path.join(OUT_DIR, "agentic_scaling.csv"))
    charts_ok = False
    if not args.no_charts:
        try:
            render_chart(rows, OUT_DIR)
            charts_ok = True
        except Exception as e:
            print(f"[charts skipped: {e}]")
    write_markdown(rows, charts_ok, os.path.join(OUT_DIR, "AGENTIC_RESULTS.md"))
    rel = os.path.relpath(OUT_DIR, _ROOT)
    print(f"wrote {rel}/agentic_scaling.csv, {rel}/AGENTIC_RESULTS.md"
          + (", agentic_scaling.png" if charts_ok else ""))


if __name__ == "__main__":
    main()
