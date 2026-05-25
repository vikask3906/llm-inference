from __future__ import annotations

"""Algorithm-level benchmark (no network, stdlib only).

Replays one identical request stream through all three routing strategies and
measures the REAL prefix-cache hit rate via independent per-backend cache models
(SimBackend), so the measurement does not depend on the gateway's own beliefs.

Workload: every request = shared system prompt + one of D documents + a unique
question. The shared system prompt means all requests share their FIRST block --
which is exactly what makes naive consistent-hashing-on-first-block collapse the
whole fleet onto one backend.
"""

import os
import random
import sys
from collections import Counter, OrderedDict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.config import Config            # noqa: E402
from gateway.load_tracker import LoadTracker  # noqa: E402
from gateway.radix_tree import RadixTree      # noqa: E402
from gateway.router import Router             # noqa: E402

BACKENDS = ["b0", "b1", "b2"]
CAP_BLOCKS = 600            # per-backend KV capacity (blocks). Fleet = 1800.
N_DOCS = 15                 # 15 docs * 96 blocks = 1440 blocks > one backend cap (600)
DOC_CHARS = 6144           # 96 blocks per document
N_REQUESTS = 3000
CONCURRENCY = 48            # sliding window modelling in-flight requests


class SimBackend:
    """A backend's real KV prefix cache: contiguous-prefix LRU over blocks."""

    def __init__(self, cap_blocks: int) -> None:
        self.cap = cap_blocks
        self.cache: "OrderedDict[int, bool]" = OrderedDict()

    def process(self, hashes: list[int]) -> int:
        hit = 0
        for h in hashes:                       # longest contiguous cached prefix
            if h in self.cache:
                hit += 1
            else:
                break
        for h in hashes:                       # cache the whole prompt (LRU + evict)
            if h in self.cache:
                self.cache.move_to_end(h)
            else:
                self.cache[h] = True
                if len(self.cache) > self.cap:
                    self.cache.popitem(last=False)
        return hit


def make_workload(n: int, n_docs: int, seed: int, skew: bool) -> list[str]:
    rng = random.Random(seed)
    system = "S" * 128                          # 2 blocks, identical for all requests
    docs = []
    for i in range(n_docs):
        base = f"doc{i:04d}-"
        docs.append((base * (DOC_CHARS // len(base) + 1))[:DOC_CHARS])   # distinct blocks
    if skew:
        weights = [1.0 / (r + 1) for r in range(n_docs)]       # Zipf-ish hot docs
    else:
        weights = [1.0] * n_docs
    reqs = []
    for k in range(n):
        di = rng.choices(range(n_docs), weights=weights, k=1)[0]
        q = f"q{k:06d}-unique-question"          # < 64 chars -> partial block, dropped
        reqs.append(system + docs[di] + q)
    return reqs


def run(strategy: str, requests: list[str], cfg: Config):
    tree = RadixTree(CAP_BLOCKS)
    load = LoadTracker()
    router = Router(cfg, tree, load)
    sim = {b: SimBackend(CAP_BLOCKS) for b in BACKENDS}
    window: deque = deque()
    total_blocks = hit_blocks = 0
    dist: Counter = Counter()

    for prompt in requests:
        r = router.choose(prompt, BACKENDS, strategy=strategy)
        hit_blocks += sim[r.backend_id].process(r.hashes)
        tree.insert(r.hashes, r.backend_id)     # gateway updates its belief
        load.on_dispatch(r.backend_id, r.tokens)
        window.append((r.backend_id, r.tokens))
        if len(window) > CONCURRENCY:
            ob, ot = window.popleft()
            load.on_complete(ob, ot)
        total_blocks += len(r.hashes)
        dist[r.backend_id] += 1

    return hit_blocks / total_blocks, dist


def report(title: str, requests: list[str], cfg: Config) -> None:
    print(f"\n=== {title} ===")
    print(f"{'strategy':<16}{'block hit rate':<18}{'load distribution (b0/b1/b2)'}")
    for strat in ("round_robin", "consistent_hash", "prefix_tree"):
        hit, dist = run(strat, requests, cfg)
        d = "/".join(str(dist.get(b, 0)) for b in BACKENDS)
        print(f"{strat:<16}{hit*100:>6.1f}%           {d}")


def main() -> None:
    cfg = Config()
    cfg.backend_cache_blocks = CAP_BLOCKS
    # Calibration: prefill saving must dominate the soft queue tiebreaker so cached
    # prefixes stay pinned; the HARD saturation cutoff (max_inflight) provides
    # balance / hot-prefix spill instead of the soft term swamping affinity.
    cfg.service_ms_per_request = 1.0    # gentle load tiebreaker
    cfg.max_inflight = 24               # hard cutoff -> excludes overloaded nodes
    cfg.hysteresis_ms = 2.0
    print(f"backends={len(BACKENDS)} cap={CAP_BLOCKS} blocks  docs={N_DOCS}  "
          f"requests={N_REQUESTS}  concurrency={CONCURRENCY}")
    report("Uniform document popularity", make_workload(N_REQUESTS, N_DOCS, 1, skew=False), cfg)
    report("Skewed (hot documents)", make_workload(N_REQUESTS, N_DOCS, 2, skew=True), cfg)


if __name__ == "__main__":
    main()
