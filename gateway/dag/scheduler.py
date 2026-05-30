from __future__ import annotations

"""Cache-locality-aware scheduling of a request DAG across a backend fleet.

The live router (gateway/router.py) makes a locally optimal, per-request
longest-prefix-match decision. This scheduler lifts that idea to a whole
workflow: it sees the entire DAG up front, so it can deliberately co-locate
cache-sharing nodes (siblings that share a system prompt + retrieved context, a
child that continues its parent's prefix) on the same backend, instead of
reacting one request at a time.

Method: greedy list-scheduling over topological layers. Each layer is a
dependency barrier (all parents finished); within a layer, independent nodes
run concurrently across backends. For each node we pick the eligible backend
minimizing projected finish time, where a node's work is its *uncached* prefill
cost -- so a longest-prefix cache hit (scored by the same RadixTree the live
router uses) shrinks its cost to near zero. A backend that already ran a parent
gets a small affinity bonus (the family's working set stays resident).

Two strategies, for an apples-to-apples contrast:
  * "locality"    -- the affinity-aware scheduler described above.
  * "round_robin" -- cache-blind cyclic assignment (the baseline to beat); it
                     still populates the cache model so we can measure the
                     hit-rate it leaves on the table.

Standalone: imported by nothing on the hot path, like gateway/disagg and
gateway/rag, so benchmark numbers are untouched.
"""

import dataclasses
from typing import Optional

from ..hashing import block_hashes
from ..radix_tree import RadixTree
from .config import DagSchedConfig
from .graph import RequestDag


@dataclasses.dataclass
class NodePlacement:
    node_id: str
    backend_id: str
    layer: int
    match_blocks: int          # prefix blocks already cached on the chosen backend
    total_blocks: int
    est_ms: float              # this node's marginal (uncached) prefill cost


@dataclasses.dataclass
class Schedule:
    placements: list[NodePlacement]
    est_makespan_ms: float
    cache_hit_blocks: int
    total_blocks: int
    strategy: str

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hit_blocks / self.total_blocks if self.total_blocks else 0.0


def _initial_busy(load, backend_id: str, cfg: DagSchedConfig) -> float:
    """Pre-existing queue (ms) from non-DAG traffic already on this backend."""
    if load is None:
        return 0.0
    try:
        inflight = load.inflight[backend_id]
    except (KeyError, TypeError):
        return 0.0
    return inflight * cfg.service_ms_per_request


def _saturated(load, backend_id: str, cfg: DagSchedConfig) -> bool:
    if load is None:
        return False
    try:
        return load.inflight[backend_id] > cfg.max_inflight
    except (KeyError, TypeError):
        return False


def schedule(*, dag: RequestDag, backends: list[str], cfg: DagSchedConfig,
             load=None, strategy: str = "locality") -> Optional[Schedule]:
    """Plan an assignment of every DAG node to a backend.

    Returns None if there are no backends. Raises DagError (via validate) on a
    malformed graph.
    """
    if not backends:
        return None
    dag.validate()

    tree = RadixTree(cfg.backend_cache_blocks)
    hashes_of: dict[str, list[int]] = {
        nid: block_hashes(node.prompt, cfg.block_chars, cfg.hash_cutoff_blocks)
        for nid, node in dag.nodes.items()
    }

    placements: list[NodePlacement] = []
    node_backend: dict[str, str] = {}
    hit_blocks = 0
    total_blocks = 0
    makespan = 0.0
    rr = 0

    for layer_idx, layer in enumerate(dag.layers()):
        # Each backend starts the layer with whatever non-DAG queue it carries;
        # nodes placed earlier in THIS layer push later ones out (serial on a
        # backend, parallel across backends).
        busy = {b: _initial_busy(load, b, cfg) for b in backends}

        for nid in layer:
            node = dag.nodes[nid]
            hashes = hashes_of[nid]
            nblocks = len(hashes)
            total_blocks += nblocks
            match = tree.match(hashes)

            eligible = [b for b in backends if not _saturated(load, b, cfg)] or list(backends)

            if strategy == "round_robin":
                chosen = eligible[rr % len(eligible)]
                rr += 1
                m_blocks = match.get(chosen, 0)
                work = cfg.prefill_ms_per_token * max(0, nblocks - m_blocks) * cfg.block_tokens
            else:
                parent_backends = {node_backend[p] for p in node.parents if p in node_backend}
                chosen = None
                chosen_finish = float("inf")
                chosen_m = 0
                chosen_work = 0.0
                for b in eligible:
                    m_blocks = match.get(b, 0)
                    work = cfg.prefill_ms_per_token * max(0, nblocks - m_blocks) * cfg.block_tokens
                    bonus = cfg.parent_affinity_bonus_ms if b in parent_backends else 0.0
                    finish = busy[b] + work - bonus
                    if finish < chosen_finish:
                        chosen_finish, chosen, chosen_m, chosen_work = finish, b, m_blocks, work
                m_blocks, work = chosen_m, chosen_work

            busy[chosen] += work
            hit_blocks += m_blocks
            tree.insert(hashes, chosen)
            node_backend[nid] = chosen
            placements.append(NodePlacement(
                node_id=nid, backend_id=chosen, layer=layer_idx,
                match_blocks=m_blocks, total_blocks=nblocks, est_ms=work))

        # Layer is a barrier: it finishes when its busiest backend finishes.
        makespan += max(busy.values()) if busy else 0.0

    return Schedule(placements=placements, est_makespan_ms=makespan,
                    cache_hit_blocks=hit_blocks, total_blocks=total_blocks,
                    strategy=strategy)
