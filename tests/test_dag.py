"""Unit tests for the cache-locality-aware DAG scheduler (gateway/dag).

Two central properties:
  * the graph layers correctly and rejects malformed DAGs, and
  * the locality strategy keeps cache-sharing families co-located, beating a
    cache-blind round-robin on prefix-cache hit rate.
"""

import pytest

from gateway.dag import DagError, DagNode, RequestDag, schedule
from gateway.dag.config import DagSchedConfig
from gateway.load_tracker import LoadTracker

BLOCK = 64  # cfg.block_chars default


def _p(letter: str, blocks: int) -> str:
    """A prompt of `blocks` identical full blocks of `letter` (shared prefix)."""
    return letter * (BLOCK * blocks)


# --- graph: layering + validation ---

def test_layers_map_reduce():
    dag = RequestDag([
        DagNode("m0", "x"), DagNode("m1", "y"), DagNode("m2", "z"),
        DagNode("reduce", "combine", parents=("m0", "m1", "m2")),
    ])
    layers = dag.layers()
    assert layers == [["m0", "m1", "m2"], ["reduce"]]


def test_cycle_is_rejected():
    dag = RequestDag([
        DagNode("a", "x", parents=("b",)),
        DagNode("b", "y", parents=("a",)),
    ])
    with pytest.raises(DagError):
        dag.validate()


def test_missing_parent_is_rejected():
    dag = RequestDag([DagNode("a", "x", parents=("ghost",))])
    with pytest.raises(DagError):
        dag.validate()


def test_duplicate_node_id_is_rejected():
    dag = RequestDag([DagNode("a", "x")])
    with pytest.raises(DagError):
        dag.add(DagNode("a", "y"))


# --- scheduler: basics ---

def test_no_backends_returns_none():
    dag = RequestDag([DagNode("a", "x")])
    assert schedule(dag=dag, backends=[], cfg=DagSchedConfig()) is None


def test_every_node_placed_exactly_once():
    dag = RequestDag([
        DagNode("m0", _p("A", 6)), DagNode("m1", _p("B", 6)),
        DagNode("r", _p("C", 6), parents=("m0", "m1")),
    ])
    sched = schedule(dag=dag, backends=["b0", "b1"], cfg=DagSchedConfig())
    placed = {p.node_id for p in sched.placements}
    assert placed == {"m0", "m1", "r"}
    assert len(sched.placements) == 3


def test_disjoint_prompts_yield_no_cache_hits():
    dag = RequestDag([
        DagNode("a", _p("A", 6)), DagNode("b", _p("B", 6)), DagNode("c", _p("C", 6)),
    ])
    sched = schedule(dag=dag, backends=["b0", "b1"], cfg=DagSchedConfig())
    assert sched.cache_hit_blocks == 0
    assert sched.cache_hit_rate == 0.0


# --- the headline property: locality beats cache-blind round-robin ---

def _three_chains() -> RequestDag:
    # Three independent parent->child chains, each chain sharing its own prefix.
    return RequestDag([
        DagNode("a0", _p("A", 6)), DagNode("b0", _p("B", 6)), DagNode("c0", _p("C", 6)),
        DagNode("a1", _p("A", 6), parents=("a0",)),
        DagNode("b1", _p("B", 6), parents=("b0",)),
        DagNode("c1", _p("C", 6), parents=("c0",)),
    ])


def test_locality_beats_round_robin_hit_rate():
    cfg = DagSchedConfig()
    loc = schedule(dag=_three_chains(), backends=["b0", "b1"], cfg=cfg, strategy="locality")
    rr = schedule(dag=_three_chains(), backends=["b0", "b1"], cfg=cfg, strategy="round_robin")
    assert loc.cache_hit_rate > rr.cache_hit_rate
    assert loc.est_makespan_ms <= rr.est_makespan_ms


def test_parent_affinity_keeps_child_with_parent():
    # a1 reuses a0's whole prefix; the bonus + cache match should pin it to a0's
    # backend even though the other backend is idle.
    dag = RequestDag([
        DagNode("a0", _p("A", 6)),
        DagNode("a1", _p("A", 6), parents=("a0",)),
    ])
    sched = schedule(dag=dag, backends=["b0", "b1"], cfg=DagSchedConfig())
    where = {p.node_id: p.backend_id for p in sched.placements}
    assert where["a1"] == where["a0"]
    a1 = next(p for p in sched.placements if p.node_id == "a1")
    assert a1.match_blocks == 6        # full prefix reuse
    assert a1.est_ms == 0.0            # nothing uncached -> no prefill cost


def test_parallel_siblings_cut_makespan():
    # Two independent, distinct-prefix nodes over two backends run concurrently,
    # so makespan is one node's work, not the sum.
    dag = RequestDag([DagNode("a", _p("A", 6)), DagNode("b", _p("B", 6))])
    sched = schedule(dag=dag, backends=["b0", "b1"], cfg=DagSchedConfig())
    one_node_ms = sum(p.est_ms for p in sched.placements) / 2
    assert sched.est_makespan_ms == pytest.approx(one_node_ms)


def test_saturated_backend_is_excluded():
    cfg = DagSchedConfig()
    load = LoadTracker()
    load.inflight["b0"] = cfg.max_inflight + 1     # b0 saturated
    load.inflight["b1"] = 0
    dag = RequestDag([DagNode("a", _p("A", 6)), DagNode("c", _p("C", 6))])
    sched = schedule(dag=dag, backends=["b0", "b1"], cfg=cfg, load=load)
    assert all(p.backend_id == "b1" for p in sched.placements)
