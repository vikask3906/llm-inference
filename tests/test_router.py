from gateway.config import Config
from gateway.hashing import block_hashes
from gateway.load_tracker import LoadTracker
from gateway.radix_tree import RadixTree
from gateway.router import Router

CANDS = ["b0", "b1", "b2"]


def _make():
    cfg = Config()
    tree = RadixTree(cfg.backend_cache_blocks)
    load = LoadTracker()
    return cfg, tree, load, Router(cfg, tree, load)


def _hashes(cfg, prompt):
    return block_hashes(prompt, cfg.block_chars, cfg.hash_cutoff_blocks)


def test_round_robin_cycles():
    _, _, _, r = _make()
    seq = [r.choose("p", CANDS, strategy="round_robin").backend_id for _ in range(6)]
    assert seq == ["b0", "b1", "b2", "b0", "b1", "b2"]


def test_consistent_hash_is_deterministic_on_first_block():
    _, _, _, r = _make()
    a = "S" * 64 + "document-a-tail"
    b = "S" * 64 + "document-b-different-tail"   # same first block, different later
    ra = r.choose(a, CANDS, strategy="consistent_hash").backend_id
    rb = r.choose(b, CANDS, strategy="consistent_hash").backend_id
    assert ra == rb                               # routed by first block only
    assert ra == r.choose(a, CANDS, strategy="consistent_hash").backend_id


def test_prefix_tree_prefers_cached_backend():
    cfg, tree, load, r = _make()
    prompt = "X" * 1300                            # ~20 blocks -> affinity > hysteresis
    tree.insert(_hashes(cfg, prompt), "b1")        # b1 holds this prefix
    res = r.choose(prompt, CANDS, strategy="prefix_tree")
    assert res.backend_id == "b1"
    assert res.match_blocks == len(_hashes(cfg, prompt))


def test_prefix_tree_saturation_cutoff_excludes_overloaded():
    cfg, tree, load, r = _make()
    prompt = "X" * 1300
    tree.insert(_hashes(cfg, prompt), "b1")        # b1 has the cache...
    load.inflight["b1"] = cfg.max_inflight + 1     # ...but is saturated
    res = r.choose(prompt, CANDS, strategy="prefix_tree")
    assert res.backend_id != "b1"                  # affinity yields to saturation


def test_prefix_tree_balances_when_no_affinity():
    cfg, tree, load, r = _make()
    prompt = "X" * 1300
    load.inflight["b0"] = 10                        # b0 busiest, no cache anywhere
    res = r.choose(prompt, CANDS, strategy="prefix_tree")
    assert res.backend_id != "b0"                   # picks a less-loaded node
