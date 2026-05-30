"""Unit tests for disaggregated prefill/decode routing (gateway/disagg).

Covers pool-role parsing, the KV-transfer cost model, and the load-dependent
decision to split phases vs co-locate. No GPU / no network -- the routing
algorithm is exercised purely against the analytic cost model.
"""

import math

from gateway.disagg import (
    DisaggConfig,
    PoolRegistry,
    choose_disaggregated,
    kv_transfer_ms,
    parse_pool_config,
)
from gateway.load_tracker import LoadTracker


# --- pool config parsing ---

def test_parse_pool_config_basic():
    roles = parse_pool_config("b0:prefill;b1:decode;b2:prefill,decode")
    assert roles == {"b0": {"prefill"}, "b1": {"decode"}, "b2": {"prefill", "decode"}}


def test_parse_pool_config_tolerates_whitespace_and_junk():
    roles = parse_pool_config(" b0 : prefill , decode ; garbage ; b1:bogus;b2:decode ")
    assert roles["b0"] == {"prefill", "decode"}
    assert "b1" not in roles            # only an invalid role -> dropped
    assert roles["b2"] == {"decode"}


def test_pool_registry_defaults_unlisted_to_both():
    reg = PoolRegistry.from_spec(["b0", "b1", "b2"], "b0:prefill;b1:decode")
    assert reg.roles("b2") == {"prefill", "decode"}      # unlisted -> both
    assert reg.prefill_pool() == ["b0", "b2"]
    assert reg.decode_pool() == ["b1", "b2"]
    assert reg.colocatable() == ["b2"]


# --- KV transfer cost model ---

def test_kv_transfer_ms_known_value():
    # 2000 tokens * 200KB / (100 GB/s) = 0.4GB / 100GB/s = 4ms
    assert math.isclose(kv_transfer_ms(2000, 200_000.0, 100.0), 4.0, rel_tol=1e-9)


def test_kv_transfer_ms_edge_cases():
    assert kv_transfer_ms(0, 200_000.0, 100.0) == 0.0
    assert kv_transfer_ms(1000, 200_000.0, 0.0) == 0.0


# --- decision: unloaded fleet co-locates (handoff is pure overhead) ---

def test_unloaded_fleet_colocates():
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill,decode;b1:prefill,decode")
    load = LoadTracker()
    cfg = DisaggConfig()
    d = choose_disaggregated(prompt_tokens=4000, pools=reg, load=load, cfg=cfg)
    assert d is not None
    assert d.disaggregated is False
    assert d.est_handoff_ms == 0.0


# --- decision: strict disjoint pools always disaggregate ---

def test_disjoint_pools_always_disaggregate():
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill;b1:decode")
    load = LoadTracker()
    cfg = DisaggConfig()
    d = choose_disaggregated(prompt_tokens=512, pools=reg, load=load, cfg=cfg)
    assert d.disaggregated is True
    assert d.prefill_backend == "b0"
    assert d.decode_backend == "b1"
    assert d.est_handoff_ms > 0.0


# --- decision: load on the co-located node makes splitting worth it ---

def test_load_triggers_disaggregation():
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill,decode;b1:decode")
    load = LoadTracker()
    load.inflight["b0"] = 5          # busy co-locatable node
    load.inflight["b1"] = 0          # free decode node
    cfg = DisaggConfig()
    d = choose_disaggregated(prompt_tokens=4000, pools=reg, load=load, cfg=cfg)
    assert d.disaggregated is True
    assert d.prefill_backend == "b0"
    assert d.decode_backend == "b1"   # decode offloaded to the free node
    # disaggregated total must beat the co-located total
    coloc = (cfg.prefill_ms_per_token * 4000 + 5 * cfg.prefill_service_ms
             + cfg.decode_ms_per_token * cfg.default_output_tokens + 5 * cfg.decode_service_ms)
    assert d.est_total_ms < coloc


# --- decision: a huge prompt makes the handoff cost exceed the queue saving ---

def test_huge_prompt_keeps_colocation():
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill,decode;b1:decode")
    load = LoadTracker()
    load.inflight["b0"] = 2          # mild load -> small queue saving
    load.inflight["b1"] = 0
    cfg = DisaggConfig()
    d = choose_disaggregated(prompt_tokens=100_000, pools=reg, load=load, cfg=cfg)
    # handoff (~200ms) dwarfs the 80ms decode-queue saving -> co-locate on b0
    assert d.disaggregated is False
    assert d.prefill_backend == "b0"


# --- decision: saturation guard excludes the overloaded node ---

def test_saturation_excludes_overloaded_node():
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill,decode;b1:prefill,decode")
    load = LoadTracker()
    cfg = DisaggConfig()
    load.inflight["b0"] = cfg.max_inflight + 1     # b0 saturated
    d = choose_disaggregated(prompt_tokens=512, pools=reg, load=load, cfg=cfg)
    assert d.prefill_backend == "b1"
    assert d.decode_backend == "b1"


# --- decision: prefix-cache locality biases ONLY the prefill backend ---

def test_prefix_match_biases_prefill_selection():
    reg = PoolRegistry.from_spec(["b0", "b1", "b2"], "b0:prefill;b1:prefill;b2:decode")
    load = LoadTracker()
    cfg = DisaggConfig()
    # b1 already holds the prompt's prefix (250 blocks * 16 = 4000 tokens cached).
    d = choose_disaggregated(prompt_tokens=4000, pools=reg, load=load, cfg=cfg,
                             prefill_match={"b1": 250})
    assert d.prefill_backend == "b1"   # cached prefix -> zero prefill compute
    assert d.decode_backend == "b2"
    assert d.disaggregated is True
    assert math.isclose(d.est_prefill_ms, 0.0, abs_tol=1e-9)


# --- decision: no backends at all -> None ---

def test_no_backends_returns_none():
    reg = PoolRegistry.from_spec([], "")
    d = choose_disaggregated(prompt_tokens=100, pools=reg, load=LoadTracker(),
                             cfg=DisaggConfig())
    assert d is None


# --- decision fields are internally consistent ---

def test_decision_totals_are_consistent():
    reg = PoolRegistry.from_spec(["b0", "b1"], "b0:prefill;b1:decode")
    d = choose_disaggregated(prompt_tokens=2000, pools=reg, load=LoadTracker(),
                             cfg=DisaggConfig())
    assert math.isclose(d.est_total_ms,
                        d.est_prefill_ms + d.est_handoff_ms + d.est_decode_ms,
                        rel_tol=1e-9)
