"""Unit tests for the SLO-driven autoscaler (gateway/autoscale).

Covers the queueing math (against textbook Erlang-C values) and the
controller's sizing + anti-flapping logic.
"""

import math

import pytest

from gateway.autoscale import (
    AutoscaleConfig,
    ScalingSignal,
    erlang_b,
    erlang_c,
    expected_wait_s,
    min_replicas_for_wait_slo,
    plan,
)


# --- queueing math ---

def test_erlang_b_known_value():
    # B(3, 2) = 0.2105... (textbook)
    assert erlang_b(3, 2.0) == pytest.approx(0.210526, abs=1e-5)


def test_erlang_c_known_value():
    # C(3, 2) = 0.4444... (textbook Erlang-C)
    assert erlang_c(3, 2.0) == pytest.approx(0.444444, abs=1e-5)


def test_erlang_c_unstable_is_one():
    assert erlang_c(2, 2.0) == 1.0      # a == c
    assert erlang_c(1, 5.0) == 1.0      # overloaded


def test_erlang_c_no_load_is_zero():
    assert erlang_c(3, 0.0) == 0.0


def test_expected_wait_unstable_is_inf():
    assert math.isinf(expected_wait_s(2, lam=2.0, mu=1.0))   # a=2, c=2


def test_expected_wait_known_value():
    # a=2 (lam=2, mu=1), c=3: Wq = C/(c*mu - lam) = 0.4444/1 = 0.4444 s
    assert expected_wait_s(3, lam=2.0, mu=1.0) == pytest.approx(0.444444, abs=1e-5)


def test_min_replicas_for_slo():
    # a=2; SLO 100ms. c=3 -> 444ms (too slow), c=4 -> ~87ms (ok).
    assert min_replicas_for_wait_slo(lam=2.0, mu=1.0, slo_s=0.1, max_replicas=20) == 4


def test_min_replicas_caps_at_max():
    # Impossible SLO -> returns the ceiling rather than looping forever.
    assert min_replicas_for_wait_slo(lam=100.0, mu=1.0, slo_s=1e-9,
                                     max_replicas=8) == 8


def test_min_replicas_zero_load():
    assert min_replicas_for_wait_slo(lam=0.0, mu=5.0, slo_s=0.1, max_replicas=20) == 0


# --- controller: sizing ---

def test_scale_up_under_load():
    cfg = AutoscaleConfig(service_rate_per_replica_rps=5.0, target_wait_ms=50.0,
                          max_scale_step=10, min_replicas=1)
    sig = ScalingSignal(offered_rps=40.0, current_replicas=2, now_s=10_000.0)
    d = plan(sig, cfg)
    assert d.direction == "up"
    assert d.desired_replicas > 2
    assert d.est_wait_ms <= cfg.target_wait_ms + 1e-6


def test_scale_down_when_overprovisioned():
    cfg = AutoscaleConfig(service_rate_per_replica_rps=5.0, target_wait_ms=100.0)
    sig = ScalingSignal(offered_rps=5.0, current_replicas=15, now_s=10_000.0)
    d = plan(sig, cfg)
    assert d.direction == "down"
    assert d.desired_replicas < 15


def test_hold_within_deadband():
    cfg = AutoscaleConfig(service_rate_per_replica_rps=5.0, target_wait_ms=100.0,
                          max_scale_step=100)
    # size for the load first (big step so we reach the true target), then feed
    # that exact count back in
    target = plan(ScalingSignal(offered_rps=20.0, current_replicas=1, now_s=1.0),
                  cfg).desired_replicas
    d = plan(ScalingSignal(offered_rps=20.0, current_replicas=target, now_s=10_000.0),
             cfg)
    assert d.direction == "hold"
    assert d.desired_replicas == target


def test_respects_min_and_max_bounds():
    cfg = AutoscaleConfig(min_replicas=3, max_replicas=6,
                          service_rate_per_replica_rps=5.0, max_scale_step=100)
    # zero load -> floor at min
    lo = plan(ScalingSignal(offered_rps=0.0, current_replicas=3, now_s=1.0), cfg)
    assert lo.desired_replicas == 3
    # crushing load -> capped at max
    hi = plan(ScalingSignal(offered_rps=500.0, current_replicas=3, now_s=1.0), cfg)
    assert hi.desired_replicas == 6


def test_scale_step_limits_jump():
    cfg = AutoscaleConfig(service_rate_per_replica_rps=5.0, target_wait_ms=10.0,
                          max_scale_step=2)
    d = plan(ScalingSignal(offered_rps=100.0, current_replicas=2, now_s=10_000.0), cfg)
    assert d.direction == "up"
    assert d.desired_replicas == 4          # 2 + step, not the full target


# --- controller: anti-flapping ---

def test_up_cooldown_suppresses_scale_up():
    cfg = AutoscaleConfig(service_rate_per_replica_rps=5.0, scale_up_cooldown_s=30.0,
                          max_scale_step=10)
    sig = ScalingSignal(offered_rps=80.0, current_replicas=2,
                        now_s=100.0, last_scale_up_s=90.0)   # 10s < 30s cooldown
    d = plan(sig, cfg)
    assert d.direction == "hold"
    assert d.blocked_by_cooldown is True
    assert d.desired_replicas == 2


def test_down_cooldown_suppresses_scale_down():
    cfg = AutoscaleConfig(service_rate_per_replica_rps=5.0,
                          scale_down_cooldown_s=300.0)
    sig = ScalingSignal(offered_rps=5.0, current_replicas=15,
                        now_s=100.0, last_scale_down_s=0.0)  # 100s < 300s cooldown
    d = plan(sig, cfg)
    assert d.direction == "hold"
    assert d.blocked_by_cooldown is True
    assert d.desired_replicas == 15
