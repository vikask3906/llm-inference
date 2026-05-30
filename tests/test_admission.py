"""Unit tests for SLO-aware admission control (gateway/admission).

Two central properties:
  * deadline-infeasible requests fail fast regardless of tier, and
  * under fleet saturation, shedding climbs the tier ladder (anonymous, then
    bronze, then silver) so premium SLOs are protected last.
"""

from gateway.admission import (
    ADMIT,
    QUEUE,
    REJECT,
    AdmissionConfig,
    AdmissionRequest,
    FleetState,
    decide,
    fleet_pressure,
    tier_priority,
)


def _req(ttft=100.0, tier="bronze"):
    return AdmissionRequest(est_ttft_ms=ttft, tier=tier)


# --- helpers ---

def test_tier_priority_order():
    assert tier_priority("gold") > tier_priority("silver") > tier_priority("bronze")
    assert tier_priority("anon") == 0
    assert tier_priority("") == 0


def test_fleet_pressure_mean_kv():
    fleet = FleetState(backend_kv_usage={"b0": 0.8, "b1": 0.6})
    assert fleet_pressure(fleet, AdmissionConfig()) == 0.7


def test_fleet_pressure_saturation_fraction():
    cfg = AdmissionConfig(max_inflight_per_backend=64)
    fleet = FleetState(backend_inflight={"b0": 100, "b1": 0})  # half saturated
    assert fleet_pressure(fleet, cfg) == 0.5


def test_fleet_pressure_empty_is_zero():
    assert fleet_pressure(FleetState(), AdmissionConfig()) == 0.0


# --- TTFT-budget gate ---

def test_reject_deadline_infeasible_even_for_gold():
    cfg = AdmissionConfig(ttft_slo_ms=500.0)
    d = decide(_req(ttft=800.0, tier="gold"), FleetState(), cfg)
    assert d.action == REJECT
    assert "deadline" in d.reason
    assert d.retry_after_ms > 0.0


def test_retry_after_is_capped():
    cfg = AdmissionConfig(ttft_slo_ms=500.0, retry_after_max_ms=10_000.0)
    d = decide(_req(ttft=1e9, tier="gold"), FleetState(), cfg)
    assert d.retry_after_ms == 10_000.0


# --- below shed threshold: admit everything that meets its deadline ---

def test_admit_when_below_shed_threshold():
    cfg = AdmissionConfig(shed_pressure=0.75)
    fleet = FleetState(backend_kv_usage={"b0": 0.5})       # pressure 0.5 < 0.75
    d = decide(_req(tier="bronze"), fleet, cfg)
    assert d.action == ADMIT
    assert d.admit_floor == 0.0


# --- priority shedding under pressure ---

def _fleet_at(pressure: float) -> FleetState:
    return FleetState(backend_kv_usage={"b0": pressure})


def test_bronze_shed_silver_and_gold_admitted_under_pressure():
    cfg = AdmissionConfig(shed_pressure=0.75)               # pressure 0.875 -> floor 2.0
    fleet = _fleet_at(0.875)
    assert decide(_req(tier="bronze"), fleet, cfg).action == QUEUE
    assert decide(_req(tier="silver"), fleet, cfg).action == ADMIT
    assert decide(_req(tier="gold"), fleet, cfg).action == ADMIT


def test_only_gold_admitted_at_full_pressure():
    cfg = AdmissionConfig(shed_pressure=0.75)               # pressure 1.0 -> floor 3.0
    fleet = _fleet_at(1.0)
    assert decide(_req(tier="silver"), fleet, cfg).action == QUEUE
    assert decide(_req(tier="gold"), fleet, cfg).action == ADMIT


def test_anonymous_shed_at_shed_threshold():
    cfg = AdmissionConfig(shed_pressure=0.75)               # pressure 0.75 -> floor 1.0
    fleet = _fleet_at(0.75)
    assert decide(_req(tier="anon"), fleet, cfg).action == QUEUE
    assert decide(_req(tier="bronze"), fleet, cfg).action == ADMIT


# --- queue vs reject ---

def test_shed_rejects_when_queue_full():
    cfg = AdmissionConfig(shed_pressure=0.75, queue_enabled=True, max_queue_depth=10)
    fleet = FleetState(backend_kv_usage={"b0": 0.875}, queue_depth=10)
    assert decide(_req(tier="bronze"), fleet, cfg).action == REJECT


def test_shed_rejects_when_queue_disabled():
    cfg = AdmissionConfig(shed_pressure=0.75, queue_enabled=False)
    fleet = _fleet_at(0.875)
    assert decide(_req(tier="bronze"), fleet, cfg).action == REJECT


def test_decision_reports_pressure_and_floor():
    cfg = AdmissionConfig(shed_pressure=0.75)
    d = decide(_req(tier="gold"), _fleet_at(0.875), cfg)
    assert d.pressure == 0.875
    assert d.admit_floor == 2.0
