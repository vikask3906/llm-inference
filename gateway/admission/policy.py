from __future__ import annotations

"""SLO-aware admission control + priority load shedding.

Autoscaling fixes capacity over minutes; admission control protects the SLO in
the moment. Before a request is dispatched we decide admit / queue / reject from
three signals:

  1. TTFT budget  -- if even the best backend's estimated TTFT already blows the
     deadline, reject early ("fail fast") instead of burning prefill compute on
     a request that is doomed to miss its SLO.
  2. Fleet pressure -- a [0,1] saturation signal (mean KV usage and the fraction
     of saturated backends). Below `shed_pressure` everything that meets its
     deadline is admitted.
  3. Priority shedding -- above `shed_pressure`, shed low tiers first. The
     admit floor rises with pressure: bronze sheds before silver before gold, so
     a load spike degrades cheap traffic to protect premium SLOs. Shed requests
     are queued (backpressure) when a queue is enabled and has room, else
     rejected with a Retry-After hint.

Standalone: imported by nothing on the hot path (like gateway/disagg,
gateway/rag, gateway/dag, gateway/autoscale), so benchmark numbers are
untouched.
"""

import dataclasses

from .config import AdmissionConfig

# tier -> priority; higher survives longer under shedding
_TIER_PRIORITY = {"gold": 3, "silver": 2, "bronze": 1}

ADMIT = "admit"
QUEUE = "queue"
REJECT = "reject"


def tier_priority(tier: str) -> int:
    return _TIER_PRIORITY.get((tier or "").lower(), 0)


@dataclasses.dataclass
class FleetState:
    backend_inflight: dict[str, int] = dataclasses.field(default_factory=dict)
    backend_kv_usage: dict[str, float] = dataclasses.field(default_factory=dict)
    queue_depth: int = 0


@dataclasses.dataclass
class AdmissionRequest:
    est_ttft_ms: float          # best achievable TTFT (from the router cost model)
    tier: str = "bronze"
    est_output_tokens: int = 256


@dataclasses.dataclass
class AdmissionDecision:
    action: str                 # admit | queue | reject
    reason: str
    retry_after_ms: float       # backoff hint (0 when admitted)
    pressure: float             # fleet pressure used for the decision
    admit_floor: float          # priority floor applied (0 when not shedding)


def fleet_pressure(fleet: FleetState, cfg: AdmissionConfig) -> float:
    """A [0,1] saturation signal: the worse of mean KV usage and the fraction of
    backends at/over the per-backend in-flight cap."""
    kv = fleet.backend_kv_usage.values()
    mean_kv = sum(kv) / len(kv) if kv else 0.0
    inflight = fleet.backend_inflight.values()
    if inflight:
        saturated = sum(1 for n in inflight if n >= cfg.max_inflight_per_backend)
        sat_frac = saturated / len(inflight)
    else:
        sat_frac = 0.0
    return max(0.0, min(1.0, max(mean_kv, sat_frac)))


def _admit_floor(pressure: float, cfg: AdmissionConfig) -> float:
    """Priority floor as pressure climbs from shed_pressure (floor 1: shed only
    anonymous) to 1.0 (floor 3: gold only)."""
    span = max(1e-9, 1.0 - cfg.shed_pressure)
    t = max(0.0, min(1.0, (pressure - cfg.shed_pressure) / span))
    return 1.0 + 2.0 * t


def _retry_after(pressure: float, cfg: AdmissionConfig, overshoot_ms: float = 0.0) -> float:
    ms = max(cfg.retry_after_base_ms * (1.0 + pressure), overshoot_ms)
    return min(cfg.retry_after_max_ms, ms)


def decide(req: AdmissionRequest, fleet: FleetState,
           cfg: AdmissionConfig) -> AdmissionDecision:
    pressure = fleet_pressure(fleet, cfg)

    # 1. TTFT budget: doomed requests fail fast (don't waste prefill).
    if req.est_ttft_ms > cfg.ttft_slo_ms:
        overshoot = req.est_ttft_ms - cfg.ttft_slo_ms
        return AdmissionDecision(
            REJECT, f"deadline infeasible: est TTFT {req.est_ttft_ms:.0f}ms "
                    f"> SLO {cfg.ttft_slo_ms:.0f}ms",
            _retry_after(pressure, cfg, overshoot), pressure, 0.0)

    # 2. Below the shed threshold: admit everything that meets its deadline.
    if pressure < cfg.shed_pressure:
        return AdmissionDecision(ADMIT, "within SLO, fleet below shed threshold",
                                 0.0, pressure, 0.0)

    # 3. Priority shedding: keep high tiers, shed (or queue) low tiers.
    floor = _admit_floor(pressure, cfg)
    if tier_priority(req.tier) >= floor:
        return AdmissionDecision(ADMIT, f"tier {req.tier} clears admit floor "
                                 f"{floor:.2f} under pressure {pressure:.2f}",
                                 0.0, pressure, floor)

    if cfg.queue_enabled and fleet.queue_depth < cfg.max_queue_depth:
        return AdmissionDecision(QUEUE, f"tier {req.tier} below floor {floor:.2f}; "
                                 "queued for backpressure",
                                 _retry_after(pressure, cfg), pressure, floor)

    return AdmissionDecision(REJECT, f"tier {req.tier} below floor {floor:.2f} "
                             "and queue full; shed",
                             _retry_after(pressure, cfg), pressure, floor)
