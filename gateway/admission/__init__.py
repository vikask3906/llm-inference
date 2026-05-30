"""SLO-aware admission control + load shedding (standalone, not on the hot path).

The router picks the *best* backend; this layer decides whether to admit the
request at all. It fails fast on deadline-infeasible requests (TTFT budget) and,
once the fleet is saturated, sheds low-priority tiers first (queueing them under
backpressure when possible) so a load spike degrades cheap traffic before it
touches premium SLOs.

  * policy.fleet_pressure -- a [0,1] saturation signal from KV usage + in-flight.
  * policy.decide         -- admit / queue / reject with a Retry-After hint.

Isolated from server.py like gateway/disagg, gateway/rag, gateway/dag, and
gateway/autoscale: the benchmarked single-request hot path is untouched.
"""

from .config import AdmissionConfig
from .policy import (
    ADMIT,
    QUEUE,
    REJECT,
    AdmissionDecision,
    AdmissionRequest,
    FleetState,
    decide,
    fleet_pressure,
    tier_priority,
)

__all__ = [
    "AdmissionConfig",
    "ADMIT",
    "QUEUE",
    "REJECT",
    "AdmissionDecision",
    "AdmissionRequest",
    "FleetState",
    "decide",
    "fleet_pressure",
    "tier_priority",
]
