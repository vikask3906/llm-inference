"""SLO-driven autoscaling / capacity planning (standalone, not on the hot path).

Given the offered load and a wait-time SLO, decide how many replicas the fleet
should run. The sizing uses an Erlang-C (M/M/c) queueing model -- the standard
"how many servers meet a wait SLO" tool -- combined with a utilization cap for
steady-state headroom, then tempered with clamping, step limits, cooldowns, and
a deadband so the controller doesn't thrash.

  1. queueing -- Erlang-B/C math + min-replicas-for-wait-SLO solver.
  2. planner  -- the scale up/down/hold controller a K8s HPA / operator drives.

Isolated from server.py like gateway/disagg, gateway/rag, and gateway/dag: the
benchmarked single-request hot path is untouched.
"""

from .config import AutoscaleConfig
from .planner import ScalingDecision, ScalingSignal, plan
from .queueing import (
    erlang_b,
    erlang_c,
    expected_wait_s,
    min_replicas_for_wait_slo,
)

__all__ = [
    "AutoscaleConfig",
    "ScalingDecision",
    "ScalingSignal",
    "plan",
    "erlang_b",
    "erlang_c",
    "expected_wait_s",
    "min_replicas_for_wait_slo",
]
