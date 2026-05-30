from __future__ import annotations

"""SLO-driven autoscaling controller.

Turns an observed offered load into a target replica count and a scale
up/down/hold decision. The desired count is the max of two requirements:

  * latency : fewest replicas whose Erlang-C expected queue wait <= the SLO, and
  * utilization : enough replicas to keep average utilization under target
    (this is the steady-state headroom that absorbs bursts).

The raw target is then tempered for production safety: clamped to [min, max],
rate-limited to `max_scale_step` per tick, gated by separate up/down cooldowns
(scale out fast, scale in slow), and a deadband so a one-replica overshoot
doesn't cause thrash. The output is what a Kubernetes HPA / custom operator
would actuate.

Standalone: imported by nothing on the hot path (like gateway/disagg,
gateway/rag, gateway/dag), so benchmark numbers are untouched.
"""

import dataclasses
import math

from .config import AutoscaleConfig
from .queueing import expected_wait_s, min_replicas_for_wait_slo


@dataclasses.dataclass
class ScalingSignal:
    offered_rps: float
    current_replicas: int
    now_s: float = 0.0
    last_scale_up_s: float = float("-inf")
    last_scale_down_s: float = float("-inf")


@dataclasses.dataclass
class ScalingDecision:
    current_replicas: int
    desired_replicas: int
    direction: str             # "up" | "down" | "hold"
    reason: str
    slo_replicas: int          # latency-driven requirement (pre-clamp)
    util_replicas: int         # utilization-driven requirement (pre-clamp)
    est_wait_ms: float         # expected queue wait at desired_replicas
    blocked_by_cooldown: bool


def _util_replicas(lam: float, mu: float, target_util: float) -> int:
    if lam <= 0.0 or mu <= 0.0:
        return 0
    if target_util <= 0.0:
        return 1
    return math.ceil((lam / mu) / target_util)


def plan(signal: ScalingSignal, cfg: AutoscaleConfig) -> ScalingDecision:
    mu = cfg.service_rate_per_replica_rps
    lam = max(0.0, signal.offered_rps)
    cur = signal.current_replicas

    slo_n = min_replicas_for_wait_slo(lam, mu, cfg.target_wait_ms / 1000.0,
                                      cfg.max_replicas)
    util_n = _util_replicas(lam, mu, cfg.target_utilization)
    target = max(slo_n, util_n)
    target = max(cfg.min_replicas, min(cfg.max_replicas, target))

    blocked = False
    if target > cur:
        if signal.now_s - signal.last_scale_up_s < cfg.scale_up_cooldown_s:
            direction, desired, blocked = "hold", cur, True
            reason = f"scale-up to {target} suppressed by up-cooldown"
        else:
            desired = min(target, cur + cfg.max_scale_step)
            direction = "up"
            reason = (f"latency needs {slo_n}, utilization needs {util_n} "
                      f"-> target {target}")
    elif target <= cur - cfg.scale_down_deadband:
        if signal.now_s - signal.last_scale_down_s < cfg.scale_down_cooldown_s:
            direction, desired, blocked = "hold", cur, True
            reason = f"scale-down to {target} suppressed by down-cooldown"
        else:
            desired = max(target, cur - cfg.max_scale_step)
            direction = "down"
            reason = f"load fits in {target} (deadband {cfg.scale_down_deadband})"
    else:
        direction, desired, reason = "hold", cur, "within deadband of current"

    wait_ms = expected_wait_s(desired, lam, mu) * 1000.0
    return ScalingDecision(
        current_replicas=cur, desired_replicas=desired, direction=direction,
        reason=reason, slo_replicas=slo_n, util_replicas=util_n,
        est_wait_ms=wait_ms, blocked_by_cooldown=blocked)
