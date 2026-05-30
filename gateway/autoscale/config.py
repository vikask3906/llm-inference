from __future__ import annotations

import os
from dataclasses import dataclass

"""Config for the SLO-driven autoscaler (standalone)."""


@dataclass
class AutoscaleConfig:
    enabled: bool = False

    # --- Fleet bounds ---
    min_replicas: int = 1
    max_replicas: int = 20

    # --- Capacity model ---
    service_rate_per_replica_rps: float = 5.0   # mu: requests/sec one replica clears
    target_wait_ms: float = 100.0               # SLO on expected queue wait
    target_utilization: float = 0.70            # cap avg utilization (the headroom)

    # --- Anti-flapping ---
    scale_up_cooldown_s: float = 30.0           # react fast to load
    scale_down_cooldown_s: float = 300.0        # shed slow (avoid thrash)
    max_scale_step: int = 4                     # max replicas added/removed per tick
    scale_down_deadband: int = 1                # don't shrink for a <= this drop

    @classmethod
    def from_env(cls) -> "AutoscaleConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_AUTOSCALE_{f.upper()}")
            if env is None:
                continue
            cur = getattr(cfg, f)
            if isinstance(cur, bool):
                setattr(cfg, f, env.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, f, int(env))
            elif isinstance(cur, float):
                setattr(cfg, f, float(env))
            else:
                setattr(cfg, f, env)
        return cfg
