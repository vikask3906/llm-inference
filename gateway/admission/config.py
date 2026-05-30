from __future__ import annotations

import os
from dataclasses import dataclass

"""Config for the SLO-aware admission controller (standalone)."""


@dataclass
class AdmissionConfig:
    enabled: bool = False

    # --- TTFT-budget gate ---
    ttft_slo_ms: float = 500.0              # deadline; reject work that can't meet it

    # --- Load-shedding under saturation ---
    max_inflight_per_backend: int = 64      # a backend at/over this counts as saturated
    shed_pressure: float = 0.75             # fleet pressure above which shedding begins

    # --- Backpressure queue (vs. outright reject) ---
    queue_enabled: bool = True
    max_queue_depth: int = 256

    # --- Retry-After hinting ---
    retry_after_base_ms: float = 200.0
    retry_after_max_ms: float = 10_000.0

    @classmethod
    def from_env(cls) -> "AdmissionConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_ADMISSION_{f.upper()}")
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
