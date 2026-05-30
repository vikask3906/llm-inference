from __future__ import annotations

import os
from dataclasses import dataclass

"""Config for the cache-locality-aware DAG scheduler (standalone)."""


@dataclass
class DagSchedConfig:
    enabled: bool = False

    # --- Prefix / block model (mirrors the live router's hashing) ---
    block_chars: int = 64
    block_tokens: int = 16
    hash_cutoff_blocks: int = 512
    backend_cache_blocks: int = 2000        # per-backend KV cache model (blocks)

    # --- Cost model (ms) ---
    prefill_ms_per_token: float = 0.05      # uncached prefill cost per token
    service_ms_per_request: float = 200.0   # queue delay per already-inflight request
    max_inflight: int = 64                  # saturation cutoff (mirror the router)

    # --- DAG-aware locality ---
    # A child that shares its upstream context (system prompt + retrieved docs)
    # with a parent benefits from co-location beyond the literal front-prefix the
    # radix tree already credits: the family's working set stays resident, so we
    # bias a node toward a backend that already ran one of its parents.
    parent_affinity_bonus_ms: float = 8.0

    @classmethod
    def from_env(cls) -> "DagSchedConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_DAG_{f.upper()}")
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
