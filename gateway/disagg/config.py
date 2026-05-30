from __future__ import annotations

"""Configuration for disaggregated routing.

Kept separate from the gateway's main Config so this subsystem can be developed
and tuned without touching (or risking a regression in) the benchmarked hot
path. `from_env()` reads GW_DISAGG_* variables, mirroring Config.from_env().
"""

import os
from dataclasses import dataclass


@dataclass
class DisaggConfig:
    # Pool roles: "b0:prefill;b1:decode;b2:prefill,decode". A backend may serve
    # one or both phases. Disjoint pools => every request is disaggregated
    # (pure Splitwise). Overlapping "prefill,decode" backends enable the
    # adaptive DistServe mode where co-location competes with disaggregation.
    pools: str = ""

    # --- phase cost models (milliseconds) ---
    # Prefill is compute-bound and cheap per token; decode is memory-bandwidth
    # bound and several times slower per token. Queue terms approximate how long
    # a new request waits behind the work already in flight on that backend.
    prefill_ms_per_token: float = 0.05      # per UNCACHED prompt token
    decode_ms_per_token: float = 0.20       # per OUTPUT token
    prefill_service_ms: float = 80.0        # queue delay added per in-flight prefill
    decode_service_ms: float = 40.0         # queue delay added per in-flight decode

    block_tokens: int = 16                  # tokens per prefix-cache block (match Config)

    # --- KV-cache handoff model ---
    # bytes/token = 2 (K,V) * num_layers * hidden_dim * dtype_bytes. ~200KB/token
    # is representative of a 7-13B model in fp16. link_gbps is the prefill->decode
    # interconnect (NVLink ~600, InfiniBand ~25-100 GB/s effective).
    kv_bytes_per_token: float = 200_000.0
    link_gbps: float = 100.0

    # --- guardrails / decision threshold ---
    max_inflight: int = 64                  # exclude a backend above this in-flight count
    kv_pressure_cutoff: float = 0.90        # exclude a backend above this KV usage (0..1)
    disagg_margin_ms: float = 1.0           # disaggregate only if it beats co-location by this

    # Output-length estimate used when the request doesn't pin max_tokens.
    default_output_tokens: int = 256

    @classmethod
    def from_env(cls) -> "DisaggConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_DISAGG_{f.upper()}")
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
