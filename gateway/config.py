from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Prefix / block model (mirrors vLLM automatic prefix caching) ---
    # MVP approximation: a "block" is a fixed run of CHARACTERS, standing in for
    # vLLM's 16-TOKEN blocks. Phase 2 swaps this for token-accurate blocks.
    block_chars: int = 64                # ~16 tokens * ~4 chars/token
    block_tokens: int = 16               # nominal tokens per block (for cost math)

    # Suggestion #3: bound hot-path hashing to the first N blocks.
    # 512 blocks * 16 tokens = ~8k tokens. Trades matcher fidelity for bounded CPU.
    hash_cutoff_blocks: int = 512

    # --- Per-backend KV-cache model (LRU eviction) ---
    # Approximates a backend's vLLM num_gpu_blocks. The gateway evicts its belief
    # set per backend down to this cap, mirroring the backend's own LRU.
    backend_cache_blocks: int = 2000

    # --- Cost function (est-TTFT, in milliseconds) ---
    # MVP uses a LINEAR prefill model. Phase 2 fits a polynomial service-time
    # model a*(N-m)*N + b*(N-m) from observed (tokens, prefill_ms) pairs.
    prefill_ms_per_token: float = 0.05
    service_ms_per_request: float = 200.0   # avg backend service time, for queue delay

    # --- Guardrails ---
    kv_pressure_cutoff: float = 0.90        # exclude backend above this KV usage
    max_inflight: int = 64                  # exclude backend above this in-flight count
    hysteresis_ms: float = 5.0              # alt must beat current best by this margin

    # --- Routing ---
    # round_robin | consistent_hash | prefix_tree
    strategy: str = "prefix_tree"

    # --- Fault tolerance ---
    max_retries: int = 2                 # failover attempts before first byte
    circuit_fail_threshold: int = 3      # consecutive failures -> open circuit
    circuit_cooldown_s: float = 5.0      # open duration before a half-open probe

    # --- Multi-tenancy / fairness ---
    rate_limit_enabled: bool = False        # enforcement is opt-in (needs tenants/tiers)
    # "key=tenant:tier,..."  tier in {gold, silver, bronze}; unknown keys -> anonymous
    tenants: str = ""
    prefix_isolation: str = "tenant"        # tenant | global (cross-tenant cache sharing)
    default_output_tokens: int = 256        # TPS reservation when max_tokens is absent
    max_output_tokens: int = 4096           # cap on the output reservation
    chars_per_token: int = 4                # prompt token estimate for quota accounting

    # --- Backends (used by the HTTP layer) ---
    # Comma-separated "id=url" pairs, all assumed to serve `default_model`.
    backends: str = "b0=http://localhost:9001,b1=http://localhost:9002,b2=http://localhost:9003"
    default_model: str = "mock-model"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_{f.upper()}")
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
