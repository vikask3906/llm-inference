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
    # round_robin | consistent_hash | prefix_tree | speculative
    strategy: str = "prefix_tree"

    # --- Extensions: BPE token-ID prefix hashing ---
    # When True, hash blocks of real BPE token IDs instead of raw character
    # blocks. Matches how vLLM keys its KV cache, so prefixes that tokenize
    # identically but render slightly differently (whitespace, casing) hit the
    # same cache entry. Falls back to char-based hashing if the tokenizer
    # can't be loaded.
    use_bpe_hashing: bool = False
    tokenizer_model: str = "gpt2"           # any HF Hub repo with tokenizer.json
    # When True, quota (TPS) + admission accounting count REAL tokens via the
    # tokenizer instead of the len(prompt)//chars_per_token heuristic (which is
    # off by 2-3x on code / non-English). Costs a tokenize pass per request, so
    # opt-in; falls back to the heuristic if the tokenizer can't load.
    token_accurate_accounting: bool = False

    # --- Extensions: predictive TTFT load scoring ---
    # When set, the gateway appends per-request (features, observed_ttft) to
    # a JSONL file for offline training. When a trained model exists at
    # ttft_model_path, the router blends its prediction with the static
    # linear-prefill formula (weight=0 -> pure static, 1 -> pure learned).
    ttft_observations_path: str = ""        # empty disables logging
    ttft_model_path: str = ""               # empty disables prediction
    ttft_predictor_weight: float = 0.5      # blend weight in [0, 1]

    # --- Extensions: speculative / shadow routing ---
    # When strategy=="speculative", dispatch the request to the top-K candidates
    # in parallel and return the first response. Cuts tail latency at a
    # K-multiplicative GPU cost; only enable when p99 dominates the SLO.
    speculative_k: int = 2

    # --- Extensions: LoRA-aware routing ---
    # "b0:adapter1,adapter2;b1:adapter3" -- declares which LoRA adapters are
    # loaded on which backend. Request model field "base:adapter" routes only
    # to backends with that adapter. Empty disables LoRA awareness.
    backend_adapters: str = ""
    lora_fallback_to_base: bool = True      # serve from base pool if no adapter-capable backend

    # --- Extensions: semantic prompt cache ---
    # Catches paraphrased prompts that share no byte-identical prefix.
    # Cosine similarity above semantic_cache_threshold triggers a hit.
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = 0.97
    semantic_cache_max_entries_per_tenant: int = 1024
    semantic_cache_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Fault tolerance ---
    max_retries: int = 2                 # failover attempts before first byte
    circuit_fail_threshold: int = 3      # consecutive failures -> open circuit
    circuit_cooldown_s: float = 5.0      # open duration before a half-open probe

    # --- Authentication ---
    # When True, a request must carry Authorization: Bearer <key> with a key in
    # the valid set (configured tenant keys + api_keys); otherwise -> 401. Default
    # OFF so an unconfigured gateway stays open (dev / behind a service mesh).
    require_auth: bool = False
    api_keys: str = ""                      # extra valid keys (comma-sep) not tied to a tenant
    # Control-plane admin API (drain backends, inspect routing). The /admin/*
    # endpoints are DISABLED until this is set, then require it as a bearer/
    # X-Admin-Token. Empty = no admin surface exposed (secure default).
    admin_token: str = ""

    # --- Multi-tenancy / fairness ---
    rate_limit_enabled: bool = False        # enforcement is opt-in (needs tenants/tiers)
    # "key=tenant:tier,..."  tier in {gold, silver, bronze}; unknown keys -> anonymous
    tenants: str = ""
    prefix_isolation: str = "tenant"        # tenant | global (cross-tenant cache sharing)
    default_output_tokens: int = 256        # TPS reservation when max_tokens is absent
    max_output_tokens: int = 4096           # cap on the output reservation
    chars_per_token: int = 4                # prompt token estimate for quota accounting

    # --- Observability ---
    log_level: str = "INFO"
    # Per-route SLO: when > 0, requests whose TTFT exceeds this (ms) increment
    # gateway_slo_violations_total{model}. The TTFT + total-latency histograms
    # (gateway_ttft_seconds / gateway_request_duration_seconds, labeled by model)
    # are always emitted regardless. 0 = no violation counting.
    slo_ttft_ms: float = 0.0

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
