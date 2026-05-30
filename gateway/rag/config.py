from __future__ import annotations

"""Configuration for RAG-aware caching/routing.

Standalone (like gateway/disagg) so it can't perturb the benchmarked hot path.
`from_env()` reads GW_RAG_* variables, mirroring Config.from_env().
"""

import os
from dataclasses import dataclass


@dataclass
class RagConfig:
    enabled: bool = False

    # Where the structured RAG payload lives on the request body:
    #   body[request_field] = {"system": str, "chunks": [str, ...], "query": str}
    request_field: str = "rag"

    # Canonicalize (dedupe + stable-sort) the retrieved chunks so two queries
    # that retrieve the SAME set of chunks produce the SAME prefix, turning
    # set-overlap into prefix-cache hits on a stock vLLM backend. Trades the
    # retriever's relevance ordering for cache reuse -- toggle off if answer
    # quality is sensitive to chunk order ("lost in the middle").
    canonicalize: bool = True

    # --- cost model for chunk-affinity routing ---
    tokens_per_chunk: int = 256             # estimate, for prefill-cost scoring
    prefill_ms_per_token: float = 0.05
    service_ms_per_request: float = 200.0
    max_inflight: int = 64                  # exclude a backend above this in-flight count

    # Per-backend chunk-cache capacity (LRU), approximating how many distinct
    # chunk KVs a backend retains before eviction.
    cache_capacity_chunks: int = 4096

    @classmethod
    def from_env(cls) -> "RagConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            env = os.environ.get(f"GW_RAG_{f.upper()}")
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
