from __future__ import annotations

"""Chunk-affinity routing for RAG requests.

Routes a RAG request to the backend that already has the most of its retrieved
chunks cached (set overlap), traded off against current load. This is the
set-based analogue of the prefix router: more cached chunks => fewer uncached
tokens to prefill => lower estimated TTFT.
"""

import dataclasses
from typing import Optional

from .config import RagConfig
from .index import ChunkAffinityIndex


@dataclasses.dataclass
class RagRouteResult:
    backend_id: str
    overlap: int               # chunks already cached on the chosen backend
    total_chunks: int
    est_ms: float


def _inflight(load, b: str) -> int:
    try:
        return load.inflight[b]
    except (KeyError, TypeError):
        return 0


def choose_rag_backend(*, chunk_ids: list[int], backends: list[str],
                       index: ChunkAffinityIndex, load, cfg: RagConfig
                       ) -> Optional[RagRouteResult]:
    """Pick the backend minimizing estimated prefill+queue cost given chunk
    cache affinity. Returns None if there are no backends."""
    if not backends:
        return None
    total = len(chunk_ids)

    def est(b: str) -> tuple[float, int]:
        ov = index.overlap(b, chunk_ids)
        uncached_tokens = max(0, total - ov) * cfg.tokens_per_chunk
        prefill = cfg.prefill_ms_per_token * uncached_tokens
        queue = _inflight(load, b) * cfg.service_ms_per_request
        return prefill + queue, ov

    # Prefer un-saturated backends; fall back to the whole pool if all are.
    eligible = [b for b in backends if _inflight(load, b) <= cfg.max_inflight] or backends

    best_b = None
    best_cost = float("inf")
    best_ov = 0
    for b in eligible:
        cost, ov = est(b)
        if cost < best_cost:
            best_cost, best_b, best_ov = cost, b, ov

    return RagRouteResult(backend_id=best_b, overlap=best_ov,
                          total_chunks=total, est_ms=best_cost)
