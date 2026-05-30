from __future__ import annotations

"""Multi-modal capability + affinity routing.

Two-stage decision:
  1. Capability filter (hard): drop backends that can't serve every modality the
     request needs -- an image-bearing request only goes to a vision backend.
  2. Cost score (soft): among capable, un-saturated backends, minimize estimated
     prefill (text tokens + the *uncached* image/audio tokens, crediting media
     a backend has already encoded) plus queue delay.
"""

import dataclasses
from typing import Optional

from .config import MultiModalConfig
from .index import MediaAffinityIndex
from .request import ModalRequest


def parse_capabilities(spec: str) -> dict[str, set[str]]:
    """'b0:text,image;b1:text' -> {'b0': {'text','image'}, 'b1': {'text'}}."""
    caps: dict[str, set[str]] = {}
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        bid, mods = entry.split(":", 1)
        bid = bid.strip()
        mset = {m.strip() for m in mods.split(",") if m.strip()}
        if bid and mset:
            caps[bid] = mset
    return caps


@dataclasses.dataclass
class MultiModalRouteResult:
    backend_id: str
    est_tokens: int            # total prompt tokens (text + media)
    media_overlap: int         # media items already cached on the chosen backend
    total_media: int
    est_ms: float


def _inflight(load, b: str) -> int:
    try:
        return load.inflight[b]
    except (KeyError, TypeError):
        return 0


def choose_multimodal_backend(*, req: ModalRequest, backends: list[str],
                              capabilities: dict[str, set[str]],
                              index: MediaAffinityIndex, load,
                              cfg: MultiModalConfig) -> Optional[MultiModalRouteResult]:
    """Pick the cheapest capable backend, or None if no backend can serve the
    request's modalities (the caller should answer 4xx/415)."""
    if not backends:
        return None
    required = req.required_modalities
    capable = [b for b in backends if required <= capabilities.get(b, {"text"})]
    if not capable:
        return None

    text_tokens = len(req.text) // max(1, cfg.chars_per_token)
    media_tok = [(m.content_id, m.tokens(cfg)) for m in req.media]
    total_media = len(media_tok)
    total_tokens = text_tokens + sum(t for _, t in media_tok)

    eligible = [b for b in capable if _inflight(load, b) <= cfg.max_inflight] or capable

    best = None
    best_cost = float("inf")
    best_overlap = 0
    for b in eligible:
        cached = sum(1 for mid, _ in media_tok if index.is_cached(b, mid))
        uncached_media_tokens = sum(t for mid, t in media_tok
                                    if not index.is_cached(b, mid))
        prefill = cfg.prefill_ms_per_token * (text_tokens + uncached_media_tokens)
        queue = _inflight(load, b) * cfg.service_ms_per_request
        cost = prefill + queue
        if cost < best_cost:
            best_cost, best, best_overlap = cost, b, cached

    return MultiModalRouteResult(backend_id=best, est_tokens=total_tokens,
                                 media_overlap=best_overlap, total_media=total_media,
                                 est_ms=best_cost)
