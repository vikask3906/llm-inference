from __future__ import annotations

"""Semantic prompt cache.

Prefix caching (the gateway's main job) only catches *byte-identical*
prefixes. Two prompts that paraphrase the same question -- "summarise
this contract" vs "give me a summary of this contract" -- share no
prefix and re-prefill from scratch. A semantic cache embeds the prompt
into a vector and returns a previously-cached response if a sufficiently
similar prompt was seen recently (cosine similarity above threshold).

This module is independent of the prefix tree and sits *before* routing:
on a hit, the gateway returns the cached response and never touches a
backend. On a miss, the request flows through normal routing, then the
response is added to the cache.

Per-tenant isolation: each tenant has its own LRU. No cross-tenant leak.

Default embedder = sentence-transformers/all-MiniLM-L6-v2 (~80MB, ~10ms
on CPU per query). For tests / offline CI, any callable str -> np.ndarray
of fixed dimension works.
"""

import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

try:
    import numpy as np  # type: ignore
except ImportError:  # numpy is a hard dep for sklearn anyway, but guard
    np = None  # type: ignore


Embedder = Callable[[str], "np.ndarray"]


def default_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Lazy-load sentence-transformers. Raises ImportError if unavailable."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(model_name)
    def _embed(text: str):
        v = model.encode(text, normalize_embeddings=True)
        return v
    return _embed


class _TenantCache:
    """Per-tenant LRU. Stores up to max_entries (embedding, response, ts)."""

    __slots__ = ("max_entries", "embeddings", "responses", "_order", "_lock")

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        # Parallel arrays kept in OrderedDict insertion order for LRU.
        # Key = a monotonically-increasing int id.
        self.embeddings: dict[int, "np.ndarray"] = {}
        self.responses: dict[int, str] = {}
        self._order: OrderedDict[int, float] = OrderedDict()
        self._lock = threading.Lock()

    def find_similar(self, query_vec: "np.ndarray", threshold: float
                     ) -> Optional[tuple[int, str, float]]:
        """Return (entry_id, response, similarity) for the highest-similarity
        cached entry above threshold, or None."""
        if np is None or not self.embeddings:
            return None
        with self._lock:
            ids = list(self.embeddings.keys())
            mat = np.stack([self.embeddings[i] for i in ids], axis=0)
            sims = mat @ query_vec  # both unit-normalised -> cosine
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= threshold:
                entry_id = ids[best_idx]
                # touch -> move to end (most-recent)
                self._order.move_to_end(entry_id)
                return entry_id, self.responses[entry_id], best_sim
        return None

    def insert(self, embedding: "np.ndarray", response: str) -> int:
        with self._lock:
            entry_id = (max(self._order.keys()) + 1) if self._order else 1
            self.embeddings[entry_id] = embedding
            self.responses[entry_id] = response
            self._order[entry_id] = time.time()
            while len(self._order) > self.max_entries:
                old_id, _ = self._order.popitem(last=False)
                self.embeddings.pop(old_id, None)
                self.responses.pop(old_id, None)
            return entry_id

    def __len__(self) -> int:
        return len(self._order)


class SemanticCache:
    """Multi-tenant semantic cache. Threshold and capacity are tunable."""

    def __init__(self, embedder: Embedder, similarity_threshold: float = 0.97,
                 max_entries_per_tenant: int = 1024):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_entries = max_entries_per_tenant
        self._tenants: dict[str, _TenantCache] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _tenant(self, tenant_id: str) -> _TenantCache:
        with self._lock:
            t = self._tenants.get(tenant_id)
            if t is None:
                t = _TenantCache(self.max_entries)
                self._tenants[tenant_id] = t
            return t

    def lookup(self, tenant_id: str, prompt: str
               ) -> Optional[tuple[str, float]]:
        """Return (cached_response, similarity) on hit, None on miss."""
        if np is None:
            return None
        try:
            vec = self.embedder(prompt)
            if hasattr(vec, "astype"):
                vec = vec.astype("float32")
        except Exception:
            return None
        hit = self._tenant(tenant_id).find_similar(vec, self.threshold)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        _entry_id, resp, sim = hit
        return resp, sim

    def store(self, tenant_id: str, prompt: str, response: str) -> bool:
        """Cache a (prompt, response) pair. Returns False if embedding fails."""
        if np is None:
            return False
        try:
            vec = self.embedder(prompt)
            if hasattr(vec, "astype"):
                vec = vec.astype("float32")
        except Exception:
            return False
        self._tenant(tenant_id).insert(vec, response)
        return True

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "tenants": len(self._tenants),
            "total_entries": sum(len(t) for t in self._tenants.values()),
        }
