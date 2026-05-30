from __future__ import annotations

"""Per-backend chunk-cache tracker.

Mirrors what each backend's KV cache holds, but at chunk granularity: a set of
chunk ids the backend has recently served (and therefore likely still has
cached), with LRU eviction down to a capacity. The router uses overlap with a
request's chunk set as the cache-affinity signal -- the set-based analogue of
the radix tree's prefix match.
"""

from collections import OrderedDict


class ChunkAffinityIndex:
    def __init__(self, capacity_chunks: int = 4096):
        self.capacity = capacity_chunks
        # backend_id -> OrderedDict[chunk_id, None] used as an LRU set
        self._by_backend: dict[str, "OrderedDict[int, None]"] = {}

    def _cache(self, backend_id: str) -> "OrderedDict[int, None]":
        c = self._by_backend.get(backend_id)
        if c is None:
            c = OrderedDict()
            self._by_backend[backend_id] = c
        return c

    def record(self, backend_id: str, ids: list[int]) -> None:
        """Mark these chunks as cached on a backend (most-recently-used)."""
        c = self._cache(backend_id)
        for cid in ids:
            c[cid] = None
            c.move_to_end(cid)
        while len(c) > self.capacity:
            c.popitem(last=False)

    def overlap(self, backend_id: str, ids: list[int]) -> int:
        """Count how many of `ids` this backend already has cached."""
        c = self._by_backend.get(backend_id)
        if not c:
            return 0
        return sum(1 for cid in ids if cid in c)

    def cached_fraction(self, backend_id: str, ids: list[int]) -> float:
        if not ids:
            return 0.0
        return self.overlap(backend_id, ids) / len(ids)

    def remove_backend(self, backend_id: str) -> None:
        self._by_backend.pop(backend_id, None)
