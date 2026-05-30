from __future__ import annotations

"""Per-backend media-affinity tracker.

Mirrors which media items (images/audio) a backend has recently encoded, so its
vision/audio-encoder KV is likely still warm. An LRU set of media content ids
per backend; the router prefers a backend that already holds a request's media,
saving the (expensive) re-encode of those tokens. The media analogue of the
text radix tree's prefix match.
"""

from collections import OrderedDict


class MediaAffinityIndex:
    def __init__(self, capacity_media: int = 1024):
        self.capacity = capacity_media
        self._by_backend: dict[str, "OrderedDict[int, None]"] = {}

    def _cache(self, backend_id: str) -> "OrderedDict[int, None]":
        c = self._by_backend.get(backend_id)
        if c is None:
            c = OrderedDict()
            self._by_backend[backend_id] = c
        return c

    def record(self, backend_id: str, ids: list[int]) -> None:
        c = self._cache(backend_id)
        for mid in ids:
            c[mid] = None
            c.move_to_end(mid)
        while len(c) > self.capacity:
            c.popitem(last=False)

    def is_cached(self, backend_id: str, media_id: int) -> bool:
        c = self._by_backend.get(backend_id)
        return bool(c) and media_id in c

    def overlap(self, backend_id: str, ids: list[int]) -> int:
        c = self._by_backend.get(backend_id)
        if not c:
            return 0
        return sum(1 for mid in ids if mid in c)

    def remove_backend(self, backend_id: str) -> None:
        self._by_backend.pop(backend_id, None)
