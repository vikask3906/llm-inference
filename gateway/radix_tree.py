from __future__ import annotations

"""Approximate global prefix tree mapping prefixes -> backends that hold them.

MVP uses a plain trie (one node per block); path compression is a Phase-2
memory optimization. Each node records which backends are believed to cache the
prefix ending at that node. A per-backend LRU bounds the gateway's belief to
~backend_cache_blocks, mirroring vLLM's own LRU eviction under VRAM pressure.

match() answers, in a single downward walk: for every backend, how many leading
blocks of this request it already holds (longest-prefix match).
"""

from collections import OrderedDict, defaultdict


class _Node:
    __slots__ = ("children", "holders")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.holders: dict[str, int] = {}   # backend_id -> last_seen clock


class RadixTree:
    def __init__(self, backend_cache_blocks: int) -> None:
        self._root = _Node()
        self._cap = backend_cache_blocks
        self._clock = 0
        # per-backend LRU: ordered map node_id -> node (front = oldest)
        self._lru: dict[str, "OrderedDict[int, _Node]"] = defaultdict(OrderedDict)

    def match(self, hashes: list[int]) -> dict[str, int]:
        """backend_id -> longest CONTIGUOUS-from-front matched prefix (in blocks).

        KV reuse requires every block from the front to be present, so a backend
        only counts while it remains a holder at *every* node so far. If a front
        block was evicted, the prefix is cold (match drops to 0) -- matching real
        block-cache contiguity.
        """
        node = self._root
        best: dict[str, int] = {}
        alive: set[str] | None = None
        for depth, h in enumerate(hashes, start=1):
            child = node.children.get(h)
            if child is None:
                break
            if alive is None:
                alive = set(child.holders.keys())
            else:
                alive &= child.holders.keys()
            if not alive:
                break
            for b in alive:
                best[b] = depth
            node = child
        return best

    def insert(self, hashes: list[int], backend_id: str) -> None:
        """Record that `backend_id` now holds every prefix of `hashes`."""
        self._clock += 1
        lru = self._lru[backend_id]
        node = self._root
        for h in hashes:
            child = node.children.get(h)
            if child is None:
                child = _Node()
                node.children[h] = child
            child.holders[backend_id] = self._clock
            nid = id(child)
            if nid in lru:
                lru.move_to_end(nid)
            else:
                lru[nid] = child
            node = child
        self._evict(backend_id)

    def remove_backend(self, backend_id: str) -> None:
        """Membership eviction: a dead backend holds nothing."""
        lru = self._lru.pop(backend_id, None)
        if not lru:
            return
        for node in lru.values():
            node.holders.pop(backend_id, None)

    def _evict(self, backend_id: str) -> None:
        lru = self._lru[backend_id]
        while len(lru) > self._cap:
            _, node = lru.popitem(last=False)     # LRU tail
            node.holders.pop(backend_id, None)
            # MVP: leave structurally-empty nodes in place. Phase 2 reclaims them
            # via copy-on-write snapshots + epoch-based reclamation (off hot path).

    # --- introspection for benchmarks ---
    def held_blocks(self, backend_id: str) -> int:
        return len(self._lru.get(backend_id, ()))
