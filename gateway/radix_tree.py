from __future__ import annotations

"""Path-compressed (radix/PATRICIA) prefix tree mapping prefixes -> backends.

Each edge carries a *run* of block-hashes, so a chain of single-child blocks is
one node instead of one-per-block: memory is O(unique blocks) but the node count
collapses to O(branch points). A divergent insert splits an edge in two.

Holder model: a backend recorded on node N holds the entire prefix ending at N.
`match` returns, for every backend, the longest CONTIGUOUS-from-front prefix it
holds (KV reuse needs every block from the front), found in one downward walk.

Eviction: a per-backend LRU over held nodes, bounded by `backend_cache_blocks`
(measured in BLOCKS = sum of edge lengths), mirroring vLLM's LRU. Eviction is
node-granular (a whole cached segment is dropped), an acceptable approximation
of the backend's block-level LRU.
"""

from collections import OrderedDict, defaultdict


class _Node:
    __slots__ = ("edge", "children", "holders")

    def __init__(self, edge: list[int]) -> None:
        self.edge = edge                       # run of block-hashes into this node
        self.children: dict[int, _Node] = {}   # first-hash-of-edge -> child
        self.holders: dict[str, int] = {}      # backend_id -> last_seen clock


class RadixTree:
    def __init__(self, backend_cache_blocks: int) -> None:
        self._root = _Node([])
        self._cap = backend_cache_blocks
        self._clock = 0
        self._lru: dict[str, "OrderedDict[int, _Node]"] = defaultdict(OrderedDict)

    # ------------------------------------------------------------------ match
    def match(self, hashes: list[int]) -> dict[str, int]:
        node = self._root
        best: dict[str, int] = {}
        alive: set[str] | None = None
        depth = 0
        i, n = 0, len(hashes)
        while i < n:
            child = node.children.get(hashes[i])
            if child is None:
                break
            e = child.edge
            le = len(e)
            j = 0
            while j < le and i + j < n and e[j] == hashes[i + j]:
                j += 1
            # backends holding `child` hold all `le` edge blocks, hence the first j
            hk = child.holders.keys()
            alive = set(hk) if alive is None else (alive & hk)
            if not alive:
                break
            depth += j
            for b in alive:
                best[b] = depth
            if j < le:
                break                          # diverged inside this edge
            i += j
            node = child
        return best

    # ----------------------------------------------------------------- insert
    def insert(self, hashes: list[int], backend_id: str) -> None:
        if not hashes:
            return
        self._clock += 1
        node = self._root
        i, n = 0, len(hashes)
        while i < n:
            first = hashes[i]
            child = node.children.get(first)
            if child is None:
                leaf = _Node(hashes[i:])
                node.children[first] = leaf
                self._hold(backend_id, leaf)
                break

            e = child.edge
            le = len(e)
            j = 0
            while j < le and i + j < n and e[j] == hashes[i + j]:
                j += 1

            if j == le:                        # consumed the whole edge
                self._hold(backend_id, child)
                i += j
                node = child
                continue

            # divergence inside the edge -> split `child` at offset j
            split = _Node(e[:j])
            child.edge = e[j:]
            split.children[child.edge[0]] = child
            node.children[first] = split
            split.holders = dict(child.holders)        # they hold the shorter prefix too
            for b in split.holders:
                self._track(b, split)
            self._hold(backend_id, split)              # current backend holds up to split
            if i + j < n:                              # new branch diverging from split
                leaf = _Node(hashes[i + j:])
                split.children[hashes[i + j]] = leaf
                self._hold(backend_id, leaf)
            break

        self._evict(backend_id)

    def remove_backend(self, backend_id: str) -> None:
        """Membership eviction: a dead backend holds nothing."""
        lru = self._lru.pop(backend_id, None)
        if not lru:
            return
        for node in lru.values():
            node.holders.pop(backend_id, None)

    # ------------------------------------------------------------- internals
    def _hold(self, backend_id: str, node: _Node) -> None:
        node.holders[backend_id] = self._clock
        self._track(backend_id, node)

    def _track(self, backend_id: str, node: _Node) -> None:
        lru = self._lru[backend_id]
        nid = id(node)
        if nid in lru:
            lru.move_to_end(nid)
        else:
            lru[nid] = node

    def _evict(self, backend_id: str) -> None:
        lru = self._lru[backend_id]
        total = sum(len(n.edge) for n in lru.values())
        while total > self._cap and lru:
            _, node = lru.popitem(last=False)          # LRU tail
            total -= len(node.edge)
            node.holders.pop(backend_id, None)

    # ------------------------------------------------------------ introspect
    def held_blocks(self, backend_id: str) -> int:
        return sum(len(n.edge) for n in self._lru.get(backend_id, {}).values())
