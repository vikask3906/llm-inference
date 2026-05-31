from __future__ import annotations

"""Last-writer-wins (LWW) fleet-membership map for gossip anti-entropy.

Runtime backend add/remove propagate as live deltas, but a restarted / late-
joining replica misses them (it only has its startup `GW_BACKENDS`). So
membership is also kept as an LWW-Map CRDT and gossiped as a periodic digest:
each backend carries `(present, url, model, ts, origin)` where `ts` is a Lamport
timestamp and `origin` the tie-breaker. A removed backend is a *tombstone*
(`present=False`) rather than a deletion, so a stale "still present" can't
resurrect it. Merges are commutative/associative/idempotent -> replicas converge.

Tracks only RUNTIME changes; backends from a replica's startup config that were
never touched at runtime are simply absent from the map and left alone.
"""


class MembershipState:
    def __init__(self) -> None:
        # backend_id -> (present, url, model, ts, origin)
        self._s: dict[str, tuple[bool, str, str, int, str]] = {}
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _observe(self, ts: int) -> None:
        if ts > self._clock:
            self._clock = ts

    def set_local(self, backend_id: str, present: bool, url: str, model: str,
                  origin: str) -> int:
        ts = self._tick()
        self._s[backend_id] = (present, url, model, ts, origin)
        return ts

    def apply(self, backend_id: str, present: bool, url: str, model: str,
              ts: int, origin: str) -> bool:
        self._observe(ts)
        cur = self._s.get(backend_id)
        if cur is None or (ts, origin) > (cur[3], cur[4]):
            changed = cur is None or cur[0] != present or cur[1] != url
            self._s[backend_id] = (present, url, model, ts, origin)
            return changed
        return False

    def merge(self, entries: dict) -> bool:
        changed = False
        for bid, v in entries.items():
            if self.apply(bid, bool(v[0]), v[1], v[2], int(v[3]), v[4]):
                changed = True
        return changed

    def present(self) -> dict[str, tuple[str, str]]:
        """Currently-present runtime backends → (url, model)."""
        return {b: (url, model) for b, (p, url, model, _, _) in self._s.items() if p}

    def removed(self) -> set[str]:
        """Tombstoned (runtime-removed) backend ids."""
        return {b for b, (p, *_rest) in self._s.items() if not p}

    def digest(self) -> dict:
        return {b: [p, url, model, ts, o] for b, (p, url, model, ts, o) in self._s.items()}

    def __len__(self) -> int:
        return len(self._s)
