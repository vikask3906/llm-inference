from __future__ import annotations

"""Last-writer-wins (LWW) drain state for gossip anti-entropy.

Drain/undrain are concurrent operations across replicas, so a naive "union of
drained backends" can never undrain (an old drain wins forever). Instead each
backend carries a version `(ts, origin)` — a Lamport timestamp plus the replica
id as a deterministic tie-breaker — and the highest version wins. This is a tiny
LWW-Map CRDT: merges are commutative, associative, idempotent, so replicas
converge regardless of message order or duplication.

Periodically every replica gossips its full map (`digest`); receivers `merge` it.
A late-joining or restarted replica therefore converges within a gossip round
with no central store — the piece that lets the gossip transport drop Redis
entirely for drain state too.
"""


class DrainState:
    def __init__(self) -> None:
        # backend_id -> (draining: bool, ts: int, origin: str)
        self._s: dict[str, tuple[bool, int, str]] = {}
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _observe(self, ts: int) -> None:
        if ts > self._clock:
            self._clock = ts

    def set_local(self, backend_id: str, draining: bool, origin: str) -> int:
        """Record a local drain/undrain; returns its Lamport timestamp."""
        ts = self._tick()
        self._s[backend_id] = (draining, ts, origin)
        return ts

    def apply(self, backend_id: str, draining: bool, ts: int, origin: str) -> bool:
        """Apply a remote op; adopt iff (ts, origin) beats the current version.
        Returns True if the drained-set changed."""
        self._observe(ts)
        cur = self._s.get(backend_id)
        if cur is None or (ts, origin) > (cur[1], cur[2]):
            changed = cur is None or cur[0] != draining
            self._s[backend_id] = (draining, ts, origin)
            return changed
        return False

    def merge(self, entries: dict) -> bool:
        """Merge a peer's full digest (LWW per backend). Returns True if changed."""
        changed = False
        for bid, ver in entries.items():
            draining, ts, origin = ver[0], int(ver[1]), ver[2]
            if self.apply(bid, bool(draining), ts, origin):
                changed = True
        return changed

    def drained(self) -> set[str]:
        return {b for b, (d, _, _) in self._s.items() if d}

    def digest(self) -> dict:
        return {b: [d, ts, o] for b, (d, ts, o) in self._s.items()}

    def __len__(self) -> int:
        return len(self._s)
