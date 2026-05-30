from __future__ import annotations

"""Fleet-wide load view: each replica's aggregate of its PEERS' load snapshots.

A replica keeps its own in-flight counters in the LoadTracker (synchronous, exact)
and folds in peers' last-reported snapshots here. The router then scores a backend
by `local_inflight + peer_inflight`, so a backend already busy on another replica
looks busy here too -- preventing two gateways from stampeding the same node.

Holds only PEERS' snapshots (a replica never receives its own echo), so the
router adds the local LoadTracker value separately -- no double counting. Stale
peers simply contribute their last snapshot until they publish a fresher one; a
crashed peer's snapshot lingers but is bounded and harmless (slightly over-counts
load on the backends it was using, biasing AWAY from them -- the safe direction).
"""

from .events import LoadEvent


class FleetLoadView:
    def __init__(self) -> None:
        self._inflight: dict[str, dict[str, int]] = {}   # origin -> {backend: inflight}
        self._unhealthy: dict[str, set[str]] = {}        # origin -> {backend, ...}

    def apply(self, ev: LoadEvent) -> None:
        self._inflight[ev.origin] = dict(ev.inflight)
        self._unhealthy[ev.origin] = set(ev.unhealthy)

    def peer_inflight(self, backend_id: str) -> int:
        return sum(d.get(backend_id, 0) for d in self._inflight.values())

    def peer_unhealthy(self, backend_id: str) -> bool:
        return any(backend_id in s for s in self._unhealthy.values())

    def peers(self) -> int:
        return len(self._inflight)
