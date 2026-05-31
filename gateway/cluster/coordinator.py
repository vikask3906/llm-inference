from __future__ import annotations

"""Ties a replica's local RadixTree to the replication bus.

Write path (hot path, when enabled): after the gateway inserts a dispatched
prefix into its LOCAL tree, it also calls `publish_insert`, which buffers a
mutation event on the bus (cheap, non-blocking).

Sync path (background loop, off the hot path): `sync()` drains the bus and
applies peers' mutations to the local tree, so every replica converges on which
backend holds which prefix. Reads (router longest-prefix match) stay purely
local and fast -- only writes fan out.

Convergence is eventual: each replica's tree is the union of its own and its
peers' inserts (which actually models the backend's real cache -- it serves all
replicas' traffic -- better than a single replica's view). Per-replica LRU
eviction may diverge slightly; routing degrades gracefully, never breaks.
"""

from .events import DRAIN, INSERT, REMOVE_BACKEND, UNDRAIN, LoadEvent, PrefixEvent
from .fleet import FleetLoadView


class ClusterCoordinator:
    def __init__(self, tree, bus, replica_id: str,
                 fleet: FleetLoadView | None = None, registry=None, store=None) -> None:
        self._tree = tree
        self._bus = bus
        self._replica_id = replica_id
        # Optional backend registry: drain/undrain events from peers are applied
        # here so a maintenance decision on one replica reaches the whole fleet.
        self._registry = registry
        # Optional durable snapshot of drain state (survives restart / late join).
        self._store = store
        # peers' load snapshots land here; the router reads it for fleet-wide load.
        self.fleet = fleet if fleet is not None else FleetLoadView()
        self._seq = 0
        # counters for /metrics + introspection
        self.published = 0
        self.applied_remote = 0

    @property
    def replica_id(self) -> str:
        return self._replica_id

    # ----------------------------------------------------------- write path
    def publish_insert(self, hashes, backend_id: str) -> None:
        if not hashes:
            return
        self._seq += 1
        self._bus.publish(PrefixEvent(INSERT, backend_id, self._replica_id,
                                      self._seq, list(hashes)))
        self.published += 1

    def publish_remove(self, backend_id: str) -> None:
        self._seq += 1
        self._bus.publish(PrefixEvent(REMOVE_BACKEND, backend_id,
                                      self._replica_id, self._seq))
        self.published += 1

    def publish_load(self, inflight: dict[str, int],
                     unhealthy: list[str] | None = None) -> None:
        """Broadcast this replica's per-backend in-flight + locally-shed backends
        so peers can route against fleet-wide load."""
        self._seq += 1
        self._bus.publish(LoadEvent(self._replica_id, self._seq,
                                    {k: int(v) for k, v in inflight.items() if v},
                                    list(unhealthy or [])))
        self.published += 1

    def publish_drain(self, backend_id: str, draining: bool) -> None:
        """Tell peers to drain (or restore) a backend, so a maintenance decision
        on this replica takes effect fleet-wide. Also write-through to the durable
        snapshot so restarting / late-joining replicas don't miss it."""
        self._seq += 1
        self._bus.publish(PrefixEvent(DRAIN if draining else UNDRAIN,
                                      backend_id, self._replica_id, self._seq))
        if self._store is not None:
            self._store.record_drain(backend_id, draining)
        self.published += 1

    def warm_start(self) -> set[str]:
        """Seed this replica's drain state from the durable snapshot on boot, so a
        restart or a late join doesn't route to a backend under maintenance.
        Returns the set of backends drained."""
        if self._store is None or self._registry is None:
            return set()
        drained = self._store.drained()
        for bid in drained:
            self._registry.set_draining(bid, True)
        return drained

    # ------------------------------------------------------------ sync path
    def sync(self) -> int:
        """Apply all pending remote mutations to the local tree. Returns the
        number of events applied. Safe to call repeatedly from a background loop.
        """
        events = self._bus.poll()
        for e in events:
            if isinstance(e, LoadEvent):
                self.fleet.apply(e)
            elif e.kind == INSERT:
                self._tree.insert(e.hashes, e.backend_id)
            elif e.kind == REMOVE_BACKEND:
                self._tree.remove_backend(e.backend_id)
            elif e.kind in (DRAIN, UNDRAIN) and self._registry is not None:
                self._registry.set_draining(e.backend_id, e.kind == DRAIN)
            self.applied_remote += 1
        return len(events)

    def close(self) -> None:
        self._bus.close()
        if self._store is not None:
            self._store.close()
