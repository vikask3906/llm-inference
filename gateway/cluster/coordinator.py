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

from .drain_state import DrainState
from .events import (
    INSERT,
    MEMBER_ADD,
    REMOVE_BACKEND,
    DrainDigest,
    DrainEvent,
    LoadEvent,
    MemberEvent,
    MembershipDigest,
    PrefixEvent,
    decode_event,
)
from .fleet import FleetLoadView
from .membership_state import MembershipState


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
        # LWW drain map: source of truth the registry's draining flags reconcile to.
        self._drain = DrainState()
        # LWW membership map: runtime backend add/removes the registry reconciles to.
        self._membership = MembershipState()
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
        on this replica takes effect fleet-wide. Versioned (LWW) so concurrent
        drain/undrain converge; also write-through to the durable snapshot."""
        ts = self._drain.set_local(backend_id, draining, self._replica_id)
        self._seq += 1
        self._bus.publish(DrainEvent(self._replica_id, self._seq, ts,
                                     backend_id, draining))
        if self._store is not None:
            self._store.record_drain(backend_id, draining)
        self._reconcile_drains()
        self.published += 1

    def publish_member_add(self, backend_id: str, url: str, model: str = "") -> None:
        ts = self._membership.set_local(backend_id, True, url, model, self._replica_id)
        self._seq += 1
        self._bus.publish(MemberEvent(self._replica_id, self._seq, MEMBER_ADD,
                                      backend_id, url, model, ts))
        self.published += 1

    def publish_member_remove(self, backend_id: str) -> None:
        ts = self._membership.set_local(backend_id, False, "", "", self._replica_id)
        self._seq += 1
        self._bus.publish(MemberEvent(self._replica_id, self._seq, "member_remove",
                                      backend_id, "", "", ts))
        self.published += 1

    def publish_membership_digest(self) -> None:
        """Broadcast the full LWW membership map (anti-entropy) so a restarted /
        late-joining replica converges on runtime backend add/removes."""
        if len(self._membership) == 0:
            return
        self._seq += 1
        self._bus.publish(MembershipDigest(self._replica_id, self._seq,
                                           self._membership.digest()))
        self.published += 1

    def _reconcile_membership(self) -> None:
        """Make the registry match the LWW membership map (runtime deltas only;
        startup-config backends not in the map are left untouched)."""
        if self._registry is None:
            return
        for bid, (url, model) in self._membership.present().items():
            self._registry.add(bid, url, model or None)
        for bid in self._membership.removed():
            if self._registry.remove(bid):
                self._tree.remove_backend(bid)

    def publish_drain_digest(self) -> None:
        """Broadcast this replica's full LWW drain map (anti-entropy). Lets a
        late-joining / restarted replica converge with no central store."""
        if len(self._drain) == 0:
            return
        self._seq += 1
        self._bus.publish(DrainDigest(self._replica_id, self._seq, self._drain.digest()))
        self.published += 1

    def _reconcile_drains(self) -> None:
        """Make the registry's draining flags match the LWW drain map."""
        if self._registry is None:
            return
        drained = self._drain.drained()
        for b in self._registry.all():
            self._registry.set_draining(b.id, b.id in drained)

    def warm_start(self) -> set[str]:
        """Seed drain state from the durable snapshot on boot, so a restart / late
        join doesn't route to a backend under maintenance. Returns the drained set."""
        if self._store is None:
            return set()
        drained = self._store.drained()
        for bid in drained:
            self._drain.set_local(bid, True, self._replica_id)
        self._reconcile_drains()
        return drained

    # ------------------------------------------------------------ sync path
    def sync(self) -> int:
        """Apply all pending remote mutations to the local tree. Returns the
        number of events applied. Safe to call repeatedly from a background loop.
        """
        events = self._bus.poll()
        drains_changed = False
        members_changed = False
        for e in events:
            if isinstance(e, LoadEvent):
                self.fleet.apply(e)
            elif isinstance(e, DrainEvent):
                if self._drain.apply(e.backend_id, e.draining, e.ts, e.origin):
                    drains_changed = True
            elif isinstance(e, DrainDigest):
                if self._drain.merge(e.entries):
                    drains_changed = True
            elif isinstance(e, MemberEvent):
                if self._membership.apply(e.backend_id, e.action == MEMBER_ADD,
                                          e.url, e.model, e.ts, e.origin):
                    members_changed = True
            elif isinstance(e, MembershipDigest):
                if self._membership.merge(e.entries):
                    members_changed = True
            elif e.kind == INSERT:
                self._tree.insert(e.hashes, e.backend_id)
            elif e.kind == REMOVE_BACKEND:
                self._tree.remove_backend(e.backend_id)
            self.applied_remote += 1
        if drains_changed:
            self._reconcile_drains()
        if members_changed:
            self._reconcile_membership()
        return len(events)

    def receive_gossip(self, raw_events) -> int:
        """Ingest raw event JSON pushed by a peer (gossip transport). Decoded
        events are queued on the bus and applied on the next sync(). Returns the
        number accepted. No-op if the transport isn't gossip."""
        ingest = getattr(self._bus, "ingest", None)
        if ingest is None:
            return 0
        decoded = []
        for raw in raw_events or []:
            try:
                decoded.append(decode_event(raw))
            except Exception:
                continue
        ingest(decoded)
        return len(decoded)

    def close(self) -> None:
        self._bus.close()
        if self._store is not None:
            self._store.close()
