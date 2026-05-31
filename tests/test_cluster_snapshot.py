"""Durable drain snapshot + warm-start (gateway/cluster/snapshot).

Live drain rides pub/sub, which has no replay -- a restarting or late-joining
replica would miss it. The snapshot store persists drain state so boot-time
`warm_start()` seeds it. Headline: a replica that joins AFTER a drain still
keeps the backend out of rotation.
"""

from gateway.backends import BackendRegistry
from gateway.cluster import (
    ClusterCoordinator,
    InMemoryBroker,
    InMemoryBus,
    InMemorySnapshotStore,
)
from gateway.config import Config
from gateway.radix_tree import RadixTree

CAP = 2000
MODEL = "mock-model"


def _registry(spec="b0=http://b0:9000,b1=http://b1:9000"):
    cfg = Config()
    cfg.backends = spec
    return BackendRegistry(cfg)


def _coord(broker, store, rid, registry):
    return ClusterCoordinator(RadixTree(CAP), InMemoryBus(broker, rid), rid,
                              registry=registry, store=store)


# --- store basics -----------------------------------------------------------

def test_store_records_and_clears():
    s = InMemorySnapshotStore()
    s.record_drain("b0", True)
    s.record_drain("b1", True)
    assert s.drained() == {"b0", "b1"}
    s.record_drain("b0", False)
    assert s.drained() == {"b1"}


# --- warm-start -------------------------------------------------------------

def test_warm_start_seeds_drain_from_store():
    shared: dict = {}
    store = InMemorySnapshotStore(shared)
    store.record_drain("b0", True)          # a drain persisted by an earlier life

    reg = _registry()
    coord = ClusterCoordinator(RadixTree(CAP), InMemoryBus(InMemoryBroker(), "X"),
                               "X", registry=reg, store=store)
    assert "b0" in reg.ids_for(MODEL)       # not applied until warm_start
    applied = coord.warm_start()
    assert applied == {"b0"}
    assert "b0" not in reg.ids_for(MODEL)   # drained on boot
    assert reg.get("b0").draining is True


def test_warm_start_noop_without_store_or_registry():
    coord = ClusterCoordinator(RadixTree(CAP), InMemoryBus(InMemoryBroker(), "X"), "X")
    assert coord.warm_start() == set()      # nothing to do, no crash


# --- the headline: a LATE-JOINING replica doesn't miss a drain --------------

def test_late_joiner_inherits_drain_via_snapshot():
    broker = InMemoryBroker()
    shared: dict = {}                        # the durable store, shared across replicas

    # Replica A drains b0 (write-through to the shared store + live publish).
    regA = _registry()
    coordA = _coord(broker, InMemorySnapshotStore(shared), "A", regA)
    regA.set_draining("b0", True)
    coordA.publish_drain("b0", True)

    # Replica C starts up LATER -- it never received the live event...
    regC = _registry()
    coordC = _coord(broker, InMemorySnapshotStore(shared), "C", regC)
    coordC.sync()                            # nothing for it on the bus (joined late)
    assert "b0" in regC.ids_for(MODEL)       # so live propagation alone misses it
    # ...but warm-start from the durable snapshot catches it up.
    coordC.warm_start()
    assert "b0" not in regC.ids_for(MODEL)


def test_undrain_write_through_clears_snapshot():
    shared: dict = {}
    reg = _registry()
    coord = _coord(InMemoryBroker(), InMemorySnapshotStore(shared), "A", reg)
    coord.publish_drain("b0", True)
    assert "b0" in InMemorySnapshotStore(shared).drained()
    coord.publish_drain("b0", False)
    assert "b0" not in InMemorySnapshotStore(shared).drained()
