"""Multi-replica prefix-state replication (gateway/cluster).

The headline property: two gateway replicas sharing a bus CONVERGE -- a prefix
inserted on replica A becomes routable on replica B after a sync, so prefix-aware
routing works horizontally instead of each replica routing blind.
"""

from gateway.cluster import (
    ClusterConfig,
    ClusterCoordinator,
    InMemoryBroker,
    InMemoryBus,
    PrefixEvent,
    make_bus,
)
from gateway.radix_tree import RadixTree

CAP = 2000


def _replica(broker: InMemoryBroker, rid: str):
    tree = RadixTree(CAP)
    bus = InMemoryBus(broker, rid)
    return tree, ClusterCoordinator(tree, bus, rid)


# --- serialization ----------------------------------------------------------

def test_event_json_roundtrip():
    e = PrefixEvent("insert", "b0", "rep-1", 7, [11, 22, 33])
    back = PrefixEvent.from_json(e.to_json())
    assert (back.kind, back.backend_id, back.origin, back.seq, back.hashes) == \
           ("insert", "b0", "rep-1", 7, [11, 22, 33])


# --- the headline: convergence ----------------------------------------------

def test_insert_on_A_becomes_routable_on_B():
    broker = InMemoryBroker()
    treeA, coordA = _replica(broker, "A")
    treeB, coordB = _replica(broker, "B")

    hashes = [101, 102, 103, 104]
    # A dispatches a prefix to b0 and records it locally + publishes.
    treeA.insert(hashes, "b0")
    coordA.publish_insert(hashes, "b0")

    # Before B syncs, B has no idea b0 holds this prefix -> would route blind.
    assert treeB.match(hashes) == {}

    # After B drains the bus, B's tree knows b0 holds the full prefix.
    applied = coordB.sync()
    assert applied == 1
    assert treeB.match(hashes).get("b0") == len(hashes)


def test_origin_does_not_apply_its_own_events():
    broker = InMemoryBroker()
    treeA, coordA = _replica(broker, "A")
    _replica(broker, "B")          # a peer must exist for broadcast to fan out

    coordA.publish_insert([1, 2, 3], "b0")
    # A must not receive its own event back (no echo loop).
    assert coordA.sync() == 0


def test_remove_backend_propagates():
    broker = InMemoryBroker()
    treeA, coordA = _replica(broker, "A")
    treeB, coordB = _replica(broker, "B")

    hashes = [5, 6, 7, 8]
    treeA.insert(hashes, "b1")
    coordA.publish_insert(hashes, "b1")
    coordB.sync()
    assert treeB.match(hashes).get("b1") == len(hashes)

    # b1 goes unhealthy on A; the removal must reach B.
    treeA.remove_backend("b1")
    coordA.publish_remove("b1")
    coordB.sync()
    assert "b1" not in treeB.match(hashes)


def test_three_replicas_all_converge():
    broker = InMemoryBroker()
    reps = {rid: _replica(broker, rid) for rid in ("A", "B", "C")}
    (treeA, coordA) = reps["A"]

    hashes = [9, 9, 9, 1, 2]
    treeA.insert(hashes, "b2")
    coordA.publish_insert(hashes, "b2")

    for rid in ("B", "C"):
        tree, coord = reps[rid]
        coord.sync()
        assert tree.match(hashes).get("b2") == len(hashes)
    assert coordA.published == 1


def test_bidirectional_union():
    # Each replica's tree should end up holding BOTH replicas' inserts.
    broker = InMemoryBroker()
    treeA, coordA = _replica(broker, "A")
    treeB, coordB = _replica(broker, "B")

    ha, hb = [10, 11, 12], [20, 21, 22]
    treeA.insert(ha, "b0"); coordA.publish_insert(ha, "b0")
    treeB.insert(hb, "b1"); coordB.publish_insert(hb, "b1")

    coordA.sync()
    coordB.sync()

    assert treeA.match(ha).get("b0") == 3 and treeA.match(hb).get("b1") == 3
    assert treeB.match(hb).get("b1") == 3 and treeB.match(ha).get("b0") == 3


# --- config + factory -------------------------------------------------------

def test_config_defaults_off_and_autoassigns_replica_id():
    cfg = ClusterConfig.from_env()
    assert cfg.enabled is False
    assert cfg.transport == "memory"
    assert cfg.replica_id           # auto-derived host-pid

def test_make_bus_memory_shared_broker():
    cfg = ClusterConfig(replica_id="A", transport="memory")
    broker = InMemoryBroker()
    bus = make_bus(cfg, broker)
    assert isinstance(bus, InMemoryBus)
