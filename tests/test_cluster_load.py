"""Fleet-wide load sharing (gateway/cluster load replication).

Headline property: a backend already busy on a PEER replica is treated as busy
here too, so two gateways don't independently stampede the same "least-loaded"
node. Also covers peer-shed (circuit) propagation into routing.
"""

from gateway.cluster import (
    ClusterCoordinator,
    FleetLoadView,
    InMemoryBroker,
    InMemoryBus,
    LoadEvent,
)
from gateway.config import Config
from gateway.load_tracker import LoadTracker
from gateway.radix_tree import RadixTree
from gateway.router import Router

CAP = 2000


def _coord(broker, rid, tree):
    return ClusterCoordinator(tree, InMemoryBus(broker, rid), rid)


# --- serialization + view ---------------------------------------------------

def test_load_event_roundtrip():
    e = LoadEvent("A", 3, {"b0": 5, "b1": 0}, ["b2"])
    back = LoadEvent.from_json(e.to_json())
    assert back.origin == "A" and back.seq == 3
    assert back.inflight == {"b0": 5, "b1": 0} and back.unhealthy == ["b2"]


def test_fleet_view_sums_peers():
    fv = FleetLoadView()
    fv.apply(LoadEvent("A", 1, {"b0": 3, "b1": 1}, []))
    fv.apply(LoadEvent("B", 1, {"b0": 2}, ["b2"]))
    assert fv.peer_inflight("b0") == 5      # 3 + 2
    assert fv.peer_inflight("b1") == 1
    assert fv.peer_unhealthy("b2") is True
    assert fv.peer_unhealthy("b0") is False
    assert fv.peers() == 2


def test_latest_snapshot_replaces_previous():
    fv = FleetLoadView()
    fv.apply(LoadEvent("A", 1, {"b0": 9}, []))
    fv.apply(LoadEvent("A", 2, {"b0": 1}, []))   # newer snapshot from same peer
    assert fv.peer_inflight("b0") == 1            # replaced, not summed


# --- the headline: fleet-aware routing --------------------------------------

def _router_with_fleet(fleet):
    cfg = Config()
    cfg.strategy = "prefix_tree"
    r = Router(cfg, RadixTree(CAP), LoadTracker(), fleet=fleet)
    return r


def test_router_avoids_backend_busy_on_a_peer():
    fleet = FleetLoadView()
    r = _router_with_fleet(fleet)
    # Locally both backends look idle; the router would normally round-robin.
    # A peer reports b0 heavily loaded -> the router must prefer b1.
    fleet.apply(LoadEvent("peer", 1, {"b0": 20}, []))
    picks = {r.choose("hello world prompt", ["b0", "b1"]).backend_id for _ in range(6)}
    # b1 (idle fleet-wide) should win every time; b0 is busy on the peer.
    assert picks == {"b1"}


def test_without_fleet_routing_is_unchanged():
    # No fleet view -> single-replica behaviour: ignores peer state entirely.
    r = Router(Config(), RadixTree(CAP), LoadTracker())   # fleet=None
    assert r.fleet is None
    # round-robin spreads across both with no peer signal
    seen = {r.choose("p", ["b0", "b1"]).backend_id for _ in range(4)}
    assert seen == {"b0", "b1"}


def test_router_sheds_backend_a_peer_marked_unhealthy():
    fleet = FleetLoadView()
    r = _router_with_fleet(fleet)
    fleet.apply(LoadEvent("peer", 1, {}, ["b0"]))    # peer's circuit opened on b0
    picks = {r.choose("prompt text here", ["b0", "b1"]).backend_id for _ in range(5)}
    assert picks == {"b1"}                            # b0 avoided fleet-wide


# --- end-to-end through the coordinator/bus ----------------------------------

def test_load_propagates_across_replicas_via_bus():
    broker = InMemoryBroker()
    treeA, treeB = RadixTree(CAP), RadixTree(CAP)
    coordA = _coord(broker, "A", treeA)
    coordB = _coord(broker, "B", treeB)

    # A is hammering b0; it publishes its load. B drains and sees it.
    coordA.publish_load({"b0": 12}, [])
    coordB.sync()
    assert coordB.fleet.peer_inflight("b0") == 12
    # A doesn't apply its own load event (no echo).
    coordA.sync()
    assert coordA.fleet.peer_inflight("b0") == 0
