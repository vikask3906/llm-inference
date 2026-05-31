"""Fleet-wide drain: a maintenance decision on one replica reaches the others.

Without this, `/admin/.../drain` is replica-local -- peers keep routing to a
backend an operator pulled for maintenance. The drain/undrain mutation rides the
same replication bus as prefix/load state.
"""

from gateway.backends import BackendRegistry
from gateway.cluster import ClusterCoordinator, InMemoryBroker, InMemoryBus
from gateway.config import Config
from gateway.radix_tree import RadixTree

CAP = 2000
MODEL = "mock-model"


def _registry(spec="b0=http://b0:9000,b1=http://b1:9000"):
    cfg = Config()
    cfg.backends = spec
    return BackendRegistry(cfg)


def _coord(broker, rid, registry):
    return ClusterCoordinator(RadixTree(CAP), InMemoryBus(broker, rid), rid,
                              registry=registry)


def test_drain_propagates_then_undrain_restores():
    broker = InMemoryBroker()
    regA, regB = _registry(), _registry()
    coordA = _coord(broker, "A", regA)
    coordB = _coord(broker, "B", regB)

    # Operator drains b0 on replica A (admin endpoint does this locally + publishes).
    regA.set_draining("b0", True)
    coordA.publish_drain("b0", True)

    # B doesn't know yet -> still routes to b0.
    assert "b0" in regB.ids_for(MODEL)
    # After B drains the bus, b0 is out of rotation on B too.
    coordB.sync()
    assert "b0" not in regB.ids_for(MODEL)
    assert regB.get("b0").draining is True
    assert regB.get("b0").healthy is True          # drained, not killed

    # Undrain propagates the same way.
    coordA.publish_drain("b0", False)
    coordB.sync()
    assert "b0" in regB.ids_for(MODEL)
    assert regB.get("b0").draining is False


def test_origin_does_not_reapply_its_own_drain():
    broker = InMemoryBroker()
    regA = _registry()
    _coord(broker, "B", _registry())               # a peer must exist to fan out
    coordA = _coord(broker, "A", regA)
    coordA.publish_drain("b0", True)
    # A's own event is not echoed back to A; its registry is whatever it set locally.
    assert coordA.sync() == 0


def test_member_add_remove_propagates():
    broker = InMemoryBroker()
    regA, regB = _registry(), _registry()
    coordA = _coord(broker, "A", regA)
    coordB = _coord(broker, "B", regB)

    # A adds a backend at runtime -> B learns of it via the bus.
    coordA.publish_member_add("b9", "http://b9:9000", "")
    coordB.sync()
    assert regB.get("b9") is not None
    assert regB.get("b9").url == "http://b9:9000"

    # A removes it -> B drops it too.
    coordA.publish_member_remove("b9")
    coordB.sync()
    assert regB.get("b9") is None


def test_drain_without_registry_is_noop_safe():
    # A coordinator with no registry (e.g. cluster used only for prefix state)
    # must tolerate drain events without crashing.
    broker = InMemoryBroker()
    coordA = ClusterCoordinator(RadixTree(CAP), InMemoryBus(broker, "A"), "A")
    coordB = ClusterCoordinator(RadixTree(CAP), InMemoryBus(broker, "B"), "B")  # registry=None
    coordA.publish_drain("b0", True)
    assert coordB.sync() == 1                       # applied (no-op) without error
