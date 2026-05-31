"""Gossip anti-entropy: LWW drain map + periodic digest reconcile.

The live drain delta can be missed by a restarted / late-joining replica. The
LWW drain map (versioned per backend) + a periodic full-state digest let any
replica converge with no central store -- and resolve concurrent drain/undrain
correctly, which a naive union cannot.
"""

from gateway.backends import BackendRegistry
from gateway.cluster import (
    ClusterCoordinator,
    DrainState,
    InMemoryBroker,
    InMemoryBus,
)
from gateway.config import Config
from gateway.radix_tree import RadixTree

CAP = 2000
MODEL = "mock-model"


def _registry(spec="b0=http://b0:9000,b1=http://b1:9000"):
    cfg = Config()
    cfg.backends = spec
    return BackendRegistry(cfg)


# --- LWW map semantics ------------------------------------------------------

def test_lww_higher_timestamp_wins():
    s = DrainState()
    assert s.apply("b0", True, ts=5, origin="A") is True
    assert s.drained() == {"b0"}
    # older undrain (ts=3) is ignored
    assert s.apply("b0", False, ts=3, origin="B") is False
    assert s.drained() == {"b0"}
    # newer undrain (ts=9) wins
    assert s.apply("b0", False, ts=9, origin="B") is True
    assert s.drained() == set()


def test_lww_tiebreak_by_origin():
    s = DrainState()
    s.apply("b0", True, ts=5, origin="A")
    # same ts, higher origin id wins the tie deterministically
    assert s.apply("b0", False, ts=5, origin="Z") is True
    assert s.drained() == set()
    assert s.apply("b0", True, ts=5, origin="A") is False   # lower origin loses


def test_merge_is_idempotent_and_commutative():
    a = DrainState(); a.apply("b0", True, 5, "A"); a.apply("b1", True, 2, "A")
    b = DrainState(); b.apply("b1", False, 7, "B")
    digest_a, digest_b = a.digest(), b.digest()
    # merge both ways -> same converged drained set
    a.merge(digest_b)
    b.merge(digest_a)
    assert a.drained() == b.drained() == {"b0"}     # b1 undrained by B's newer ts
    # re-merging changes nothing (idempotent)
    assert a.merge(digest_b) is False


# --- anti-entropy across replicas via the coordinator -----------------------

def _coord(broker, rid, reg):
    return ClusterCoordinator(RadixTree(CAP), InMemoryBus(broker, rid), rid, registry=reg)


def test_late_joiner_converges_via_digest_no_store():
    broker = InMemoryBroker()
    regA = _registry()
    coordA = _coord(broker, "A", regA)
    # A drains b0 BEFORE C exists -> the live delta only reaches current members.
    coordA.publish_drain("b0", True)

    # C joins later; it never saw the delta.
    regC = _registry()
    coordC = _coord(broker, "C", regC)
    coordC.sync()
    assert "b0" in regC.ids_for(MODEL)               # missed the live event

    # A's periodic anti-entropy digest carries the full drain map -> C converges.
    coordA.publish_drain_digest()
    coordC.sync()
    assert "b0" not in regC.ids_for(MODEL)


def test_digest_reconciles_undrain_too():
    broker = InMemoryBroker()
    regA, regB = _registry(), _registry()
    coordA = _coord(broker, "A", regA)
    coordB = _coord(broker, "B", regB)
    coordA.publish_drain("b0", True)
    coordB.sync()
    assert "b0" not in regB.ids_for(MODEL)

    # A undrains, then only a digest reaches B (e.g. B missed the live delta).
    coordA.publish_drain("b0", False)
    broker.drain("B")                                # simulate B missing the live op
    coordA.publish_drain_digest()
    coordB.sync()
    assert "b0" in regB.ids_for(MODEL)               # digest carried the undrain
