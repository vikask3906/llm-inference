"""Peer-to-peer gossip transport (gateway/cluster HttpGossipBus + /cluster/gossip).

Replicas push replication events directly to each other over HTTP -- no broker.
The `sender` is injected so the gossip logic is exercised in-process without
real sockets; the server endpoint is tested through the ASGI app.
"""

import asyncio

import httpx

import gateway.server as gw
from gateway.cluster import (
    ClusterCoordinator,
    HttpGossipBus,
    PrefixEvent,
)
from gateway.cluster.events import DRAIN, INSERT, decode_event
from gateway.backends import BackendRegistry
from gateway.config import Config
from gateway.radix_tree import RadixTree

CAP = 2000


def _registry(spec="b0=http://b0:9000,b1=http://b1:9000"):
    cfg = Config()
    cfg.backends = spec
    return BackendRegistry(cfg)


# --- bus logic --------------------------------------------------------------

def test_publish_flushes_to_every_peer():
    sent = []
    bus = HttpGossipBus(["http://p1", "http://p2"], "ch", "A",
                        sender=lambda peer, batch: sent.append((peer, batch)))
    bus.publish(PrefixEvent(INSERT, "b0", "A", 1, [1, 2, 3]))
    assert bus.poll() == []                       # nothing inbound yet
    assert {p for p, _ in sent} == {"http://p1", "http://p2"}   # flushed to both
    assert all(len(batch) == 1 for _, batch in sent)
    # outbox cleared after flush
    assert bus.poll() == [] and len(sent) == 2


def test_ingest_returned_on_poll_and_self_echo_dropped():
    bus = HttpGossipBus([], "ch", "A", sender=lambda p, b: None)
    bus.ingest([PrefixEvent(INSERT, "b0", "B", 1, [1]),
                PrefixEvent(INSERT, "b1", "A", 2, [2])])   # 2nd is our own echo
    got = bus.poll()
    assert len(got) == 1 and got[0].origin == "B"


def test_down_peer_does_not_raise():
    def boom(peer, batch):
        raise ConnectionError("peer down")
    bus = HttpGossipBus(["http://dead"], "ch", "A", sender=boom)
    bus.publish(PrefixEvent(INSERT, "b0", "A", 1, [1]))
    assert bus.poll() == []                       # swallowed, no crash


# --- two replicas converge via gossip (no broker, no sockets) ---------------

def _pair():
    busA = HttpGossipBus(["B"], "ch", "A", sender=lambda p, b: None)
    busB = HttpGossipBus(["A"], "ch", "B", sender=lambda p, b: None)
    # wire each bus's sender to deliver into the other's inbox (decode like HTTP would)
    busA._send = lambda peer, batch: busB.ingest([decode_event(x) for x in batch])
    busB._send = lambda peer, batch: busA.ingest([decode_event(x) for x in batch])
    regA, regB = _registry(), _registry()
    treeA, treeB = RadixTree(CAP), RadixTree(CAP)
    coordA = ClusterCoordinator(treeA, busA, "A", registry=regA)
    coordB = ClusterCoordinator(treeB, busB, "B", registry=regB)
    return (treeA, regA, coordA), (treeB, regB, coordB)


def test_prefix_insert_gossips_to_peer():
    (treeA, _, coordA), (treeB, _, coordB) = _pair()
    hashes = [101, 102, 103, 104]
    treeA.insert(hashes, "b0")
    coordA.publish_insert(hashes, "b0")
    coordA.sync()                                 # flush A's outbox -> B's inbox
    assert treeB.match(hashes) == {}              # B hasn't applied yet
    coordB.sync()                                 # B drains inbox -> applies
    assert treeB.match(hashes).get("b0") == len(hashes)


def test_drain_gossips_to_peer():
    (_, _, coordA), (_, regB, coordB) = _pair()
    coordA.publish_drain("b0", True)
    coordA.sync()
    coordB.sync()
    assert "b0" not in regB.ids_for("mock-model")


# --- server /cluster/gossip endpoint ----------------------------------------

def _gwclient():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")


def test_gossip_endpoint_ingests_and_applies():
    async def run():
        tree = RadixTree(CAP)
        gw.cluster = ClusterCoordinator(
            tree, HttpGossipBus([], "ch", "S", sender=lambda p, b: None), "S")
        try:
            async with _gwclient() as c:
                ev = PrefixEvent(INSERT, "b0", "peer", 1, [10, 11, 12]).to_json()
                r = await c.post("/cluster/gossip", json={"events": [ev]})
                assert r.status_code == 200 and r.json()["accepted"] == 1
            gw.cluster.sync()                      # apply what the endpoint queued
            assert tree.match([10, 11, 12]).get("b0") == 3
        finally:
            gw.cluster = None
    asyncio.run(run())


def test_gossip_endpoint_404_when_cluster_disabled():
    async def run():
        gw.cluster = None
        async with _gwclient() as c:
            r = await c.post("/cluster/gossip", json={"events": []})
            assert r.status_code == 404
    asyncio.run(run())


def test_gossip_endpoint_requires_secret_when_configured():
    async def run():
        gw.cluster = ClusterCoordinator(
            RadixTree(CAP), HttpGossipBus([], "ch", "S", sender=lambda p, b: None), "S")
        gw.cluster_cfg.secret = "topsecret"
        ev = PrefixEvent(INSERT, "b0", "peer", 1, [1, 2]).to_json()
        try:
            async with _gwclient() as c:
                # no secret -> 403
                assert (await c.post("/cluster/gossip", json={"events": [ev]})).status_code == 403
                # wrong secret -> 403
                bad = await c.post("/cluster/gossip", json={"events": [ev]},
                                   headers={"x-cluster-secret": "nope"})
                assert bad.status_code == 403
                # correct secret -> 200
                ok = await c.post("/cluster/gossip", json={"events": [ev]},
                                  headers={"x-cluster-secret": "topsecret"})
                assert ok.status_code == 200 and ok.json()["accepted"] == 1
        finally:
            gw.cluster_cfg.secret = ""
            gw.cluster = None
    asyncio.run(run())
