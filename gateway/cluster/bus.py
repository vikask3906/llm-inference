from __future__ import annotations

"""Replication transports.

A `ReplicationBus` is a one-to-many mutation channel between gateway replicas:

  * publish(event)  -- CHEAP and non-blocking. The request hot path calls this,
                       so it must never do network I/O. Implementations buffer.
  * poll()          -- flush any buffered outbound events and return the inbound
                       events produced by OTHER replicas since the last poll.
                       Called from a background loop, never the hot path.

Two implementations:
  * InMemoryBus  -- a shared in-process broker. For tests and single-process
                    demos (no real network). Fans out to peers' inboxes.
  * RedisBus     -- Redis pub/sub for a real multi-replica deployment. Buffers
                    outbound and flushes on poll() so the hot path stays I/O-free.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque

from .events import PrefixEvent


class ReplicationBus(ABC):
    @abstractmethod
    def publish(self, event: PrefixEvent) -> None:
        """Buffer an outbound event. Cheap, non-blocking (hot-path safe)."""

    @abstractmethod
    def poll(self) -> list[PrefixEvent]:
        """Flush outbound + return inbound events from other replicas."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# --------------------------------------------------------------------------- #
# In-memory transport (tests / single process)
# --------------------------------------------------------------------------- #

class InMemoryBroker:
    """A process-local broker: each registered replica gets an inbox; a publish
    is fanned out to every OTHER replica's inbox."""

    def __init__(self) -> None:
        self._inboxes: dict[str, deque[PrefixEvent]] = defaultdict(deque)

    def register(self, replica_id: str) -> None:
        self._inboxes.setdefault(replica_id, deque())

    def broadcast(self, event: PrefixEvent) -> None:
        for rid, box in self._inboxes.items():
            if rid != event.origin:          # don't echo to the producer
                box.append(event)

    def drain(self, replica_id: str) -> list[PrefixEvent]:
        box = self._inboxes[replica_id]
        out = list(box)
        box.clear()
        return out


class InMemoryBus(ReplicationBus):
    def __init__(self, broker: InMemoryBroker, replica_id: str) -> None:
        self._broker = broker
        self._replica_id = replica_id
        broker.register(replica_id)

    def publish(self, event: PrefixEvent) -> None:
        self._broker.broadcast(event)        # cheap: appends to peer deques

    def poll(self) -> list[PrefixEvent]:
        return self._broker.drain(self._replica_id)


# --------------------------------------------------------------------------- #
# Redis transport (real multi-replica)
# --------------------------------------------------------------------------- #

class RedisBus(ReplicationBus):
    """Redis pub/sub transport. Outbound events are buffered and flushed in
    poll() so the hot path never blocks on the network; inbound events are read
    from the subscription and filtered to drop our own echoes."""

    def __init__(self, redis_url: str, channel: str, replica_id: str) -> None:
        try:
            import redis  # lazy: only needed when transport=redis
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "GW_CLUSTER_TRANSPORT=redis requires the 'redis' package "
                "(pip install redis)"
            ) from exc
        self._channel = channel
        self._replica_id = replica_id
        self._client = redis.Redis.from_url(redis_url)
        self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(channel)
        self._outbox: deque[PrefixEvent] = deque()

    def publish(self, event: PrefixEvent) -> None:
        self._outbox.append(event)           # buffer; flushed in poll()

    def poll(self) -> list[PrefixEvent]:
        # 1) flush outbound
        while self._outbox:
            ev = self._outbox.popleft()
            try:
                self._client.publish(self._channel, ev.to_json())
            except Exception:                # never let a broker hiccup break routing
                break
        # 2) drain inbound, dropping our own echoes
        inbound: list[PrefixEvent] = []
        while True:
            msg = self._pubsub.get_message(timeout=0.0)
            if not msg:
                break
            data = msg.get("data")
            if not data:
                continue
            try:
                ev = PrefixEvent.from_json(data)
            except Exception:
                continue
            if ev.origin != self._replica_id:
                inbound.append(ev)
        return inbound

    def close(self) -> None:  # pragma: no cover - environment dependent
        try:
            self._pubsub.close()
            self._client.close()
        except Exception:
            pass


def make_bus(cfg, broker: InMemoryBroker | None = None) -> ReplicationBus:
    """Construct the transport named by cfg.transport.

    For 'memory', pass a shared `broker` (multiple buses on one broker simulate
    a fleet); if omitted a fresh isolated broker is created.
    """
    if cfg.transport == "redis":
        return RedisBus(cfg.redis_url, cfg.channel, cfg.replica_id)
    return InMemoryBus(broker or InMemoryBroker(), cfg.replica_id)
