from __future__ import annotations

"""Durable snapshot of operator intent (which backends are drained).

Live drain/undrain rides the pub/sub bus, but pub/sub has no replay: a replica
that RESTARTS or JOINS after a drain would miss it and route to a backend under
maintenance. So drain state is also write-through to a small durable store (a
Redis set) and read once on boot (`warm_start`).

Only drain state is persisted -- it's durable operator intent. The radix tree
and load are deliberately NOT snapshotted: they're large/transient and self-heal
from a few seconds of traffic, so persisting them would be cost without benefit.
"""

from abc import ABC, abstractmethod


class SnapshotStore(ABC):
    @abstractmethod
    def record_drain(self, backend_id: str, draining: bool) -> None:
        """Persist (or clear) a backend's drained flag. Must be best-effort -- a
        store hiccup may never break routing."""

    @abstractmethod
    def drained(self) -> set[str]:
        """The set of currently-drained backend ids (read on boot)."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class InMemorySnapshotStore(SnapshotStore):
    """Process-local store. Pass a shared dict to simulate a durable store across
    replicas in tests; otherwise isolated (fine for a single-process gateway)."""

    def __init__(self, shared: dict | None = None) -> None:
        self._drained: dict = shared if shared is not None else {}

    def record_drain(self, backend_id: str, draining: bool) -> None:
        if draining:
            self._drained[backend_id] = True
        else:
            self._drained.pop(backend_id, None)

    def drained(self) -> set[str]:
        return set(self._drained)


class RedisSnapshotStore(SnapshotStore):
    """Drain set persisted as a Redis SET, survives replica restarts + lets a
    newly-joined replica warm-start. Best-effort: a Redis error degrades to
    'no persisted drains' rather than failing the gateway."""

    def __init__(self, redis_url: str, key: str) -> None:
        import redis  # lazy: only when transport=redis
        self._client = redis.Redis.from_url(redis_url)
        self._key = key

    def record_drain(self, backend_id: str, draining: bool) -> None:
        try:
            if draining:
                self._client.sadd(self._key, backend_id)
            else:
                self._client.srem(self._key, backend_id)
        except Exception:
            pass

    def drained(self) -> set[str]:
        try:
            return {x.decode() if isinstance(x, bytes) else str(x)
                    for x in self._client.smembers(self._key)}
        except Exception:
            return set()

    def close(self) -> None:  # pragma: no cover - environment dependent
        try:
            self._client.close()
        except Exception:
            pass


def make_store(cfg, shared: dict | None = None) -> SnapshotStore:
    """Construct the snapshot store named by cfg.transport (mirrors make_bus)."""
    if cfg.transport == "redis":
        return RedisSnapshotStore(cfg.redis_url, cfg.channel + ":drained")
    return InMemorySnapshotStore(shared)
