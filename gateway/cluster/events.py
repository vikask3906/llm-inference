from __future__ import annotations

"""Replication events: the mutations one gateway replica broadcasts so its peers
can keep their local radix trees in sync.

We replicate the radix tree's *writes*, not the tree itself. The two mutations
that change routing are:

  * insert         -- a backend now holds a prefix (after a dispatch)
  * remove_backend -- a backend went unhealthy and holds nothing

Each event carries the originating replica id + a per-origin sequence number so
peers can ignore their own echoes (Redis pub/sub delivers a publisher its own
messages) and so duplicates are detectable. Events are JSON for transport.
"""

import json
from dataclasses import dataclass, field

INSERT = "insert"
REMOVE_BACKEND = "remove_backend"
LOAD = "load"


@dataclass
class PrefixEvent:
    kind: str                       # INSERT | REMOVE_BACKEND
    backend_id: str
    origin: str                     # replica id that produced this event
    seq: int                        # per-origin monotonic sequence
    hashes: list[int] = field(default_factory=list)   # block hashes (INSERT only)

    def to_json(self) -> str:
        return json.dumps({
            "k": self.kind,
            "b": self.backend_id,
            "o": self.origin,
            "s": self.seq,
            "h": self.hashes,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "PrefixEvent":
        d = json.loads(raw)
        return cls(
            kind=d["k"],
            backend_id=d["b"],
            origin=d["o"],
            seq=int(d["s"]),
            hashes=list(d.get("h") or []),
        )


@dataclass
class LoadEvent:
    """A replica's snapshot of its own per-backend load + locally-failed backends.

    Peers aggregate these into a fleet-wide view so each replica routes against
    the WHOLE fleet's load (not just its own) -- closing the cross-replica
    thundering-herd hole where two gateways independently pick the same
    "least-loaded" backend.
    """

    origin: str
    seq: int
    inflight: dict[str, int] = field(default_factory=dict)   # backend_id -> in-flight
    unhealthy: list[str] = field(default_factory=list)       # backends this replica's circuit shed

    kind: str = LOAD                                         # discriminator (uniform with PrefixEvent)

    def to_json(self) -> str:
        return json.dumps({
            "k": LOAD,
            "o": self.origin,
            "s": self.seq,
            "l": self.inflight,
            "u": self.unhealthy,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "LoadEvent":
        d = json.loads(raw)
        return cls(
            origin=d["o"],
            seq=int(d["s"]),
            inflight={k: int(v) for k, v in (d.get("l") or {}).items()},
            unhealthy=list(d.get("u") or []),
        )


def decode_event(raw: str | bytes):
    """Deserialize either event type off the wire, dispatching on the 'k' tag."""
    kind = json.loads(raw).get("k")
    return LoadEvent.from_json(raw) if kind == LOAD else PrefixEvent.from_json(raw)
