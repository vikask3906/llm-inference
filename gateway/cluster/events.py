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
DRAIN = "drain"            # operator drained a backend on one replica -> tell peers
UNDRAIN = "undrain"
DRAIN_OP = "drainop"       # versioned (LWW) drain/undrain delta
DRAIN_DIGEST = "draindigest"   # full LWW drain map for anti-entropy
MEMBER_ADD = "member_add"      # runtime fleet membership: add a backend
MEMBER_REMOVE = "member_remove"  # ... or remove one
MEMBER_DIGEST = "member_digest"  # full LWW membership map for anti-entropy


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


@dataclass
class DrainEvent:
    """A versioned (LWW) drain/undrain op: `ts` is a Lamport timestamp,
    `origin` the deterministic tie-breaker."""

    origin: str
    seq: int
    ts: int
    backend_id: str
    draining: bool
    kind: str = DRAIN_OP

    def to_json(self) -> str:
        return json.dumps({"k": DRAIN_OP, "o": self.origin, "s": self.seq,
                           "t": self.ts, "b": self.backend_id,
                           "d": 1 if self.draining else 0}, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "DrainEvent":
        d = json.loads(raw)
        return cls(origin=d["o"], seq=int(d["s"]), ts=int(d["t"]),
                   backend_id=d["b"], draining=bool(d["d"]))


@dataclass
class DrainDigest:
    """A replica's full LWW drain map, broadcast periodically for anti-entropy."""

    origin: str
    seq: int
    entries: dict          # backend_id -> [draining, ts, owner]
    kind: str = DRAIN_DIGEST

    def to_json(self) -> str:
        return json.dumps({"k": DRAIN_DIGEST, "o": self.origin, "s": self.seq,
                           "e": self.entries}, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "DrainDigest":
        d = json.loads(raw)
        return cls(origin=d["o"], seq=int(d["s"]), entries=dict(d.get("e") or {}))


@dataclass
class MemberEvent:
    """A versioned (LWW) runtime fleet-membership change: add or remove a backend.
    `ts` is a Lamport timestamp so concurrent add/remove converge."""

    origin: str
    seq: int
    action: str            # MEMBER_ADD | MEMBER_REMOVE
    backend_id: str
    url: str = ""
    model: str = ""
    ts: int = 0

    @property
    def kind(self) -> str:
        return self.action

    def to_json(self) -> str:
        return json.dumps({"k": self.action, "o": self.origin, "s": self.seq,
                           "b": self.backend_id, "u": self.url, "m": self.model,
                           "t": self.ts}, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "MemberEvent":
        d = json.loads(raw)
        return cls(origin=d["o"], seq=int(d["s"]), action=d["k"], backend_id=d["b"],
                   url=d.get("u", ""), model=d.get("m", ""), ts=int(d.get("t", 0)))


@dataclass
class MembershipDigest:
    """A replica's full LWW membership map, broadcast periodically for anti-entropy."""

    origin: str
    seq: int
    entries: dict          # backend_id -> [present, url, model, ts, owner]
    kind: str = MEMBER_DIGEST

    def to_json(self) -> str:
        return json.dumps({"k": MEMBER_DIGEST, "o": self.origin, "s": self.seq,
                           "e": self.entries}, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "MembershipDigest":
        d = json.loads(raw)
        return cls(origin=d["o"], seq=int(d["s"]), entries=dict(d.get("e") or {}))


def decode_event(raw: str | bytes):
    """Deserialize any event type off the wire, dispatching on the 'k' tag."""
    kind = json.loads(raw).get("k")
    if kind == LOAD:
        return LoadEvent.from_json(raw)
    if kind == DRAIN_OP:
        return DrainEvent.from_json(raw)
    if kind == DRAIN_DIGEST:
        return DrainDigest.from_json(raw)
    if kind in (MEMBER_ADD, MEMBER_REMOVE):
        return MemberEvent.from_json(raw)
    if kind == MEMBER_DIGEST:
        return MembershipDigest.from_json(raw)
    return PrefixEvent.from_json(raw)
