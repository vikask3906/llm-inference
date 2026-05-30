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
