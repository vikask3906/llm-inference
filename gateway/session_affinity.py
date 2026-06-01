from __future__ import annotations

"""Session-affinity routing for agentic / multi-turn workflows.

An agent loop is: model emits tool call -> tool runs -> result appended -> model
called again with [everything so far]. So turn N's prompt = turn N-1's prompt
plus a bit, and every turn shares a long, growing common prefix. The prefix tree
*usually* keeps these on the same backend -- but the FIRST turn has a short
prefix, and under cache eviction / load spill subsequent turns can scatter,
forcing the backend to re-prefill the entire growing context.

Session-affinity solves that explicitly: a request that carries a session id
(header `X-Session-ID` or `session_id` in the body) is routed to the backend
that served the previous turn of the same session, guaranteeing KV reuse across
the whole agent run. Falls back to the normal cost-function pick when:
  * no session id (per-request routing, unchanged), or
  * the pinned backend is no longer in the candidate set (down, drained,
    circuit-open, removed) -- the new choice is then re-pinned for next turn.

LRU-bounded; standalone; default OFF.
"""

from collections import OrderedDict
import time


class SessionAffinity:
    def __init__(self, capacity: int = 10_000, ttl_s: float = 1800.0) -> None:
        self.capacity = max(1, capacity)
        self.ttl_s = ttl_s
        # session_id -> (backend_id, last_seen_monotonic)
        self._by_session: "OrderedDict[str, tuple[str, float]]" = OrderedDict()

    def pick(self, session_id: str | None, candidates: list[str],
             now: float | None = None) -> str | None:
        """Return the pinned backend if the session has one AND it's still
        eligible; otherwise None (caller falls through to normal routing)."""
        if not session_id:
            return None
        rec = self._by_session.get(session_id)
        if rec is None:
            return None
        backend_id, last = rec
        now = now if now is not None else time.monotonic()
        if now - last > self.ttl_s:                  # session went idle, drop the pin
            self._by_session.pop(session_id, None)
            return None
        if backend_id not in candidates:             # pinned backend ineligible
            return None
        self._by_session.move_to_end(session_id)     # LRU touch
        return backend_id

    def remember(self, session_id: str | None, backend_id: str,
                 now: float | None = None) -> None:
        """Record (or refresh) the session's backend pin."""
        if not session_id:
            return
        now = now if now is not None else time.monotonic()
        self._by_session[session_id] = (backend_id, now)
        self._by_session.move_to_end(session_id)
        while len(self._by_session) > self.capacity:
            self._by_session.popitem(last=False)     # LRU evict

    def forget_backend(self, backend_id: str) -> int:
        """Drop every pin to a backend (called on backend remove). Returns the
        number of sessions evicted."""
        gone = [s for s, (b, _) in self._by_session.items() if b == backend_id]
        for s in gone:
            self._by_session.pop(s, None)
        return len(gone)

    def __len__(self) -> int:
        return len(self._by_session)


def extract_session_id(headers, body: dict | None = None) -> str | None:
    """Pull a session id from `X-Session-ID` (preferred) or `body['session_id']`."""
    sid = headers.get("x-session-id") or headers.get("X-Session-ID")
    if sid:
        return str(sid).strip() or None
    if isinstance(body, dict):
        s = body.get("session_id")
        if s:
            return str(s).strip() or None
    return None
