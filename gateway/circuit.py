from __future__ import annotations

"""Per-backend circuit breaker.

Three states:
  closed     -> traffic flows; consecutive failures are counted.
  open       -> too many failures; traffic is blocked for `cooldown_s`.
  half_open  -> cooldown elapsed; a probe is allowed. Success closes the circuit;
                failure reopens it (resetting the cooldown).

Driven by BOTH request outcomes and the control-plane scrape loop, so a backend
recovers automatically once it starts passing health checks again.
"""

import time
from collections import defaultdict

CLOSED, HALF_OPEN, OPEN = "closed", "half_open", "open"
_STATE_CODE = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, cooldown_s: float = 5.0) -> None:
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self._fails: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}

    def state(self, backend_id: str, now: float | None = None) -> str:
        opened = self._opened_at.get(backend_id)
        if opened is None:
            return CLOSED
        now = time.monotonic() if now is None else now
        return HALF_OPEN if (now - opened) >= self.cooldown_s else OPEN

    def state_code(self, backend_id: str, now: float | None = None) -> int:
        return _STATE_CODE[self.state(backend_id, now)]

    def allow(self, backend_id: str, now: float | None = None) -> bool:
        """True if a request may be routed to this backend (closed or half-open)."""
        return self.state(backend_id, now) != OPEN

    def record_success(self, backend_id: str) -> None:
        self._fails[backend_id] = 0
        self._opened_at.pop(backend_id, None)

    def record_failure(self, backend_id: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state(backend_id, now) == HALF_OPEN:
            # the probe failed -> reopen and restart the cooldown
            self._opened_at[backend_id] = now
            self._fails[backend_id] = self.fail_threshold
            return
        self._fails[backend_id] += 1
        if self._fails[backend_id] >= self.fail_threshold:
            self._opened_at[backend_id] = now
