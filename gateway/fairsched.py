from __future__ import annotations

"""Async dispatch scheduler — makes Weighted Fair Queuing a *live* admission gate.

`gateway/fairqueue.WeightedFairQueue` is the pure scheduling primitive (no I/O):
it answers "given a set of waiting requests, whose runs next?" in weighted-fair
(start-time fair queuing) order. This module wraps it in an async concurrency
gate so it actually controls request flow on the hot path:

    scheduler = AsyncFairScheduler(max_concurrency=64, weights={"gold":3,...})
    await scheduler.acquire(tenant.tier)   # blocks here only when all slots busy
    try:
        ... dispatch + stream the request ...
    finally:
        scheduler.release()

Behaviour
---------
- While fewer than `max_concurrency` requests are in flight, `acquire` returns
  immediately (no queueing, zero added latency) — so under-load behaviour is
  identical to immediate dispatch.
- Once all slots are busy, further `acquire` calls park in the WFQ keyed by their
  tenant class/weight. When a slot frees, `release` hands it to the waiting
  request with the smallest virtual-finish tag — i.e. the weighted-fair winner.
  A backlogged low tier therefore can't starve a premium tier's share.

Concurrency model
------------------
FastAPI runs on a single-threaded asyncio event loop. The only suspension point
is `await fut`; every state mutation (`_active`, the heap) happens in a
synchronous span between awaits, so it's atomic under cooperative scheduling and
needs no lock. Cancellation (client disconnect) is handled so a parked or
just-granted waiter never leaks a slot or deadlocks the queue.
"""

import asyncio

from .fairqueue import WeightedFairQueue


def parse_weights(spec: str, default: float = 1.0) -> dict[str, float]:
    """Parse "gold=3,silver=2,bronze=1" -> {"gold":3.0,...}. Bad entries skipped."""
    out: dict[str, float] = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, val = pair.partition("=")
        try:
            w = float(val)
        except ValueError:
            continue
        if w > 0:
            out[name.strip()] = w
    return out


class AsyncFairScheduler:
    def __init__(self, max_concurrency: int, weights: dict[str, float] | None = None,
                 default_weight: float = 1.0, cost: float = 1.0) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self._weights = dict(weights or {})
        self._default_weight = float(default_weight)
        self._cost = float(cost)
        self._active = 0
        self._wfq = WeightedFairQueue()

    @property
    def active(self) -> int:
        """Requests currently holding a dispatch slot."""
        return self._active

    @property
    def queue_depth(self) -> int:
        """Requests parked waiting for a slot (may include not-yet-reaped
        cancelled waiters; an upper bound on live waiters)."""
        return len(self._wfq)

    def weight_for(self, cls: str) -> float:
        return self._weights.get(cls, self._default_weight)

    async def acquire(self, cls: str) -> None:
        """Take a dispatch slot, blocking in weighted-fair order if all are busy.

        Invariant: live waiters exist only while `_active == max_concurrency`
        (a slot is taken immediately whenever one is free), so a parked request
        is always eventually woken by some in-flight request's `release`.
        """
        if self._active < self.max_concurrency:
            self._active += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._wfq.enqueue(fut, cls, self.weight_for(cls), self._cost)
        try:
            await fut
        except asyncio.CancelledError:
            # If we were granted a slot in the same tick we got cancelled, the
            # request won't use it — hand it on so the queue doesn't stall.
            if fut.done() and not fut.cancelled():
                self._release_one()
            raise

    def _grant_next(self) -> bool:
        """Give a freed slot to the next *live* waiter (smallest finish tag).
        Cancelled waiters are reaped and skipped. Returns True if granted."""
        while len(self._wfq):
            fut = self._wfq.dequeue()
            if fut.cancelled():
                continue                       # waiter gave up; slot still free
            fut.set_result(None)               # hand this in-flight slot over
            return True
        return False

    def _release_one(self) -> None:
        if not self._grant_next():
            self._active = max(0, self._active - 1)

    def release(self) -> None:
        """Free a dispatch slot, waking the weighted-fair next waiter if any."""
        self._release_one()
