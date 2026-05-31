from __future__ import annotations

"""Weighted fair queuing across tenant classes (Start-time Fair Queuing).

Rate limiting + admission decide *whether* a request runs; this decides, when
requests *compete* for a scarce dispatch slot, *whose* runs next — so throughput
is shared in proportion to tier weight (e.g. gold 3 : silver 2 : bronze 1)
rather than by arrival luck (FIFO) or starving low tiers (strict priority).

Algorithm (SFQ): each enqueued request gets a virtual finish time
    finish = max(virtual_now, last_finish[class]) + cost / weight
and `dequeue` serves the smallest finish time. A class that has been served a
lot has a larger `last_finish`, so it yields to under-served classes; backlogged
classes converge to a throughput share equal to their weight share. Work-
conserving: an idle class's unused share goes to whoever is backlogged.

Pure data structure (no I/O, no async). A scheduler can wrap this to release
requests under a concurrency limit; the benchmark drives it directly.
"""

import heapq


class WeightedFairQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, object]] = []   # (finish_vt, seq, item)
        self._last_finish: dict[str, float] = {}           # class -> last finish vt
        self._vt = 0.0                                      # virtual time (last served finish)
        self._seq = 0

    def __len__(self) -> int:
        return len(self._heap)

    def enqueue(self, item: object, cls: str, weight: float, cost: float = 1.0) -> float:
        """Add `item` for tenant class `cls` (relative `weight`, `cost` units of
        work). Returns its virtual finish time. Higher weight => earlier finish
        => served sooner / more often."""
        w = weight if weight > 0 else 1e-9
        start = max(self._vt, self._last_finish.get(cls, 0.0))   # SFQ start tag
        finish = start + cost / w
        self._last_finish[cls] = finish
        heapq.heappush(self._heap, (finish, self._seq, item))
        self._seq += 1
        return finish

    def dequeue(self) -> object:
        """Pop the next item in weighted-fair order. Raises IndexError if empty."""
        finish, _, item = heapq.heappop(self._heap)
        if finish > self._vt:                  # advance virtual time to the served tag
            self._vt = finish
        return item

    def peek_class(self) -> str | None:
        """The class that would be served next, without dequeuing (for tests)."""
        if not self._heap:
            return None
        return getattr(self._heap[0][2], "cls", None)
