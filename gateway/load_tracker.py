from __future__ import annotations

"""Real-time load accounting.

The gateway trusts its OWN in-flight counters (updated synchronously at dispatch
and completion) for routing, because scraped Prometheus metrics lag by seconds
and routing on stale snapshots causes herd behaviour. Scraped metrics are folded
in as slower ground-truth via EWMA.
"""

from collections import defaultdict


class LoadTracker:
    def __init__(self) -> None:
        self.inflight: dict[str, int] = defaultdict(int)
        self.inflight_tokens: dict[str, int] = defaultdict(int)
        self.kv_usage: dict[str, float] = defaultdict(float)  # 0..1, reconciled

    def on_dispatch(self, backend_id: str, tokens: int) -> None:
        self.inflight[backend_id] += 1
        self.inflight_tokens[backend_id] += tokens

    def on_complete(self, backend_id: str, tokens: int) -> None:
        self.inflight[backend_id] = max(0, self.inflight[backend_id] - 1)
        self.inflight_tokens[backend_id] = max(0, self.inflight_tokens[backend_id] - tokens)

    def update_scraped(self, backend_id: str, kv_usage: float, alpha: float = 0.5) -> None:
        prev = self.kv_usage[backend_id]
        self.kv_usage[backend_id] = alpha * kv_usage + (1 - alpha) * prev
