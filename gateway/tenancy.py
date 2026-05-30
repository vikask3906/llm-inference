from __future__ import annotations

"""Multi-tenant fairness: identification, token-bucket rate limiting, QoS.

Admission control sits BEFORE routing. A request must pass both a per-tenant
requests-per-second (RPS) bucket and a tokens-per-second (TPS) bucket -- mirroring
OpenAI's RPM+TPM -- plus a per-tenant in-flight cap. TPS is resource-aligned;
RPS is a cheap abuse guard. The router / est-TTFT cost function are untouched.
"""

import time
from collections import defaultdict
from dataclasses import dataclass

from .hashing import stable_seed


@dataclass(frozen=True)
class Tier:
    rps: float
    tps: float
    max_inflight: int


# Quota tiers. Production would load these from config/DB; sensible defaults here.
DEFAULT_TIERS: dict[str, Tier] = {
    "gold": Tier(rps=50, tps=100_000, max_inflight=64),
    "silver": Tier(rps=20, tps=40_000, max_inflight=24),
    "bronze": Tier(rps=5, tps=10_000, max_inflight=8),
    "anonymous": Tier(rps=2, tps=4_000, max_inflight=4),
}


@dataclass(frozen=True)
class Tenant:
    id: str
    rps: float
    tps: float
    max_inflight: int
    tier: str = "anonymous"          # tier name, kept for priority-based admission

    @classmethod
    def from_tier(cls, tenant_id: str, tier: Tier, tier_name: str = "anonymous") -> "Tenant":
        return cls(tenant_id, tier.rps, tier.tps, tier.max_inflight, tier_name)


class TokenBucket:
    """Lazy-refill token bucket: O(1), no timers. Starts full (allows a burst)."""

    __slots__ = ("capacity", "refill", "tokens", "ts")

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self.tokens = float(capacity)
        # Start at 0 (not monotonic): an idle bucket simply refills to full on first
        # use, and explicit `now=...` values in tests aren't seen as "in the past".
        self.ts = 0.0

    def _replenish(self, now: float) -> None:
        if now > self.ts:
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.refill)
            self.ts = now

    def try_consume(self, n: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        self._replenish(now)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def deficit_seconds(self, n: float, now: float | None = None) -> float:
        """Seconds until `n` tokens would be available (0 if already)."""
        now = time.monotonic() if now is None else now
        self._replenish(now)
        if self.tokens >= n:
            return 0.0
        return float("inf") if self.refill <= 0 else (n - self.tokens) / self.refill

    def adjust(self, delta: float) -> None:
        """Refund (delta>0) or extra-debit (delta<0); capped at capacity."""
        self.tokens = min(self.capacity, self.tokens + delta)


@dataclass
class Admission:
    allowed: bool
    reason: str | None = None            # "rps" | "tps" | "inflight"
    retry_after: float = 0.0
    remaining_rps: float = 0.0
    remaining_tps: float = 0.0


class TenantRegistry:
    """Resolves a request to a Tenant via the Authorization bearer key."""

    def __init__(self, spec: str = "", tiers: dict[str, Tier] | None = None) -> None:
        self._tiers = tiers or DEFAULT_TIERS
        self._by_key: dict[str, Tenant] = {}
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, rest = pair.partition("=")
            tid, _, tier_name = rest.partition(":")
            name = tier_name.strip() or "bronze"
            if name not in self._tiers:
                name = "bronze"
            tier = self._tiers[name]
            key = key.strip()
            self._by_key[key] = Tenant.from_tier(tid.strip() or key, tier, name)
        self._anon = Tenant.from_tier("anonymous", self._tiers["anonymous"], "anonymous")

    def resolve(self, headers) -> Tenant:
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if auth[:7].lower() == "bearer ":
            key = auth[7:].strip()
            if key in self._by_key:
                return self._by_key[key]
        return self._anon


class RateLimiter:
    """Per-tenant RPS + TPS token buckets and an in-flight cap."""

    def __init__(self) -> None:
        self._rps: dict[str, TokenBucket] = {}
        self._tps: dict[str, TokenBucket] = {}
        self.inflight: dict[str, int] = defaultdict(int)

    def _buckets(self, t: Tenant) -> tuple[TokenBucket, TokenBucket]:
        if t.id not in self._rps:
            self._rps[t.id] = TokenBucket(max(1.0, t.rps), t.rps)
            self._tps[t.id] = TokenBucket(t.tps, t.tps)
        return self._rps[t.id], self._tps[t.id]

    def admit(self, t: Tenant, cost_tokens: float, now: float | None = None) -> Admission:
        rps_b, tps_b = self._buckets(t)
        if self.inflight[t.id] >= t.max_inflight:
            return Admission(False, "inflight", retry_after=1.0,
                             remaining_rps=rps_b.tokens, remaining_tps=tps_b.tokens)
        # TPS first (the resource-aligned limit); refund if RPS then fails.
        if not tps_b.try_consume(cost_tokens, now):
            return Admission(False, "tps", tps_b.deficit_seconds(cost_tokens, now),
                             remaining_rps=rps_b.tokens, remaining_tps=tps_b.tokens)
        if not rps_b.try_consume(1, now):
            tps_b.adjust(cost_tokens)            # give the reserved tokens back
            return Admission(False, "rps", rps_b.deficit_seconds(1, now),
                             remaining_rps=rps_b.tokens, remaining_tps=tps_b.tokens)
        self.inflight[t.id] += 1
        return Admission(True, remaining_rps=rps_b.tokens, remaining_tps=tps_b.tokens)

    def release(self, t: Tenant, reserved_output: float, actual_output: float) -> None:
        """Free an in-flight slot and reconcile the TPS bucket with reality."""
        self.inflight[t.id] = max(0, self.inflight[t.id] - 1)
        _, tps_b = self._buckets(t)
        tps_b.adjust(reserved_output - actual_output)   # refund over-reserve / debit under


def tenant_seed(tenant_id: str) -> int:
    """Hash-chain seed that namespaces a tenant's prefix cache (isolation)."""
    return stable_seed(tenant_id)
