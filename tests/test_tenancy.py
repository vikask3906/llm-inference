from gateway.hashing import block_hashes
from gateway.tenancy import (DEFAULT_TIERS, RateLimiter, Tenant, TenantRegistry,
                             TokenBucket, tenant_seed)


# --- token bucket ---

def test_bucket_consume_then_empty():
    b = TokenBucket(capacity=10, refill_per_sec=5)
    assert b.try_consume(10, now=0)
    assert not b.try_consume(1, now=0)


def test_bucket_refills_over_time():
    b = TokenBucket(10, 5)
    b.try_consume(10, now=0)
    assert b.try_consume(5, now=1.0)        # 5 tokens/sec * 1s
    assert not b.try_consume(1, now=1.0)


def test_bucket_caps_at_capacity():
    b = TokenBucket(10, 5)
    b.try_consume(10, now=0)
    assert b.try_consume(10, now=100.0)     # would refill 500 but capped at 10
    assert not b.try_consume(1, now=100.0)


def test_bucket_deficit_seconds():
    b = TokenBucket(10, 5)
    b.try_consume(10, now=0)
    assert b.deficit_seconds(5, now=0) == 1.0
    assert b.deficit_seconds(0, now=0) == 0.0


# --- tenant resolution ---

def test_resolve_bearer_key_and_anonymous_fallback():
    reg = TenantRegistry("sk-acme=acme:gold,sk-beta=beta:silver")
    assert reg.resolve({"authorization": "Bearer sk-acme"}).id == "acme"
    assert reg.resolve({"Authorization": "Bearer sk-beta"}).id == "beta"
    assert reg.resolve({}).id == "anonymous"
    assert reg.resolve({"authorization": "Bearer nope"}).id == "anonymous"


def test_tier_quotas_applied():
    t = TenantRegistry("sk-acme=acme:gold").resolve({"authorization": "Bearer sk-acme"})
    assert (t.rps, t.tps) == (DEFAULT_TIERS["gold"].rps, DEFAULT_TIERS["gold"].tps)


# --- rate limiter ---

def test_rps_limit_then_throttle():
    rl = RateLimiter()
    t = Tenant("t", rps=2, tps=1_000_000, max_inflight=100)
    assert rl.admit(t, 1, now=0).allowed
    assert rl.admit(t, 1, now=0).allowed
    adm = rl.admit(t, 1, now=0)
    assert not adm.allowed and adm.reason == "rps"


def test_tps_limit_then_throttle():
    rl = RateLimiter()
    t = Tenant("t", rps=1000, tps=100, max_inflight=100)
    adm = rl.admit(t, 1000, now=0)             # cost exceeds TPS capacity
    assert not adm.allowed and adm.reason == "tps"


def test_tps_refunded_when_rps_fails():
    rl = RateLimiter()
    t = Tenant("t", rps=1, tps=1000, max_inflight=100)
    assert rl.admit(t, 100, now=0).allowed     # consumes 1 rps + 100 tps -> tps 900
    adm = rl.admit(t, 100, now=0)              # rps empty -> reject, tps refunded
    assert not adm.allowed and adm.reason == "rps"
    assert adm.remaining_tps == 900            # not 800 -> the reservation was refunded


def test_inflight_cap():
    rl = RateLimiter()
    t = Tenant("t", rps=1000, tps=1_000_000, max_inflight=2)
    assert rl.admit(t, 1, now=0).allowed
    assert rl.admit(t, 1, now=0).allowed
    adm = rl.admit(t, 1, now=0)
    assert not adm.allowed and adm.reason == "inflight"
    rl.release(t, 0, 0)                         # free a slot
    assert rl.admit(t, 1, now=0).allowed


def test_release_reconciles_overestimate():
    rl = RateLimiter()
    t = Tenant("t", rps=1000, tps=1000, max_inflight=10)
    rl.admit(t, 300, now=0)                     # reserve 300 -> tps 700
    rl.release(t, reserved_output=250, actual_output=50)  # used 200 fewer -> refund 200
    _, tps_b = rl._buckets(t)
    assert tps_b.tokens == 900


# --- prefix isolation seed ---

def test_tenant_seed_namespaces_hashes():
    sa, sb = tenant_seed("acme"), tenant_seed("beta")
    assert sa == tenant_seed("acme")           # deterministic
    assert sa != sb
    p = "X" * 200
    assert block_hashes(p, 64, 100, seed=sa) != block_hashes(p, 64, 100, seed=sb)
