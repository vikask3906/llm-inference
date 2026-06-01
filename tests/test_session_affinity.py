"""Session-affinity routing (gateway/session_affinity).

Headline: an agent loop carrying X-Session-ID pins every turn of one session to
the same backend, so the growing context's KV cache stays warm across N turns.
"""

from gateway.session_affinity import SessionAffinity, extract_session_id


def test_no_session_id_is_a_passthrough():
    a = SessionAffinity()
    assert a.pick(None, ["b0", "b1"]) is None
    assert a.pick("", ["b0", "b1"]) is None
    a.remember(None, "b0")
    a.remember("", "b0")
    assert len(a) == 0


def test_remember_then_pick_returns_same_backend():
    a = SessionAffinity()
    a.remember("s1", "b1", now=10.0)
    assert a.pick("s1", ["b0", "b1", "b2"], now=11.0) == "b1"   # pinned to b1


def test_falls_through_when_pinned_backend_ineligible():
    a = SessionAffinity()
    a.remember("s1", "b1", now=10.0)
    # b1 is down/drained/circuit-open -> not in the candidate set -> no pin
    assert a.pick("s1", ["b0", "b2"], now=11.0) is None


def test_ttl_evicts_idle_session():
    a = SessionAffinity(ttl_s=60.0)
    a.remember("s1", "b1", now=0.0)
    assert a.pick("s1", ["b0", "b1"], now=30.0) == "b1"        # within TTL
    assert a.pick("s1", ["b0", "b1"], now=120.0) is None        # past TTL
    assert "s1" not in [k for k in a._by_session]               # actually evicted


def test_lru_capacity_evicts_oldest():
    a = SessionAffinity(capacity=2)
    a.remember("s1", "b0", now=0.0)
    a.remember("s2", "b0", now=1.0)
    a.remember("s3", "b0", now=2.0)        # evicts s1 (oldest)
    assert a.pick("s1", ["b0"], now=3.0) is None
    assert a.pick("s2", ["b0"], now=3.0) == "b0"
    assert a.pick("s3", ["b0"], now=3.0) == "b0"


def test_pick_touches_lru_recency():
    a = SessionAffinity(capacity=2)
    a.remember("s1", "b0", now=0.0)
    a.remember("s2", "b0", now=1.0)
    a.pick("s1", ["b0"], now=2.0)          # s1 becomes most recent
    a.remember("s3", "b0", now=3.0)        # should evict s2, not s1
    assert a.pick("s1", ["b0"], now=4.0) == "b0"
    assert a.pick("s2", ["b0"], now=4.0) is None


def test_forget_backend_drops_pins_to_it():
    a = SessionAffinity()
    a.remember("s1", "b0"); a.remember("s2", "b0"); a.remember("s3", "b1")
    n = a.forget_backend("b0")
    assert n == 2
    assert a.pick("s1", ["b0", "b1"]) is None and a.pick("s2", ["b0", "b1"]) is None
    assert a.pick("s3", ["b0", "b1"]) == "b1"   # untouched


def test_extract_session_id_header_then_body():
    assert extract_session_id({"x-session-id": "abc"}, {}) == "abc"
    assert extract_session_id({"X-Session-ID": "  xyz "}, {}) == "xyz"
    assert extract_session_id({}, {"session_id": "from-body"}) == "from-body"
    assert extract_session_id({}, {}) is None
    assert extract_session_id({"x-session-id": "  "}, {}) is None     # blank -> None
