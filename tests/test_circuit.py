from gateway.circuit import CLOSED, HALF_OPEN, OPEN, CircuitBreaker


def test_opens_after_threshold():
    cb = CircuitBreaker(fail_threshold=3, cooldown_s=5.0)
    assert cb.allow("b", now=0)
    cb.record_failure("b", now=0)
    cb.record_failure("b", now=0)
    assert cb.state("b", now=0) == CLOSED       # 2 < threshold
    cb.record_failure("b", now=0)
    assert cb.state("b", now=0) == OPEN
    assert not cb.allow("b", now=0)


def test_half_open_after_cooldown():
    cb = CircuitBreaker(3, 5.0)
    for _ in range(3):
        cb.record_failure("b", now=0)
    assert cb.state("b", now=4.9) == OPEN
    assert cb.state("b", now=5.0) == HALF_OPEN
    assert cb.allow("b", now=5.0)               # probe permitted


def test_success_closes_circuit():
    cb = CircuitBreaker(3, 5.0)
    for _ in range(3):
        cb.record_failure("b", now=0)
    cb.record_success("b")
    assert cb.state("b", now=0) == CLOSED
    assert cb.allow("b", now=0)


def test_half_open_failure_reopens_and_restarts_cooldown():
    cb = CircuitBreaker(3, 5.0)
    for _ in range(3):
        cb.record_failure("b", now=0)
    assert cb.state("b", now=5.0) == HALF_OPEN
    cb.record_failure("b", now=5.0)             # probe fails
    assert cb.state("b", now=5.0) == OPEN
    assert cb.state("b", now=9.9) == OPEN       # cooldown restarted from t=5
    assert cb.state("b", now=10.0) == HALF_OPEN


def test_state_isolated_per_backend():
    cb = CircuitBreaker(2, 5.0)
    cb.record_failure("a", now=0)
    cb.record_failure("a", now=0)
    assert not cb.allow("a", now=0)
    assert cb.allow("b", now=0)                 # unaffected


def test_state_code_mapping():
    cb = CircuitBreaker(1, 5.0)
    assert cb.state_code("b", now=0) == 0       # closed
    cb.record_failure("b", now=0)
    assert cb.state_code("b", now=0) == 2       # open
    assert cb.state_code("b", now=5.0) == 1     # half_open
