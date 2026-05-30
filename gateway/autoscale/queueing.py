from __future__ import annotations

"""Erlang-C queueing math for SLO-driven capacity planning.

Model the fleet as an M/M/c queue: Poisson arrivals at rate lambda (req/s), c
identical replicas each serving at rate mu (req/s). Offered load a = lambda/mu
(Erlangs). The Erlang-C formula gives the probability an arriving request has to
wait, from which the expected queue wait follows. We use it to find the fewest
replicas that hold a wait-time SLO -- the same question a call-center sizes
agents for, applied to LLM backends.

This is an approximation: real LLM service time isn't exponential and capacity
is KV-memory-bound, not a clean rate. But M/M/c is the standard, defensible
first-order model for "how many servers meet a wait SLO," and it degrades
gracefully (it over-provisions slightly vs. deterministic service).
"""

import math


def erlang_b(c: int, a: float) -> float:
    """Blocking probability of an M/M/c/c (Erlang-B), via the stable recursion.

    B(0,a)=1; B(k,a) = a*B(k-1,a) / (k + a*B(k-1,a)). No factorials, no overflow.
    """
    if c <= 0:
        return 1.0
    b = 1.0
    for k in range(1, c + 1):
        b = (a * b) / (k + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Probability an arriving request must queue (M/M/c).

    Derived from Erlang-B: C = B / (1 - rho*(1 - B)), rho = a/c. Returns 1.0 when
    the system is unstable (a >= c): every request eventually waits.
    """
    if a <= 0.0:
        return 0.0
    if c <= a:
        return 1.0
    b = erlang_b(c, a)
    rho = a / c
    return b / (1.0 - rho * (1.0 - b))


def expected_wait_s(c: int, lam: float, mu: float) -> float:
    """Expected time a request spends waiting in queue (seconds).

    Wq = C(c,a) / (c*mu - lambda). Infinite when unstable.
    """
    if mu <= 0.0 or lam <= 0.0:
        return 0.0
    a = lam / mu
    if c <= a:
        return math.inf
    return erlang_c(c, a) / (c * mu - lam)


def min_replicas_for_wait_slo(lam: float, mu: float, slo_s: float,
                              max_replicas: int) -> int:
    """Fewest replicas whose expected queue wait is within `slo_s`.

    Starts just above the stability floor (c > a) and climbs until the SLO is
    met or `max_replicas` is hit (returns max_replicas if it can't be met).
    """
    if lam <= 0.0 or mu <= 0.0:
        return 0
    a = lam / mu
    c = max(1, math.floor(a) + 1)        # smallest stable replica count
    while c < max_replicas:
        if expected_wait_s(c, lam, mu) <= slo_s:
            return c
        c += 1
    return max_replicas
