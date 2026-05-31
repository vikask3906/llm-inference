"""Weighted fair queuing (gateway/fairqueue) — Start-time Fair Queuing.

Headline property: under contention, served throughput share converges to each
tenant class's weight share — not arrival order (FIFO) and not starvation
(strict priority).
"""

from collections import Counter

from gateway.fairqueue import WeightedFairQueue

WEIGHTS = {"gold": 3.0, "silver": 2.0, "bronze": 1.0}


def test_single_class_is_fifo():
    q = WeightedFairQueue()
    for i in range(5):
        q.enqueue(i, "gold", 3.0)
    assert [q.dequeue() for _ in range(5)] == [0, 1, 2, 3, 4]


def test_backlogged_share_matches_weights():
    # All three tiers fully backlogged; serve a long run and check the split.
    q = WeightedFairQueue()
    n = 600
    for i in range(n):
        for cls in WEIGHTS:
            q.enqueue((cls, i), cls, WEIGHTS[cls])
    served = Counter()
    for _ in range(600):                       # serve 600 of the 1800 enqueued
        cls, _ = q.dequeue()
        served[cls] += 1
    total_w = sum(WEIGHTS.values())            # 6
    for cls, w in WEIGHTS.items():
        expected = 600 * (w / total_w)
        assert abs(served[cls] - expected) <= 12, (cls, served[cls], expected)
    # ordering of shares follows weights
    assert served["gold"] > served["silver"] > served["bronze"]


def test_idle_class_share_goes_to_backlogged():
    # Only bronze is backlogged -> it gets everything (work-conserving), even
    # though gold has the highest weight, because gold has nothing queued.
    q = WeightedFairQueue()
    for i in range(10):
        q.enqueue(("bronze", i), "bronze", 1.0)
    served = Counter(q.dequeue()[0] for _ in range(10))
    assert served["bronze"] == 10


def test_underserved_class_catches_up():
    # gold floods first; one bronze arrives -> bronze isn't starved behind the
    # whole gold backlog, it interleaves within a bounded number of serves.
    q = WeightedFairQueue()
    for i in range(20):
        q.enqueue(("gold", i), "gold", 3.0)
    q.enqueue(("bronze", 0), "bronze", 1.0)
    order = [q.dequeue()[0] for _ in range(len(q) and 21)]
    # bronze (weight 1) should be served within ~the first few, not last
    assert order.index("bronze") <= 4


def test_higher_weight_served_first_when_tied_arrival():
    q = WeightedFairQueue()
    q.enqueue("b", "bronze", 1.0)
    q.enqueue("g", "gold", 3.0)
    # both arrive at vt=0; gold's finish (1/3) < bronze's (1/1) -> gold first
    assert q.dequeue() == "g"
    assert q.dequeue() == "b"
