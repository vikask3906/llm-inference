from gateway.load_tracker import LoadTracker


def test_dispatch_and_complete_accounting():
    lt = LoadTracker()
    lt.on_dispatch("b0", 100)
    lt.on_dispatch("b0", 50)
    assert lt.inflight["b0"] == 2
    assert lt.inflight_tokens["b0"] == 150
    lt.on_complete("b0", 100)
    assert lt.inflight["b0"] == 1
    assert lt.inflight_tokens["b0"] == 50


def test_complete_never_goes_negative():
    lt = LoadTracker()
    lt.on_complete("b0", 100)
    assert lt.inflight["b0"] == 0
    assert lt.inflight_tokens["b0"] == 0


def test_scraped_ewma():
    lt = LoadTracker()
    lt.update_scraped("b0", 1.0, alpha=0.5)        # from 0 -> 0.5
    assert lt.kv_usage["b0"] == 0.5
    lt.update_scraped("b0", 1.0, alpha=0.5)        # 0.5 -> 0.75
    assert lt.kv_usage["b0"] == 0.75
