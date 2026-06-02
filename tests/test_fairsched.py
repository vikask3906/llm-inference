"""AsyncFairScheduler — WFQ wired as a live async dispatch gate.

Verifies the property that matters on the hot path: when slots are scarce,
waiting requests are released in weighted-fair (SFQ) order by tenant tier, so a
flooding low tier can't jump ahead of a premium tier. Also covers the no-block
fast path, slot accounting, and cancellation safety (no leaked slots, no
deadlock). Async tests run via asyncio.run so no pytest-asyncio dependency is
needed.
"""

import asyncio

from gateway.fairsched import AsyncFairScheduler, parse_weights

WEIGHTS = {"gold": 3.0, "silver": 2.0, "bronze": 1.0}


def test_parse_weights():
    assert parse_weights("gold=3,silver=2,bronze=1") == {
        "gold": 3.0, "silver": 2.0, "bronze": 1.0}
    # bad / empty entries are skipped; non-positive dropped
    assert parse_weights("gold=3, junk ,silver=x,bronze=0,plat=5") == {
        "gold": 3.0, "plat": 5.0}
    assert parse_weights("") == {}


def test_under_capacity_acquires_immediately():
    async def go():
        s = AsyncFairScheduler(max_concurrency=2, weights=WEIGHTS)
        await s.acquire("bronze")
        await s.acquire("gold")
        assert s.active == 2
        assert s.queue_depth == 0
    asyncio.run(go())


def test_release_without_waiters_decrements():
    async def go():
        s = AsyncFairScheduler(max_concurrency=2, weights=WEIGHTS)
        await s.acquire("gold")
        assert s.active == 1
        s.release()
        assert s.active == 0
    asyncio.run(go())


def test_waiters_released_in_weighted_fair_order():
    async def go():
        s = AsyncFairScheduler(max_concurrency=1, weights=WEIGHTS)
        await s.acquire("gold")                 # the single slot is now busy
        order: list[str] = []

        async def waiter(name, tier):
            await s.acquire(tier)
            order.append(name)

        # Enqueue in an order that is NOT the fair order: bronze, silver, gold.
        # Tasks enqueue in creation order on the first scheduler turn.
        tasks = [
            asyncio.create_task(waiter("bronze", "bronze")),
            asyncio.create_task(waiter("silver", "silver")),
            asyncio.create_task(waiter("gold2", "gold")),
        ]
        await asyncio.sleep(0)                   # let all three park in the WFQ
        assert s.queue_depth == 3

        # Free the slot three times; each release wakes the weighted-fair winner.
        for _ in range(3):
            s.release()
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)

        # gold (w=3, finish 1/3) < silver (w=2, 1/2) < bronze (w=1, 1/1)
        assert order == ["gold2", "silver", "bronze"]
        assert s.active == 1                     # last grant inherited the slot
    asyncio.run(go())


def test_cancelled_waiter_is_skipped_no_deadlock():
    async def go():
        s = AsyncFairScheduler(max_concurrency=1, weights=WEIGHTS)
        await s.acquire("gold")                  # slot busy
        served: list[str] = []

        async def waiter(name, tier):
            await s.acquire(tier)
            served.append(name)

        doomed = asyncio.create_task(waiter("bronze", "bronze"))
        await asyncio.sleep(0)                    # bronze parks
        live = asyncio.create_task(waiter("silver", "silver"))
        await asyncio.sleep(0)                    # silver parks
        assert s.queue_depth == 2

        doomed.cancel()                           # bronze gives up while parked
        with_cancel = asyncio.gather(doomed, return_exceptions=True)
        await with_cancel

        s.release()                               # should skip cancelled bronze
        await asyncio.sleep(0)
        await live
        assert served == ["silver"]               # silver granted, no deadlock
    asyncio.run(go())


def test_throughput_share_under_saturation():
    # Saturate a 1-slot scheduler with a continuous backlog from gold and bronze;
    # over many serves gold (w=3) should win ~3x as often as bronze (w=1).
    async def go():
        s = AsyncFairScheduler(max_concurrency=1, weights=WEIGHTS)
        await s.acquire("gold")                   # occupy the slot
        served = {"gold": 0, "bronze": 0}
        done = asyncio.Event()

        async def flood(tier):
            # Each task re-queues itself: a perpetually backlogged class.
            while not done.is_set():
                await s.acquire(tier)
                served[tier] += 1
                s.release()
                await asyncio.sleep(0)

        tasks = [asyncio.create_task(flood("gold")),
                 asyncio.create_task(flood("bronze"))]
        await asyncio.sleep(0)
        s.release()                               # kick off the contended loop
        # let the two floods contend for a while
        for _ in range(400):
            await asyncio.sleep(0)
        done.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # gold should be served clearly more than bronze (weight 3:1). We assert a
        # loose lower bound to stay robust to event-loop scheduling jitter.
        assert served["gold"] > served["bronze"], served
    asyncio.run(go())


def test_server_hot_path_gate_no_deadlock_and_clean_accounting():
    # End-to-end through the real server: with a 1-slot gate, fire 5 concurrent
    # requests. They must all succeed (the gate serializes dispatch, the next
    # waiter is woken on each release) and every slot must be returned through
    # body_iter's finally -> active and queue_depth back to 0, no leak/deadlock.
    import httpx

    import gateway.server as gw
    from gateway.backends import BackendRegistry
    from gateway.load_tracker import LoadTracker
    from gateway.radix_tree import RadixTree
    from gateway.router import Router
    from mock_backend.app import create_app

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self, by_host):
            self.by_host = by_host

        async def handle_async_request(self, request):
            return await self.by_host[request.url.host].handle_async_request(request)

    saved_scheduler = gw.scheduler
    try:
        mock = create_app("b0", 50)
        gw.cfg.backends = "b0=http://b0:9000"
        gw.cfg.rate_limit_enabled = False
        gw.cfg.tenants = ""
        gw.registry = BackendRegistry(gw.cfg)
        gw.tree = RadixTree(gw.cfg.backend_cache_blocks)
        gw.load = LoadTracker()
        gw.router = Router(gw.cfg, gw.tree, gw.load)
        gw.app.state.client = httpx.AsyncClient(
            transport=_Transport({"b0": httpx.ASGITransport(app=mock)}))
        gw.scheduler = AsyncFairScheduler(max_concurrency=1, weights=WEIGHTS)

        async def go():
            async def one():
                c = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app),
                                      base_url="http://gw")
                async with c:
                    async with c.stream(
                            "POST", "/v1/chat/completions",
                            json={"model": "mock-model",
                                  "messages": [{"role": "user", "content": "hi there"}]}) as resp:
                        async for _ in resp.aiter_raw():
                            pass
                        return resp.status_code

            results = await asyncio.gather(*[one() for _ in range(5)])
            assert all(s == 200 for s in results), results
            assert gw.scheduler.active == 0, gw.scheduler.active
            assert gw.scheduler.queue_depth == 0, gw.scheduler.queue_depth

        asyncio.run(go())
    finally:
        gw.scheduler = saved_scheduler        # don't leak the gate into other tests
