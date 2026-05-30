from __future__ import annotations

"""Unit tests for the speculative routing extension.

`top_k_backends` is tested against the real Router. `race` is tested with
an httpx MockTransport so we can deterministically delay one backend and
verify the faster one wins.
"""

import asyncio

import httpx
import pytest

from gateway.config import Config
from gateway.extensions.speculative import race, top_k_backends
from gateway.load_tracker import LoadTracker
from gateway.radix_tree import RadixTree
from gateway.router import Router


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def router_with_state(cfg):
    tree = RadixTree(cfg.backend_cache_blocks)
    load = LoadTracker()
    return Router(cfg, tree, load), tree, load


def test_top_k_returns_k_distinct_backends(router_with_state):
    r, _, _ = router_with_state
    res = top_k_backends(r, "hello world " * 50, ["b0", "b1", "b2"], k=2)
    assert len(res.backend_ids) == 2
    assert len(set(res.backend_ids)) == 2


def test_top_k_falls_back_when_only_one_survives(router_with_state, cfg):
    r, _, load = router_with_state
    # saturate b1 and b2 -> only b0 is a real candidate
    load.kv_usage["b1"] = 0.99
    load.kv_usage["b2"] = 0.99
    res = top_k_backends(r, "hi", ["b0", "b1", "b2"], k=3)
    assert res.backend_ids == ["b0"]


def test_top_k_prefers_backend_with_cached_prefix(router_with_state, cfg):
    r, tree, _ = router_with_state
    # warm b1 with a prefix that matches our request
    p = "shared prompt " * 40
    from gateway.hashing import block_hashes
    hs = block_hashes(p, cfg.block_chars, cfg.hash_cutoff_blocks)
    tree.insert(hs, "b1")
    res = top_k_backends(r, p, ["b0", "b1", "b2"], k=2)
    # b1 should be the top-ranked (lowest est-TTFT) backend
    assert res.backend_ids[0] == "b1"


def test_top_k_records_hash_mode(router_with_state):
    r, _, _ = router_with_state
    res = top_k_backends(r, "x" * 200, ["b0", "b1"], k=2)
    assert res.hash_mode == "char"


@pytest.mark.asyncio
async def test_race_faster_backend_wins():
    """Mock two backends; b1 responds immediately, b0 after a delay.
    Winner must be b1 and b0 must end up in loser_ids."""
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, text="slow")

    async def fast_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="fast")

    def dispatch(request: httpx.Request):
        if "9001" in str(request.url):
            return slow_handler(request)
        return fast_handler(request)

    transport = httpx.MockTransport(dispatch)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await race(client,
                         candidates=[("b0", "http://x:9001"),
                                     ("b1", "http://x:9002")],
                         body={"model": "m", "messages": []})
        assert out is not None
        assert out.winner_id == "b1"
        assert out.loser_ids == ["b0"]
        # caller must close winner stream
        await out.winner_cm.__aexit__(None, None, None)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_race_returns_none_when_all_fail():
    def fail_all(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    transport = httpx.MockTransport(fail_all)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await race(client,
                         candidates=[("b0", "http://x:9001"),
                                     ("b1", "http://x:9002")],
                         body={"model": "m", "messages": []})
        assert out is None


@pytest.mark.asyncio
async def test_race_empty_candidates_returns_none():
    async with httpx.AsyncClient() as client:
        out = await race(client, candidates=[], body={})
        assert out is None
