"""Tests for the enriched mock backend (mock_backend/app.py).

Covers the GPU-realistic effects added on top of the prefix cache: decode length
(``max_tokens`` -> token count + bandwidth-bound time), 503 failure injection,
and TTFT latency spikes. The defaults must leave the original behaviour intact.
"""

import asyncio
import json
import time

import httpx

from mock_backend.app import create_app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b0")


async def _post(app, **body_extra):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi there"}]}
    body.update(body_extra)
    async with _client(app) as c:
        async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
            raw = b""
            async for chunk in resp.aiter_raw():
                raw += chunk
            return resp, raw


def _content_tokens(raw: bytes) -> int:
    n = 0
    for line in raw.decode().split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        if "content" in json.loads(payload)["choices"][0]["delta"]:
            n += 1
    return n


# --- backward compatibility: defaults reproduce the original 4-token stream ---

def test_defaults_unchanged():
    async def run():
        resp, raw = await _post(create_app("b0", 600))
        assert resp.status_code == 200
        assert b"[DONE]" in raw
        assert _content_tokens(raw) == 4
        assert resp.headers["x-decode-tokens"] == "4"
        assert resp.headers.get("x-prefix-cache-hit") in ("true", "false")
        assert "x-total-blocks" in resp.headers
    asyncio.run(run())


# --- decode length: honour max_tokens ---

def test_respects_max_tokens():
    async def run():
        resp, raw = await _post(create_app("b0", 600), max_tokens=8)
        assert resp.headers["x-decode-tokens"] == "8"
        assert _content_tokens(raw) == 8
    asyncio.run(run())


def test_short_max_tokens():
    async def run():
        resp, raw = await _post(create_app("b0", 600), max_tokens=2)
        assert resp.headers["x-decode-tokens"] == "2"
        assert _content_tokens(raw) == 2
    asyncio.run(run())


def test_default_max_tokens_param():
    async def run():
        resp, raw = await _post(create_app("b0", 600, default_max_tokens=6))
        assert _content_tokens(raw) == 6
    asyncio.run(run())


# --- decode is bandwidth-bound: time scales with output length ---

def test_decode_time_scales_with_tokens():
    async def run():
        app = create_app("b0", 600, prefill_s_per_block=0.0, decode_s_per_token=0.005)

        t0 = time.perf_counter()
        await _post(app, max_tokens=2)
        short = time.perf_counter() - t0

        t0 = time.perf_counter()
        await _post(app, max_tokens=20)
        long = time.perf_counter() - t0

        assert long > short          # 18 extra tokens * 5ms is well above noise
    asyncio.run(run())


# --- failure injection ---

def test_fail_rate_returns_503():
    async def run():
        resp, _ = await _post(create_app("b0", 600, fail_rate=1.0))
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"
    asyncio.run(run())


def test_zero_fail_rate_never_fails():
    async def run():
        app = create_app("b0", 600, fail_rate=0.0)
        for _ in range(5):
            resp, _ = await _post(app)
            assert resp.status_code == 200
    asyncio.run(run())


# --- latency spike injection (adds to TTFT) ---

def test_spike_adds_latency():
    async def run():
        plain = create_app("b0", 600, prefill_s_per_block=0.0, token_delay_s=0.0)
        spiky = create_app("b0", 600, prefill_s_per_block=0.0, token_delay_s=0.0,
                           spike_rate=1.0, spike_s=0.08)

        t0 = time.perf_counter()
        await _post(plain)
        base = time.perf_counter() - t0

        t0 = time.perf_counter()
        await _post(spiky)
        spiked = time.perf_counter() - t0

        assert spiked > base + 0.04          # the ~80ms stall dominates
    asyncio.run(run())


# --- /metrics still reports the expected keys ---

def test_metrics_keys():
    async def run():
        app = create_app("b0", 600)
        await _post(app)
        async with _client(app) as c:
            m = (await c.get("/metrics")).json()
        assert set(m) == {"node", "kv_usage", "running"}
        assert m["node"] == "b0"
    asyncio.run(run())
