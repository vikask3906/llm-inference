from __future__ import annotations

"""Mock vLLM-style backend (factory).

Independently models a KV prefix cache (contiguous-prefix LRU over block hashes,
using the SAME hashing as the gateway) so the cache-hit measurement is the
backend's own truth, not the gateway's belief. Streams an OpenAI-style SSE
response and exposes a /metrics endpoint the gateway scrapes.

Beyond the prefix cache it models two more GPU-realistic effects, both OFF by
default so existing benchmarks are byte-for-byte unchanged:

  * decode length -- honours the request's ``max_tokens`` and bills a constant
    ``decode_s_per_token`` per generated token. Decode is memory-bandwidth-bound,
    so it scales with output length, unlike compute-bound prefill which scales
    with the *uncached* prompt. This is what makes disaggregation legible.
  * unreliability -- injects 503s at ``fail_rate`` and TTFT latency spikes at
    ``spike_rate`` (adding ``spike_s`` before prefill), so the circuit-breaker,
    failover and admission-control paths have something real to react to.
"""

import asyncio
import json
import os
import random
import time
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import Config
from gateway.hashing import block_hashes, stable_seed

CFG = Config()


def _decode_tokens(n: int, node_id: str) -> list[str]:
    """A length-``n`` token stream. n<=4 reproduces the original fixed output."""
    base = ["Hello", " from", f" {node_id}", "."]
    if n <= len(base):
        return base[:n]
    return base[:-1] + [f" t{i}" for i in range(n - len(base) + 1)]


def create_app(node_id: str = "mock", cap_blocks: int = 600,
               prefill_s_per_block: float = 0.001, token_delay_s: float = 0.002,
               *, default_max_tokens: int = 4, decode_s_per_token: float | None = None,
               fail_rate: float = 0.0, spike_rate: float = 0.0, spike_s: float = 0.0,
               seed: int = 0) -> FastAPI:
    app = FastAPI()
    cache: "OrderedDict[int, bool]" = OrderedDict()
    state = {"inflight": 0}
    rng = random.Random(seed)
    # decode bills per generated token; default mirrors the old per-token cadence
    decode_s = token_delay_s if decode_s_per_token is None else decode_s_per_token

    def process(hashes: list[int]) -> int:
        hit = 0
        for h in hashes:
            if h in cache:
                hit += 1
            else:
                break
        for h in hashes:
            if h in cache:
                cache.move_to_end(h)
            else:
                cache[h] = True
                if len(cache) > cap_blocks:
                    cache.popitem(last=False)
        return hit

    @app.get("/health")
    async def health():
        # Real vLLM serves /health; the gateway's scrape loop marks a backend
        # ready only on a 200 here, so without it every node stays unhealthy.
        return {"node": node_id, "status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return {"node": node_id, "kv_usage": len(cache) / cap_blocks, "running": state["inflight"]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()

        # Draw the unreliability dice synchronously, before any await (asyncio is
        # single-threaded, so the shared RNG stays deterministic per request).
        if fail_rate > 0 and rng.random() < fail_rate:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"{node_id} overloaded", "type": "server_error"}},
                headers={"Retry-After": "1"},
            )
        spike = spike_s if (spike_rate > 0 and rng.random() < spike_rate) else 0.0

        prompt = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}"
                           for m in body.get("messages", []))
        # cache_salt namespaces the prefix cache per tenant (vLLM-style isolation)
        salt = body.get("cache_salt")
        seed_ = stable_seed(salt) if salt else 0
        hashes = block_hashes(prompt, CFG.block_chars, CFG.hash_cutoff_blocks, seed=seed_)
        total = len(hashes)
        hit_blocks = process(hashes)
        uncached = max(0, total - hit_blocks)
        cache_hit = hit_blocks > 2          # reused beyond the shared system prompt

        req_max = body.get("max_tokens")
        n_tokens = int(req_max) if isinstance(req_max, int) and req_max > 0 else default_max_tokens

        state["inflight"] += 1

        async def gen():
            try:
                if spike:
                    await asyncio.sleep(spike)                         # latency spike (TTFT stall)
                await asyncio.sleep(uncached * prefill_s_per_block)    # prefill (compute-bound TTFT)
                created = int(time.time())
                for i, tok in enumerate(_decode_tokens(n_tokens, node_id)):
                    chunk = {
                        "id": f"chatcmpl-{created}-{i}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.get("model", "mock-model"),
                        "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                    if decode_s:
                        await asyncio.sleep(decode_s)                  # decode (bandwidth-bound)
                yield b"data: [DONE]\n\n"
            finally:
                state["inflight"] -= 1

        headers = {
            "x-prefix-cache-hit": "true" if cache_hit else "false",
            "x-cache-hit-blocks": str(hit_blocks),
            "x-total-blocks": str(total),
            "x-decode-tokens": str(n_tokens),
        }
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    return app


def _opt_float(name: str) -> float | None:
    v = os.environ.get(name)
    return float(v) if v is not None else None


app = create_app(
    node_id=os.environ.get("MOCK_NODE_ID", "mock"),
    cap_blocks=int(os.environ.get("MOCK_CAP_BLOCKS", "600")),
    prefill_s_per_block=float(os.environ.get("MOCK_PREFILL_S_PER_BLOCK", "0.001")),
    token_delay_s=float(os.environ.get("MOCK_TOKEN_DELAY_S", "0.002")),
    default_max_tokens=int(os.environ.get("MOCK_DEFAULT_MAX_TOKENS", "4")),
    decode_s_per_token=_opt_float("MOCK_DECODE_S_PER_TOKEN"),
    fail_rate=float(os.environ.get("MOCK_FAIL_RATE", "0.0")),
    spike_rate=float(os.environ.get("MOCK_SPIKE_RATE", "0.0")),
    spike_s=float(os.environ.get("MOCK_SPIKE_S", "0.0")),
    seed=int(os.environ.get("MOCK_SEED", "0")),
)
