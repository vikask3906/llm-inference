from __future__ import annotations

"""Mock vLLM-style backend (factory).

Independently models a KV prefix cache (contiguous-prefix LRU over block hashes,
using the SAME hashing as the gateway) so the cache-hit measurement is the
backend's own truth, not the gateway's belief. Streams an OpenAI-style SSE
response and exposes a /metrics endpoint the gateway scrapes.
"""

import asyncio
import json
import os
import time
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from gateway.config import Config
from gateway.hashing import block_hashes, stable_seed

CFG = Config()


def create_app(node_id: str = "mock", cap_blocks: int = 600,
               prefill_s_per_block: float = 0.001, token_delay_s: float = 0.002) -> FastAPI:
    app = FastAPI()
    cache: "OrderedDict[int, bool]" = OrderedDict()
    state = {"inflight": 0}

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

    @app.get("/metrics")
    async def metrics():
        return {"node": node_id, "kv_usage": len(cache) / cap_blocks, "running": state["inflight"]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        prompt = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}"
                           for m in body.get("messages", []))
        # cache_salt namespaces the prefix cache per tenant (vLLM-style isolation)
        salt = body.get("cache_salt")
        seed = stable_seed(salt) if salt else 0
        hashes = block_hashes(prompt, CFG.block_chars, CFG.hash_cutoff_blocks, seed=seed)
        total = len(hashes)
        hit_blocks = process(hashes)
        uncached = max(0, total - hit_blocks)
        cache_hit = hit_blocks > 2          # reused beyond the shared system prompt
        state["inflight"] += 1

        async def gen():
            try:
                await asyncio.sleep(uncached * prefill_s_per_block)   # simulate prefill (TTFT)
                created = int(time.time())
                for i, tok in enumerate(["Hello", " from", f" {node_id}", "."]):
                    chunk = {
                        "id": f"chatcmpl-{created}-{i}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.get("model", "mock-model"),
                        "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                    if token_delay_s:
                        await asyncio.sleep(token_delay_s)
                yield b"data: [DONE]\n\n"
            finally:
                state["inflight"] -= 1

        headers = {
            "x-prefix-cache-hit": "true" if cache_hit else "false",
            "x-cache-hit-blocks": str(hit_blocks),
            "x-total-blocks": str(total),
        }
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    return app


app = create_app(
    node_id=os.environ.get("MOCK_NODE_ID", "mock"),
    cap_blocks=int(os.environ.get("MOCK_CAP_BLOCKS", "600")),
    prefill_s_per_block=float(os.environ.get("MOCK_PREFILL_S_PER_BLOCK", "0.001")),
    token_delay_s=float(os.environ.get("MOCK_TOKEN_DELAY_S", "0.002")),
)
