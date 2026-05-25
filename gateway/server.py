from __future__ import annotations

"""OpenAI-compatible reverse proxy (the data plane).

Hot path: parse -> filter by model -> prefix-affinity + load route -> stream the
upstream SSE response back without buffering. A background loop scrapes backend
/metrics (the control-plane job) and reconciles load + health off the hot path.

This is the Python MVP. Phase 2 rewrites this hot path in Rust (Tokio/hyper) for
predictable sub-2ms added latency; the control loop stays in Python.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import BackendRegistry
from .config import Config
from .load_tracker import LoadTracker
from .radix_tree import RadixTree
from .router import Router

cfg = Config.from_env()
registry = BackendRegistry(cfg)
tree = RadixTree(cfg.backend_cache_blocks)
load = LoadTracker()
router = Router(cfg, tree, load)


def extract_prompt(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


async def scrape_loop(client: httpx.AsyncClient) -> None:
    while True:
        for b in registry.all():
            try:
                resp = await client.get(f"{b.url}/metrics", timeout=1.0)
                data = resp.json()
                load.update_scraped(b.id, float(data.get("kv_usage", 0.0)))
                registry.set_health(b.id, True)
            except Exception:
                registry.set_health(b.id, False)
                tree.remove_backend(b.id)        # membership eviction
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    task = asyncio.create_task(scrape_loop(app.state.client))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"backends": {b.id: b.healthy for b in registry.all()},
            "inflight": dict(load.inflight)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    prompt = extract_prompt(body.get("messages", []))
    strategy = request.headers.get("x-routing-strategy")

    ids = registry.ids_for(model)
    if not ids:
        return JSONResponse({"error": f"no healthy backend for model {model!r}"},
                            status_code=503)

    r = router.choose(prompt, ids, strategy=strategy)
    tree.insert(r.hashes, r.backend_id)          # commit belief
    load.on_dispatch(r.backend_id, r.tokens)

    client: httpx.AsyncClient = request.app.state.client
    cm = client.stream("POST", f"{registry.url(r.backend_id)}/v1/chat/completions", json=body)
    try:
        upstream = await cm.__aenter__()
    except Exception:
        load.on_complete(r.backend_id, r.tokens)
        registry.set_health(r.backend_id, False)
        return JSONResponse({"error": "backend unreachable"}, status_code=502)

    headers = {"x-gw-backend": r.backend_id, "x-gw-match-blocks": str(r.match_blocks)}
    if "x-prefix-cache-hit" in upstream.headers:
        headers["x-prefix-cache-hit"] = upstream.headers["x-prefix-cache-hit"]

    async def body_iter():
        # NOTE: a stream cannot be safely retried after the first byte is sent
        # (it would duplicate tokens). Retry is only valid before first byte.
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await cm.__aexit__(None, None, None)
            load.on_complete(r.backend_id, r.tokens)

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )
