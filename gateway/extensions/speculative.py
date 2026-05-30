from __future__ import annotations

"""Speculative / shadow-traffic routing.

Dispatches the same request to the top-K candidate backends in parallel and
returns whichever responds first. The slower N-1 are cancelled before
streaming the body, which means the loser GPUs paid for prefill but not for
decode. Net effect: lower tail latency (p99 follows the FASTEST backend,
not the slowest), at a multiplicative compute cost. Useful when:
  - tail latency dominates the SLO,
  - the gateway sits in front of a heterogeneous fleet where one node is
    sometimes much slower (queue spike, cold cache),
  - per-request cost matters less than user-visible latency.

This module gives the two primitives:
  * top_k_backends(...) -- ranks candidates by est-TTFT (reusing the
    router's _est_ttft so speculative shares its load model).
  * race(...) -- fires concurrent upstream connects, returns the winner's
    response handle, cancels the rest.

Integration with server.py is opt-in: when cfg.strategy == "speculative"
or the header x-routing-strategy=speculative is set, swap the failover
loop for `race(top_k_backends(...))`.
"""

import asyncio
import dataclasses
from typing import Awaitable, Callable

import httpx


@dataclasses.dataclass
class TopKResult:
    """Top-K candidates ranked by est-TTFT (lowest first). Each entry's
    match_blocks lets the caller update the prefix tree's belief for the
    winner only -- losers didn't actually serve the request."""
    backend_ids: list[str]
    match_blocks: dict[str, int]
    hashes: list[int]
    tokens: int
    hash_mode: str


def top_k_backends(router, prompt: str, backends: list[str], k: int,
                   seed: int = 0) -> TopKResult:
    """Return the top-K backends by predicted TTFT. Drops saturated ones
    (same guardrails as Router.choose). Falls back to the single best
    backend when fewer than K survive the cutoff."""
    from .bpe_hashing import block_hashes_with_fallback
    from ..hashing import block_hashes

    cfg = router.cfg
    if cfg.use_bpe_hashing:
        hashes, mode = block_hashes_with_fallback(
            prompt, cfg.block_chars, cfg.block_tokens,
            cfg.hash_cutoff_blocks, cfg.tokenizer_model,
            use_bpe=True, seed=seed,
        )
    else:
        hashes = block_hashes(prompt, cfg.block_chars,
                              cfg.hash_cutoff_blocks, seed=seed)
        mode = "char"
    tokens = len(hashes) * cfg.block_tokens
    match = router.tree.match(hashes)

    scored: list[tuple[float, str, int]] = []
    saturated: list[tuple[int, str]] = []
    for b in backends:
        if router.load.kv_usage[b] > cfg.kv_pressure_cutoff or \
           router.load.inflight[b] > cfg.max_inflight:
            saturated.append((router.load.inflight[b], b))
            continue
        m = match.get(b, 0)
        scored.append((router._est_ttft(tokens, m, b), b, m))

    scored.sort(key=lambda t: t[0])
    if not scored:
        # all saturated: pick least-inflight as a single fallback
        saturated.sort()
        chosen = [saturated[0][1]] if saturated else (backends[:1])
    else:
        chosen = [b for _, b, _ in scored[:max(1, k)]]

    return TopKResult(
        backend_ids=chosen,
        match_blocks={b: match.get(b, 0) for b in chosen},
        hashes=hashes,
        tokens=tokens,
        hash_mode=mode,
    )


@dataclasses.dataclass
class RaceOutcome:
    """Result of a race. `winner_id` always set on success; `loser_ids` is
    the list whose requests were cancelled. `winner_cm` is the async-ctx
    httpx stream handle the caller must __aexit__ when done streaming."""
    winner_id: str
    winner_cm: object             # httpx async context manager
    winner_response: httpx.Response
    loser_ids: list[str]


async def race(client: httpx.AsyncClient,
               candidates: list[tuple[str, str]],   # [(backend_id, url), ...]
               body: dict,
               connect_timeout_s: float = 10.0) -> RaceOutcome | None:
    """Open POSTs to all candidates concurrently, return the first one to
    yield a response, cancel the rest. Returns None if every connection
    failed. Caller is responsible for streaming + closing winner_cm."""
    if not candidates:
        return None

    cms: dict[str, object] = {}
    tasks: dict[asyncio.Task, str] = {}

    async def open_one(bid: str, url: str):
        cm = client.stream("POST", f"{url}/v1/chat/completions",
                           json=body, timeout=connect_timeout_s)
        cms[bid] = cm
        resp = await cm.__aenter__()
        return bid, resp

    for bid, url in candidates:
        t = asyncio.create_task(open_one(bid, url))
        tasks[t] = bid

    winner_id: str | None = None
    winner_resp: httpx.Response | None = None
    failed: list[str] = []

    while tasks and winner_id is None:
        done, _pending = await asyncio.wait(
            list(tasks.keys()), return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            bid = tasks.pop(t)
            try:
                wbid, resp = t.result()
                winner_id = wbid
                winner_resp = resp
                break
            except Exception:
                failed.append(bid)
                cms.pop(bid, None)

    if winner_id is None or winner_resp is None:
        for t in list(tasks.keys()):
            t.cancel()
        for cm in cms.values():
            try:
                await cm.__aexit__(None, None, None)  # type: ignore[attr-defined]
            except Exception:
                pass
        return None

    loser_ids: list[str] = []
    for t, bid in list(tasks.items()):
        if bid == winner_id:
            continue
        t.cancel()
        loser_ids.append(bid)
    for bid, cm in list(cms.items()):
        if bid == winner_id:
            continue
        try:
            await cm.__aexit__(None, None, None)  # type: ignore[attr-defined]
        except Exception:
            pass

    return RaceOutcome(
        winner_id=winner_id,
        winner_cm=cms[winner_id],
        winner_response=winner_resp,
        loser_ids=loser_ids + failed,
    )
