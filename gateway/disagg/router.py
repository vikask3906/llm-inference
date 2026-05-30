from __future__ import annotations

"""The disaggregation decision.

Given a request's prompt/output sizes and the current per-backend load, pick a
prefill backend and a decode backend, and decide whether splitting the phases
across them beats running both on one node.

Key model insight (why this is load-dependent):
  * For an unloaded fleet, disaggregation is pure overhead -- the KV handoff
    only ADDS latency -- so co-location wins. Correct: Splitwise/DistServe help
    under load, not on an idle box.
  * Under load, a long prefill queues behind other prefills AND blocks decodes
    on the same node. If a decode node is free, paying the handoff to move the
    KV there can beat waiting out the co-located queue. Disaggregate exactly
    when (decode queue saved) > (handoff cost).

Prefix-cache locality matters only for PREFILL (the decode node receives the KV
via handoff), so prefix-match bias is applied to prefill-backend selection
alone -- a property unique to the disaggregated setting.
"""

import dataclasses
from typing import Optional

from .config import DisaggConfig
from .kv_transfer import kv_transfer_ms
from .pools import PoolRegistry


@dataclasses.dataclass
class DisaggDecision:
    prefill_backend: str
    decode_backend: str
    disaggregated: bool          # True iff prefill and decode run on different nodes
    est_prefill_ms: float
    est_decode_ms: float
    est_handoff_ms: float
    est_total_ms: float
    reason: str


def _at(mapping, key, default=0):
    """Read a per-backend value, tolerating both defaultdict and plain dict."""
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return default


def _prefill_ms(b: str, prompt_tokens: int, prefill_match: dict[str, int],
                load, cfg: DisaggConfig) -> float:
    cached = prefill_match.get(b, 0) * cfg.block_tokens
    uncached = max(0, prompt_tokens - cached)
    compute = cfg.prefill_ms_per_token * uncached
    queue = _at(load.inflight, b) * cfg.prefill_service_ms
    return compute + queue


def _decode_ms(b: str, output_tokens: int, load, cfg: DisaggConfig) -> float:
    compute = cfg.decode_ms_per_token * output_tokens
    queue = _at(load.inflight, b) * cfg.decode_service_ms
    return compute + queue


def _eligible(b: str, load, cfg: DisaggConfig) -> bool:
    return (_at(load.inflight, b) <= cfg.max_inflight
            and _at(load.kv_usage, b, 0.0) <= cfg.kv_pressure_cutoff)


def _candidates(pool: list[str], load, cfg: DisaggConfig) -> list[str]:
    # Prefer un-saturated nodes; if every node in the pool is saturated, keep
    # the whole pool so we degrade to least-bad rather than failing to route.
    elig = [b for b in pool if _eligible(b, load, cfg)]
    return elig or pool


def choose_disaggregated(*, prompt_tokens: int, pools: PoolRegistry, load,
                         cfg: DisaggConfig, output_tokens: Optional[int] = None,
                         prefill_match: Optional[dict[str, int]] = None
                         ) -> Optional[DisaggDecision]:
    """Return the chosen (prefill, decode) routing, or None if no backend can
    serve the request at all."""
    output_tokens = cfg.default_output_tokens if output_tokens is None else output_tokens
    prefill_match = prefill_match or {}

    prefill_cands = _candidates(pools.prefill_pool(), load, cfg)
    decode_cands = _candidates(pools.decode_pool(), load, cfg)
    if not prefill_cands and not decode_cands:
        return None
    # Degrade gracefully if a misconfigured spec leaves one pool empty.
    if not prefill_cands:
        prefill_cands = decode_cands
    if not decode_cands:
        decode_cands = prefill_cands

    best_pb = min(prefill_cands, key=lambda b: _prefill_ms(b, prompt_tokens, prefill_match, load, cfg))
    best_pb_ms = _prefill_ms(best_pb, prompt_tokens, prefill_match, load, cfg)
    best_db = min(decode_cands, key=lambda b: _decode_ms(b, output_tokens, load, cfg))
    best_db_ms = _decode_ms(best_db, output_tokens, load, cfg)

    handoff = kv_transfer_ms(prompt_tokens, cfg.kv_bytes_per_token, cfg.link_gbps)
    disagg_handoff = 0.0 if best_pb == best_db else handoff
    disagg_total = best_pb_ms + disagg_handoff + best_db_ms

    # Best co-located node (serves both phases on one GPU, no handoff).
    coloc_pool = _candidates(pools.colocatable(), load, cfg) if pools.colocatable() else []
    coloc_choice: Optional[str] = None
    coloc_total = float("inf")
    for b in coloc_pool:
        total = (_prefill_ms(b, prompt_tokens, prefill_match, load, cfg)
                 + _decode_ms(b, output_tokens, load, cfg))
        if total < coloc_total:
            coloc_total, coloc_choice = total, b

    # Decide.
    if coloc_choice is None:
        disaggregate = True
        reason = "no co-locatable backend; pure prefill/decode split"
    elif best_pb == best_db:
        disaggregate = False
        reason = "single node is best at both phases; handoff would add cost"
    elif disagg_total + cfg.disagg_margin_ms < coloc_total:
        disaggregate = True
        reason = (f"split saves {coloc_total - disagg_total:.1f}ms vs co-location "
                  f"(handoff {disagg_handoff:.1f}ms)")
    else:
        disaggregate = False
        reason = (f"co-location wins; split would cost "
                  f"{disagg_total - coloc_total:+.1f}ms incl. {disagg_handoff:.1f}ms handoff")

    if disaggregate:
        pf, dc = best_pb, best_db
        est_pf, est_dc = best_pb_ms, best_db_ms
        ho = disagg_handoff
    else:
        node = coloc_choice if coloc_choice is not None else best_pb
        pf = dc = node
        est_pf = _prefill_ms(node, prompt_tokens, prefill_match, load, cfg)
        est_dc = _decode_ms(node, output_tokens, load, cfg)
        ho = 0.0

    return DisaggDecision(
        prefill_backend=pf,
        decode_backend=dc,
        disaggregated=(pf != dc),
        est_prefill_ms=est_pf,
        est_decode_ms=est_dc,
        est_handoff_ms=ho,
        est_total_ms=est_pf + ho + est_dc,
        reason=reason,
    )
