from __future__ import annotations

"""LoRA-aware routing.

Production fleets often load several LoRA adapters per backend GPU (a
SQL-finetune adapter alongside a customer-support one, alongside the base
model). Naive round-robin sends an adapter request to a backend that
doesn't have it loaded, forcing a multi-second on-demand adapter load that
craters TTFT for that request.

This module gives:
  * parse_model_spec(model_str) -- splits "base:adapter" into (base, adapter)
  * parse_adapter_config(cfg) -- reads "b0:adapter1,adapter2;b1:adapter3"
  * lora_filter(registry, base, adapter, fallback_to_base) -- returns the
    subset of healthy backends serving `base` that also have `adapter`
    loaded. If none and fallback_to_base is True, returns the full
    base-only pool (request runs slower but doesn't fail).

Backwards-compatible: when no adapter is requested or no adapters are
configured, falls through to the existing registry.ids_for() behaviour.
"""


def parse_model_spec(model_str: str | None) -> tuple[str | None, str | None]:
    """Split a request's model field into (base, adapter).

    "mock-model"             -> ("mock-model", None)
    "mock-model:summary-v2"  -> ("mock-model", "summary-v2")
    None / ""                -> (None, None)
    """
    if not model_str:
        return None, None
    if ":" not in model_str:
        return model_str, None
    base, _, adapter = model_str.partition(":")
    base = base.strip() or None
    adapter = adapter.strip() or None
    return base, adapter


def parse_adapter_config(spec: str) -> dict[str, set[str]]:
    """Parse "b0:adapter1,adapter2;b1:adapter3" into a {backend_id: {adapters}}
    map. Whitespace is tolerated; malformed segments are silently dropped."""
    out: dict[str, set[str]] = {}
    if not spec:
        return out
    for seg in spec.split(";"):
        seg = seg.strip()
        if not seg or ":" not in seg:
            continue
        bid, _, adapters = seg.partition(":")
        bid = bid.strip()
        if not bid:
            continue
        adapter_set = {a.strip() for a in adapters.split(",") if a.strip()}
        if adapter_set:
            out[bid] = adapter_set
    return out


def apply_adapter_config(registry, spec: str) -> None:
    """Attach parsed adapters to existing Backend objects. Unknown backend
    ids in the spec are silently ignored."""
    parsed = parse_adapter_config(spec)
    for b in registry.all():
        if b.id in parsed:
            b.adapters = parsed[b.id]


def lora_filter(registry, base: str | None, adapter: str | None,
                fallback_to_base: bool) -> list[str]:
    """Filter healthy backends serving `base` to those that also have
    `adapter` loaded. Returns the unfiltered base-pool if adapter is None.

    Fallback policy:
      fallback_to_base=True  -> if zero adapter-capable backends, route to
                                the base pool anyway (slower first call,
                                but the request still completes).
      fallback_to_base=False -> return empty list; caller returns 503.
    """
    base_pool = registry.ids_for(base)
    if not adapter:
        return base_pool
    capable = [
        bid for bid in base_pool
        if adapter in registry._by_id[bid].adapters
    ]
    if capable:
        return capable
    return base_pool if fallback_to_base else []
