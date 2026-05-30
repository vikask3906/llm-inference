"""Disaggregated prefill/decode routing (Splitwise / DistServe style).

LLM inference has two phases with opposite resource profiles:

  * prefill -- processes the whole prompt in one compute-bound forward pass,
    produces the first token and the prompt's KV cache. Cost scales with the
    UNCACHED prompt length. Saturates GPU compute.
  * decode  -- generates the rest of the tokens one at a time, memory-bandwidth
    bound. Cost scales with the OUTPUT length. Leaves compute idle.

Co-locating both phases on one GPU means a long prefill head-of-line-blocks
short decodes (and vice versa). Disaggregation runs them on separate pools and
ships the KV cache from the prefill node to the decode node over a fast
interconnect. This package decides, per request, whether disaggregating beats
co-locating given current load -- and which prefill/decode backends to pick.

It is deliberately standalone: nothing in the gateway hot path imports it, so
the validated single-pool benchmark is untouched. The real KV handoff requires
a vLLM KV connector (NIXL / LMCache) on actual GPUs; `kv_transfer` models the
handoff analytically so the routing logic can be developed and tested offline.
"""

from .config import DisaggConfig
from .kv_transfer import kv_transfer_ms
from .pools import PoolRegistry, parse_pool_config
from .router import DisaggDecision, choose_disaggregated

__all__ = [
    "DisaggConfig",
    "kv_transfer_ms",
    "PoolRegistry",
    "parse_pool_config",
    "DisaggDecision",
    "choose_disaggregated",
]
