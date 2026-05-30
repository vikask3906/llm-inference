from __future__ import annotations

"""KV-cache handoff between a prefill node and a decode node.

In a real disaggregated deployment the prefill backend computes the prompt's
KV cache and ships it to the decode backend over a fast interconnect before
decoding starts. vLLM does this via a KV connector (NIXL, LMCache, Mooncake).
That requires real GPUs and RDMA, so here we provide:

  * kv_transfer_ms(...)  -- the analytic latency of moving the KV tensors,
    used by the router to weigh disaggregation against co-location.
  * simulate_handoff(...) -- an async stand-in that sleeps for the modelled
    duration, so an end-to-end disaggregated flow can be exercised offline.
    The production path would instead await the connector's transfer future.
"""

import asyncio


def kv_transfer_ms(num_tokens: int, kv_bytes_per_token: float,
                   link_gbps: float) -> float:
    """Time to move the KV cache for `num_tokens` over a `link_gbps` GB/s link.

    bytes = num_tokens * kv_bytes_per_token
    ms    = bytes / (link_gbps * 1e9 bytes/s) * 1000
          = num_tokens * kv_bytes_per_token / (link_gbps * 1e6)
    """
    if num_tokens <= 0 or link_gbps <= 0:
        return 0.0
    return (num_tokens * kv_bytes_per_token) / (link_gbps * 1e6)


async def simulate_handoff(num_tokens: int, kv_bytes_per_token: float,
                           link_gbps: float) -> float:
    """Offline stand-in for a real KV transfer: sleeps for the modelled
    duration and returns the elapsed ms. Replace with the vLLM KV-connector
    transfer future when running on GPUs."""
    ms = kv_transfer_ms(num_tokens, kv_bytes_per_token, link_gbps)
    await asyncio.sleep(ms / 1000.0)
    return ms
