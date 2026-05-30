from __future__ import annotations

"""Chained block hashing that mirrors vLLM's automatic prefix caching.

Each block's hash depends on the previous block's hash, so block i is a cache
hit only if the ENTIRE prefix up to block i is identical -- exactly vLLM's
prefix-exact, block-structured semantics. MVP hashes characters; Phase 2 will
hash real token blocks.
"""

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = 0xFFFFFFFFFFFFFFFF


def _fnv1a(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK
    return h


def stable_seed(s: str) -> int:
    """Deterministic, process-independent seed from a string (e.g. a tenant id).

    Python's built-in hash() is randomized per process, which would make multiple
    gateway replicas disagree -- this stays stable across processes.
    """
    return _fnv1a(s.encode("utf-8", "ignore"))


def block_hashes(prompt: str, block_chars: int, cutoff_blocks: int,
                 seed: int = 0) -> list[int]:
    """Return the chained hash of each FULL block, truncated to cutoff_blocks.

    Partial trailing blocks are dropped: vLLM only caches full blocks, so a
    partial block is not a stable cache key. A non-zero `seed` namespaces the
    whole chain (used for per-tenant prefix-cache isolation).
    """
    raw = prompt.encode("utf-8", "ignore")
    n_full = len(raw) // block_chars
    n = min(n_full, cutoff_blocks)
    out: list[int] = []
    prev = seed & _MASK
    for i in range(n):
        chunk = raw[i * block_chars:(i + 1) * block_chars]
        # fold the previous chain hash into this block's input -> chaining
        prev = _fnv1a(prev.to_bytes(8, "little") + chunk)
        out.append(prev)
    return out
