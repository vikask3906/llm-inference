from __future__ import annotations

"""BPE token-ID block hashing.

Replaces the char-based prefix hasher with one that hashes blocks of real
token IDs, matching how vLLM keys its KV cache. Char hashing miscounts:
"Hello, world!" and "Hello,  world!" tokenize to the same IDs but produce
different char-block hashes, so the gateway would miss a prefix match the
backend actually has.

Same chained FNV-1a scheme as `gateway.hashing` so behaviour is identical
modulo the input units (tokens vs chars). API parity with `block_hashes`
makes this a drop-in swap controlled by `Config.use_bpe_hashing`.

Tokenizer is lazy-loaded on first use to keep import-time cost zero when the
feature is off.
"""

import os
import threading

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = 0xFFFFFFFFFFFFFFFF

_tokenizer = None
_tokenizer_name = None
_lock = threading.Lock()


class TokenizerUnavailable(RuntimeError):
    """Raised when BPE hashing is enabled but `tokenizers` is not importable
    or the requested model cannot be loaded. Callers should fall back to the
    char-based hasher rather than crash the request."""


def _load_tokenizer(name: str):
    global _tokenizer, _tokenizer_name
    with _lock:
        if _tokenizer is not None and _tokenizer_name == name:
            return _tokenizer
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise TokenizerUnavailable(
                "tokenizers not installed; pip install 'tokenizers>=0.20'"
            ) from e
        try:
            tok = Tokenizer.from_pretrained(name)
        except Exception as e:
            raise TokenizerUnavailable(
                f"could not load tokenizer '{name}': {e}"
            ) from e
        _tokenizer = tok
        _tokenizer_name = name
        return tok


def tokenize_ids(prompt: str, tokenizer_name: str) -> list[int]:
    """Return the BPE token-ID sequence for `prompt`. Raises
    TokenizerUnavailable on import or model-load failure."""
    tok = _load_tokenizer(tokenizer_name)
    enc = tok.encode(prompt, add_special_tokens=False)
    return list(enc.ids)


def count_tokens(prompt: str, tokenizer_name: str) -> int:
    """Exact BPE token count for `prompt`. Raises TokenizerUnavailable on
    import/model-load failure."""
    return len(tokenize_ids(prompt, tokenizer_name))


def count_tokens_with_fallback(prompt: str, chars_per_token: int,
                               tokenizer_name: str, use_tokenizer: bool
                               ) -> tuple[int, str]:
    """Token count for quota/admission accounting. Uses the real tokenizer when
    asked (and available), else the char/token heuristic. Returns (count, mode)
    where mode is "bpe" or "char" for metrics attribution. Always >= 1."""
    if use_tokenizer:
        try:
            return max(1, count_tokens(prompt, tokenizer_name)), "bpe"
        except TokenizerUnavailable:
            pass
    return max(1, len(prompt) // max(1, chars_per_token)), "char"


def _hash_block(prev: int, ids: list[int]) -> int:
    h = _FNV_OFFSET
    for b in prev.to_bytes(8, "little"):
        h ^= b
        h = (h * _FNV_PRIME) & _MASK
    for tid in ids:
        for b in tid.to_bytes(4, "little", signed=False):
            h ^= b
            h = (h * _FNV_PRIME) & _MASK
    return h


def bpe_block_hashes(prompt: str, block_tokens: int, cutoff_blocks: int,
                     tokenizer_name: str, seed: int = 0) -> list[int]:
    """Same contract as `gateway.hashing.block_hashes`, but operates on token
    IDs. Returns chained hashes of each FULL token-block, truncated to
    `cutoff_blocks`. Partial trailing blocks are dropped (vLLM only caches
    full blocks).

    Raises TokenizerUnavailable if the tokenizer cannot be loaded.
    """
    ids = tokenize_ids(prompt, tokenizer_name)
    n_full = len(ids) // block_tokens
    n = min(n_full, cutoff_blocks)
    out: list[int] = []
    prev = seed & _MASK
    for i in range(n):
        chunk = ids[i * block_tokens:(i + 1) * block_tokens]
        prev = _hash_block(prev, chunk)
        out.append(prev)
    return out


def block_hashes_with_fallback(prompt: str, block_chars: int, block_tokens: int,
                               cutoff_blocks: int, tokenizer_name: str,
                               use_bpe: bool, seed: int = 0) -> tuple[list[int], str]:
    """Try BPE, fall back to char-based on tokenizer failure. Returns
    (hashes, mode) where mode is "bpe" or "char" for metrics attribution."""
    if use_bpe:
        try:
            return bpe_block_hashes(prompt, block_tokens, cutoff_blocks,
                                    tokenizer_name, seed=seed), "bpe"
        except TokenizerUnavailable:
            pass
    from gateway.hashing import block_hashes as char_block_hashes
    return char_block_hashes(prompt, block_chars, cutoff_blocks, seed=seed), "char"
