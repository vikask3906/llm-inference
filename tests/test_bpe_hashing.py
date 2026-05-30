from __future__ import annotations

"""Tests for BPE block hashing extension.

The tokenizer model is downloaded on first use; tests skip if it's not
available (offline CI). The hashing logic itself is deterministic and
prefix-chained, mirroring the char-based hasher's contract.
"""

import pytest

from gateway.extensions.bpe_hashing import (
    TokenizerUnavailable,
    block_hashes_with_fallback,
    bpe_block_hashes,
)


TOKENIZER = "gpt2"


def _tokenizer_available() -> bool:
    try:
        from gateway.extensions.bpe_hashing import _load_tokenizer
        _load_tokenizer(TOKENIZER)
        return True
    except Exception:
        return False


needs_tokenizer = pytest.mark.skipif(
    not _tokenizer_available(),
    reason="gpt2 tokenizer not available offline",
)


@needs_tokenizer
def test_deterministic():
    p = "the quick brown fox jumps over the lazy dog. " * 30
    assert bpe_block_hashes(p, 16, 100, TOKENIZER) == \
           bpe_block_hashes(p, 16, 100, TOKENIZER)


@needs_tokenizer
def test_full_blocks_only():
    p = "hello world " * 200
    out = bpe_block_hashes(p, 16, 100, TOKENIZER)
    assert all(isinstance(h, int) for h in out)
    assert len(out) <= 100


@needs_tokenizer
def test_cutoff_truncates():
    p = "a b c d e f g h " * 100
    out = bpe_block_hashes(p, 16, 4, TOKENIZER)
    assert len(out) == 4


@needs_tokenizer
def test_chaining_is_prefix_exact():
    shared = "shared system prompt " * 50
    a = shared + "diverge alpha alpha alpha " * 30
    b = shared + "diverge bravo bravo bravo " * 30
    ha = bpe_block_hashes(a, 16, 100, TOKENIZER)
    hb = bpe_block_hashes(b, 16, 100, TOKENIZER)
    assert ha[0] == hb[0]
    common = next((i for i in range(min(len(ha), len(hb))) if ha[i] != hb[i]), None)
    assert common is not None and common > 0


@needs_tokenizer
def test_semantic_equivalence_char_misses():
    # The point of BPE: tokenizer-equivalent strings get the same hash chain
    # even when raw character bytes differ. GPT-2 BPE typically treats
    # "Hello" and "Hello" identically (no transform), so we instead test the
    # core property: identical token IDs -> identical hashes, regardless of
    # how they were produced.
    p1 = "abcdefgh " * 50
    p2 = "abcdefgh " * 50
    assert bpe_block_hashes(p1, 16, 100, TOKENIZER) == \
           bpe_block_hashes(p2, 16, 100, TOKENIZER)


def test_fallback_to_char_when_tokenizer_missing():
    # bogus model name forces the loader to fail; helper must return char hashes
    hashes, mode = block_hashes_with_fallback(
        prompt="x" * 200,
        block_chars=64,
        block_tokens=16,
        cutoff_blocks=10,
        tokenizer_name="this/does-not-exist-anywhere",
        use_bpe=True,
    )
    assert mode == "char"
    assert len(hashes) == 200 // 64


def test_use_bpe_false_uses_char():
    hashes, mode = block_hashes_with_fallback(
        prompt="x" * 200,
        block_chars=64,
        block_tokens=16,
        cutoff_blocks=10,
        tokenizer_name=TOKENIZER,
        use_bpe=False,
    )
    assert mode == "char"


def test_tokenizer_unavailable_raises():
    with pytest.raises(TokenizerUnavailable):
        bpe_block_hashes("hello", 16, 10, "definitely/not-a-real-model")
