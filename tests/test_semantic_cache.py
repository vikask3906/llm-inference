from __future__ import annotations

"""Tests use a deterministic fake embedder so no model download is needed."""

import hashlib

import numpy as np
import pytest

from gateway.extensions.semantic_cache import SemanticCache


DIM = 32


def _fake_embedder(text: str) -> np.ndarray:
    """Deterministic, normalised embedding driven by SHA-256 of the text.
    Equal strings -> identical vectors; tiny edits -> near-orthogonal.
    Good enough for verifying the cache mechanics, not for semantic ranking."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    v = rng.randn(DIM).astype("float32")
    v /= np.linalg.norm(v) + 1e-9
    return v


def test_miss_on_empty_cache():
    c = SemanticCache(_fake_embedder)
    assert c.lookup("t1", "hello world") is None
    assert c.stats()["misses"] == 1


def test_exact_match_is_a_hit():
    c = SemanticCache(_fake_embedder, similarity_threshold=0.99)
    c.store("t1", "what is the capital of France?", "Paris")
    hit = c.lookup("t1", "what is the capital of France?")
    assert hit is not None
    resp, sim = hit
    assert resp == "Paris"
    assert sim > 0.99


def test_different_prompts_miss_with_fake_embedder():
    """Fake embedder is hash-based: any byte difference -> near-orthogonal.
    Real model would give a hit on paraphrases; we test the threshold logic."""
    c = SemanticCache(_fake_embedder, similarity_threshold=0.97)
    c.store("t1", "what is the capital of France", "Paris")
    assert c.lookup("t1", "What is the capital of France") is None


def test_per_tenant_isolation():
    c = SemanticCache(_fake_embedder, similarity_threshold=0.99)
    c.store("alice", "secret", "alice-only-response")
    # bob does the same query -- shouldn't see alice's cache
    assert c.lookup("bob", "secret") is None
    # alice still sees it
    assert c.lookup("alice", "secret") is not None


def test_lru_eviction():
    c = SemanticCache(_fake_embedder, similarity_threshold=0.99,
                      max_entries_per_tenant=3)
    for i in range(5):
        c.store("t1", f"prompt-{i}", f"resp-{i}")
    # only the most recent 3 should remain
    assert c.lookup("t1", "prompt-0") is None
    assert c.lookup("t1", "prompt-1") is None
    assert c.lookup("t1", "prompt-4") is not None


def test_lru_touch_on_hit():
    c = SemanticCache(_fake_embedder, similarity_threshold=0.99,
                      max_entries_per_tenant=3)
    c.store("t1", "p0", "r0")
    c.store("t1", "p1", "r1")
    c.store("t1", "p2", "r2")
    # touch p0 -> now most-recent
    assert c.lookup("t1", "p0") is not None
    # add p3 -> p1 should be evicted (LRU), not p0
    c.store("t1", "p3", "r3")
    assert c.lookup("t1", "p1") is None
    assert c.lookup("t1", "p0") is not None


def test_stats_tracks_hits_and_misses():
    c = SemanticCache(_fake_embedder, similarity_threshold=0.99)
    c.store("t1", "q", "r")
    c.lookup("t1", "q")        # hit
    c.lookup("t1", "other")    # miss
    c.lookup("t1", "q")        # hit
    s = c.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3)


def test_failing_embedder_returns_none_no_raise():
    def boom(_text: str):
        raise RuntimeError("embedder down")
    c = SemanticCache(boom)
    # store should fail silently
    assert c.store("t1", "p", "r") is False
    # lookup should miss-silently
    assert c.lookup("t1", "p") is None


def test_threshold_filters_below_cutoff():
    """Manually inject a low-similarity entry and verify threshold rejects it."""
    c = SemanticCache(_fake_embedder, similarity_threshold=0.999)
    # store a wildly different prompt
    c.store("t1", "completely unrelated query", "wrong")
    # query something different -> similarity < threshold -> miss
    assert c.lookup("t1", "another totally distinct prompt") is None
