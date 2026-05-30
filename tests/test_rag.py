"""Unit tests for RAG-aware caching/routing (gateway/rag).

The central property: two queries that retrieve the SAME set of document
chunks (in any order) must produce an identical prefix, so a stock prefix
cache reuses the chunk KVs. Plus chunk-affinity routing and the LRU index.
"""

from gateway.load_tracker import LoadTracker
from gateway.rag import (
    ChunkAffinityIndex,
    RagConfig,
    build_messages,
    canonicalize_chunks,
    choose_rag_backend,
    chunk_id,
    chunk_ids,
    parse_rag_request,
)
from gateway.rag.structure import RagPrompt


# --- chunk ids ---

def test_chunk_id_is_deterministic():
    assert chunk_id("the quick brown fox") == chunk_id("the quick brown fox")
    assert chunk_id("a") != chunk_id("b")


# --- request parsing ---

def test_parse_rag_request_valid():
    body = {"rag": {"system": "You are helpful.",
                    "chunks": ["doc one", "doc two"], "query": "what?"}}
    rag = parse_rag_request(body)
    assert rag is not None
    assert rag.system == "You are helpful."
    assert rag.chunks == ["doc one", "doc two"]
    assert rag.query == "what?"


def test_parse_rag_request_non_rag_returns_none():
    assert parse_rag_request({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert parse_rag_request({"rag": {"chunks": []}}) is None
    assert parse_rag_request({"rag": {"query": "no chunks"}}) is None
    assert parse_rag_request({"rag": "not-a-dict"}) is None


def test_parse_rag_request_filters_non_string_chunks():
    rag = parse_rag_request({"rag": {"chunks": ["ok", 5, None, "", "fine"]}})
    assert rag.chunks == ["ok", "fine"]


# --- canonicalization: the core cache-sharing property ---

def test_canonicalize_is_order_independent():
    a = ["doc-x", "doc-y", "doc-z"]
    b = ["doc-z", "doc-x", "doc-y"]      # same set, different retrieval order
    assert canonicalize_chunks(a) == canonicalize_chunks(b)
    assert chunk_ids(canonicalize_chunks(a)) == chunk_ids(canonicalize_chunks(b))


def test_canonicalize_dedupes():
    assert canonicalize_chunks(["d1", "d1", "d2"]) == canonicalize_chunks(["d2", "d1"])


def test_build_messages_puts_chunks_in_system_query_in_tail():
    rag = RagPrompt(system="SYS", chunks=["c1", "c2"], query="Q")
    canon = canonicalize_chunks(rag.chunks)
    msgs = build_messages(rag, canon)
    assert msgs[0]["role"] == "system"
    assert "SYS" in msgs[0]["content"]
    assert "c1" in msgs[0]["content"] and "c2" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "Q"}


# --- chunk affinity index ---

def test_index_overlap_and_fraction():
    idx = ChunkAffinityIndex()
    idx.record("b0", [1, 2, 3])
    assert idx.overlap("b0", [2, 3, 4]) == 2
    assert idx.cached_fraction("b0", [2, 3, 4, 5]) == 0.5
    assert idx.overlap("b1", [1]) == 0       # untouched backend


def test_index_lru_eviction():
    idx = ChunkAffinityIndex(capacity_chunks=2)
    idx.record("b0", [1, 2])
    idx.record("b0", [3])                    # capacity 2 -> evicts oldest (1)
    assert idx.overlap("b0", [1]) == 0
    assert idx.overlap("b0", [2, 3]) == 2


# --- chunk-affinity routing ---

def test_routes_to_best_overlap_backend():
    idx = ChunkAffinityIndex()
    ids = chunk_ids(canonicalize_chunks(["a", "b", "c", "d"]))
    idx.record("b1", ids[:3])                # b1 cached 3 of the 4 chunks
    res = choose_rag_backend(chunk_ids=ids, backends=["b0", "b1", "b2"],
                             index=idx, load=LoadTracker(), cfg=RagConfig())
    assert res.backend_id == "b1"
    assert res.overlap == 3
    assert res.total_chunks == 4


def test_saturated_overlap_backend_is_excluded():
    idx = ChunkAffinityIndex()
    ids = chunk_ids(["a", "b", "c", "d"])
    idx.record("b1", ids)                    # b1 has everything cached...
    load = LoadTracker()
    cfg = RagConfig()
    load.inflight["b1"] = cfg.max_inflight + 1   # ...but is saturated
    res = choose_rag_backend(chunk_ids=ids, backends=["b0", "b1"],
                             index=idx, load=load, cfg=cfg)
    assert res.backend_id == "b0"


def test_no_backends_returns_none():
    res = choose_rag_backend(chunk_ids=[1, 2], backends=[], index=ChunkAffinityIndex(),
                             load=LoadTracker(), cfg=RagConfig())
    assert res is None


# --- end-to-end: reordered retrieval of the same docs is a full cache hit ---

def test_reordered_retrieval_hits_cache():
    idx = ChunkAffinityIndex()
    # Query 1 retrieves docs in one order; record what the chosen backend caches.
    q1 = canonicalize_chunks(["alpha", "beta", "gamma"])
    idx.record("b0", chunk_ids(q1))
    # Query 2 retrieves the SAME docs in a different order.
    q2_ids = chunk_ids(canonicalize_chunks(["gamma", "alpha", "beta"]))
    res = choose_rag_backend(chunk_ids=q2_ids, backends=["b0", "b1"],
                             index=idx, load=LoadTracker(), cfg=RagConfig())
    assert res.backend_id == "b0"
    assert res.overlap == 3                   # full reuse despite different order
