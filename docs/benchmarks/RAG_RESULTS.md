# RAG-Routing Benchmark: Chunk-Affinity vs Cache-Blind

RAG-style workload: a corpus of `n_chunks_pool` chunks; each request
retrieves `k_per_request` of them via a Zipf-weighted popularity
distribution (the canonical RAG access pattern where a few chunks
dominate traffic). The chunk-affinity router routes to the backend
already holding the most of the request's chunks; round-robin and
consistent-hash ignore chunk state.

Fleet: 3 backends, per-backend chunk-cache capacity 64 (constrained so eviction binds -- the realistic
case where the corpus exceeds per-GPU KV).

## Baseline (pool=600, k=8, alpha=1.2, topics=3, n_requests=2000)

| strategy | chunk hit rate | mean est-ms |
|---|---|---|
| round_robin | 50.2% | 102.07 |
| consistent_hash | 52.6% | 97.08 |
| chunk_affinity | 60.0% | 81.88 |

**Chunk-affinity wins**: 60.0% hit rate vs 50.2% (RR) and 52.6% (consistent-hash) -- a 1.2x lift, and a 1.2x reduction in mean per-request prefill cost.

## Sweeps

### sweep: n_chunks_pool (corpus size (chunks in the pool))

| n_chunks_pool | round_robin hit rate | consistent_hash hit rate | chunk_affinity hit rate |
|---|---|---|---|
| 150 | 70.5% | 80.9% | 85.2% |
| 300 | 57.6% | 63.0% | 68.7% |
| 600 | 50.2% | 52.6% | 60.0% |
| 1200 | 46.2% | 49.5% | 54.4% |
| 2400 | 42.1% | 45.3% | 49.2% |

### sweep: k_per_request (chunks retrieved per request)

| k_per_request | round_robin hit rate | consistent_hash hit rate | chunk_affinity hit rate |
|---|---|---|---|
| 3 | 60.1% | 63.6% | 63.1% |
| 5 | 55.4% | 58.5% | 61.1% |
| 8 | 50.2% | 52.6% | 60.0% |
| 12 | 44.3% | 46.6% | 60.1% |
| 20 | 35.8% | 37.0% | 64.0% |

### sweep: skew_alpha (Zipf skew (chunk popularity))

| skew_alpha | round_robin hit rate | consistent_hash hit rate | chunk_affinity hit rate |
|---|---|---|---|
| 0.0 | 10.9% | 11.9% | 17.4% |
| 0.5 | 15.3% | 16.4% | 24.2% |
| 1.0 | 39.4% | 42.2% | 51.2% |
| 1.5 | 64.2% | 67.5% | 73.2% |
| 2.0 | 81.5% | 84.4% | 86.6% |

### sweep: n_topics (disjoint sub-corpora (multi-tenant))

| n_topics | round_robin hit rate | consistent_hash hit rate | chunk_affinity hit rate |
|---|---|---|---|
| 1 | 64.8% | 65.3% | 68.3% |
| 2 | 57.0% | 59.1% | 64.2% |
| 3 | 50.2% | 52.6% | 60.0% |
| 5 | 40.7% | 49.4% | 57.3% |
| 9 | 30.4% | 38.9% | 51.0% |

## Caveat

Hit rate is measured against the same `ChunkAffinityIndex` the live
router uses (LRU set per backend) -- the backend's *own* truth, not a
gateway belief. est-ms is the prefill proxy (uncached tokens x
ms/token), not wall-clock. Absolute ms validation is the GPU job; same
caveat as `bench/matrix.py` and `bench/dag_bench.py`.
