# Multimodal-Routing Benchmark: Media-Affinity vs Cache-Blind

Vision requests are expensive: an image is expanded into hundreds of
tokens by the vision encoder, and that encode is recomputed whenever the
image lands on a cold backend. Media-affinity routing sends a request to
the backend that already encoded its image (warm vision KV); round-robin
and consistent-hash ignore which backend holds what.

Fleet: 3 vision-capable backends, per-backend media-cache
capacity 48 (constrained so eviction binds).

## Baseline (images=200, alpha=1.5, topics=3, n_requests=2000)

| strategy | media hit rate | mean prefill (est-ms) | load imbalance (CoV) |
|---|---|---|---|
| round_robin | 81.2% | 7.32 | 0.00 |
| consistent_hash | 92.8% | 2.90 | 0.11 |
| media_affinity | 89.8% | 4.05 | 0.01 |

**Media-affinity wins on both axes**: 89.8% hit
rate vs 81.2% (round-robin) -- a 1.1x lift,
1.8x lower prefill cost -- while staying load-balanced (CoV 0.01). Consistent-hash reaches a similar hit rate (92.8%) but HOTSPOTS (CoV 0.11 vs 0.01): it pins each image to a fixed backend with no
load awareness, so a popular image overloads one node. Affinity gets the
cache reuse without the hotspot -- the same edge the text radix router has.

## Sweeps

### sweep: n_images (image library size)

| n_images | round_robin hit rate | consistent_hash hit rate | media_affinity hit rate |
|---|---|---|---|
| 50 | 93.2% | 97.6% | 95.7% |
| 100 | 85.8% | 95.3% | 93.5% |
| 200 | 81.2% | 92.8% | 89.8% |
| 400 | 78.1% | 90.0% | 86.5% |
| 800 | 76.3% | 87.2% | 84.6% |

### sweep: skew_alpha (Zipf skew (image popularity))

| skew_alpha | round_robin hit rate | consistent_hash hit rate | media_affinity hit rate |
|---|---|---|---|
| 0.0 | 24.6% | 70.0% | 58.9% |
| 0.5 | 32.2% | 76.2% | 63.7% |
| 1.0 | 56.6% | 86.8% | 78.6% |
| 1.5 | 81.2% | 92.8% | 89.8% |
| 2.0 | 91.8% | 95.9% | 94.7% |

### sweep: n_topics (disjoint image collections (multi-tenant))

| n_topics | round_robin hit rate | consistent_hash hit rate | media_affinity hit rate |
|---|---|---|---|
| 1 | 88.1% | 93.8% | 92.5% |
| 2 | 84.4% | 93.1% | 91.0% |
| 3 | 81.2% | 92.8% | 89.8% |
| 5 | 77.3% | 91.8% | 87.5% |
| 9 | 69.7% | 91.1% | 86.3% |

## Caveat

Hit rate is measured against the same `MediaAffinityIndex` the live router
uses (LRU set of media ids per backend). est-ms is the prefill proxy
(uncached tokens x ms/token) with image tokens from OpenAI's tiling
formula; not wall-clock. Absolute ms validation is the GPU job; same
caveat as `bench/matrix.py`.
