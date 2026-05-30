# DAG-Scheduler Benchmark: Locality vs Round-Robin

Cache-locality-aware DAG scheduling on multi-step LLM workflows.
Workload: N independent chains, each a sequence of LLM calls sharing a long
context prefix (the agentic / map-reduce pattern).

Fleet: 3 backends, KV cap 2000 blocks each.

## Baseline (n_chains=5, chain_depth=4, context_blocks=16)

| strategy | hit rate | makespan (ms) |
|---|---|---|
| round_robin | 25.0% | 76.8 |
| locality | 75.0% | 25.6 |

**Locality wins**: hit rate 75.0% vs 25.0%, makespan 3.00x lower.

## Sweeps

### sweep: n_chains (number of chains (parallel workflows))

| n_chains | round_robin hit rate | locality hit rate | round_robin makespan (ms) | locality makespan (ms) |
|---|---|---|---|---|
| 3 | 75.0% | 75.0% | 12.8 | 12.8 |
| 4 | 25.0% | 75.0% | 76.8 | 25.6 |
| 5 | 25.0% | 75.0% | 76.8 | 25.6 |
| 6 | 75.0% | 75.0% | 25.6 | 25.6 |
| 7 | 25.0% | 75.0% | 115.2 | 38.4 |
| 9 | 75.0% | 75.0% | 38.4 | 38.4 |
| 11 | 25.0% | 75.0% | 153.6 | 51.2 |
| 13 | 25.0% | 75.0% | 192.0 | 64.0 |

### sweep: chain_depth (chain depth (nodes per chain))

| chain_depth | round_robin hit rate | locality hit rate | round_robin makespan (ms) | locality makespan (ms) |
|---|---|---|---|---|
| 2 | 0.0% | 50.0% | 51.2 | 25.6 |
| 3 | 0.0% | 66.7% | 76.8 | 25.6 |
| 4 | 25.0% | 75.0% | 76.8 | 25.6 |
| 6 | 50.0% | 83.3% | 76.8 | 25.6 |
| 8 | 62.5% | 87.5% | 76.8 | 25.6 |

### sweep: context_blocks (shared-context blocks per chain)

| context_blocks | round_robin hit rate | locality hit rate | round_robin makespan (ms) | locality makespan (ms) |
|---|---|---|---|---|
| 4 | 25.0% | 75.0% | 19.2 | 6.4 |
| 8 | 25.0% | 75.0% | 38.4 | 12.8 |
| 16 | 25.0% | 75.0% | 76.8 | 25.6 |
| 32 | 25.0% | 75.0% | 153.6 | 51.2 |
| 64 | 25.0% | 75.0% | 307.2 | 102.4 |

## A note on the n_chains sweep

Round-robin's cyclic counter co-locates a chain by accident whenever
`n_chains % len(backends) == 0` (e.g. 3, 6, 9, 12 chains on a 3-backend
fleet). At those points RR ties locality. At every other point RR scatters
each chain across backends and pays the full uncached prefill at every
child node -- ~3x makespan and ~3x lower hit rate. The takeaway: RR's
best case is a coincidence of integer arithmetic; locality is robust.

## Caveat

Makespan is the scheduler's own model (prefill ms = compute-bound proxy +
queue terms), not wall-clock. It captures the work a cache hit removes,
which is precisely what the locality scheduler optimizes for. Absolute ms
validation is the GPU job; see the same caveat in `bench/matrix.py` and
`docs/benchmarks/RESULTS.md`.
