# Prefix-Aware LLM Inference Gateway

[![CI](https://github.com/vikask3906/llm-inferance/actions/workflows/ci.yml/badge.svg)](https://github.com/vikask3906/llm-inferance/actions/workflows/ci.yml)

An OpenAI-compatible inference gateway that routes requests to the GPU **already
holding the matching KV-cache prefix** — balanced against per-node saturation —
so a cluster of LLM servers reuses cache instead of recomputing it. A standalone,
platform-agnostic take on Google's GKE Inference Gateway.

> **~2.4× the prefix-cache hit rate of round-robin** (99% vs 40%) while keeping
> load balanced — **validated on real vLLM + 2× A40 GPUs: 53% lower mean TTFT at
> low load and +10 pp vLLM cache hit rate** — plus fault tolerance, multi-tenant
> fairness, and full observability (logs + metrics + traces).

---

## Why

LLM inference is **stateful**: a Transformer's prompt prefill is cached in GPU
memory as the **KV cache**, and engines like vLLM reuse it when a later request
shares a prefix (a system prompt, a document, a conversation), skipping prefill
and slashing time-to-first-token. But ordinary HTTP load balancers are
**stateless** — round-robin sends request B to a different GPU than request A,
which recomputes the identical 10k-token prefix and duplicates the VRAM. The
cluster wastes compute and memory, and tail latency spikes.

This gateway makes routing **prefix-aware**: it tracks which backend holds which
prefix and routes for cache reuse, while spilling load when a node saturates.

## Results

Workload: every request = shared system prompt + one of 15 documents + a unique
question. Fleet of 3 backends; hit rate measured by **independent per-backend
cache models** (the backend's own truth, not the gateway's belief).

| Strategy | uniform docs | skewed/hot docs | HTTP e2e | load balance |
|---|---|---|---|---|
| round-robin | 40.5% | 62.0% | 45.2% | even |
| consistent-hash (1st block) | 40.1% | 62.7% | 45.8% | **collapses to 1 node** |
| **prefix-tree (this)** | **99.3%** | **99.3%** | **98.3%** | **even** |

**Multi-tenant fairness** — a greedy tenant floods at 10× quota while a polite
tenant stays within quota (same total budget):

| limiter | polite tenant served |
|---|---|
| shared global bucket | 16% (starved) |
| **per-tenant buckets (this)** | **100%** |

**Python gateway vs Rust hot-path rewrite** — measured on the same fast backend:

**Full-parity Rust** matches every per-request thing the Python gateway does
(metrics + circuit + failover + tenancy + structured JSON logging) — see
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** for methodology:

| metric (c=64) | Python | **full-parity Rust** | ratio |
|---|---|---|---|
| throughput | 209 req/s | **10,922 req/s** | **~52× higher** |
| p99 under load | 399.51 ms | **12.20 ms** | **~33× lower** |
| added latency (c=1) | +3.8 ms | **+0.8 ms** | ~4.5× lower |

**Real-vLLM GPU validation** — 2× NVIDIA A40, Qwen2.5-1.5B-Instruct, vLLM 0.7.3
with prefix caching, KV capped to force cache pressure (30 docs × 12 KB, n=600
per point, concurrency swept); TTFT measured client-side (first SSE byte):

| concurrency | round-robin TTFT mean | **prefix-tree (this)** | improvement |
|---|---|---|---|
| **c = 1** | 166.4 ms | **77.4 ms** | **−53%** |
| c = 8 | 309.7 ms | 304.3 ms | −2% (tied) |
| c = 32 | 893.1 ms | 935.9 ms | +5% mean / **−9% p95** |

vLLM prefix-cache hit rate **62.6% → 72.6% (+10 pp)**; avg prefix-match blocks
**0 → 166** (routing affinity confirmed). The win is largest at low load where
cache locality dominates, and is gracefully traded for load balance as
concurrency rises — see **[docs/BENCHMARKS.md §D](docs/BENCHMARKS.md)**.

Reproduce: `python bench/sim.py` · `python bench/e2e_inproc.py` ·
`python bench/fairness_sim.py`.

### Standalone-package benchmarks

Each extension ships with its own before/after benchmark, written up in
`docs/benchmarks/`:

| package | benchmark | headline |
|---|---|---|
| **DAG scheduler** | `bench/dag_bench.py` | **3.0×** lower makespan, **3.0×** higher prefix-cache hit rate vs round-robin on a multi-chain workflow (5 chains × 4 nodes × 16-block shared context) — [`DAG_RESULTS.md`](docs/benchmarks/DAG_RESULTS.md) |
| **RAG chunk-affinity** | `bench/rag_bench.py` | **60% chunk-cache hit rate** vs round-robin's 50% on a multi-tenant RAG workload (3 disjoint sub-corpora, 600 chunks, Zipf α=1.2), **20% lower** per-request prefill cost — [`RAG_RESULTS.md`](docs/benchmarks/RAG_RESULTS.md) |
| **Admission control** | `bench/admission_bench.py` | At **2× offered load**: baseline collapses to **2% gold-tier SLO compliance**, admission keeps gold at **100%** (bronze shed to 29% served, by design) — [`ADMISSION_RESULTS.md`](docs/benchmarks/ADMISSION_RESULTS.md) |
| **Disaggregation** | `bench/disagg_bench.py` | Adaptive prefill/decode split is the **lower-latency envelope** — co-locates when idle (0% split, matches colocate), splits under load (**33% lower mean latency than colocate-only** at 8× load) — [`DISAGG_RESULTS.md`](docs/benchmarks/DISAGG_RESULTS.md) |
| **Multimodal** | `bench/multimodal_bench.py` | Media-affinity hits **90% image-cache** vs round-robin's 81%, while staying **11× more load-balanced** than consistent-hash (CoV 0.01 vs 0.11) — best of both — [`MULTIMODAL_RESULTS.md`](docs/benchmarks/MULTIMODAL_RESULTS.md) |

## How it works

Two layers of routing intelligence, scored by **estimated time-to-first-token**:

1. **Prefix affinity** — a path-compressed **radix tree** of token-block hashes
   maps each cached prefix to the backends holding it, so routing does
   *longest-prefix match* (not a coarse hash). A per-backend LRU model mirrors the
   backend's own KV eviction.
2. **Load awareness** — `est_TTFT = queue_delay + prefill(uncached_tokens)`, with
   a hard saturation cutoff and a least-loaded + round-robin tiebreak so a tiny
   shared prefix can't pin all traffic to one node. Hot prefixes **replicate**
   across nodes automatically as one saturates.

```
            ┌───────────────────────── Gateway ─────────────────────────┐
 client ──▶ │ DATA PLANE: identify tenant → admit (rate limit) →         │
            │   block-hash → radix-tree match → est-TTFT pick → stream   │
            │ CONTROL PLANE: scrape /metrics → reconcile load + health   │
            └────────────────────────────┬──────────────────────────────┘
                       ┌──────────────────┼──────────────────┐
                       ▼                  ▼                  ▼
                   ┌───────┐          ┌───────┐          ┌───────┐
                   │ vLLM  │          │ vLLM  │          │ vLLM  │
                   │  b0   │          │  b1   │          │  b2   │
                   └───────┘          └───────┘          └───────┘
```

## Features

- **Routing** — path-compressed radix tree, est-TTFT cost function, load-spread
  tiebreak, emergent hot-prefix replication; pluggable strategies (`round_robin`,
  `consistent_hash`, `prefix_tree`).
- **Fault tolerance** — per-backend circuit breaker (closed/open/half-open with
  auto-recovery) and safe pre-first-byte failover (mid-stream errors propagate, no
  duplicated tokens).
- **Authentication** — opt-in API-key auth (`GW_REQUIRE_AUTH`): a request must
  carry `Authorization: Bearer <key>` with a key in the valid set (tenant keys +
  `GW_API_KEYS`) or it's rejected `401` with `WWW-Authenticate`. Reuses the
  tenant-key scheme, so an authenticated key still resolves to its tier.
- **Multi-tenant fairness** — per-tenant RPS + TPS token buckets (OpenAI RPM+TPM
  style) + in-flight caps, `429` with `Retry-After`, and per-tenant prefix
  isolation (routing seed + backend `cache_salt`) to close the cross-tenant TTFT
  side channel. Optional **token-accurate accounting** (`GW_TOKEN_ACCURATE_ACCOUNTING`)
  counts real tokenizer tokens for quota/admission instead of the `chars/4`
  heuristic (which is off by 2-3× on code / non-English).
- **Observability** — structured JSON logs (request_id + trace_id), Prometheus
  `/metrics`, OpenTelemetry traces, and a provisioned Grafana dashboard.
- **RAG-aware routing** — structured RAG payloads are canonicalized (deduped,
  sorted chunks → identical prefix), routed by chunk-affinity (set-overlap) to
  the backend already holding the most chunks. Opt-in via `GW_RAG_ENABLED`.
- **SLO-aware admission** — TTFT-budget fail-fast, fleet-pressure priority
  shedding (gold protected, bronze shed first), Retry-After headers. Opt-in via
  `GW_ADMISSION_ENABLED`.
- **Disaggregated prefill/decode** — Splitwise/DistServe-style phase splitting:
  assign prefill and decode to specialized backends when the split beats
  co-location. Opt-in via `GW_DISAGG_POOLS`.
- **Multi-modal routing** — capability filtering (only vision backends serve
  images) + media-affinity cache (prefer the backend that already encoded an
  image). Opt-in via `GW_MULTIMODAL_ENABLED`.
- **DAG scheduling** — `POST /v1/dag/schedule` plans cache-locality-aware
  placement of multi-step workflows (map-reduce, tool-use chains). Opt-in via
  `GW_DAG_ENABLED`.
- **Autoscaling** — SLO-driven replica planner (Erlang-C + utilization target,
  anti-flapping cooldowns). Emits scaling recommendations via `/autoscale` and
  Prometheus metrics. Opt-in via `GW_AUTOSCALE_ENABLED`.
- **Horizontal scaling** — run multiple gateway replicas behind a load balancer
  with **shared state**: each keeps its local radix tree for fast longest-prefix
  match and replicates over a bus (Redis pub/sub) both (a) tree *mutations* — so
  a request can hit any replica and still route to the backend holding the cached
  prefix — and (b) per-backend *load + circuit state* — so the cost function
  scores by fleet-wide in-flight and two replicas don't stampede the same
  "least-loaded" node. Reads stay local; only writes fan out. Opt-in via
  `GW_CLUSTER_ENABLED` — see [Horizontal scaling](#horizontal-scaling-multi-replica).
- **OpenAI-compatible** — `POST /v1/chat/completions` with SSE streaming.

## Quickstart

```bash
pip install -r requirements-dev.txt

python -m pytest -q            # 252 tests
python bench/sim.py            # routing hit-rate proof (no network)
python bench/e2e_inproc.py     # full HTTP path through 3 mock backends
python bench/fairness_sim.py   # per-tenant fairness demo
```

Full stack (gateway + 3 mock backends + Prometheus + Grafana) — one command
brings it up, waits for health, and drives sustained traffic so the
pre-provisioned Grafana dashboard fills with live data:

```bash
make demo                      # or: python scripts/demo.py   (no make needed, Windows-friendly)
# gateway     -> http://localhost:8000
# Prometheus  -> http://localhost:9090
# Grafana     -> http://localhost:3000   (anonymous admin; dashboard pre-loaded)

make traffic                   # drive 60s more prefix-heavy traffic at the running gateway
make down                      # tear the stack down

# a single streaming request by hand:
curl -N http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

### Horizontal scaling (multi-replica)

A single gateway holds its radix tree in-process — a second replica would route
blind. With `GW_CLUSTER_ENABLED`, replicas **share prefix-routing state**: each
keeps its local tree for fast matching and replicates tree *mutations*
(insert / backend-removal) over Redis pub/sub, converging so a request can land
on **any** replica and still route to the backend holding the cached prefix.

```bash
# two gateway replicas behind an nginx LB, sharing state via Redis:
docker compose -f docker-compose.cluster.yml up --build
# load balancer -> http://localhost:8080   (round-robins gw1 / gw2)
python bench/loadtest.py --url http://localhost:8080 --duration 60
```

Two kinds of state are replicated: **prefix→backend mappings** (so any replica
routes to the warm backend) and **per-backend load + circuit state** (so the
cost function scores by fleet-wide in-flight and a node that's busy or shed on
one replica is avoided everywhere). Design: writes fan out, reads stay local
(the hot path adds only a buffered, non-blocking publish; a background loop
publishes this replica's load and drains peers' mutations off the request path).
Eventual consistency — each replica's tree becomes the union of the fleet's
inserts, which models the backend's real cache better than any single replica's
view. See [`gateway/cluster/`](gateway/cluster/).

## Project layout

```
gateway/           data plane + control plane
  server.py           OpenAI-compatible async proxy (the hot path)
  router.py           strategies + est-TTFT cost function
  radix_tree.py       path-compressed prefix tree + LRU eviction
  hashing.py          chained block hashing (mirrors vLLM APC)
  circuit.py          per-backend circuit breaker
  tenancy.py          token buckets + tenant registry + rate limiter
  metrics.py          Prometheus exposition
  tracing.py          OpenTelemetry spans
  logging_setup.py    structured JSON logs
  rag/                RAG structuring + chunk-affinity routing
  admission/          SLO-aware admission control + load shedding
  disagg/             disaggregated prefill/decode routing
  multimodal/         multi-modal capability + affinity routing
  dag/                cache-locality-aware DAG scheduler
  autoscale/          SLO-driven autoscaler / capacity planner
  extensions/         LoRA, semantic cache, speculative, TTFT predictor
mock_backend/      fake vLLM (prefix cache sim + SSE + /health + /metrics)
bench/             sim, e2e, fairness, load test, benchmark matrix
scripts/           demo orchestrator + standalone package demos
deploy/            Docker Compose + Prometheus + Grafana provisioning + Helm
docs/DESIGN.md     full design doc (architecture, trade-offs, roadmap)
```

## Design

See **[docs/DESIGN.md](docs/DESIGN.md)** for the architecture, the routing
algorithm (radix tree, eviction model, cost function), trade-offs, failure
handling, and the Phase-2 roadmap — and **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)**
for a complete file-by-file account of everything implemented.

## Roadmap

- Real **vLLM** on GPUs — **done ✓** (2× A40, −53% mean TTFT at low load, +10 pp
  cache hit rate; see [Results](#results) and **[docs/BENCHMARKS.md §D](docs/BENCHMARKS.md)**).
  Reproduce in one command on a 2-GPU pod via **[docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md)**.
- Multi-replica gateway with **shared prefix + load + circuit state** — **done ✓**
  ([`gateway/cluster/`](gateway/cluster/), `docker-compose.cluster.yml`):
  Redis-replicated radix-tree mutations *and* fleet-wide load/health, local
  reads. Next: a CRDT/gossip transport to drop the Redis dependency.
- **Rust** hot-path rewrite ([`rust/`](rust/)): data-plane core + axum/reqwest
  streaming proxy done ✓ (20 core tests; e2e smoke-tested vs the mock backend);
  next port metrics/circuit/tenancy + the profiled before/after latency vs Python.
- Backend Prometheus scraping, token-accurate tokenization, weighted fair
  queuing, and a multi-replica gateway with shared prefix state.

Built in Python (FastAPI · httpx · OpenTelemetry · Prometheus · Grafana · Docker).
