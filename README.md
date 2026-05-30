# Prefix-Aware LLM Inference Gateway

[![CI](https://github.com/vikask3906/llm-inferance/actions/workflows/ci.yml/badge.svg)](https://github.com/vikask3906/llm-inferance/actions/workflows/ci.yml)

An OpenAI-compatible inference gateway that routes requests to the GPU **already
holding the matching KV-cache prefix** — balanced against per-node saturation —
so a cluster of LLM servers reuses cache instead of recomputing it. A standalone,
platform-agnostic take on Google's GKE Inference Gateway.

> **~2.4× the prefix-cache hit rate of round-robin** (99% vs 40%) while keeping
> load balanced — plus fault tolerance, multi-tenant fairness, and full
> observability (logs + metrics + traces).

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

Reproduce: `python bench/sim.py` · `python bench/e2e_inproc.py` ·
`python bench/fairness_sim.py`.

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
- **Multi-tenant fairness** — per-tenant RPS + TPS token buckets (OpenAI RPM+TPM
  style) + in-flight caps, `429` with `Retry-After`, and per-tenant prefix
  isolation (routing seed + backend `cache_salt`) to close the cross-tenant TTFT
  side channel.
- **Observability** — structured JSON logs (request_id + trace_id), Prometheus
  `/metrics`, OpenTelemetry traces, and a provisioned Grafana dashboard.
- **OpenAI-compatible** — `POST /v1/chat/completions` with SSE streaming.

## Quickstart

```bash
pip install -r requirements-dev.txt

python -m pytest -q            # 61 tests
python bench/sim.py            # routing hit-rate proof (no network)
python bench/e2e_inproc.py     # full HTTP path through 3 mock backends
python bench/fairness_sim.py   # per-tenant fairness demo
```

Full stack (gateway + 3 mock backends + Prometheus + Grafana):

```bash
docker compose up --build
# gateway   -> http://localhost:8000
# Prometheus-> http://localhost:9090
# Grafana   -> http://localhost:3000   (dashboard pre-loaded)

curl -N http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Project layout

```
gateway/        data plane + control plane
  router.py        strategies + est-TTFT cost function
  radix_tree.py    path-compressed prefix tree + LRU eviction
  hashing.py       chained block hashing (mirrors vLLM APC)
  circuit.py       per-backend circuit breaker
  tenancy.py       token buckets + tenant registry + rate limiter
  metrics.py       Prometheus exposition
  tracing.py       OpenTelemetry spans
  logging_setup.py structured JSON logs
  server.py        OpenAI-compatible async proxy
mock_backend/   fake vLLM (prefix cache sim + SSE + /metrics)
bench/          sim, e2e, fairness, load test
deploy/         Prometheus + Grafana provisioning
docs/DESIGN.md  full design doc (architecture, trade-offs, roadmap)
```

## Design

See **[docs/DESIGN.md](docs/DESIGN.md)** for the architecture, the routing
algorithm (radix tree, eviction model, cost function), trade-offs, failure
handling, and the Phase-2 roadmap — and **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)**
for a complete file-by-file account of everything implemented.

## Roadmap

- Real **vLLM** on GPUs for the headline TTFT-reduction number.
- **Rust** hot-path rewrite (Tokio/hyper) with profiled before/after latency.
- Backend Prometheus scraping, token-accurate tokenization, weighted fair
  queuing, and a multi-replica gateway with shared prefix state.

Built in Python (FastAPI · httpx · OpenTelemetry · Prometheus · Grafana · Docker).
