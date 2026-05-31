# Architecture — Prefix-Aware LLM Inference Gateway

A visual, end-to-end map of everything the project covers. Each box names the
module that implements it. Read top-to-bottom: system topology → the request hot
path → the routing decision → the off-path control & cluster planes →
observability → the Rust parity port → a feature-coverage matrix.

> One-line thesis: route each request to the GPU **already holding its KV-cache
> prefix** (radix-tree longest-prefix match), balanced against per-node load —
> and do it with production concerns handled: auth, fairness, admission, fault
> tolerance, horizontal scaling, and observability.

---

## 1. System topology (horizontally scaled)

```
                              ┌──────────────┐        ┌──────────────────────┐
   clients (OpenAI SDK) ────▶ │ load balancer │ ─────▶ │ operator / CI         │
        SSE streaming         │   (nginx)     │        │ curl /admin, make ... │
                              └──────┬───────┘         └───────────┬──────────┘
                       round-robin   │                            │ /admin/* (token)
                ┌─────────────┬──────┴───────┬─────────────┐      │
                ▼             ▼              ▼             ▼      ▼
          ┌───────────┐ ┌───────────┐  ┌───────────┐   (… N gateway replicas)
          │ gateway 1 │ │ gateway 2 │  │ gateway 3 │   each: local radix tree,
          │           │ │           │  │           │   load, circuit (fast reads)
          └─────┬─────┘ └─────┬─────┘  └─────┬─────┘
                │   replication bus (pluggable): Redis pub/sub │  ← GW_CLUSTER_*
                └──────────────┴──── OR ──────┴───────────────┘
                                 peer-to-peer HTTP gossip (no broker)
                                          │  shares: prefix map, load,
                                          │          circuit, drain (LWW)
                ┌─────────────────────────┼─────────────────────────┐
                ▼                         ▼                         ▼
          ┌───────────┐            ┌───────────┐            ┌───────────┐
          │ vLLM  b0  │            │ vLLM  b1  │            │ vLLM  b2  │   backend fleet
          │ KV prefix │            │ KV prefix │            │ KV prefix │   (or mock_backend
          │  cache    │            │  cache    │            │  cache    │    in dev/bench)
          └───────────┘            └───────────┘            └───────────┘
                ▲                         ▲                         ▲
                └─────────── Prometheus scrape /metrics ───────────┘ ──▶ Grafana
```

A single replica is fully functional on its own; the bus + shared state are
opt-in (`GW_CLUSTER_ENABLED`) and only add **fleet-wide** awareness.

---

## 2. Request hot path (data plane, one replica)

`server.py :: chat_completions` — every gate is ordered to fail cheap before
spending compute. Opt-in stages are no-ops unless their flag is set.

```
 POST /v1/chat/completions  (OpenAI-compatible, stream=true)
        │
        ▼
 (1)  parse messages → prompt string ........................ server.py
 (2)  API-key auth ........... auth.py ....................... ─▶ 401 (WWW-Authenticate)
 (3)  RAG structuring (opt) .. rag/ .......................... dedupe+sort chunks →
        │                                                       cacheable system prefix
 (4)  resolve tenant ......... tenancy.py .................... key → tenant + tier
 (5)  rate limit ............. tenancy.py .................... RPS+TPS buckets + inflight
        │                                                      ─▶ 429 (Retry-After, X-RateLimit-*)
 (6)  SLO admission (opt) .... admission/ .................... TTFT budget fail-fast +
        │                                                      fleet-pressure priority shed
        │                                                      ─▶ 503 (Retry-After)
 (7)  model + LoRA filter .... backends.py / extensions/lora.py  candidates serving model
 (8)  prefix-isolation salt .. hashing.py .................... per-tenant cache_salt (opt)
 (9)  semantic cache (opt) ... extensions/semantic_cache.py .. paraphrase hit ─▶ cached resp
 (10) circuit + drain filter . circuit.py / backends.py ...... drop open / draining backends
        │
        ▼
 (11) ROUTE  ───────────────────────────────────▶  [ §3 routing decision ]
        │                                            picks backend(s) + hashes
        ▼
 (12) commit: tree.insert(hashes,B) ; load.on_dispatch(B) ;     radix_tree / load_tracker
        cluster.publish_insert(B) (opt) ........................ cluster/
        ▼
 (13) proxy upstream SSE; pre-first-byte failover on error ... server.py + circuit.py
        ▼
 (14) stream chunks to client (no buffering) ................. StreamingResponse
        ▼
 (15) finalize (in stream teardown):
        • per-route SLO histograms (TTFT, total) ............. metrics.py
        • structured JSON log (request_id + trace_id) ........ logging_setup.py
        • OTel span end ...................................... tracing.py
        • semantic-cache store / RAG + media affinity record . extensions/, rag/, multimodal/
        • load.on_complete(B) ; TPS reconcile ................ load_tracker / tenancy
```

---

## 3. The routing decision (`router.py`)

The intelligence: longest-prefix match scored by estimated time-to-first-token,
balanced against fleet-wide load.

```
  prompt
    │  block_hashes()  — chained FNV-1a over fixed blocks, cutoff to first N
    │     char blocks (default)  ............ hashing.py
    │     OR real BPE token-ID blocks (opt) . extensions/bpe_hashing.py  (matches vLLM keys)
    ▼
  radix_tree.match(hashes) ──▶ { backend : matched_prefix_blocks }    radix_tree.py
    │   path-compressed PATRICIA tree; per-backend LRU mirrors the
    │   backend's own KV eviction (held_blocks bounded by cap)
    ▼
  for each candidate backend B:
        cached   = match[B] · block_tokens
        uncached = max(0, prompt_tokens − cached)
        est_TTFT[B] = prefill_ms·uncached  +  queue_ms·inflight[B]      router.py
                                              └ inflight = local + peers (fleet view)
                                                                        cluster/fleet.py
        guardrails: skip B if  kv_usage>cutoff │ inflight>cap │ peer-shed
    ▼
  strategy  (per-request override via x-routing-strategy):
        round_robin       cache-blind cycle ............... the baseline to beat
        consistent_hash   first-block hash ................ collapses to one node
        prefix_tree  ★    argmin est_TTFT, then least-load + RR tiebreak within
                          a hysteresis band  → emergent hot-prefix replication
        speculative       dispatch top-K in parallel, first byte wins
                                                           extensions/speculative.py
    ▼
  optional routing AXES that pre-empt the strategy (opt-in flags):
        disagg/      split prefill vs decode across specialized pools (Splitwise)
        multimodal/  hard capability filter (vision/audio) + media-affinity cache
        dag/         POST /v1/dag/schedule — cache-locality plan for multi-step DAGs
    ▼
  chosen backend(s) + RouteResult(hashes, tokens, match_blocks)
                          │
            (predictive TTFT blend, opt) ── extensions/ttft_predictor.py
```

---

## 4. Control plane (off the hot path)

Two background loops in `server.py` keep state fresh without touching request
latency.

```
  scrape_loop (every ~2s) ........................................ server.py
    ├─ GET {backend}/health  → registry.set_health + circuit record   backends/circuit
    ├─ reconcile KV usage / inflight (EWMA) ........................ load_tracker.py
    └─ autoscale planner: offered-RPS → desired replicas (Erlang-C)  autoscale/
            → gateway_autoscale_* metrics + GET /autoscale

  cluster_sync_loop (every ~250ms, opt) ......................... server.py + cluster/
    ├─ publish_load(inflight, shed)        → peers (fleet-wide load)
    ├─ publish_drain_digest() every ~2s    → anti-entropy (LWW drain map)
    └─ sync(): apply peers' tree/load/drain mutations to local state

  Admin API (token-guarded, GW_ADMIN_TOKEN) ...................... server.py
    POST   /admin/backends                     add a backend at runtime (no restart)
    DELETE /admin/backends/{id}                remove a backend at runtime
    POST   /admin/backends/{id}/drain|undrain  maintenance, propagates fleet-wide
    GET    /admin/backends , /admin/state       live routing inspection
    (add/remove/drain all propagate over the cluster bus when clustered)
  Cluster ingest (gossip transport) ............................. server.py
    POST /cluster/gossip                       receive a peer's event batch
```

---

## 5. Cluster replication (`gateway/cluster/`)

Reads stay local & fast; only **writes** fan out. Eventually consistent.

```
              ┌────────────────────── ClusterCoordinator ──────────────────────┐
  hot path ──▶│ publish_insert / publish_remove   (radix-tree mutations)        │
  bg loop  ──▶│ publish_load                       (per-backend in-flight)       │
  admin    ──▶│ publish_drain / publish_drain_digest (LWW drain map)             │
              │ sync(): poll bus → apply to { tree │ fleet view │ drain state }  │
              └───────────────────────────────┬─────────────────────────────────┘
                                               ▼  ReplicationBus (one interface)
        ┌───────────────┬──────────────────────┴───────────────────┐
        ▼               ▼                                           ▼
  InMemoryBus      RedisBus                                   HttpGossipBus
  (tests/1-proc)   (pub/sub, central broker)                  (peer-to-peer, no broker;
                                                               POST /cluster/gossip)

  Replicated state                         Durability / convergence
  ─────────────────────────────────        ─────────────────────────────────────
  prefix → backend   (RadixTree)           tree self-heals from a few s of traffic
  load               (FleetLoadView)       re-published every tick
  circuit/health     (peer-shed set)       re-published every tick
  drain intent       (DrainState, LWW)     Redis: write-through snapshot + boot
                     (ts,origin) per node   warm_start;  gossip: periodic digest
                                            (anti-entropy) → late joiners converge
```

`events.py` defines the wire types (PrefixEvent / LoadEvent / DrainEvent /
DrainDigest); `decode_event` dispatches them; `drain_state.py` is the LWW-Map
CRDT (commutative, associative, idempotent merges).

---

## 6. Core state & data structures

| Structure | File | Role |
|---|---|---|
| `RadixTree` | `radix_tree.py` | path-compressed prefix→backend map; longest-prefix match; per-backend LRU eviction |
| `LoadTracker` | `load_tracker.py` | real-time in-flight + token counts; EWMA-reconciled KV usage |
| `CircuitBreaker` | `circuit.py` | per-backend closed/open/half-open with cooldown + probe recovery |
| `BackendRegistry` | `backends.py` | membership, model filter, health, **draining** flag |
| `TokenBucket`/`RateLimiter` | `tenancy.py` | per-tenant RPS+TPS buckets + in-flight caps |
| `FleetLoadView` | `cluster/fleet.py` | peers' load snapshots → fleet-wide in-flight |
| `DrainState` | `cluster/drain_state.py` | LWW-Map of drain intent for gossip anti-entropy |
| `MetricsCollector` | `metrics.py` | Prometheus counters / gauges / histograms |

---

## 7. Observability

```
  /metrics (Prometheus)            structured logs (JSON)        traces (OTel)
  ───────────────────────          ───────────────────          ─────────────
  gateway_requests_total           one line / request:           one chat.completion
  gateway_cache_{hits,misses}        request_id + trace_id        span / request with
  gateway_inflight / kv_usage        tenant, model, backend,      model, tenant, backend,
  gateway_circuit_state              cache_hit, retries,          cache hit, retries,
  gateway_ttft_seconds{model} ★      status, duration_ms          status, output tokens
  gateway_request_duration{model}★                               (OTLP-exportable)
  gateway_slo_violations_total{model}
  gateway_tenant_* / admission_* / rag_* / cluster_* / autoscale_*
            │
            ▼
     Prometheus ──▶ Grafana dashboards (deploy/grafana, provisioned)
   ★ per-route SLO: histogram_quantile → p50/p95/p99 TTFT per model
```

---

## 8. Rust parity hot path (`rust/`)

A faithful port of the per-request hot path — same behavior, ~52× throughput /
~33× lower p99 (see `docs/BENCHMARKS.md`).

```
  rust/src/  hashing · radix_tree · router · load · circuit · tenancy ·
             metrics · logging · admission  +  bin/{gateway,loadgen,mockbackend}
             (53 tests; axum/reqwest streaming proxy)
```

---

## 9. Validation

- **Algorithm sims / matrices** — `bench/matrix.py` (+ dag/rag/admission/disagg/
  multimodal/cluster benches): each routing mode has a before/after chart in
  `docs/benchmarks/`. The cluster bench shows shared prefix state holds ~95% hit
  rate at any replica count while unshared collapses to ~55% by 16 replicas.
- **Real-vLLM GPU run** — 2× A40: prefix routing cuts mean TTFT **53%** at low
  load, **+10pp** cache hit rate (`docs/BENCHMARKS.md §D`,
  `bench/run_vllm_benchmark.sh`, runbook `docs/GPU_RUNBOOK.md`).
- **Tests** — 281 Python + 53 Rust.
- **One-command demos** — `make demo` (single stack) / `make cluster-demo`
  (2 replicas + Redis + LB), both with Prometheus + Grafana.

---

## 10. Feature-coverage matrix

| Capability | Module(s) | Flag | Proof |
|---|---|---|---|
| Prefix-aware routing (radix tree + est-TTFT) | `router`, `radix_tree`, `hashing` | default | `bench/matrix.py`, GPU §D |
| Strategies: round_robin / consistent_hash / prefix_tree / speculative | `router`, `extensions/speculative` | `GW_STRATEGY` | `test_server`, `test_speculative` |
| Fault tolerance: circuit breaker + pre-byte failover | `circuit`, `server` | default | `test_circuit`, `test_server` |
| Multi-tenant fairness (RPS+TPS+inflight, isolation) | `tenancy`, `hashing` | `GW_RATE_LIMIT_ENABLED` | `test_tenancy`, `test_server` |
| **API-key auth** | `auth` | `GW_REQUIRE_AUTH` | `test_auth` |
| **Token-accurate accounting** | `extensions/bpe_hashing` | `GW_TOKEN_ACCURATE_ACCOUNTING` | `test_token_accounting` |
| SLO-aware admission + shedding | `admission/` | `GW_ADMISSION_ENABLED` | `test_admission`, `admission_bench` |
| RAG chunk-affinity routing | `rag/` | `GW_RAG_ENABLED` | `test_rag`, `rag_bench` |
| Disaggregated prefill/decode | `disagg/` | `GW_DISAGG_POOLS` | `test_disagg`, `disagg_bench` |
| Multimodal capability + media affinity | `multimodal/` | `GW_MULTIMODAL_ENABLED` | `test_multimodal`, `multimodal_bench` |
| DAG cache-locality scheduler | `dag/` | `GW_DAG_ENABLED` | `test_dag`, `dag_bench` |
| SLO autoscaler | `autoscale/` | `GW_AUTOSCALE_ENABLED` | `test_autoscale` |
| Semantic cache / LoRA / TTFT predictor | `extensions/` | per-flag | `test_semantic_cache`, `test_lora`, `test_ttft_predictor` |
| **Horizontal scaling** (shared prefix+load+circuit+drain) | `cluster/` | `GW_CLUSTER_ENABLED` | `test_cluster*` (32+ tests) |
| Replication transports: in-mem / Redis / **gossip** | `cluster/bus` | `GW_CLUSTER_TRANSPORT` | `test_cluster_gossip` |
| Drain durability: snapshot + **LWW anti-entropy** | `cluster/{snapshot,drain_state}` | — | `test_cluster_snapshot`, `_antientropy` |
| **Control plane** (drain / inspect, fleet-wide) | `server` (`/admin/*`) | `GW_ADMIN_TOKEN` | `test_admin` |
| Observability + **per-route SLO** | `metrics`, `logging_setup`, `tracing` | default | `test_slo_metrics`, `test_logging` |
| Rust hot-path parity (+ admission) | `rust/` | — | 53 cargo tests |

Legend: ★ newest additions · "default" = always on · flags read from `GW_*` env
(see `gateway/config.py` + each package's `config.py`).
