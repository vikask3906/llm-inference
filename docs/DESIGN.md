# Prefix-Aware LLM Inference Gateway — Design Doc

> An open-source, platform-agnostic gateway that routes OpenAI-compatible LLM
> requests to the backend that already holds the request's KV-cache prefix,
> balanced against per-node saturation. A standalone version of Google's GKE
> Inference Gateway.

---

## 1. Problem & motivation

LLM inference is **stateful** at the hardware level, but HTTP load balancers are
**stateless**. Transformer inference has two phases:

- **Prefill** (compute-bound): the whole prompt is processed at once; cost grows
  with prompt length. Dominates **time-to-first-token (TTFT)** for long prompts.
- **Decode** (memory-bound): tokens are generated one at a time, reusing the
  intermediate attention state stored in GPU memory — the **KV cache**.

vLLM implements **automatic prefix caching (APC)**: KV blocks for a prompt prefix
are kept in VRAM and reused if a later request shares that prefix, skipping its
prefill. But a round-robin balancer is blind to this: if request A (a 10k-token
document) lands on GPU 1 and identical request B lands on GPU 2, GPU 2
recomputes all 10k tokens and allocates duplicate VRAM. The cluster wastes
compute and memory, and tail latency spikes.

Google reported that solving this with the **GKE Inference Gateway** roughly
**doubled** prefix-cache hit rate and cut **P95 TTFT by ~52%**. This project
builds that intelligence as a standalone gateway.

**Why it's relevant:** maps to big-tech serving infra (Google/Microsoft/Meta)
*and* to quant low-latency/SLO concerns (predictable tail latency, reliability,
multi-tenant fairness).

---

## 2. Goals / non-goals

**Goals**
- Maximize cluster-wide prefix-cache hit rate without overloading any node.
- OpenAI-compatible (`/v1/chat/completions`), streaming (SSE) and non-streaming.
- Add minimal, predictable latency on the hot path.
- Pluggable routing strategies; observable; fault-tolerant.

**Non-goals (MVP)**
- Token-accurate tokenization (MVP approximates with byte-blocks).
- Multi-replica gateway HA (single gateway; see §12).
- Disaggregated prefill/decode routing (noted as future work).

---

## 3. Architecture

Strict **data-plane / control-plane** split: routing decisions are hot and must
be fast; metric scraping and table maintenance are off the critical path.

```
            ┌───────────────────────── Gateway ─────────────────────────┐
 client ──▶ │ DATA PLANE (hot path)                                      │
            │   parse → filter by model → block_hashes → tree.match      │
            │   → est-TTFT score (+ cutoffs, hysteresis) → pick backend  │
            │   → tree.insert → in-flight++ → proxy SSE stream           │
            │        ▲ lock-free reads of routing state                  │
            │ CONTROL PLANE (off hot path)                               │
            │   scrape /metrics → reconcile load + health → update state │
            └────────────────────────────┬──────────────────────────────┘
                       ┌──────────────────┼──────────────────┐
                       ▼                  ▼                  ▼
                   ┌───────┐          ┌───────┐          ┌───────┐
                   │ vLLM  │          │ vLLM  │          │ vLLM  │
                   │  b0   │          │  b1   │          │  b2   │
                   └───────┘          └───────┘          └───────┘
                  (each holds its own KV prefix cache, LRU-evicted)
```

**Components**
- `RequestRouter` / `Router` — chooses a backend per request.
- `RadixTree` — approximate global prefix → backend map (longest-prefix match).
- `LoadTracker` — real-time in-flight accounting + scraped-metric reconciliation.
- `BackendRegistry` — membership, model filtering, health.
- `CircuitBreaker` — per-backend 3-state breaker (closed/open/half-open) with
  cooldown + half-open probe recovery, driven by request outcomes and scrapes.
- `MetricsCollector` — Prometheus metrics at `/metrics` (request/cache/error
  counters, routing-latency + match-block histograms, inflight/health/circuit gauges).
- `TenantRegistry` / `RateLimiter` — tenant resolution + per-tenant RPS/TPS token
  buckets and in-flight caps (admission control, before routing).
- `tracing` — OpenTelemetry: one `chat.completion` span per request (model,
  tenant, backend, cache hit, retries, status, output tokens), OTLP-exportable.
- `server` — async reverse proxy + SSE streaming + failover + control-plane loop.

---

## 4. Request lifecycle

1. Parse request → extract `model`, build prompt string from `messages`.
2. Compute **chained block hashes** `[h1..hk]` (MVP: byte-blocks; Phase 2: tokens).
3. Filter candidates → healthy backends serving `model`.
4. `tree.match(hashes)` → `match[B]` (cached prefix blocks) for each candidate.
5. Score `est_TTFT[B]`; apply saturation cutoff + hysteresis; `argmin`.
6. Commit: `inflight[B]++`, `tree.insert(hashes, B)`.
7. Proxy the upstream SSE stream back; on completion `inflight[B]--`.

---

## 5. Routing algorithm

### 5.1 Block hashing (mirrors vLLM APC)
KV cache is block-structured (vLLM default 16 tokens/block). Block *i*'s key is a
**chained hash**: `h_i = H(h_{i-1}, block_i)`, making caching **prefix-exact** —
block *i* hits only if the entire prefix up to *i* is identical. The gateway
mirrors this so its "match length" corresponds to the backend's real hit length.
Only **full blocks** are hashed (partial trailing blocks aren't cacheable).

**Hashing cutoff:** only the first `hash_cutoff_blocks` (~8k tokens) are hashed,
bounding hot-path CPU. Trade-off: underestimates `match[B]` for deep-sharing
workloads (multi-turn agents, same-doc RAG), which can cause mild over-spill;
routing usually still picks the right node since a partial match is enough to win.

### 5.2 Radix tree (longest-prefix match)
One tree shared across backends; path from root to a node = a prefix. Each node
holds `holders: {backend_id → last_seen}`. On dispatch to B, B is recorded as a
holder on **every node along its path** (B caches all prefixes of what it
processed). Then:

> `match[B]` = the deepest matched-path node where `B ∈ holders` — found in a
> single O(matched_depth) downward walk for *all* candidates at once.

The tree is **path-compressed** (radix/PATRICIA): a chain of single-child blocks
is one edge carrying a run of hashes, so node count collapses to O(branch points)
while memory stays O(unique blocks); a divergent insert splits an edge. A flat
`prefix→backend` hash (the consistent-hash baseline) *cannot* express
longest-prefix match — that's why it loses.

**Tie-break / load spread.** Among backends whose est-TTFT is within hysteresis
of the best, the router spreads by least-load + round-robin, so a tiny shared
prefix (e.g. a common system prompt) doesn't pin all traffic to one node, while a
large real affinity still keeps a single backend the sole winner. Dispatch state
(tree belief + in-flight) is recorded *before* connecting, so concurrent requests
for the same prefix converge instead of duplicating cache.

### 5.3 Eviction model
The tree is an **approximation** of what each backend *still* holds (backends
evict LRU under VRAM pressure). Three tiers:

- **Tier 1 (MVP):** per-backend LRU capped at ~`num_gpu_blocks`; evicting the LRU
  tail removes B from those nodes — mirrors vLLM's own LRU, no backend changes.
- **Tier 2 (Phase 2):** closed-loop correction from observed TTFT / vLLM's
  `gpu_prefix_cache_hit_rate` — demote beliefs that turned out to be misses.
- **Tier 3 (ideal):** backend pushes block add/evict events (high coupling).

Distinct from **membership eviction**: a failed health check drops a backend from
*all* holders immediately.

### 5.4 Cost function (est-TTFT)
Convert affinity and load to the **same unit (ms)** and minimize:

```
est_TTFT[B] = queue_delay_ms[B]                                  # load
            + prefill_ms(prompt_tokens − match[B]·block_tokens)  # affinity
            + overflow_penalty_ms[B]                             # load
route to argmin_B est_TTFT[B]
```

- MVP `prefill_ms` is **linear**. Phase 2 uses a **bilinear** service-time model
  `a·(N−m)·N + b·(N−m)` fit from real timings (new tokens still attend over the
  full prefix; for ranking one request N is fixed, so the quadratic's real value
  is **queue-delay accuracy**, not per-candidate ranking).

**Guardrails**
- **Hard saturation cutoff:** if `kv_usage[B] > 0.9` or `inflight[B] > Qmax`,
  *exclude* B entirely (affinity yields to saturation).
- **Hysteresis:** an alternative must beat the current best by a margin to switch
  (prevents flapping on noisy load samples).

**Emergent hot-prefix replication:** a viral prefix drives one node's queue up
until `est_TTFT[fresh_node] < est_TTFT[hot_node]`; the gateway spills to a second
node, which then caches the prefix too. Replication falls out of the cost model
rather than being hand-coded.

### 5.5 Real-time load accounting
Routing on scraped Prometheus metrics alone (seconds stale) causes **herd
behavior**. The gateway trusts its **own** `inflight` / `inflight_tokens`
counters (updated synchronously at dispatch/completion) and folds in scraped
`kv_usage` via EWMA as slower ground truth.

---

## 6. Key data models

```python
TreeNode { children: {hash → TreeNode}; holders: {backend_id → last_seen} }
RouteResult { backend_id, hashes, tokens, match_blocks }
Backend { id, url, model, healthy }
LoadTracker { inflight, inflight_tokens, kv_usage }  # per backend_id
Config { block_chars, block_tokens, hash_cutoff_blocks, backend_cache_blocks,
         prefill_ms_per_token, service_ms_per_request, kv_pressure_cutoff,
         max_inflight, hysteresis_ms, strategy }
```

---

## 7. Baselines (so the numbers mean something)
1. **round_robin** — cache-blind (worst hit rate).
2. **consistent_hash (first block)** — deterministic affinity, no longest-match,
   no eviction awareness; collapses families sharing a system prompt onto one node.
3. **prefix_tree (this design)** — should beat both on hit rate *and* balance.

---

## 8. Benchmark methodology & results

Workload: every request = shared system prompt + one of 15 documents + a unique
question. The shared system prompt means all requests share their first block.
Backend KV cap = 600 blocks; fleet of 3. Cache hits are measured by **independent
per-backend cache models** (the backend's own truth), not the gateway's belief.

| Scenario | round-robin | consistent-hash | **prefix-tree** |
|---|---|---|---|
| sim, uniform docs | 40.5% | 40.1% (load → 1 node) | **97.6%** (balanced) |
| sim, skewed/hot docs | 62.0% | 62.7% (load → 1 node) | **96.6%** (balanced) |
| HTTP e2e (ASGI) | 45.2% | 45.8% (load → 1 node) | **98.3%** (balanced) |

Prefix-aware routing ≈ **2.4× the cache hit rate** while keeping load balanced;
the est-TTFT cutoff holds balance even under hot-document skew.

Reproduce: `python bench/sim.py` · `python bench/e2e_inproc.py` ·
`docker compose up --build` then `python bench/loadtest.py --strategy prefix_tree`.

---

## 9. Concurrency & complexity
- `match` O(matched_depth); `insert`/`evict` O(path) / O(1) amortized; memory
  O(unique fleet blocks) bounded by per-backend caps.
- Tree is read on every request, written on every dispatch → reads must not block.
  - **MVP (Python):** single event loop; mutations safe without locks.
  - **Phase 2 (Rust):** `arc-swap`/RCU copy-on-write snapshots for lock-free reads;
    deferred (epoch-based) reclamation so freeing a subtree never stalls readers.

---

## 10. Trade-offs & decisions

- **Python MVP → Rust hot path.** The intellectual core (routing + hit-rate
  proof) needs no fast proxy, so Python proved it fastest; Rust later matches
  C++'s perf ceiling *memory-safely* and yields a measured before/after story.
  Control plane stays Python (off hot path, faster policy iteration).
- **est-TTFT calibration bug (found during benchmarking):** the soft queue term
  initially swamped the affinity term (queue ~200ms vs ~20ms prefill saving),
  bouncing cached prefixes between nodes. Fix: make the soft term a gentle
  tiebreaker and let the **hard cutoff** provide balance. Hit rate jumped 66%→98%.
- **Reviewer suggestions:** hashing cutoff → integrated (MVP). Quadratic prefill →
  Phase 2 as *bilinear* service-time model. Lazy pruning → Phase 2 as COW/epoch
  reclamation (cleaner than per-node tombstones, required for lock-free reads).

---

## 11. Failure handling
- **Safe failover:** a pre-first-byte connect failure re-routes optimally on the
  shrinking candidate set, up to `max_retries`. After the first byte, retrying
  would duplicate tokens, so a mid-stream failure propagates instead.
- **Circuit breaking:** consecutive failures open a backend's circuit; it's
  excluded from routing for `cooldown_s`, then a half-open probe either closes it
  (success) or reopens it (failure). If *all* circuits are open, the gateway
  degrades to trying all backends rather than hard-failing.
- **Automatic recovery:** the control-plane scrape loop records success/failure
  into the breaker, so a backend recovers without needing live request traffic.
- **Backend unreachable (all candidates):** gateway returns 502.
- **Health failure:** membership eviction removes the node from all holders.
- **All candidates saturated:** fall back to the least-loaded node.

---

## 12. Multi-tenant fairness & QoS
Admission control sits **before** routing; the router/est-TTFT logic is untouched.

- **Identification:** `Authorization: Bearer <key>` → `TenantRegistry` maps key →
  tenant + quota tier; unknown/missing key → a low-quota `anonymous` tenant.
- **Rate limiting:** per-tenant **token buckets** on **both RPS and TPS** (mirrors
  OpenAI RPM+TPM). TPS is the resource-aligned limit; RPS is a cheap abuse guard;
  a request must pass both, plus a per-tenant **in-flight cap** for fairness.
- **TPS accounting:** reserve `input_tokens + estimated_output` at admission
  (`max_tokens` or a default), then **reconcile** the bucket with actual streamed
  output on completion.
- **Over-quota → `429`** with `Retry-After` + `X-RateLimit-*` headers (correct
  backpressure, no gateway memory growth), not queuing.
- **Prefix isolation:** per-tenant by default — the routing hash chain is seeded by
  tenant **and** the backend is told to salt its KV cache (vLLM `cache_salt`), so
  tenants neither share nor leak (via TTFT timing) each other's cache. `global`
  keeps caches shared for a trusted single-org deployment.
- **Enforcement is opt-in** (`rate_limit_enabled`); tenant attribution metrics are
  always emitted (`gateway_tenant_{requests,throttled,tokens,inflight}_total`).

---

## 13. Roadmap (Phase 2+)
- Real **vLLM** on ≥2 cheap cloud GPUs; fit the bilinear cost model from timings.
- Observability: Prometheus `/metrics` ✓ and OpenTelemetry tracing ✓ (per-request
  spans, OTLP-exportable to Jaeger); next: Grafana dashboard (TTFT + hit rate vs
  NGINX round-robin) and Prometheus-format scraping of backends.
- **Rust** hot-path rewrite with profiled latency/throughput before/after.
- COW/epoch reclamation in the tree (path compression is implemented ✓).
- **Multi-replica gateway:** shared prefix state (here `etcd`/Redis earns its
  place) or a deterministic shared hash ring to keep prefix routing consistent.
- **Heterogeneous fleet:** route by model first, then affinity+load.
- **Multi-tenant fairness:** per-tenant RPS+TPS limits, in-flight caps, prefix
  isolation ✓; next: weighted fair queuing across tenants, priority tiers.
- **Disaggregated prefill/decode** routing (DistServe-style) as a routing axis.

---

## 14. Interview talking points
- Why round-robin is wrong for stateful LLM serving (KV/prefix cache physics).
- Radix tree + longest-prefix match vs consistent hashing — and *why* the latter
  collapses load onto one node (demonstrated: `0/3000/0`).
- The eviction model as an *approximation* of the backend's LRU, self-correcting.
- est-TTFT as a unit-unifying cost function; emergent hot-prefix replication.
- Stale-metrics herd behavior → real-time in-flight accounting.
- The calibration bug: a concrete story of profiling-then-fixing with numbers.
- Language choice as a *measured* optimization (Python baseline → Rust hot path).
- Instrumenting the gateway's own added latency (routing-latency histogram) to
  back the "<2ms overhead" claim with data rather than assertion.
- Why TPS (not RPS) is the fair unit for LLM quotas, and isolating per-tenant KV
  cache (routing seed + `cache_salt`) to close the cross-tenant TTFT side channel.
