# Implementation Reference — Prefix-Aware LLM Inference Gateway (Python MVP)

Snapshot: tag **`v1.0.0-python-mvp`**. This document is a complete, file-by-file
account of everything implemented so far. For the *why* (architecture rationale,
trade-offs, roadmap) see [DESIGN.md](DESIGN.md); this doc is the *what* and *how*.

- **Status:** Python MVP feature-complete. 61 tests passing, CI green (3.11/3.12).
- **Stack:** Python 3.11+, FastAPI, httpx, OpenTelemetry, Prometheus, Grafana, Docker.
- **Verified results:** prefix routing **99.3%** cache hit (sim) / **98.3%** (HTTP e2e)
  vs ~40–45% round-robin; multi-tenant fairness **100% vs 16%**.

---

## 1. What this system is

An OpenAI-compatible reverse proxy that sits between clients and a fleet of LLM
servers (vLLM) and routes each request to the backend that already holds its
KV-cache **prefix**, balanced against per-node load. It adds fault tolerance,
multi-tenant rate limiting/fairness, and full observability around that core.

Data-plane / control-plane split:
- **Data plane** (hot path, per request): parse → identify tenant → admission
  (rate limit) → filter by model & circuit → **route** (radix-tree prefix match +
  est-TTFT) → stream the upstream SSE response back.
- **Control plane** (background): scrape backend `/metrics`, reconcile load &
  health, drive circuit-breaker recovery.

---

## 2. Repository layout

```
gateway/
  config.py         all tunables (dataclass + GW_* env overrides)
  hashing.py        chained block hashing (mirrors vLLM APC) + stable_seed
  radix_tree.py     path-compressed prefix tree, longest-match, LRU eviction
  load_tracker.py   real-time per-backend in-flight + scraped kv_usage
  router.py         round_robin / consistent_hash / prefix_tree (est-TTFT)
  backends.py       backend registry, model filtering, health
  circuit.py        per-backend circuit breaker (closed/open/half-open)
  tenancy.py        token buckets, tenant registry, rate limiter, isolation seed
  metrics.py        Prometheus text-exposition collector
  tracing.py        OpenTelemetry provider setup
  logging_setup.py  structured JSON logging
  server.py         FastAPI app: the proxy + control-plane loop + endpoints
mock_backend/
  app.py            fake vLLM: prefix-cache sim + SSE + /metrics + cache_salt
bench/
  sim.py            algorithm-level routing benchmark (stdlib)
  e2e_inproc.py     full HTTP path through 3 in-process mock backends
  fairness_sim.py   per-tenant fairness contention demo
  loadtest.py       async load test against a running gateway
tests/              61 tests (unit + async integration)
deploy/             Prometheus config + Grafana provisioning + dashboard
docs/               DESIGN.md, IMPLEMENTATION.md
Dockerfile, docker-compose.yml, requirements*.txt, .github/workflows/ci.yml
```

---

## 3. Request lifecycle (`server.py::chat_completions`)

For `POST /v1/chat/completions`:

1. **Parse** body → `model`, `messages` (joined into a prompt string), optional
   `x-routing-strategy` header.
2. **Identify tenant** via `TenantRegistry.resolve` (Authorization bearer key →
   tenant+tier; else `anonymous`). Emit `gateway_tenant_requests_total`.
3. **Start a trace span** `chat.completion`; assign a `request_id`
   (`x-request-id` header or a fresh uuid); record `t_request`.
4. **Admission control** (if `rate_limit_enabled`): reserve
   `input_tokens + estimated_output` against the tenant's TPS bucket and 1 against
   the RPS bucket, checking the in-flight cap. On denial → **429** with
   `Retry-After` + `X-RateLimit-*` headers, error span, structured log.
5. **Candidate selection**: backends serving `model` (`registry.ids_for`), minus
   any with an **open circuit** (degrade to all if every circuit is open).
6. **Prefix isolation**: if `prefix_isolation == "tenant"`, compute a per-tenant
   hash `seed` and set `body["cache_salt"] = tenant.id`.
7. **Failover routing loop** (up to `max_retries + 1`): `router.choose` →
   `tree.insert` + `load.on_dispatch` (recorded *before* connect so concurrent
   same-prefix requests converge) → open the upstream stream. On a pre-first-byte
   connect failure: undo dispatch, record circuit failure, drop the backend,
   `gateway_retries_total++`, retry the next-best. If all fail → **502**.
8. **Success**: record `gateway_requests_total`, tenant tokens, match-block
   histogram; propagate `x-prefix-cache-hit`, set `x-gw-backend` / `x-request-id`.
9. **Stream** the SSE body back unbuffered. In the streaming `finally`: close the
   upstream, `load.on_complete`, **reconcile** the TPS bucket with actual output
   tokens, end the span (OK), and emit the per-request structured log.

---

## 4. Module-by-module

### `config.py`
A single `@dataclass Config` with all tunables and a `from_env()` classmethod that
overrides any field from `GW_<FIELD>` (bool/int/float/str coerced by current type).

| Field | Default | Meaning |
|---|---|---|
| `block_chars` | 64 | chars per block (≈16 tokens) — MVP stands in for token blocks |
| `block_tokens` | 16 | nominal tokens/block for cost math |
| `hash_cutoff_blocks` | 512 | cap hot-path hashing (~8k tokens) |
| `backend_cache_blocks` | 2000 | per-backend LRU cap (≈ num_gpu_blocks) |
| `prefill_ms_per_token` | 0.05 | linear prefill cost (MVP) |
| `service_ms_per_request` | 200.0 | queue-delay coefficient |
| `kv_pressure_cutoff` | 0.90 | exclude backend above this KV usage |
| `max_inflight` | 64 | exclude backend above this in-flight |
| `hysteresis_ms` | 5.0 | "equivalent" band for the load-spread tiebreak |
| `strategy` | prefix_tree | round_robin / consistent_hash / prefix_tree |
| `max_retries` | 2 | failover attempts before first byte |
| `circuit_fail_threshold` | 3 | consecutive failures → open |
| `circuit_cooldown_s` | 5.0 | open duration before half-open probe |
| `rate_limit_enabled` | False | enforcement opt-in |
| `tenants` | "" | `key=tenant:tier,...` |
| `prefix_isolation` | tenant | tenant / global |
| `default_output_tokens` | 256 | TPS reservation when `max_tokens` absent |
| `max_output_tokens` | 4096 | cap on output reservation |
| `chars_per_token` | 4 | prompt-token estimate for quotas |
| `log_level` | INFO | gateway logger level |
| `backends` | b0..b2 localhost | `id=url` pairs |
| `default_model` | mock-model | model all backends serve |

### `hashing.py`
- `block_hashes(prompt, block_chars, cutoff_blocks, seed=0)` → list of 64-bit
  **chained** FNV-1a hashes, one per *full* block. Chaining (`h_i = fnv(h_{i-1} ‖
  block_i)`) makes the key **prefix-exact**: block *i* matches only if the entire
  prefix up to *i* is identical — mirroring vLLM's automatic prefix caching.
  Partial trailing block is dropped (not cacheable). `cutoff_blocks` bounds CPU;
  `seed` namespaces the chain (per-tenant isolation).
- `stable_seed(s)` → deterministic process-independent FNV seed from a string
  (Python's `hash()` is per-process salted, unusable across replicas).

### `radix_tree.py`
Path-compressed (PATRICIA) tree mapping prefixes → backends that hold them.
- `_Node{edge: [hash], children: {hash→node}, holders: {backend→last_seen}}` —
  each edge carries a *run* of block hashes; node count is O(branch points).
- `match(hashes)` → `{backend: longest_contiguous_prefix_blocks}` in one downward
  walk. Uses a running set intersection across edges so a backend only counts
  while it's a holder at **every** node from the front — if a front block was
  evicted, the prefix is cold (correct KV contiguity).
- `insert(hashes, backend)` — walks/extends; on a divergent prefix it **splits**
  an edge (new intermediate node inherits the old holders), marks the backend a
  holder along its path, touches the per-backend LRU.
- Eviction: per-backend LRU bounded by `backend_cache_blocks` (measured in
  blocks = Σ edge lengths); node-granular, mirrors vLLM LRU.
- `remove_backend` — membership eviction (dead node dropped from all holders).

### `load_tracker.py`
Real-time, gateway-owned signals (not stale scrapes): `inflight[b]`,
`inflight_tokens[b]` (incremented at dispatch, decremented at completion) and
`kv_usage[b]` reconciled from scraped metrics via EWMA. Avoids herd behavior.

### `router.py`
`choose(prompt, backends, strategy, seed)` → `RouteResult{backend_id, hashes,
tokens, match_blocks}`.
- `round_robin` — rotating index (cache-blind baseline).
- `consistent_hash` — first-block hash mod N (deterministic affinity, no
  longest-match; collapses a shared-system-prompt fleet onto one node).
- `prefix_tree` — `est_TTFT[b] = prefill_ms(tokens − match·block_tokens) +
  inflight·service_ms`; hard saturation cutoff (`kv_usage>cutoff` or
  `inflight>max_inflight` excluded); among backends within `hysteresis_ms` of the
  best, pick **least-loaded + round-robin** so a tiny shared prefix can't pin all
  traffic while a large real affinity still isolates one backend.

### `backends.py`
`BackendRegistry` parses `id=url` pairs, tracks per-backend `healthy`, and
`ids_for(model)` returns healthy backends serving the model (two-stage routing:
model filter first, then affinity+load).

### `circuit.py`
`CircuitBreaker` with 3 states via lazy time math: `closed` →(threshold
consecutive failures)→ `open` →(cooldown elapsed)→ `half_open` →(probe
success)→ `closed` / →(probe failure)→ `open`. `allow`, `state`, `state_code`,
`record_success`, `record_failure`. Driven by both request outcomes and the
scrape loop, so backends recover without live traffic.

### `tenancy.py`
- `TokenBucket` — lazy-refill (O(1), no timers): `try_consume`, `deficit_seconds`,
  `adjust` (refund/debit). Starts full; `ts=0` so explicit test clocks work.
- `Tier` / `Tenant` + `DEFAULT_TIERS` (gold 50rps/100k tps/64; silver
  20/40k/24; bronze 5/10k/8; anonymous 2/4k/4).
- `TenantRegistry.resolve(headers)` — Authorization bearer key → tenant, else
  anonymous.
- `RateLimiter.admit(tenant, cost)` — in-flight cap, then TPS bucket, then RPS
  bucket (TPS refunded if RPS then fails); `release(tenant, reserved, actual)`
  frees the slot and reconciles the TPS bucket with actual output.
- `tenant_seed(id)` — isolation seed (via `stable_seed`).

### `metrics.py`
`MetricsCollector` — instance-based (not a global registry) counters/gauges/
histograms rendered in **Prometheus text exposition format** (`render()`).
Histograms keep cumulative `le` buckets + `_sum` + `_count`. Production swap-in:
`prometheus_client`.

### `tracing.py`
`setup_tracing(exporter)` (idempotent provider) + `get_tracer()`. No-op /
zero-overhead until an exporter is configured (OTLP via
`OTEL_EXPORTER_OTLP_ENDPOINT`, or `InMemorySpanExporter` in tests).

### `logging_setup.py`
`JsonFormatter` + `configure_logging(level)` + `log_event(logger, msg, **fields)`
(drops None fields). One JSON line per request carrying `request_id` + `trace_id`
→ logs correlate with traces and metrics.

### `server.py`
The FastAPI app and module-level singletons (`registry`, `tree`, `load`,
`router`, `metrics`, `breaker`, `tenants`, `limiter`, `log`, `tracer`). Endpoints:
- `POST /v1/chat/completions` — the lifecycle in §3 (SSE streaming via
  `StreamingResponse` over `httpx` `aiter_raw`, header propagation).
- `GET /metrics` — refresh gauges, return exposition text.
- `GET /healthz` — backend health + in-flight snapshot.
- `lifespan` — owns the shared `httpx.AsyncClient` and starts `scrape_loop`
  (polls each backend `/metrics` every 2s, updates load/health, drives the
  breaker, evicts dead backends from the tree).

### `mock_backend/app.py`
`create_app(node_id, cap_blocks, prefill_s_per_block)` factory → a fake vLLM:
its own contiguous-prefix LRU cache (honoring `cache_salt` for per-tenant
isolation), simulated prefill delay, OpenAI-style SSE token stream, and a JSON
`/metrics` (kv_usage, running). Reports `x-prefix-cache-hit`.

---

## 5. Observability surface

**Metrics** (`/metrics`):
- counters: `gateway_requests_total{strategy,backend}`,
  `gateway_cache_hits_total{backend}`, `gateway_cache_misses_total{backend}`,
  `gateway_errors_total{code}`, `gateway_retries_total`,
  `gateway_tenant_requests_total{tenant}`,
  `gateway_tenant_throttled_total{tenant,reason}`,
  `gateway_tenant_tokens_total{tenant}`
- histograms: `gateway_routing_seconds` (gateway's own added latency),
  `gateway_prefix_match_blocks`
- gauges: `gateway_backend_up`, `gateway_inflight`, `gateway_kv_usage`,
  `gateway_circuit_state`, `gateway_tenant_inflight`

**Trace span** `chat.completion` attributes: `request.id`, `llm.model`,
`routing.strategy`, `tenant.id`, `routing.backend`, `routing.match_blocks`,
`routing.retries`, `http.status_code`, `cache.hit`, `ratelimit.reason`,
`output.tokens`.

**Log fields** (per request): `request_id`, `trace_id`, `tenant`, `model`,
`strategy`, `backend`, `match_blocks`, `cache_hit`, `retries`, `output_tokens`,
`status`, `reason`, `duration_ms`.

**Grafana** (`deploy/`): Prometheus scrape + provisioned datasource + 9-panel
dashboard (cache hit rate, routing-latency p50/p95/p99, per-backend req/inflight/
kv/circuit, errors+failovers, per-tenant requests/throttles).

---

## 6. Tests (61) and benchmarks

| File | n | Covers |
|---|---|---|
| test_hashing.py | 6 | determinism, prefix-exact chaining, cutoff, full-blocks-only |
| test_radix_tree.py | 9 | longest match, branching, multi-backend, split, eviction, remove |
| test_router.py | 5 | rr cycle, consistent-hash determinism, affinity, cutoff, balance |
| test_load_tracker.py | 3 | accounting, no-negative, EWMA |
| test_circuit.py | 6 | open/cooldown/half-open/reopen/isolation/state_code |
| test_metrics.py | 4 | counters, gauges, histogram buckets, exposition headers |
| test_tenancy.py | 12 | buckets, resolution, RPS/TPS, refund, inflight cap, reconcile, seed |
| test_server.py | 8 | stream+headers, warm cache, 503, /metrics, 502, failover, 429, isolation |
| test_tracing.py | 2 | success span attrs, 429 error span |
| test_fairness.py | 3 | per-tenant isolation vs shared starvation |
| test_logging.py | 3 | JSON formatter, None-drop, request↔log correlation |

**Benchmarks** (`python bench/<name>.py`):
- `sim.py` — routing hit rate: prefix_tree **99.3%** vs round-robin 40.5% / 62.0%.
- `e2e_inproc.py` — full HTTP path: prefix_tree **98.3%**, balanced.
- `fairness_sim.py` — polite tenant served **100%** (per-tenant) vs **16%** (shared).

CI (`.github/workflows/ci.yml`): pytest + the three benchmarks as gates, on 3.11 & 3.12.

---

## 7. Key decisions & bugs found (during build)

- **est-TTFT calibration**: a too-strong queue term first swamped affinity
  (66%→98% after making the soft term a tiebreaker + hard cutoff for balance).
- **`match` contiguity**: enforced contiguous-from-front so front-block eviction
  correctly cools a prefix (99.3% in sim).
- **Dispatch timing**: record tree/load state *before* connect so concurrent
  same-prefix requests converge instead of duplicating cache (recovered e2e 98.3%).
- **Token-bucket init** seeded `ts` from `monotonic()` → broke explicit test
  clocks; fixed to `ts=0` (idle bucket refills to full anyway).
- **Backend cache sharing** would have defeated tenant isolation → added
  `cache_salt` so the *backend's* cache is namespaced, not just the routing tree.

---

## 8. Boundaries (not yet implemented) → next phase

- **Real vLLM on GPUs** — current proof is correctness + (hardware-agnostic)
  cache-hit rate; the TTFT-reduction number needs real GPUs.
- **Rust hot-path rewrite** (`rust/`) — data-plane **core + axum/reqwest
  streaming proxy + Prometheus metrics + circuit breaker + failover + tenancy
  (RPS/TPS limits, 429 + Retry-After + X-RateLimit-*, per-tenant prefix
  isolation via `cache_salt`)** ported & tested (**42 Rust unit tests**), plus
  fast Rust mock + Rust loadgen. **Measured (see [BENCHMARKS.md](BENCHMARKS.md))**:
  +0.8 ms added latency vs Python's +3.8 ms (~4.5× lower); parity-port 9,822
  req/s vs Python 209 at c=64 (~47× higher); p99 13.5 ms vs 399.5 ms (~30× lower).
  Next: structured JSON logging for full parity, then a final parity-fair benchmark.
- Token-accurate tokenization (vs char-blocks), backend Prometheus-format
  scraping (mock emits JSON), weighted fair queuing, multi-replica + shared
  prefix state (etcd), disaggregated prefill/decode routing.
