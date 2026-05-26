# gateway-core (Rust)

Rust port of the Python gateway's **data-plane hot path** — the part being
rewritten for predictable sub-millisecond routing overhead and high throughput.
Mirrors the Python modules 1:1 so behaviour is identical
(`gateway/hashing.py` → `src/hashing.rs`, etc.); see
[../docs/IMPLEMENTATION.md](../docs/IMPLEMENTATION.md).

## Status

**Full-parity port: 45 unit tests passing; HTTP proxy smoke-tested end-to-end.**
The Rust gateway now matches every per-request thing the Python gateway does
(metrics + circuit + failover + tenancy + structured JSON logging), modulo
OpenTelemetry tracing.

- `hashing.rs` — chained FNV block hashing + tenant `stable_seed`
- `radix_tree.rs` — path-compressed prefix tree: longest-contiguous match,
  edge-splitting insert, per-backend LRU eviction, membership eviction
- `router.rs` — est-TTFT routing + load-spread tiebreak (round_robin /
  consistent_hash / prefix_tree)
- `load.rs`, `config.rs` — real-time load signals + tunables
- `server.rs` — **axum + reqwest streaming proxy**: OpenAI-compatible
  `/v1/chat/completions` (routes via the core, streams the upstream SSE straight
  back) + `/healthz`. The routing lock is never held across an `.await`; an
  in-flight `Drop` guard decrements load even on client disconnect.
- `bin/gateway.rs` — `cargo run --bin gateway`
- `bin/mockbackend.rs` — fast axum mock (static SSE, ~43k req/s ceiling) used
  to un-bottleneck throughput benchmarks
- `bin/loadgen.rs` — `tokio + reqwest` load generator (the Python `httpx`
  client capped at ~370 req/s, too slow to expose the gateways' real ceilings)
- `metrics.rs` — Prometheus exposition collector (same metric names as Python so
  the Grafana dashboard works against either gateway)
- `circuit.rs` — 3-state breaker (closed/open/half-open) with cooldown
- `tenancy.rs` — TokenBucket, TenantRegistry (bearer-key resolve), RateLimiter
  (RPS + TPS + in-flight cap, refund-on-second-fail, release reconciliation),
  `tenant_seed` for per-tenant prefix isolation
- `logging.rs` — structured JSON log line per request (`request_id` correlatable
  with the `x-request-id` response header + `duration_ms` + outcome fields)

Smoke test: Rust gateway → Python mock backend returns **200**, streams the SSE,
propagates `x-gw-backend` + `x-prefix-cache-hit` (verified a real cold→warm
prefix-cache hit through the Rust proxy).

**Measured (parity gateway with metrics + circuit + failover; same fast backend;
see [../docs/BENCHMARKS.md](../docs/BENCHMARKS.md)):** ~9.8k req/s at c=64 vs
Python's ~209 (~47× higher), p99 13.5 ms vs 399.5 ms (~30× lower), and +0.8 ms
added latency at c=1 vs Python's +3.8 ms (~4.5× lower).

**Smoke-tested tenancy**: bronze tenant (rps=5) passes 10 sequential requests;
anonymous (rps=2) gets `429`s with `Retry-After: 1` + `X-RateLimit-*` headers +
JSON error body matching Python. Per-tenant metrics (`gateway_tenant_requests_total`,
`gateway_tenant_throttled_total{reason}`, `gateway_tenant_tokens_total`,
`gateway_tenant_inflight`) all flow through `/metrics`.

**Final benchmark (full-parity Rust vs Python, same fast backend, c=64):**
**10,922 req/s** vs Python's 209 → **~52× higher throughput**; **p99 12.2 ms**
vs 399.5 ms → **~33× lower tail**.

**Remaining (optional polish):** OpenTelemetry tracing port, control-plane
scrape loop for backend health.

## Build & run

```bash
cd rust
cargo test                 # 20 core unit tests
cargo build --bin gateway  # build the proxy
GW_BACKENDS="b0=http://127.0.0.1:9001" cargo run --bin gateway   # serve on :8000
```
