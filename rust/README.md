# gateway-core (Rust)

Rust port of the Python gateway's **data-plane hot path** — the part being
rewritten for predictable sub-millisecond routing overhead and high throughput.
Mirrors the Python modules 1:1 so behaviour is identical
(`gateway/hashing.py` → `src/hashing.rs`, etc.); see
[../docs/IMPLEMENTATION.md](../docs/IMPLEMENTATION.md).

## Status

**Core ported & tested (20 unit tests); HTTP proxy smoke-tested end-to-end.**

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

Smoke test: Rust gateway → Python mock backend returns **200**, streams the SSE,
propagates `x-gw-backend` + `x-prefix-cache-hit` (verified a real cold→warm
prefix-cache hit through the Rust proxy).

**Measured wins** (vs the Python gateway, same fast backend; see
[../docs/BENCHMARKS.md](../docs/BENCHMARKS.md)): **+0.8 ms** added latency vs
+3.8 ms (~4.5× lower), **~16k req/s** at c=64 vs ~209 (~80× higher), and
**p99 7.4 ms** vs 399 ms (~54× lower) under sustained load.

**Next:** port metrics / circuit breaker / tenancy to Rust for feature parity.

## Build & run

```bash
cd rust
cargo test                 # 20 core unit tests
cargo build --bin gateway  # build the proxy
GW_BACKENDS="b0=http://127.0.0.1:9001" cargo run --bin gateway   # serve on :8000
```
