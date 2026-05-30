# Benchmarks: Python gateway vs Rust gateway

Measures the **gateway's own overhead** — the cost the proxy adds on top of the
backend. Run on a single Windows machine; Rust binaries built `--release`.

## TL;DR

Two Rust gateway flavors are reported below, on the same fast backend:

- **lean Rust** — routing + streaming only (the original hot-path port).
- **parity Rust** — adds Prometheus metrics + circuit breaker + failover (matches
  the Python gateway's per-request work; tenancy/structured logging still TODO).

| metric (c=64) | Python | lean Rust | **parity Rust** | parity ratio |
|---|---|---|---|---|
| throughput | 209 req/s | 16,651 req/s | **9,822 req/s** | **~47× higher** |
| p99 latency under load | 399.51 ms | 7.38 ms | **13.50 ms** | **~30× lower** |

| metric (c=128) | Python | lean Rust | **parity Rust** | parity ratio |
|---|---|---|---|---|
| throughput | 212 req/s | 15,220 req/s | **9,213 req/s** | **~43× higher** |
| p99 latency under load | 922.42 ms | 16.44 ms | **27.26 ms** | **~34× lower** |

| metric (c=1) | Python | lean Rust |
|---|---|---|
| added latency (backend subtracted) | +3.8 ms | **+0.8 ms** (~4.5× lower) |

The Rust hot path is a real, measured optimization — not just lower per-request
overhead, but a fundamentally different throughput regime under load. The
parity port narrows the gap (extra per-request work) but the win stays large.

## Methodology

Two complementary tests:

### A. Per-request overhead — concurrency 1
- Backend: Python `uvicorn` mock with `MOCK_TOKEN_DELAY_S=0`, `MOCK_PREFILL_S_PER_BLOCK=0`.
- Load: `bench/latency_bench.py` (Python `httpx` async client, n=400, c=1).
- `added latency = gateway_latency − direct_latency` at the same concurrency.
- At c=1 there's no queuing, so the difference is pure per-request overhead.

### B. Throughput + tail — concurrency 64/128
- Backend: **`rust/src/bin/mockbackend.rs`** — a tiny `axum` mock that returns
  an instant static SSE response (~43k req/s ceiling, so the **backend isn't
  the bottleneck**).
- Load: **`rust/src/bin/loadgen.rs`** — `tokio + reqwest` driver (the Python
  `httpx` client capped at ~370 req/s, too slow to differentiate the gateways'
  ceilings).
- 20–30k requests per run.

Both tests hit the **same** mock backend for direct/Python-gateway/Rust-gateway
runs, so any gateway-specific overhead shows up cleanly.

## Results — A. concurrency 1 (Python mock, delays=0)

| target | throughput | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|
| direct (backend floor) | 218 req/s | 4.33 | 5.89 | 7.03 | 4.55 |
| python gateway | 120 req/s | 7.94 | 11.42 | 15.14 | 8.31 |
| **rust gateway** | **184 req/s** | **5.12** | **6.88** | **12.84** | **5.39** |

→ **added latency:** Python **+3.8 ms**, Rust **+0.8 ms**.

## Results — B. throughput + tail (Rust mock + Rust loadgen)

| target | conc | throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|---|
| direct (backend ceiling) | 64 | 43,135 | 1.34 | 2.47 | 3.37 | 1.45 |
| direct (backend ceiling) | 128 | 43,607 | 2.70 | 4.84 | 6.55 | 2.87 |
| python gateway | 64 | 209 | 305.17 | 360.25 | 399.51 | 304.69 |
| python gateway | 128 | 212 | 577.02 | 805.89 | 922.42 | 598.08 |
| **rust gateway** | 64 | **16,651** | **3.64** | **5.95** | **7.38** | 3.79 |
| **rust gateway** | 128 | **15,220** | **7.89** | **13.36** | **16.44** | 8.28 |

→ Rust gateway sustains **~70–80× the throughput** of the Python gateway with
**~50–60× lower p99**, against the same fast backend.

## Results — C. parity Rust gateway (metrics + circuit + failover)

Same setup; the Rust gateway now emits per-request Prometheus metrics (counters,
gauges, latency histogram) and runs the circuit-breaker + failover loop just
like the Python gateway.

| target | conc | throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|---|
| parity Rust gateway | 64 | 9,822 | 6.21 | 10.89 | 13.50 | 6.43 |
| parity Rust gateway | 128 | 9,213 | 13.26 | 22.21 | 27.26 | 13.70 |

The added per-request work costs ~40% throughput vs the lean port (16.6k →
9.8k req/s) — expected, and the parity gap vs Python stays huge (~47×).

## Honest caveats

- **Still ahead of full parity:** the Rust gateway doesn't yet have
  tenancy/rate-limiting or structured JSON logging. Adding those will narrow
  the gap a bit more; the architecture conclusion (Python's GIL + per-request
  Python work is the limiter, not the algorithm) holds either way.
- **Single-machine numbers.** The load generator, gateway, and backend share
  one host. Absolute throughput would be higher on separate machines, but the
  *relative* gap between Python and Rust gateways is the meaningful signal.

## Reproduce

```bash
# fast Rust mock (backend ceiling ~43k req/s)
cd rust && cargo build --release
./target/release/mockbackend 127.0.0.1:9101

# choice 1 — Python gateway
GW_BACKENDS="b0=http://127.0.0.1:9101" GW_LOG_LEVEL=WARNING \
  python -m uvicorn gateway.server:app --host 127.0.0.1 --port 8000

# choice 2 — Rust gateway (release)
GW_BACKENDS="b0=http://127.0.0.1:9101" ./target/release/gateway   # :8000

# fast Rust load generator (point at the backend, then either gateway)
./target/release/loadgen --url http://127.0.0.1:9101 --n 20000 -c 64  --label direct
./target/release/loadgen --url http://127.0.0.1:8000 --n 20000 -c 64  --label gateway
```
