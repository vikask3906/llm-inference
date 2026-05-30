# Benchmarks: Python gateway vs Rust gateway

Measures the **gateway's own overhead** — the cost the proxy adds on top of the
backend. Run on a single Windows machine; Rust binaries built `--release`.

## TL;DR

Three Rust gateway flavors on the same fast backend:

- **lean** — routing + streaming only (original hot-path port).
- **parity (no logs)** — adds metrics + circuit breaker + failover + tenancy.
- **full parity** — adds structured JSON logging (now matches every per-request
  thing Python does, modulo OpenTelemetry tracing).

| metric (c=64) | Python | lean Rust | parity (no logs) | **full parity** | ratio vs Python |
|---|---|---|---|---|---|
| throughput | 209 req/s | 16,651 | 9,822 | **10,922 req/s** | **~52× higher** |
| p99 latency under load | 399.51 ms | 7.38 ms | 13.50 ms | **12.20 ms** | **~33× lower** |

| metric (c=128) | Python | full parity Rust | ratio |
|---|---|---|---|
| throughput | 212 req/s | **10,283 req/s** | **~49× higher** |
| p99 latency under load | 922.42 ms | **23.94 ms** | **~39× lower** |

| metric (c=1) | Python | lean Rust |
|---|---|---|
| added latency (backend subtracted) | +3.8 ms | **+0.8 ms** (~4.5× lower) |

The Rust hot path is a real, measured optimization — not just lower per-request
overhead, but a fundamentally different throughput regime under load. **Full
parity Rust** retains a ~50× throughput / ~33× p99 advantage at the same
per-request work as the Python gateway.

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

## Results — C. full-parity Rust gateway

The Rust gateway now emits per-request Prometheus metrics, runs the circuit
breaker + failover loop, applies tenancy admission with `Retry-After` /
`X-RateLimit-*` headers, injects per-tenant `cache_salt` for prefix isolation,
and writes a structured JSON log line per request — i.e., everything the Python
gateway does on every request, *except* OpenTelemetry tracing.

| target | conc | log level | throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|---|---|
| full-parity Rust | 64  | WARN | 10,922 | 5.49 | 9.73 | 12.20 | 5.77 |
| full-parity Rust | 128 | WARN | 10,283 | 11.83 | 19.52 | 23.94 | 12.24 |
| full-parity Rust | 64  | INFO | 10,389 | 5.87 | 9.49 | 11.74 | 6.07 |
| full-parity Rust | 128 | INFO | 10,271 | 11.89 | 19.67 | 24.03 | 12.30 |

Logging at INFO costs almost nothing here because stdout is fast / redirected;
in a real deployment writing to a log aggregator the cost rises.

## Honest caveats

- **Full parity except OpenTelemetry tracing.** The Rust gateway now matches
  the Python gateway's per-request work: metrics + circuit + failover +
  tenancy + structured JSON logging. OTel tracing is the one remaining feature
  not yet ported, but it's no-op when unconfigured in Python too — so the
  benchmarks above are apples-to-apples for a realistic deployment.
- **Single-machine numbers.** Load generator, gateway, and backend share one
  host. Absolute throughput would be higher on separate machines, but the
  *relative* gap between Python and Rust gateways is the meaningful signal.

## Results — D. Real vLLM on RunPod (GPU validation)

**Goal:** prove prefix-aware routing reduces real time-to-first-token (TTFT)
against real vLLM on real GPUs. **Outcome: validated — at low load prefix_tree
cuts mean TTFT by 53% (166→77 ms) and lifts the vLLM prefix-cache hit rate by
+10 pp (63%→73%) vs round_robin under cache pressure on 2× A40. The advantage is
largest at low concurrency and is gracefully traded for load balance as
concurrency rises.**

### Setup
- **Hardware:** RunPod pod, 2× NVIDIA A40 (48 GB each).
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` served by **vLLM 0.7.3** (one
  process per GPU, `--enable-prefix-caching`, `--served-model-name mock-model`,
  `--max-model-len 16384`).
- **Cache pressure:** `--num-gpu-blocks-override 8000` caps each backend's KV
  cache at 8000 blocks ≈ 128K tokens ≈ 18.8 docs. The working set is 30 docs ×
  ~6.8K tokens. So prefix_tree's per-backend half (~15 docs ≈ 102K tokens) fits
  comfortably, while round_robin's full 30 docs (~204K) overflow — the regime
  where routing matters.
- **Workload:** `bench/ttft_bench.py` (30 large docs × 12 KB each ≈ 6.8K
  tokens/doc, seeded draw, small shared system prefix, unique question/request),
  n=600 per concurrency, swept over **c = 1 / 8 / 32**, 150-request warmup.
- **Protocol:** vLLM restarted between strategies so
  `vllm:gpu_prefix_cache_hit_rate` is a clean per-strategy reading. TTFT
  measured client-side (stopwatch from send to first SSE byte) — version- and
  metric-name-independent.

### TTFT vs concurrency (mean, ms)

| concurrency | round_robin | prefix_tree | delta |
|---|---|---|---|
| **c = 1** | 166.4 | **77.4** | **−53%** |
| c = 8 | 309.7 | 304.3 | −2% (tied) |
| c = 32 | 893.1 | 935.9 | +5% mean / **−9% p95** |

Full percentile detail at **c = 1** (the cache-locality regime):

| metric | round_robin | prefix_tree | delta |
|---|---|---|---|
| TTFT mean | 166.4 ms | **77.4 ms** | **−53%** |
| TTFT p50 | 80.8 ms | **66.4 ms** | −18% |
| TTFT p95 | 309.0 ms | **280.7 ms** | −9% |
| TTFT p99 | 315.1 ms | **303.9 ms** | −4% |
| avg_match_blocks | 0.0 | **166.6** | routing affinity works |

### vLLM prefix-cache hit rate (per-strategy, fresh vLLM each)

| strategy | b0 | b1 | fleet mean |
|---|---|---|---|
| round_robin | 0.615 | 0.636 | **0.626** |
| prefix_tree | 0.721 | 0.731 | **0.726** |

→ **+10 percentage points** (62.6% → 72.6%), a 16% relative lift, with
`avg_match_blocks` of 166 vs 0 confirming the gateway routes to the backend
already holding each prefix.

### Honest read

- **Routing affinity is unambiguous:** `avg_match_blocks ≈ 166` for prefix_tree
  vs `0` for round_robin, at every concurrency. The gateway finds cached
  prefixes and routes to the backend holding them — the core thesis works
  end-to-end against real vLLM.
- **The win is largest at low load (c=1: −53% mean TTFT).** With no queue
  pressure, the router follows prefix affinity, pins each doc to one backend,
  and avoids the ~6.8K-token recompute that round_robin pays on its cache
  misses. The mean gap (53%) exceeds the p50 gap (18%) because round_robin's
  distribution is bimodal — fast hits plus slow misses — and the misses pull
  its mean up; prefix_tree is tightly clustered near its hit latency.
- **The win shrinks as concurrency rises (c=8 tied, c=32 mean tied).** This is
  by design: under load the est-TTFT cost function spills hot prefixes to the
  less-loaded backend to avoid pinning all traffic to one node, trading cache
  locality for balance. Even so, prefix_tree keeps a **−9% p95** tail at c=32 —
  it sheds the worst cache-miss tail while staying balanced.
- **Hit-rate gap is clean (+10 pp) because vLLM was restarted per strategy**, so
  the cumulative gauge reflects only that strategy's run.
- **This is a conservative floor, not a ceiling.** A40s have ample bandwidth and
  the cap creates moderate pressure. The sim's 99% vs 40% gap (bench/matrix.py)
  uses a tighter per-backend cap and more backends; the GPU run confirms the
  direction and that the magnitude scales with pressure and inversely with load.

### What would widen the gap further

- **Tighter cache** (`NUM_GPU_BLOCKS=7000` or smaller): round_robin thrashes
  harder while prefix_tree's half still fits.
- **More backends** (4–8): more room for affinity before load-spilling forces
  replication, so the c=8/c=32 rows would separate too.
- **Larger working set** (100+ docs): overwhelms the cache under every strategy
  → back toward the sim's 99% vs 40% hit-rate regime.

### Reproduce (one command on a 2-GPU pod)

```bash
# see docs/GPU_RUNBOOK.md for the full step-by-step
export GW_BACKEND_CACHE_BLOCKS=12000
NUM_GPU_BLOCKS=8000 GPUS="0 1" GPU_MEM_UTIL=0.20 bash bench/run_vllm_benchmark.sh
```

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
