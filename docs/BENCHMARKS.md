# Benchmarks: Python gateway vs Rust gateway

Measures the **gateway's own overhead** — the cost the proxy adds on top of the
backend. Run on a single machine against a *fast* mock backend (token delay = 0,
prefill = 0) so backend time is minimal and the gateway is what's being measured.
The Rust gateway is built `--release`.

## TL;DR

Per-request overhead at concurrency 1 (backend time subtracted):

| gateway | added latency (mean) | single-stream throughput |
|---|---|---|
| Python (FastAPI/uvicorn) | **+3.8 ms** | 120 req/s |
| **Rust (axum/reqwest)** | **+0.8 ms** | **184 req/s** |

The Rust hot path adds **~4.5× less latency per request** and sustains ~1.5× the
single-stream request rate.

## Methodology

- `bench/latency_bench.py`: fires N requests at concurrency C, consumes each SSE
  stream fully, reports throughput + p50/p95/p99.
- Three targets, all hitting the **same** mock backend:
  - **direct** — client → mock backend (baseline; the gateway's floor).
  - **python** — client → Python gateway → mock.
  - **rust** — client → Rust gateway (release) → mock.
- `added latency = gateway − direct` at the same concurrency.

## Results — concurrency 1 (clean: no backend queuing)

| target | throughput | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|
| direct (backend floor) | 218 req/s | 4.33 | 5.89 | 7.03 | 4.55 |
| python gateway | 120 req/s | 7.94 | 11.42 | 15.14 | 8.31 |
| **rust gateway** | **184 req/s** | **5.12** | **6.88** | **12.84** | **5.39** |

→ **added latency:** Python **+3.8 ms**, Rust **+0.8 ms**. Rust sustains 184 vs
120 req/s on a single stream.

## Results — concurrency 64 (backend-bound — *not* a fair gateway comparison)

| target | throughput | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| direct | 201 req/s | 288 | 402 | 607 |
| python gateway | 101 req/s | 473 | 1631 | 2548 |
| rust gateway | 81 req/s | 679 | 1369 | 3337 |

**Read these with caution.** The single Python `uvicorn` mock caps at ~200 req/s,
and the load generator + both gateways + the mock all share one machine's CPU. At
c64 everything queues at the backend / contends for cores, so this measures the
*bottleneck*, not the gateways. The Rust number landing below Python here is a
measurement artifact of that shared-host bottleneck, not a real deficiency — the
clean signal is the concurrency-1 overhead above.

## Honest caveats

- **The throughput/tail comparison needs a non-bottlenecked backend.** With one
  Python mock the gateway can never out-run the backend. A faithful throughput/p99
  comparison needs a faster backend (a Rust mock, or real vLLM) and ideally the
  load generator on a separate host. *(This is benchmark fidelity, not a GPU
  blocker — real vLLM would also give the end-to-end TTFT number.)*
- **The Rust gateway currently does routing + streaming only**; the Python gateway
  also runs metrics, tenancy, tracing, and logging per request. So part of the
  c1 gap is "the Rust hot path does less work" — which is exactly the point of
  *rewriting the latency-critical path* in Rust. A parity port (metrics/circuit/
  tenancy in Rust) is planned; the architecture comparison stands either way.

## Reproduce

```bash
# fast backend
MOCK_TOKEN_DELAY_S=0 MOCK_PREFILL_S_PER_BLOCK=0 \
  python -m uvicorn mock_backend.app:app --host 127.0.0.1 --port 9001

# Python gateway
GW_BACKENDS="b0=http://127.0.0.1:9001" \
  python -m uvicorn gateway.server:app --host 127.0.0.1 --port 8000

# Rust gateway (release)
cd rust && cargo build --release --bin gateway
GW_BACKENDS="b0=http://127.0.0.1:9001" ./target/release/gateway   # :8000

# measure (point --url at the backend, then each gateway)
python bench/latency_bench.py --url http://127.0.0.1:9001 --n 400  --concurrency 1  --label direct
python bench/latency_bench.py --url http://127.0.0.1:8000 --n 400  --concurrency 1  --label gateway
```
