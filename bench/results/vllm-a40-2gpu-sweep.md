# Real-vLLM TTFT benchmark — 2× A40 (canonical GPU validation run)

Raw output from `bench/run_vllm_benchmark.sh` on RunPod, 2× NVIDIA A40.
This is the data behind [docs/BENCHMARKS.md §D](../../docs/BENCHMARKS.md).

```
Model:        Qwen/Qwen2.5-1.5B-Instruct  (served as: mock-model)
Hardware:     NVIDIA A40 (x2, one vLLM process per GPU)
vLLM:         0.7.3
Mem util cap: 0.20   +  --num-gpu-blocks-override 8000  (KV ~128K tok ~ 18.8 docs)
Max model len: 16384
Workload:     30 docs x 12000 chars (~6.8K tokens/doc), n=600/point, warmup=150
Protocol:     fresh vLLM per strategy (clean cumulative gauge); TTFT client-side
```

## round_robin (fresh vLLM)

```
round_robin c=1    n=600  errors=0  avg_match_blocks=  0.0  TTFT p50= 80.8  p95=309.0  p99=315.1  mean=166.4 (ms)
round_robin c=8    n=600  errors=0  avg_match_blocks=  0.0  TTFT p50=317.9  p95=643.5  p99=879.5  mean=309.7 (ms)
round_robin c=32   n=600  errors=0  avg_match_blocks=  0.0  TTFT p50=847.5  p95=2041.1 p99=2362.9 mean=893.1 (ms)
[after round_robin] vllm:gpu_prefix_cache_hit_rate:  b0 0.6153  b1 0.6364  fleet 0.6259
```

## prefix_tree (fresh vLLM)

```
prefix_tree c=1    n=600  errors=0  avg_match_blocks=166.6  TTFT p50= 66.4  p95=280.7  p99=303.9  mean= 77.4 (ms)
prefix_tree c=8    n=600  errors=0  avg_match_blocks=166.7  TTFT p50=310.9  p95=783.7  p99=821.4  mean=304.3 (ms)
prefix_tree c=32   n=600  errors=0  avg_match_blocks=175.0  TTFT p50=914.5  p95=1861.7 p99=2444.1 mean=935.9 (ms)
[after prefix_tree] vllm:gpu_prefix_cache_hit_rate:  b0 0.7213  b1 0.7311  fleet 0.7262
```

## Headline

- **c=1 mean TTFT: 166.4 → 77.4 ms (−53%)** — prefix routing avoids the
  ~6.8K-token recompute that round_robin pays on cache misses.
- **vLLM prefix-cache hit rate: 62.6% → 72.6% (+10 pp)**.
- **avg_match_blocks 0 → 166** — the gateway routes to the backend holding each
  prefix; round_robin never matches.
- The advantage is largest at low concurrency and is traded for load balance as
  concurrency rises (c=8 tied; c=32 mean tied but −9% p95). See §D for the read.
