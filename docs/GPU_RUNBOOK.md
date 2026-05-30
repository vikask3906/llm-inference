# GPU Validation Runbook

The single-page guide to running the real-vLLM TTFT benchmark on a rented GPU
pod. **Read this end-to-end before booking the pod.** Designed so the GPU
session can't be wasted again — every pitfall from the prior run
(see [`BENCHMARKS.md §D`](BENCHMARKS.md) "Real vLLM on RunPod") is baked into
`bench/run_vllm_benchmark.sh` already.

---

## TL;DR

1. Pick a **2× A100 SXM** pod on RunPod (~$3/hr; cheapest available 2-GPU option).
2. SSH in, `git clone ...llm-inferance`, `cd llm-inferance`.
3. `bash bench/run_vllm_benchmark.sh` (handles vLLM install + spin-up +
   measurement + teardown; ~30-45 min total).
4. Copy `bench/results/vllm-*.md` off the pod.

That's it. The script forces the cache-pressure regime the prior run missed,
restarts vLLM between strategies for clean gauges, and reads
`vllm:gpu_prefix_cache_hit_rate` directly (vLLM doesn't emit
`x-prefix-cache-hit`).

---

## Step 1 — Pick the right pod

**Best pick: 2× A100 SXM (80 GB), ~$1.49/hr each → ~$2.98/hr total.**

It's the cheapest *truly available* 2-GPU config on the RunPod inventory
screenshot you shared. We force cache pressure with
`--gpu-memory-utilization 0.25` (the script does this automatically), so the
80 GB isn't a problem — vLLM is told to use only ~20 GB and the workload binds.

| Pick | VRAM | $/hr | Why |
|---|---|---|---|
| **2× A100 SXM** | 80 GB | ~$2.98 | Cheapest 2-max GPU available, plenty of room for the model |
| 2× RTX PRO 6000 | 96 GB | ~$4.18 | Backup if A100 SXM is out |
| 2× H100 NVL | 94 GB | ~$6.38 | Faster but pricier — not needed for this benchmark |

**Avoid:**
- Any GPU with `1 max` (RTX 5090, A40, A6000, A5000 in the screenshot) — only one
  GPU per pod, so you can't run two backends.
- H100/H200/B200/B300 — overkill for this benchmark.
- Bigger models (7B, 14B) — they eat headroom without changing the demonstration.

**Region / storage:** any region; ~30 GB disk is enough for the Qwen-1.5B model
+ vLLM image.

**Total cost:** ~$3 for a 1-hour pod is fine. The benchmark itself takes
~25 minutes once vLLM is up; the rest is install + model download + buffer.

---

## Step 2 — On the pod, get the repo

```bash
git clone https://github.com/vikask3906/llm-inferance
cd llm-inferance
pip install -r requirements-dev.txt
```

If the pod's Python is 3.12 and pip complains about vLLM, use the pinned
versions the script will install for you (vLLM 0.7.3 wants Python ≤3.11).
Some RunPod images give you 3.10 — that's fine.

---

## Step 3 — Run the benchmark

```bash
bash bench/run_vllm_benchmark.sh
```

What it does (no interaction needed):

1. **Pins compatible versions** (`torch 2.5.1+cu121`, `vllm 0.7.3`,
   `transformers 4.49.0`) — the CUDA-version trap from the prior run.
2. **Boots two vLLM backends**, one per GPU, with:
   - `--enable-prefix-caching` (the whole point of the benchmark)
   - `--gpu-memory-utilization 0.25` (caps KV cache so the workload binds —
     the missing flag that made the prior run inconclusive)
   - `--served-model-name mock-model` (short, stable model name)
   - `nohup` (so a closed tab doesn't kill vLLM)
3. **Runs each strategy from a fresh vLLM**:
   - Reads `vllm:gpu_prefix_cache_hit_rate` baseline (≈0)
   - Warms the cache + drives 200 TTFT-measurement requests at concurrency 8
   - Reads the final gauge → the cumulative reading equals that strategy's
     own hit rate (no contamination from a prior warmup)
4. **Tears down everything** at the end and prints the result file path.

Expected duration: ~5 min install, ~10 min model download (first run),
~12 min for both strategies. Subsequent runs reuse the downloaded model.

---

## Step 4 — Read the result

The file `bench/results/vllm-<timestamp>.md` will contain:

```
round_robin n=200 c=8   n=200 errors=0 avg_match_blocks=  0.0  TTFT p50=...  p95=...  p99=...  mean=... (ms)
[after round_robin] vllm:gpu_prefix_cache_hit_rate:
  http://127.0.0.1:9001  0.XX
  http://127.0.0.1:9002  0.XX

prefix_tree n=200 c=8   n=200 errors=0 avg_match_blocks=...  TTFT p50=...  p95=...  p99=...  mean=... (ms)
[after prefix_tree] vllm:gpu_prefix_cache_hit_rate:
  http://127.0.0.1:9001  0.XX
  http://127.0.0.1:9002  0.XX
```

### What success looks like

- **prefix_tree TTFT < round_robin TTFT** at the same concurrency, by 30-60%.
- **prefix_tree cache hit rate > round_robin** by 20+ percentage points.
- **avg_match_blocks**: ≈0 for `round_robin`, >10 for `prefix_tree` (the gateway
  is finding shared prefixes and routing for cache reuse).

### What "still inconclusive" looks like (and what to do)

If both strategies land at the same high hit rate (~90%+) and similar TTFT,
**the workload still fits the KV cache**. Tighten the screws:

```bash
# tighter KV cap (forces eviction harder)
GPU_MEM_UTIL=0.15 bash bench/run_vllm_benchmark.sh

# or bigger workload
N_DOCS=400 DOC_CHARS=32000 bash bench/run_vllm_benchmark.sh
```

If `round_robin` errors out with OOM, the KV cap is too tight — raise to 0.30.

### What real failure looks like

- **vLLM never comes up.** Check `bench/logs/vllm0.log`. Most common: CUDA
  driver < the pinned CUDA toolkit. Set `PIN_VERSIONS=0` and let the script
  use whatever vLLM `pip install vllm` picks; or pick a pod with a newer
  CUDA driver.
- **Gateway never goes healthy.** vLLM isn't ready yet, or its `/health`
  endpoint isn't responding. The script's `/health` scrape is the right
  thing (fixed in the gateway after the prior run).

---

## Step 5 — Get the result off the pod

```bash
# on the pod
cat bench/results/vllm-*.md
```

Copy the output and paste into the next Claude Code session along with the
pod tear-down note. Or `scp` it off if you have keys set up.

**Then terminate the pod.** Don't leave it running.

---

## Recipe variants

### Bigger model (validates the gateway against something realistic)
```bash
MODEL=Qwen/Qwen2.5-7B-Instruct GPU_MEM_UTIL=0.50 bash bench/run_vllm_benchmark.sh
```
7B fp16 needs ~14 GB; cap at 0.50 of 80 GB = 40 GB total (model + KV).

### Skip version pinning (use the pod image's defaults)
```bash
PIN_VERSIONS=0 bash bench/run_vllm_benchmark.sh
```
Faster but risks the CUDA-version trap. Try if `pip install vllm==0.7.3`
fails.

### Single-strategy run (debugging)
Edit `run_vllm_benchmark.sh`, comment out one of:
```bash
run_one_strategy round_robin
run_one_strategy prefix_tree
```

---

## Lessons baked in (so you can't repeat them)

| Prior-run pitfall | How it's prevented now |
|---|---|
| Web-tab close killed vLLM | `nohup` on every vllm/gateway launch |
| CUDA driver mismatch wasted ~30 min | `PIN_VERSIONS=1` installs known-good torch/vllm/transformers |
| Gateway scraped `/metrics` (Prometheus text) for liveness | Gateway patched to scrape `/health` (in mock + real vLLM) |
| vLLM doesn't emit `x-prefix-cache-hit` (loadtest reported 0%) | `bench/vllm_cache_stats.py` reads `vllm:gpu_prefix_cache_hit_rate` from `/metrics` directly |
| Cumulative gauge read after BOTH strategies = meaningless | Script restarts vLLM between strategies for clean per-strategy reads |
| KV cache too large → both strategies hit 99% → no discrimination | `--gpu-memory-utilization 0.25` caps KV so workload binds |
| Workload too small (50 docs × 8 KB) → fits in cache | Defaults bumped to 200 docs × 24 KB; tunable via env |
| Model-name mismatch between vLLM and benchmark client | `--served-model-name mock-model` + both clients default to `mock-model` |

---

## After the GPU session

1. Paste the result into the README under "Results - GPU validation".
2. Update `docs/BENCHMARKS.md §D` with the actual delta.
3. Mark task #8 complete.

That's it. The portfolio claim shifts from "routing-quality proof (mock)" to
"routing-quality proof + measured real-vLLM TTFT reduction" — the headline
number recruiters want.
