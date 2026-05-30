#!/usr/bin/env bash
# Real-vLLM TTFT benchmark — turnkey runner.
#
# Hardened against the lessons from the prior GPU run (docs/BENCHMARKS.md §D):
#   1. nohup vLLM so closing a web-terminal tab doesn't kill it
#   2. pinned torch/vllm/transformers (the CUDA-version trap)
#   3. scrape /health (not /metrics) for liveness — already patched in gateway
#   4. read vllm:gpu_prefix_cache_hit_rate from /metrics directly
#      (vLLM does NOT emit x-prefix-cache-hit)
#   5. constrain KV with --gpu-memory-utilization 0.25 so the workload binds
#      (otherwise both strategies hit ~99% and nothing discriminates)
#   6. restart vLLM between strategies for clean gauge reads
#   7. use --served-model-name mock-model so model name is short and stable
#
# Prereqs on the GPU box:
#   - 2 GPUs visible (CUDA_VISIBLE_DEVICES uses 0 and 1)
#   - `git clone https://github.com/vikask3906/llm-inferance && cd llm-inferance`
#   - `pip install -r requirements-dev.txt`
#
# Single command:
#   bash bench/run_vllm_benchmark.sh
#
# Env overrides:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct  (default)
#   GPUS="0 1"                         (which GPUs to use)
#   PORT0 / PORT1 / GW_PORT            (defaults 9001 / 9002 / 8000)
#   GPU_MEM_UTIL=0.25                  (cap KV-cache fraction; raise for big workloads)
#   N_DOCS=200 DOC_CHARS=24000         (workload size; defaults are pressure-sized)
#   PIN_VERSIONS=1                     (force pin install; set 0 to skip)

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SERVED_NAME="${SERVED_NAME:-mock-model}"
GPUS=( ${GPUS:-0 1} )
PORT0="${PORT0:-9001}"
PORT1="${PORT1:-9002}"
GW_PORT="${GW_PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
# n_docs is sized so prefix_tree's half (n_docs/2) ~fits the KV cache while
# round_robin's full n_docs doesn't -- the regime where routing discriminates.
# ~50 large docs suits a ~170K-token cache (util 0.20 on a 48GB A40).
N_DOCS="${N_DOCS:-30}"
# ~12000 chars ~= 6.8K tokens for this repetitive text (~1.8 chars/token), so a
# doc fits under MAX_MODEL_LEN and prefix_tree's half (~15 docs ~102K tokens)
# fits the ~170K-token cache while round_robin's full 30 (~204K) don't -- the
# regime where routing discriminates. Validated on 2x A40 at util 0.20.
DOC_CHARS="${DOC_CHARS:-12000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"   # headroom so a big doc is never rejected
TTFT_N="${TTFT_N:-600}"            # measurement requests (keep > n_docs for revisits)
# Concurrency sweep. Prefix routing's cache benefit is largest at low load
# (no queue pressure forcing the router to spill docs to balance), and shrinks
# as load rises and load-balancing takes over -- so sweep to show the curve.
TTFT_C="${TTFT_C:-1 8 32}"
WARMUP_N="${WARMUP_N:-150}"        # pre-populate the cache to steady state
PIN_VERSIONS="${PIN_VERSIONS:-1}"

ROOT="$(pwd)"
RESULTS_DIR="$ROOT/bench/results"
LOG_DIR="$ROOT/bench/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_DIR/vllm-$TS.md"

VLLM0_PID=""
VLLM1_PID=""
GW_PID=""

start_vllm() {
    local gpu="$1" port="$2" logname="$3"
    CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --host 127.0.0.1 --port "$port" \
        --enable-prefix-caching \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        > "$LOG_DIR/$logname.log" 2>&1 &
    echo $!
}

wait_vllm() {
    local port="$1" name="$2"
    for i in $(seq 1 300); do
        if curl -sf -o /dev/null "http://127.0.0.1:$port/v1/models"; then
            echo "   $name ready ($((i * 2))s)."
            return 0
        fi
        sleep 2
    done
    echo "   $name FAILED to come up; check $LOG_DIR/$name.log" >&2
    return 1
}

start_both_vllm() {
    echo ">> starting vLLM on GPU ${GPUS[0]} (port $PORT0)..."
    VLLM0_PID=$(start_vllm "${GPUS[0]}" "$PORT0" vllm0)
    echo ">> starting vLLM on GPU ${GPUS[1]} (port $PORT1)..."
    VLLM1_PID=$(start_vllm "${GPUS[1]}" "$PORT1" vllm1)
    echo ">> waiting for vLLM backends (model download + load: ~2-10 min first time)..."
    wait_vllm "$PORT0" vllm0
    wait_vllm "$PORT1" vllm1
}

stop_both_vllm() {
    [[ -n "$VLLM0_PID" ]] && kill "$VLLM0_PID" 2>/dev/null || true
    [[ -n "$VLLM1_PID" ]] && kill "$VLLM1_PID" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
}

start_gateway() {
    GW_BACKENDS="b0=http://127.0.0.1:$PORT0,b1=http://127.0.0.1:$PORT1" \
    GW_PREFIX_ISOLATION=global \
    GW_LOG_LEVEL=WARNING \
    nohup python -m uvicorn gateway.server:app \
        --host 127.0.0.1 --port "$GW_PORT" --log-level warning \
        > "$LOG_DIR/gateway.log" 2>&1 &
    GW_PID=$!
    for i in $(seq 1 60); do
        curl -sf -o /dev/null "http://127.0.0.1:$GW_PORT/healthz" && return 0
        sleep 1
    done
    echo "   gateway FAILED; check $LOG_DIR/gateway.log" >&2
    return 1
}

stop_gateway() {
    [[ -n "$GW_PID" ]] && kill "$GW_PID" 2>/dev/null || true
    sleep 2
}

cleanup() {
    echo
    echo "Tearing down..."
    stop_gateway
    stop_both_vllm
}
trap cleanup EXIT

echo "=== vLLM benchmark — $TS ==="
echo "Model:        $MODEL  (served as: $SERVED_NAME)"
echo "GPUs:         ${GPUS[*]}"
echo "Mem util:     $GPU_MEM_UTIL  (cap to force cache pressure)"
echo "Workload:     $N_DOCS docs x $DOC_CHARS chars"
echo "Results →     $OUT"
echo

# 1) Install vLLM + pin compatible torch/transformers (the prior CUDA-version trap)
if [[ "$PIN_VERSIONS" == "1" ]]; then
    if ! command -v vllm >/dev/null 2>&1; then
        echo ">> installing pinned versions (torch 2.5.1+cu121 / vllm 0.7.3 / transformers 4.49.0)..."
        pip install -q "torch==2.5.1" "vllm==0.7.3" "transformers==4.49.0"
    fi
fi

# 2) Run each strategy from a FRESH vLLM (clean gauge reads)
run_one_strategy() {
    local strat="$1"
    echo
    echo "========================================================================"
    echo ">> strategy: $strat   (fresh vLLM for clean gauge read)"
    echo "========================================================================"
    start_both_vllm
    start_gateway

    echo ">> baseline cache stats (should be ~0):"
    python bench/vllm_cache_stats.py \
        --label "baseline $strat" \
        "http://127.0.0.1:$PORT0" "http://127.0.0.1:$PORT1"

    echo ">> warmup pass ($WARMUP_N reqs, populates prefix cache to steady state)..."
    python bench/loadtest.py --url "http://127.0.0.1:$GW_PORT" \
        --strategy "$strat" \
        --n "$WARMUP_N" --concurrency 8 \
        --n-docs "$N_DOCS" --doc-chars "$DOC_CHARS" \
        --model "$SERVED_NAME" --max-tokens 8 \
        > "$LOG_DIR/warmup-$strat.log"

    echo ">> TTFT measurement (n=$TTFT_N, c={$TTFT_C}, $N_DOCS docs x $DOC_CHARS chars)..."
    for c in $TTFT_C; do
        python bench/ttft_bench.py \
            --url "http://127.0.0.1:$GW_PORT" \
            --strategy "$strat" \
            --n "$TTFT_N" --concurrency "$c" \
            --n-docs "$N_DOCS" --doc-chars "$DOC_CHARS" \
            --model "$SERVED_NAME" \
            --label "$strat c=$c" \
            | tee -a "$OUT"
    done

    echo ">> final cache stats (delta from baseline = $strat's contribution):"
    python bench/vllm_cache_stats.py \
        --label "after $strat" \
        "http://127.0.0.1:$PORT0" "http://127.0.0.1:$PORT1" \
        | tee -a "$OUT"

    stop_gateway
    stop_both_vllm
}

# 3) Write report header
{
    echo "# Real-vLLM TTFT benchmark — $TS"
    echo
    echo "Model:        \`$MODEL\` (served as \`$SERVED_NAME\`)"
    echo "Hardware:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "vLLM:         $(vllm --version 2>/dev/null || echo unknown)"
    echo "Mem util cap: $GPU_MEM_UTIL  (so workload binds; raise if OOM)"
    echo "Workload:     $N_DOCS docs x $DOC_CHARS chars = ~$((N_DOCS * DOC_CHARS / 4)) tokens working set"
    echo
    echo '```'
} > "$OUT"

run_one_strategy round_robin
run_one_strategy prefix_tree

{
    echo '```'
    echo
    echo "## How to read"
    echo
    echo "- **TTFT**: lower is better. The discriminating signal is the delta"
    echo "  from \`round_robin\` to \`prefix_tree\` at the same concurrency."
    echo "- **vllm:gpu_prefix_cache_hit_rate** (per backend, CUMULATIVE since"
    echo "  vLLM start): each strategy started from a fresh vLLM, so the"
    echo "  reading after each run is that strategy's own hit rate -- not"
    echo "  inflated by an earlier warmup as in the prior run."
    echo
    echo "**Expected if the workload binds** (cache pressure regime):"
    echo "  - prefix_tree TTFT < round_robin TTFT by 30-60%"
    echo "  - prefix_tree cache hit rate > round_robin by 20+ percentage points"
    echo
    echo "**If both strategies tie at high hit rate:** the workload still fits"
    echo "  in the KV cache. Lower \`GPU_MEM_UTIL\` (try 0.15) or raise"
    echo "  \`N_DOCS\` / \`DOC_CHARS\` and re-run."
} >> "$OUT"

echo
echo "=== DONE ==="
echo "Results: $OUT"
echo "Logs:    $LOG_DIR/{vllm0,vllm1,gateway,warmup-*}.log"
echo
echo "Copy $OUT off the GPU box (e.g. \`scp\` or pipe through stdout) and"
echo "paste into the next Claude session."
