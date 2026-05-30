#!/usr/bin/env bash
# Real-vLLM TTFT benchmark — turnkey runner.
#
# Boots two vLLM backends (one per GPU), the gateway, warms the prefix cache,
# runs round_robin vs prefix_tree at c=1/8/32, writes the results to
# bench/results/vllm-<timestamp>.md, then tears everything down.
#
# Prereqs on the GPU box:
#   - 2 GPUs visible (CUDA_VISIBLE_DEVICES will use 0 and 1)
#   - `git clone https://github.com/vikask3906/llm-inferance && cd llm-inferance`
#   - `pip install -r requirements-dev.txt`
#
# Single command:
#   bash bench/run_vllm_benchmark.sh
#
# Env overrides:
#   MODEL=Qwen/Qwen2.5-7B-Instruct  (default 1.5B)
#   GPUS="0 1"                       (which GPUs to use)
#   PORT0 / PORT1 / GW_PORT          (defaults 9001 / 9002 / 8000)

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GPUS=( ${GPUS:-0 1} )
PORT0="${PORT0:-9001}"
PORT1="${PORT1:-9002}"
GW_PORT="${GW_PORT:-8000}"

ROOT="$(pwd)"
RESULTS_DIR="$ROOT/bench/results"
LOG_DIR="$ROOT/bench/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_DIR/vllm-$TS.md"

cleanup() {
    echo
    echo "Tearing down..."
    [[ -n "${GW_PID:-}" ]]    && kill "$GW_PID"    2>/dev/null || true
    [[ -n "${VLLM0_PID:-}" ]] && kill "$VLLM0_PID" 2>/dev/null || true
    [[ -n "${VLLM1_PID:-}" ]] && kill "$VLLM1_PID" 2>/dev/null || true
    sleep 2
    pkill -f "vllm serve"   2>/dev/null || true
}
trap cleanup EXIT

echo "=== vLLM benchmark — $TS ==="
echo "Model:    $MODEL"
echo "GPUs:     ${GPUS[*]}"
echo "Results → $OUT"
echo

# 1) Install vLLM if missing
if ! command -v vllm >/dev/null 2>&1; then
    echo ">> installing vLLM (may take a few minutes)..."
    pip install -q vllm
fi

# 2) Launch vLLM backends (one per GPU). Models download on first run.
echo ">> starting vLLM on GPU ${GPUS[0]} (port $PORT0)..."
CUDA_VISIBLE_DEVICES="${GPUS[0]}" nohup vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT0" \
    --enable-prefix-caching --max-model-len 8192 \
    > "$LOG_DIR/vllm0.log" 2>&1 &
VLLM0_PID=$!

echo ">> starting vLLM on GPU ${GPUS[1]} (port $PORT1)..."
CUDA_VISIBLE_DEVICES="${GPUS[1]}" nohup vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT1" \
    --enable-prefix-caching --max-model-len 8192 \
    > "$LOG_DIR/vllm1.log" 2>&1 &
VLLM1_PID=$!

# 3) Wait for vLLM to be ready (model download + load can take a few minutes)
echo ">> waiting for vLLM backends to come up (~2-10 min)..."
for i in $(seq 1 300); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT0/v1/models" \
       && curl -sf -o /dev/null "http://127.0.0.1:$PORT1/v1/models"; then
        echo "   both backends ready ($((i*2))s)."
        break
    fi
    sleep 2
done

# 4) Start the Python gateway pointed at both
echo ">> starting gateway on port $GW_PORT..."
GW_BACKENDS="b0=http://127.0.0.1:$PORT0,b1=http://127.0.0.1:$PORT1" \
GW_PREFIX_ISOLATION=global \
GW_LOG_LEVEL=WARNING \
nohup python -m uvicorn gateway.server:app \
    --host 127.0.0.1 --port "$GW_PORT" --log-level warning \
    > "$LOG_DIR/gateway.log" 2>&1 &
GW_PID=$!

for i in $(seq 1 60); do
    curl -sf -o /dev/null "http://127.0.0.1:$GW_PORT/healthz" && break
    sleep 1
done

# 5) Warm up — populate prefix cache on whichever backend the router lands on
echo ">> warming up prefix cache..."
python bench/loadtest.py --url "http://127.0.0.1:$GW_PORT" \
    --strategy prefix_tree --n 50 --concurrency 4 > /dev/null

# 6) Run TTFT benchmarks
echo ">> running TTFT benchmarks (round_robin vs prefix_tree, c=1/8/32)..."
{
    echo "# Real-vLLM TTFT benchmark — $TS"
    echo
    echo "Model:       \`$MODEL\`"
    echo "Hardware:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "vLLM:        $(vllm --version 2>/dev/null || echo unknown)"
    echo
    echo '```'
    for strat in round_robin prefix_tree; do
        for conc in 1 8 32; do
            python bench/ttft_bench.py \
                --url "http://127.0.0.1:$GW_PORT" \
                --strategy "$strat" \
                --n 100 --concurrency "$conc" \
                --model "$MODEL" \
                --label "$strat c=$conc"
        done
    done
    echo '```'
    echo
    echo "**Compare:** for each concurrency, the TTFT delta from \`round_robin\` to"
    echo "\`prefix_tree\` is the real-world impact of prefix-aware routing."
} | tee "$OUT"

echo
echo "=== DONE ==="
echo "Results: $OUT"
echo "Logs:    $LOG_DIR/{vllm0,vllm1,gateway}.log"
echo
echo "Copy $OUT off the GPU box and paste into the next Claude session."
