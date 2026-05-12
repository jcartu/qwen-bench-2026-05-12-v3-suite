#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# sweep_lib.sh — Shared library for the 2026-05-11 v2-followup study.
#
# Supports BOTH speculative decoding methods on repne/vllm:v2:
#   - FP8 + MTP (no external drafter; built into the FP8 base)
#   - BF16 + DFlash (external z-lab/Qwen3.6-27B-DFlash drafter)
#
# Method is selected per-call by the SPEC_METHOD argument: "mtp" or "dflash".
#
# Provides:
#   - launch_config()        : start vLLM with method-specific args
#   - wait_for_ready()       : poll /v1/models until 200
#   - run_throughput()       : 15-cell decode matrix (c × ctx)
#   - run_prefill()          : 5-context prefill (TTFT)
#   - run_gates()            : 4-gate sanity
#   - run_quality_at()       : HumanEval + MBPP with parameterized max_tokens
#   - stop_config()          : tear down
#   - eta_remaining()        : ETA helper
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-/tmp/qwen-bench-2026-05-11-v2-followup}"
PORT="${PORT:-11435}"
IMAGE="${IMAGE:-repne/vllm:v2}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-v2-followup}"
GPU_0_UUID="GPU-ba6334bc-6fec-5f2c-df75-a887bbca476e"
GPU_1_UUID="GPU-538bf008-7ff2-0d1d-69e9-20db81a00459"

# Models — selected by config caller
MODEL_BF16="Qwen/Qwen3.6-27B"
MODEL_FP8="Qwen/Qwen3.6-27B-FP8"
DRAFTER_DFLASH="z-lab/Qwen3.6-27B-DFlash"
SERVED_NAME="Qwen3.6-27B"

BENCH_PY="/home/josh/qwen-vllm-test/llm-inference-bench/.venv/bin/python"
BENCH_SCRIPT="/home/josh/qwen-vllm-test/llm-inference-bench/llm_decode_bench.py"
STRESS_HARNESS="/home/josh/qwen-vllm-test/bench/stress-harness/stress_harness.py"
PROBLEMS_DIR="/home/josh/qwen-vllm-test/bench/stress-harness/problems"

HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || echo '')"

PY3=python3

# ──────────────────────────────────────────────────────────────────────────────
# launch_config LABEL SPEC_METHOD MODEL DRAFTER NUM_SPEC MAX_BATCHED CAPTURE OUT_DIR
#   SPEC_METHOD: "mtp" or "dflash"
#   DRAFTER:     drafter HF repo (for dflash) or "-" (for mtp)
# ──────────────────────────────────────────────────────────────────────────────
launch_config() {
  local label="$1" method="$2" model="$3" drafter="$4"
  local num_spec="$5" max_batched="$6" capture="$7" out_dir="$8"
  mkdir -p "$out_dir"

  cat > "$out_dir/server_args.txt" <<EOF
LABEL=$label
IMAGE=$IMAGE
MODEL=$model
SPEC_METHOD=$method
DRAFTER=$drafter
TP=2 (GPU0+GPU1 by UUID)
max-num-batched-tokens=$max_batched
max-cudagraph-capture-size=$capture
num_speculative_tokens=$num_spec
attention-backend=flashinfer
gpu-memory-utilization=0.85
max-model-len=262144
max-num-seqs=128
launched_at=$(date -Iseconds)
EOF

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  # Build method-specific spec config flags
  local spec_args=()
  if [ "$method" = "mtp" ]; then
    spec_args=(
      --speculative-config.method mtp
      --speculative-config.num_speculative_tokens "$num_spec"
    )
  elif [ "$method" = "dflash" ]; then
    spec_args=(
      --speculative-config.method dflash
      --speculative-config.model "$drafter"
      --speculative-config.num_speculative_tokens "$num_spec"
      --speculative-config.use_local_argmax_reduction true
      --speculative-config.attention_backend flashinfer
    )
  else
    echo "ERROR: unknown method '$method'" >&2
    return 2
  fi

  # FP8 uses instanttensor load-format; BF16 uses default
  local load_format_args=()
  if [ "$method" = "mtp" ]; then
    load_format_args=(--load-format instanttensor)
  fi

  docker run -d --name "$CONTAINER_NAME" \
    --device "nvidia.com/gpu=${GPU_0_UUID}" \
    --device "nvidia.com/gpu=${GPU_1_UUID}" \
    --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 --network host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
    -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "${HOME}/.triton/cache:/root/.triton/cache" \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "OMP_NUM_THREADS=16" \
    -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -e "VLLM_ALLREDUCE_USE_SYMM_MEM=0" \
    -e "NCCL_P2P_LEVEL=SYS" \
    -e "NCCL_NET_GDR_LEVEL=SYS" \
    -e "NCCL_MIN_NCHANNELS=8" \
    "$IMAGE" \
      -O3 \
      --model "$model" \
      --served-model-name "$SERVED_NAME" \
      --port "$PORT" \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.85}" \
      --max-model-len 262144 \
      --max-num-seqs 128 \
      --max-num-batched-tokens "$max_batched" \
      --max-cudagraph-capture-size "$capture" \
      --language-model-only \
      --enable-auto-tool-choice \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      --enable-prefix-caching \
      --attention-backend flashinfer \
      "${spec_args[@]}" \
      "${load_format_args[@]}" \
      --default-chat-template-kwargs.preserve_thinking true \
      >/dev/null
}

# ──────────────────────────────────────────────────────────────────────────────
wait_for_ready() {
  local out_dir="$1" deadline="${2:-600}"
  local start=$(date +%s) now
  while true; do
    now=$(date +%s)
    if [ $((now - start)) -ge "$deadline" ]; then
      docker logs "$CONTAINER_NAME" > "$out_dir/server.log" 2>&1 || true
      echo "  [TIMEOUT] $((now-start))s elapsed, no /v1/models" >&2
      return 2
    fi
    if curl -s -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
      docker logs "$CONTAINER_NAME" > "$out_dir/server.log" 2>&1 || true
      echo "  [READY] in $((now-start))s" >&2
      return 0
    fi
    if docker logs --tail 30 "$CONTAINER_NAME" 2>&1 \
         | grep -qiE 'TypeError|ValueError|RuntimeError|Engine core init.*failed|CUDA error|out of memory|Traceback \(most recent'; then
      docker logs "$CONTAINER_NAME" > "$out_dir/server.log" 2>&1 || true
      echo "  [ENGINE-ERROR] check $out_dir/server.log" >&2
      return 1
    fi
    sleep 6
  done
}

settle() { sleep "${1:-60}"; }

run_throughput() {
  local out_dir="$1"
  "$BENCH_PY" "$BENCH_SCRIPT" \
    --host localhost --port "$PORT" --model "$SERVED_NAME" \
    --concurrency 1,2,4 --contexts 0,16k,32k,64k,128k \
    --duration 60 --decode-warmup-seconds 20 --skip-prefill \
    --display-mode plain --output "$out_dir/throughput.json" \
    --no-calibration-cache \
    < /dev/null >> "$out_dir/throughput.log" 2>&1
}

run_prefill() {
  local out_dir="$1"
  "$BENCH_PY" "$BENCH_SCRIPT" \
    --host localhost --port "$PORT" --model "$SERVED_NAME" \
    --concurrency 1 --prefill-contexts 8k,16k,32k,64k,128k \
    --prefill-duration 10 --standalone-prefill \
    --display-mode plain --output "$out_dir/prefill.json" \
    --no-calibration-cache \
    < /dev/null >> "$out_dir/prefill.log" 2>&1
}

run_gates() {
  local out_dir="$1"
  "$PY3" - "$out_dir" <<'PY' 2>&1 | tee "$out_dir/gates.log"
import json, os, sys, requests
OUT_DIR=sys.argv[1]
URL='http://localhost:11435/v1/chat/completions'; H={'Content-Type':'application/json'}
def ask(msgs, max_tokens=4096, tools=None):
    p={'model':'Qwen3.6-27B','messages':msgs,'temperature':0.0,'max_tokens':max_tokens}
    if tools: p['tools']=tools; p['tool_choice']='auto'
    r=requests.post(URL,headers=H,json=p,timeout=300).json()
    return r['choices'][0]['message']
results={}
try:
    ok=0
    for i in range(5):
        m=ask([{'role':'user','content':'Output the first 10 Fibonacci numbers as a comma-separated list (start: 1, 1).'}])
        c=(m.get('content') or '').strip()
        if '1, 1, 2, 3, 5, 8, 13, 21, 34, 55' in c: ok+=1
    results['fib_5x']=[ok,5]; print(f"Gate 1 (Fibonacci 5x): {ok}/5")
except Exception as e:
    print(f"Gate 1: EXCEPTION {e}"); results['fib_5x']=[0,5]
try:
    m=ask([{'role':'user','content':'What is the current weather in Tokyo? Use the tool.'}],
          tools=[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}])
    tcs=m.get('tool_calls') or []
    ok=any(tc['function']['name']=='get_weather' and 'tokyo' in tc['function']['arguments'].lower() for tc in tcs)
    results['tool_call']=ok; print(f"Gate 2 (Tool call): {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"Gate 2: EXCEPTION {e}"); results['tool_call']=False
try:
    m=ask([{'role':'user','content':'What is 47 times 83? Show the result as a number only on the last line.'}], 8192)
    c=(m.get('content') or '').strip()
    ok='3901' in c
    results['reasoning_47x83']=ok; print(f"Gate 3 (47x83=3901): {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"Gate 3: EXCEPTION {e}"); results['reasoning_47x83']=False
try:
    msgs=[{'role':'user','content':'Imagine the temperature in Tokyo is 28C. Just acknowledge.'}]
    t1=ask(msgs,2048); msgs.append({'role':'assistant','content':(t1.get('content') or '')})
    msgs.append({'role':'user','content':'Now imagine Berlin is at 18C. Just acknowledge.'})
    t2=ask(msgs,2048); msgs.append({'role':'assistant','content':(t2.get('content') or '')})
    msgs.append({'role':'user','content':'Which of the two cities I mentioned is warmer? Answer in one short sentence.'})
    t3=ask(msgs,4096); t3c=(t3.get('content') or '').strip()
    ok='tokyo' in t3c.lower() and 'warm' in t3c.lower()
    results['multi_turn']=ok; print(f"Gate 4 (multi-turn): {'PASS' if ok else 'FAIL'}")
except Exception as e:
    print(f"Gate 4: EXCEPTION {e}"); results['multi_turn']=False
passed=sum([results['fib_5x'][0]==5, results['tool_call'], results['reasoning_47x83'], results['multi_turn']])
print(f"\nGates: {passed}/4")
with open(os.path.join(OUT_DIR,'gates.json'),'w') as f:
    json.dump({'results':results,'gates_passed':passed,'gates_total':4},f,indent=2)
sys.exit(0 if passed==4 else 1)
PY
}

# ──────────────────────────────────────────────────────────────────────────────
# run_quality_at OUT_DIR MAX_TOKENS
# HumanEval (164) + MBPP-sanitized (257) @ c=8, parameterized max_tokens, temp=0.0.
# ──────────────────────────────────────────────────────────────────────────────
run_quality_at() {
  local out_dir="$1" max_tokens="${2:-8192}"
  local label="$(basename "$out_dir")"
  "$PY3" "$STRESS_HARNESS" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --model "$SERVED_NAME" \
    --config-label "$label" \
    --benchmark humaneval \
    --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
    --concurrency 8 --max-tokens "$max_tokens" \
    --output "$out_dir/humaneval.jsonl" \
    > "$out_dir/humaneval.log" 2>&1
  "$PY3" "$STRESS_HARNESS" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --model "$SERVED_NAME" \
    --config-label "$label" \
    --benchmark mbpp \
    --problems-file "$PROBLEMS_DIR/mbpp.jsonl" \
    --concurrency 8 --max-tokens "$max_tokens" \
    --output "$out_dir/mbpp.jsonl" \
    > "$out_dir/mbpp.log" 2>&1
}

stop_config() {
  docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

eta_remaining() {
  local cur="$1" total="$2" elapsed="$3" label="$4"
  if [ "$cur" -eq 0 ]; then echo "  [ETA] $label $cur/$total starting"; return; fi
  local per=$(( elapsed / cur ))
  local remain=$(( per * (total - cur) ))
  local rh=$(( remain / 3600 ))
  local rm=$(( (remain % 3600) / 60 ))
  local finish_ts=$(( $(date +%s) + remain ))
  local finish_hhmm=$(date -d "@$finish_ts" '+%H:%M')
  echo "  [ETA] $label $cur/$total — ${rh}h ${rm}m remaining → ~${finish_hhmm} MSK finish"
}
