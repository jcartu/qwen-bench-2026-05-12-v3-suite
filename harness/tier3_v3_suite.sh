#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Tier 3 — Full stress-validation suite on repne/vllm:v3
# ──────────────────────────────────────────────────────────────────────────────
# Runs 4 configs back-to-back. Each config gets:
#   1. launch_config (~2-3min)
#   2. wait_for_ready
#   3. settle 60s
#   4. run_gates (sanity)
#   5. run_throughput (15-cell c×ctx matrix)
#   6. run_prefill (5-context TTFT)
#   7. run_quality_at mt=16384 (HE + MBPP)
#   8. stop_config
#
# Configs:
#   tier3-bf16-dflash-n8        — BF16+DFlash N=8 (QUALITY SOTA candidate)
#   tier3-fp8-dflash-n8         — FP8+DFlash N=8 (long-context speed)
#   tier3-fp8-mtp3              — FP8+MTP=3 (current production default)
#   tier3-fp8-mtp5              — FP8+MTP=5 (top throughput candidate)
#
# Estimated total wall: ~3h. Auto-restarts SOTA service on completion.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Override IMAGE before sourcing lib
export IMAGE="repne/vllm:v3"
export CONTAINER_NAME="vllm-v3-suite"
export STUDY_ROOT="/tmp/qwen-bench-2026-05-12-v3-suite"
export PORT=11435

# shellcheck source=./sweep_lib.sh
source "$(dirname "$0")/sweep_lib.sh"

CONFIGS_DIR="$STUDY_ROOT/configs"
LOG="$STUDY_ROOT/tier3.log"
mkdir -p "$CONFIGS_DIR"

# Preflight: pinned harness must have smart_glue
if ! grep -q smart_glue_humaneval "$STRESS_HARNESS"; then
  echo "[FATAL] harness missing smart_glue_humaneval patch: $STRESS_HARNESS" | tee -a "$LOG"
  exit 9
fi
echo "[OK] Using patched harness: $STRESS_HARNESS" | tee -a "$LOG"

# Preflight: v3 image present
if ! docker images "$IMAGE" -q | grep -q .; then
  echo "[FATAL] Image $IMAGE not present locally" | tee -a "$LOG"
  exit 10
fi
echo "[OK] Image $IMAGE ready" | tee -a "$LOG"

# Define the 4 configs as parallel arrays
LABELS=(
  "tier3-bf16-dflash-n8"
  "tier3-fp8-dflash-n8"
  "tier3-fp8-mtp3"
  "tier3-fp8-mtp5"
)
METHODS=(
  "dflash"
  "dflash"
  "mtp"
  "mtp"
)
MODELS=(
  "$MODEL_BF16"
  "$MODEL_FP8"
  "$MODEL_FP8"
  "$MODEL_FP8"
)
DRAFTERS=(
  "$DRAFTER_DFLASH"
  "$DRAFTER_DFLASH"
  "-"
  "-"
)
NUM_SPECS=(8 8 3 5)
MAX_BATCHED=(32768 32768 32768 32768)
CAPTURE=(256 256 256 256)

TOTAL=${#LABELS[@]}
SUITE_START=$(date +%s)

# ──────────────────────────────────────────────────────────────────────────────
# Trap: ensure cleanup + SOTA restart NO MATTER WHAT happens.
# ──────────────────────────────────────────────────────────────────────────────
cleanup() {
  local ec=$?
  echo "[$(date +%H:%M:%S)] [TRAP] Cleanup invoked (exit_code=$ec)" | tee -a "$LOG"
  docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "[$(date +%H:%M:%S)] [TRAP] Restarting SOTA service on :latest..." | tee -a "$LOG"
  systemctl --user start vllm-qwen36-27b-sota.service 2>&1 | tee -a "$LOG" || echo "[WARN] SOTA restart failed" | tee -a "$LOG"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] [TRAP] Done." | tee -a "$LOG"
}
trap cleanup EXIT INT TERM

# ──────────────────────────────────────────────────────────────────────────────
# server_alive: returns 0 if /v1/models responds, 1 otherwise.
# ──────────────────────────────────────────────────────────────────────────────
server_alive() {
  curl -s -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"
}


run_one() {
  local i="$1"
  local label="${LABELS[$i]}" method="${METHODS[$i]}"
  local model="${MODELS[$i]}" drafter="${DRAFTERS[$i]}"
  local num_spec="${NUM_SPECS[$i]}" max_batched="${MAX_BATCHED[$i]}" capture="${CAPTURE[$i]}"
  local out_dir="$CONFIGS_DIR/$label"
  mkdir -p "$out_dir"

  echo "================================================================" | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] CONFIG $((i+1))/$TOTAL: $label  ($method, num_spec=$num_spec)" | tee -a "$LOG"
  echo "  model=$model  drafter=$drafter" | tee -a "$LOG"
  echo "================================================================" | tee -a "$LOG"

  echo "[$(date +%H:%M:%S)]   Launching..." | tee -a "$LOG"
  launch_config "$label" "$method" "$model" "$drafter" "$num_spec" "$max_batched" "$capture" "$out_dir"

  if ! wait_for_ready "$out_dir" 600 2>&1 | tee -a "$LOG"; then
    echo "[$(date +%H:%M:%S)]   [FAIL] $label did not become ready, skipping" | tee -a "$LOG"
    stop_config
    return 1
  fi

  echo "[$(date +%H:%M:%S)]   Settling 60s..." | tee -a "$LOG"
  settle 60

  # Each bench phase: skip if server is dead. Capture engine errors immediately.
  local PHASES=(gates throughput prefill quality)

  for phase in "${PHASES[@]}"; do
    if ! server_alive; then
      echo "[$(date +%H:%M:%S)]   [SKIP] $phase: server is dead" | tee -a "$LOG"
      continue
    fi
    case "$phase" in
      gates)
        echo "[$(date +%H:%M:%S)]   Running gates..." | tee -a "$LOG"
        run_gates "$out_dir" 2>&1 | tail -8 | tee -a "$LOG" || echo "  [WARN] gates errored" | tee -a "$LOG"
        ;;
      throughput)
        echo "[$(date +%H:%M:%S)]   Running throughput sweep (c=1,2,4 × ctx=0,16k,32k,64k,128k)..." | tee -a "$LOG"
        run_throughput "$out_dir" || echo "  [WARN] throughput partial" | tee -a "$LOG"
        ;;
      prefill)
        echo "[$(date +%H:%M:%S)]   Running prefill sweep (TTFT @ 8k,16k,32k,64k,128k)..." | tee -a "$LOG"
        run_prefill "$out_dir" || echo "  [WARN] prefill partial" | tee -a "$LOG"
        ;;
      quality)
        echo "[$(date +%H:%M:%S)]   Running HumanEval + MBPP @ mt=16384..." | tee -a "$LOG"
        run_quality_at "$out_dir" 16384 || echo "  [WARN] quality partial" | tee -a "$LOG"
        ;;
    esac
    # After each phase, dump container logs as a snapshot for forensics
    docker logs "$CONTAINER_NAME" > "$out_dir/server_after_${phase}.log" 2>&1 || true
  done

  # Capture FULL server logs ONE MORE TIME after all benches finish
  docker logs "$CONTAINER_NAME" > "$out_dir/server_full.log" 2>&1 || true
  # Extract anything looking like an engine error/stack trace
  grep -iE 'TypeError|ValueError|RuntimeError|Engine core|CUDA error|out of memory|Traceback|encountered an issue|Killed|Signal|Aborted|fatal' "$out_dir/server_full.log" | head -50 > "$out_dir/server_errors.log" || true
  if [ -s "$out_dir/server_errors.log" ]; then
    echo "  [SERVER ERRORS detected in $label - see server_errors.log]" | tee -a "$LOG"
    head -15 "$out_dir/server_errors.log" | tee -a "$LOG"
  fi
  # Final alive check
  if server_alive; then
    echo "  [SERVER OK] $label finished with live server" | tee -a "$LOG"
  else
    echo "  [SERVER DEAD] $label finished with crashed server" | tee -a "$LOG"
  fi

  # Compute pass rates inline so the log shows them
  for bench in humaneval mbpp; do
    local sum="$out_dir/${bench}_summary.json"
    if [ -f "$sum" ]; then
      python3 -c "
import json
d=json.load(open('$sum')); fmb=d.get('failure_mode_breakdown',{})
print(f\"  [{('$bench').upper()}] pass={d['pass_rate']*100:.1f}%  ok={fmb.get('ok',0)}/{d['total_problems']}  empty={fmb.get('empty_response',0)}  test_fail={fmb.get('test_fail',0)}  length_truncated={fmb.get('length_truncated',0)}\")
" 2>&1 | tee -a "$LOG"
    fi
  done

  echo "[$(date +%H:%M:%S)]   Tearing down $label..." | tee -a "$LOG"
  stop_config

  local elapsed=$(( $(date +%s) - SUITE_START ))
  eta_remaining "$((i+1))" "$TOTAL" "$elapsed" "suite" 2>&1 | tee -a "$LOG"
}

echo "================================================================" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Tier 3 v3 full suite START — $TOTAL configs" | tee -a "$LOG"
echo "  Image: $IMAGE" | tee -a "$LOG"
echo "  Container: $CONTAINER_NAME" | tee -a "$LOG"
echo "  Port: $PORT" | tee -a "$LOG"
echo "  GPU 2 must remain at ~89.8 GB (oss-120b protected)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

for i in $(seq 0 $((TOTAL-1))); do
  run_one "$i" || echo "  [WARN] $i failed but continuing"
done

echo "================================================================" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] Tier 3 suite COMPLETE. Restoring SOTA on :latest..." | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# Trap handler restores SOTA. End of main.
echo "[$(date +%H:%M:%S)] DONE (main flow). Trap will restore SOTA." | tee -a "$LOG"
