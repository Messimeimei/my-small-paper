#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="qwen8b_full_seed42_44"
STATE_DIR="$PROJECT_ROOT/outputs/qwen8b_full_pipeline"
QUEUE_LOG="$STATE_DIR/queue.log"
MODEL_DIR="$PROJECT_ROOT/model/Qwen3-8B"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "missing model directory: $MODEL_DIR" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$STATE_DIR"
tmux new-session -d -s "$SESSION"   "cd '$PROJECT_ROOT' && export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && touch '$STATE_DIR/queue.started' && python scripts/run_qwen8b_full_pipeline.py 2>&1 | tee -a '$QUEUE_LOG'; rc=\${PIPESTATUS[0]}; if [ \"\$rc\" -eq 0 ]; then touch '$STATE_DIR/queue.finished'; else touch '$STATE_DIR/queue.failed'; fi; printf 'QUEUE_EXIT rc=%s\\n' \"\$rc\"; exec bash"

echo "started: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "state: cat $STATE_DIR/state.json"
echo "log: tail -f $QUEUE_LOG"
