#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="qwen06b_self_data_seed42"
STATE_DIR="$PROJECT_ROOT/outputs/qwen06b_self_data_generation"
QUEUE_LOG="$STATE_DIR/queue.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

mkdir -p "$STATE_DIR"
tmux new-session -d -s "$SESSION"   "cd '$PROJECT_ROOT' && export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 && touch '$STATE_DIR/queue.started' && python scripts/run_qwen06b_self_data.py 2>&1 | tee -a '$QUEUE_LOG'; rc=\${PIPESTATUS[0]}; if [ \"\$rc\" -eq 0 ]; then touch '$STATE_DIR/queue.finished'; else touch '$STATE_DIR/queue.failed'; fi; printf 'QUEUE_EXIT rc=%s\\n' \"\$rc\"; exec bash"

echo "started: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $QUEUE_LOG"
