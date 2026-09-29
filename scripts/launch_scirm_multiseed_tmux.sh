#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SCIRM_SESSION:-scirm7b_seed43_44}"
STATE_NAME="${SCIRM_STATE_NAME:-scirm_multiseed}"
STATE_DIR="$PROJECT_ROOT/outputs/$STATE_NAME"
DOWNLOAD_LOG="$STATE_DIR/download.log"
QUEUE_LOG="$STATE_DIR/queue.log"
MODEL_DIR_NAME="${SCIRM_MODEL_DIR:-SciRM-7B}"
MODEL_STORAGE="/root/autodl-tmp/model/$MODEL_DIR_NAME"
MODEL_LINK="$PROJECT_ROOT/model/$MODEL_DIR_NAME"
HF_REPO="${SCIRM_HF_REPO:-UKPLab/SciRM-7B}"
HF_REVISION="${SCIRM_HF_REVISION:-d0475b1725de05287d7e0b4a72a741ea6e452b6a}"
MODEL_TAG="${SCIRM_MODEL_TAG:-scirm_7b}"
MODEL_DISPLAY="${SCIRM_MODEL_DISPLAY:-$MODEL_DIR_NAME}"
SEED42_TAG="${SCIRM_SEED42_TAG:-base}"
RUN_SEEDS="${SCIRM_RUN_SEEDS:-43,44}"
HF_DOWNLOAD_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -e "$MODEL_LINK" && ! -d "$MODEL_LINK" ]]; then
  echo "refusing to replace non-symlink model path: $MODEL_LINK" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$STATE_DIR" "$(dirname -- "$MODEL_STORAGE")"
if [[ ! -L "$PROJECT_ROOT/model" ]]; then
  mkdir -p "$PROJECT_ROOT/model"
  if [[ ! -e "$MODEL_LINK" ]]; then
    ln -s "$MODEL_STORAGE" "$MODEL_LINK"
  fi
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_ROOT' && \
   export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
     VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     HF_ENDPOINT='$HF_DOWNLOAD_ENDPOINT' HF_HUB_DISABLE_XET=1 \
     HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=300 \
     SCIRM_EVAL_MODEL_DIR='$MODEL_DIR_NAME' SCIRM_EVAL_MODEL_REPO='$HF_REPO' \
     SCIRM_EVAL_MODEL_REVISION='$HF_REVISION' SCIRM_EVAL_MODEL_TAG='$MODEL_TAG' \
     SCIRM_EVAL_MODEL_DISPLAY='$MODEL_DISPLAY' SCIRM_EVAL_SEED42_TAG='$SEED42_TAG' \
     SCIRM_EVAL_RUN_SEEDS='$RUN_SEEDS' SCIRM_EVAL_ALL_SEEDS='42,43,44' \
     SCIRM_EVAL_STATE_NAME='$STATE_NAME' && \
   touch '$STATE_DIR/download.started' && \
   hf download '$HF_REPO' --revision '$HF_REVISION' \
     --local-dir '$MODEL_STORAGE' --max-workers 4 2>&1 | tee -a '$DOWNLOAD_LOG'; \
   download_rc=\${PIPESTATUS[0]}; \
   if [[ \"\$download_rc\" -ne 0 ]]; then \
     touch '$STATE_DIR/download.failed'; \
     printf 'DOWNLOAD_EXIT rc=%s\\n' \"\$download_rc\"; \
     exec bash; \
   fi; \
   touch '$STATE_DIR/download.finished' '$STATE_DIR/queue.started' && \
   python scripts/run_scirm_multiseed_evaluation.py 2>&1 | tee -a '$QUEUE_LOG'; \
   queue_rc=\${PIPESTATUS[0]}; \
   if [[ \"\$queue_rc\" -eq 0 ]]; then \
     touch '$STATE_DIR/queue.finished'; \
   else \
     touch '$STATE_DIR/queue.failed'; \
   fi; \
   printf 'QUEUE_EXIT rc=%s\\n' \"\$queue_rc\"; \
   exec bash"

echo "started: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "download log: tail -f $DOWNLOAD_LOG"
echo "queue log: tail -f $QUEUE_LOG"
echo "state: cat $STATE_DIR/state.json"
