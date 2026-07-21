#!/usr/bin/env bash
set -euo pipefail
ROOT="/data01/public/yangxin/small-paper"
PYTHON="/data01/public/yangxin/.conda/envs/small-paper/bin/python"
INFER="${ROOT}/eval_data/inference.py"
OUT_ROOT="${ROOT}/eval_data/eval_outputs"
LOG_DIR="${ROOT}/eval_data/eval_logs"
cd "${ROOT}"
mkdir -p "${LOG_DIR}"
echo "[queue] start $(date) gpu=2"
echo "============================================================"
echo "[run] revision_gemma3_4b_it | gpu=2 | rollout=1 | bs=64"
echo "============================================================"
set +e
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${INFER}" \
  --exp_name revision_gemma3_4b_it \
  --model_name "${ROOT}/model/Gemma-3-4B-It" \
  --dataset_file "${ROOT}/eval_data/prompted_revision_data.json" \
  --output_path "${OUT_ROOT}" \
  --max_model_len 16384 \
  --max_tokens 2048 \
  --temp 0 \
  --top_p 1.0 \
  --seed 42 \
  --rollout 1 \
  --batch_size 64 \
  2>&1 | tee -a "${LOG_DIR}/revision_gemma3_4b_it_gpu2_default_rerun.log"
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]]; then
  echo "[fail] revision_gemma3_4b_it exit=$ec" | tee -a "${LOG_DIR}/revision_gemma3_4b_it_gpu2_default_rerun.log"
else
  echo "[ok] revision_gemma3_4b_it $(date)" | tee -a "${LOG_DIR}/revision_gemma3_4b_it_gpu2_default_rerun.log"
fi
set -e
echo "[queue] all done $(date) gpu=2"
