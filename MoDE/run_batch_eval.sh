#!/usr/bin/env bash
# Batch evaluate MoDE mix adapter + base models on MoDE clean test sets.
# Failures are skipped; the script continues to the next job.

set -u
set +e

ROOT="/data01/public/yangxin/small-paper"
PYTHON="${PYTHON:-/data01/public/yangxin/.conda/envs/small-paper/bin/python}"
EVAL_PY="${ROOT}/training/evaluate.py"
DATA_ROOT="${ROOT}/MoDE/data"
OUT_ROOT="${ROOT}/MoDE/eval_outputs"
LOG_DIR="${ROOT}/MoDE/eval_batch_logs"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

MIX_ADAPTER="${ROOT}/MoDE/outputs/factor_mix__experts-n2-positioning_check_deepse--positioning_type_deepsee-12c5ae8f9e__target-positioning_check__k5pc__seed42__20260723T031221Z/adapter"
MIX_TAG="mix_n2_check+type_k5#ds"

MAX_MODEL_LEN=8192
MAX_TOKENS=512
TEMP=0
TOP_P=1.0
SEED=42
ROLLOUT=5
BATCH_SIZE=64
GPU_MEM=0.9

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_LOG="${LOG_DIR}/batch_${STAMP}.log"
FAIL_LOG="${LOG_DIR}/batch_${STAMP}.failed.txt"
: >"${SUMMARY_LOG}"
: >"${FAIL_LOG}"

ok=0
fail=0
skip=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${SUMMARY_LOG}"
}

declare -A DATASETS=(
  [rw_gen_coherence]="${DATA_ROOT}/coherence/clean_test1046.json"
  [rw_gen_positioning_type]="${DATA_ROOT}/positioning_type/clean_test204.json"
  [rw_gen_positioning_check]="${DATA_ROOT}/positioning_check/clean_test603.json"
  [rev_util_actionability]="${DATA_ROOT}/actionability/clean_test1000.json"
  [rev_util_grounding_specificity]="${DATA_ROOT}/grounding_specificity/clean_test1000.json"
  [rev_util_helpfulness]="${DATA_ROOT}/helpfulness/clean_test1000.json"
  [rev_util_verifiability]="${DATA_ROOT}/verifiability/clean_test788.json"
  [rev_util_verifiability_extraction]="${DATA_ROOT}/verifiability_extraction/clean_test1000.json"
  [novelty]="${DATA_ROOT}/novelty/clean_test66.json"
  [revision_relatedness]="${DATA_ROOT}/revision_relatedness/clean_test3026.json"
  [revision_correctness]="${DATA_ROOT}/revision_correctness/clean_test3026.json"
)

run_one() {
  local exp_name="$1"
  local model_name="$2"
  local adapter="$3"
  local dataset_file="$4"
  local job_log="${LOG_DIR}/${exp_name//\//_}.log"

  if [[ ! -f "${dataset_file}" ]]; then
    log "SKIP missing dataset: ${exp_name} -> ${dataset_file}"
    echo "${exp_name}|missing_dataset|${dataset_file}" >>"${FAIL_LOG}"
    skip=$((skip + 1))
    return 0
  fi
  if [[ ! -d "${model_name}" ]]; then
    log "SKIP missing model: ${exp_name} -> ${model_name}"
    echo "${exp_name}|missing_model|${model_name}" >>"${FAIL_LOG}"
    skip=$((skip + 1))
    return 0
  fi
  if [[ "${adapter}" != "none" && ! -f "${adapter}/adapter_model.safetensors" && ! -f "${adapter}/adapter_model.bin" ]]; then
    log "SKIP missing adapter: ${exp_name} -> ${adapter}"
    echo "${exp_name}|missing_adapter|${adapter}" >>"${FAIL_LOG}"
    skip=$((skip + 1))
    return 0
  fi
  if [[ -f "${OUT_ROOT}/${exp_name}/metrics.json" ]]; then
    log "SKIP already done: ${exp_name}"
    skip=$((skip + 1))
    return 0
  fi

  log "START ${exp_name}"
  log "  model=${model_name}"
  log "  adapter=${adapter}"
  log "  dataset=${dataset_file}"
  log "  job_log=${job_log}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${EVAL_PY}" \
    --exp_name "${exp_name}" \
    --model_name "${model_name}" \
    --adapter "${adapter}" \
    --dataset_file "${dataset_file}" \
    --output_path "${OUT_ROOT}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_tokens "${MAX_TOKENS}" \
    --temp "${TEMP}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
    --rollout "${ROLLOUT}" \
    --batch_size "${BATCH_SIZE}" \
    --gpu_memory_utilization "${GPU_MEM}" \
    >"${job_log}" 2>&1
  local rc=$?
  if [[ ${rc} -eq 0 && -f "${OUT_ROOT}/${exp_name}/metrics.json" ]]; then
    log "OK   ${exp_name}"
    ok=$((ok + 1))
  else
    log "FAIL ${exp_name} rc=${rc} (see ${job_log})"
    echo "${exp_name}|rc=${rc}|${job_log}" >>"${FAIL_LOG}"
    fail=$((fail + 1))
  fi
}

log "batch eval start gpu=${GPU} stamp=${STAMP}"
log "python=${PYTHON}"
log "mix_adapter=${MIX_ADAPTER}"

# ---------------------------------------------------------------------------
# Part 1: MoDE mix adapter on remaining tasks
# (positioning_check already evaluated separately)
# ---------------------------------------------------------------------------
MIX_TASKS=(
  rw_gen_positioning_type
  novelty
  revision_relatedness
  revision_correctness
)

for task in "${MIX_TASKS[@]}"; do
  run_one \
    "${task}#${MIX_TAG}" \
    "${ROOT}/model/Qwen3-4B" \
    "${MIX_ADAPTER}" \
    "${DATASETS[${task}]}"
done

# ---------------------------------------------------------------------------
# Part 2: base models (adapter=none) on rw x3 + rev_util x5 + novelty + revision x2
# ---------------------------------------------------------------------------
BASE_TASKS=(
  rw_gen_coherence
  rw_gen_positioning_type
  rw_gen_positioning_check
  rev_util_actionability
  rev_util_grounding_specificity
  rev_util_helpfulness
  rev_util_verifiability
  rev_util_verifiability_extraction
  novelty
  revision_relatedness
  revision_correctness
)

declare -A MODELS=(
  [qwen3_4b]="${ROOT}/model/Qwen3-4B"
  [scirm_7b]="${ROOT}/model/SciRM-7B"
  [scirm_ref_7b]="${ROOT}/model/SciRM-Ref-7B"
  [scilitllm_7b]="${ROOT}/model/SciLitLLM-7B"
)

MODEL_ORDER=(qwen3_4b scirm_7b scirm_ref_7b scilitllm_7b)

for model_tag in "${MODEL_ORDER[@]}"; do
  model_path="${MODELS[${model_tag}]}"
  for task in "${BASE_TASKS[@]}"; do
    run_one \
      "${task}#base#${model_tag}" \
      "${model_path}" \
      "none" \
      "${DATASETS[${task}]}"
  done
done

log "batch eval finished ok=${ok} fail=${fail} skip=${skip}"
log "summary_log=${SUMMARY_LOG}"
log "fail_log=${FAIL_LOG}"
exit 0
