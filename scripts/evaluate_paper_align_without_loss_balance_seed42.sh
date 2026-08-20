#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

train_seed="${1:-42}"
if [[ ! "${train_seed}" =~ ^[0-9]+$ ]]; then
  printf 'Training seed must be an integer, got: %s\n' "${train_seed}" >&2
  exit 2
fi

tasks=(
  rev_util_actionability
  rev_util_grounding_specificity
  rev_util_helpfulness
  rev_util_verifiability
  rw_gen_coherence
  rw_gen_positioning_check
  rw_gen_positioning_type
)

for task in "${tasks[@]}"; do
  for mode in label_only cot; do
    config="configs/evaluation/${task}/paper_align_without_loss_balance/greedy_on_${mode}.yaml"
    adapter_root="checkpoints/${task}/paper_align_without_loss_balance"
    exp_name="${task}#qwen3_4b#ft#paper_align_without_loss_balance#greedy#on_${mode}#seed_${train_seed}"
    printf 'Starting evaluation: %s on_%s with training seed %s\n' \
      "${task}" "${mode}" "${train_seed}"
    python scripts/evaluate.py \
      --config "${config}" \
      --exp_name "${exp_name}" \
      --adapter "${adapter_root}" \
      --train_seed "${train_seed}" \
      --seed "${train_seed}"
  done
done
