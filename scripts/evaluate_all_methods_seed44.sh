#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

tasks=(
  rev_util_actionability
  rev_util_grounding_specificity
  rev_util_helpfulness
  rev_util_verifiability
  rw_gen_coherence
  rw_gen_positioning_check
  rw_gen_positioning_type
)
methods=(label_only cot paper_align paper_align_without_loss_balance self_correct_align self_correct_cot)
seed="${1:-44}"
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "seed must be an integer: ${seed}" >&2; exit 2; }

for task in "${tasks[@]}"; do
  for method in "${methods[@]}"; do
    config_dir="configs/evaluation/${task}/${method}"
    adapter_root="checkpoints/${task}/${method}"
    train_config="configs/training/${task}/${method}.yaml"
    for mode in label_only cot; do
      config="$(find "${config_dir}" -maxdepth 1 -type f -name "greedy_on_${mode}*.yaml" | sort | head -n 1)"
      [[ -n "${config}" ]] || { echo "Missing eval config for ${task}/${method}/${mode}" >&2; exit 1; }
      exp_name="${task}#qwen3_4b#ft#${method}#greedy#on_${mode}#seed_${seed}"
      metrics_path="outputs/evaluations/${task}/${exp_name}/metrics.json"
      if [[ -f "${metrics_path}" ]]; then
        echo "Skipping completed evaluation: task=${task} method=${method} mode=${mode} train_seed=${seed}"
        continue
      fi
      echo "Starting evaluation: task=${task} method=${method} mode=${mode} train_seed=${seed}"
      python scripts/evaluate.py \
        --config "${config}" \
        --exp_name "${exp_name}" \
        --adapter "${adapter_root}" \
        --train_seed "${seed}" \
        --train_config "${train_config}" \
        --seed "${seed}"
    done
  done
done
