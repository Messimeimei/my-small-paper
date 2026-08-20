#!/usr/bin/env bash
set -euo pipefail

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
    config="configs/training/${task}/${method}.yaml"
    output_root="checkpoints/${task}/${method}"
    existing_run="$(find "${output_root}" -maxdepth 1 -mindepth 1 -type d -name "*seed${seed}*" -exec test -f '{}/summary.json' \; -exec test -f '{}/adapter/adapter_model.safetensors' \; -print -quit 2>/dev/null || true)"
    if [[ -n "${existing_run}" ]]; then
      echo "Skipping completed training: task=${task} method=${method} seed=${seed} run=${existing_run}"
      continue
    fi
    echo "Starting training: task=${task} method=${method} seed=${seed}"
    python scripts/train.py --config "${config}" --seed "${seed}" --fresh

    # Retain the final adapter and run metadata, but remove only this run's
    # intermediate optimizer/checkpoint directory to keep all 42 results.
    run_dir="$(find "${output_root}" -maxdepth 1 -mindepth 1 -type d -name "*seed${seed}*" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
    [[ -n "${run_dir}" && -f "${run_dir}/summary.json" && -f "${run_dir}/adapter/adapter_model.safetensors" ]] || {
      echo "Could not locate completed run for ${task}/${method}/seed${seed}" >&2
      exit 1
    }
    if [[ -d "${run_dir}/checkpoints" ]]; then
      rm -rf -- "${run_dir}/checkpoints"
      echo "Removed intermediate checkpoints from ${run_dir}"
    fi
  done
done
