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

task_selector="${1:-all}"
if (( $# > 0 )); then
  shift
fi
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then
  seeds=(42)
fi

if [[ "${task_selector}" != "all" ]]; then
  task_found=false
  for task in "${tasks[@]}"; do
    if [[ "${task}" == "${task_selector}" ]]; then
      task_found=true
      break
    fi
  done
  if [[ "${task_found}" != true ]]; then
    printf 'Unknown task: %s\nAvailable tasks: all %s\n' \
      "${task_selector}" "${tasks[*]}" >&2
    exit 2
  fi
  tasks=("${task_selector}")
fi

for seed in "${seeds[@]}"; do
  for task in "${tasks[@]}"; do
    config="configs/training/${task}/paper_align_without_loss_balance.yaml"
    printf 'Starting %s with seed %s\n' "${task}" "${seed}"
    python scripts/train.py --config "${config}" --seed "${seed}" --fresh
  done
done
