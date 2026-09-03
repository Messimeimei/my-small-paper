#!/usr/bin/env bash

set -uo pipefail

cd /root/my-small-paper

queue_dir="outputs/queues/ssa_v2_remaining_three_seeds"
mkdir -p "$queue_dir"
exec > >(tee -a "$queue_dir/queue.log") 2>&1

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

tasks=(
  rev_util_actionability
  rev_util_grounding_specificity
  rev_util_helpfulness
  rev_util_verifiability
  rw_gen_coherence
  rw_gen_positioning_check
  rw_gen_positioning_type
)
seeds=(42 43 44)

echo "===SSA_V2_REMAINING_QUEUE_START $(date -Is)==="

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    if [[ "$task" == "rev_util_actionability" && "$seed" == "42" ]]; then
      echo "===SKIP_COMPLETED ${task} seed=${seed} $(date -Is)==="
      touch "$queue_dir/${task}_seed${seed}.skipped_completed"
      continue
    fi

    echo "===RUN_START ${task} seed=${seed} $(date -Is)==="
    if python scripts/run_ssa_v2.py \
      --tasks "$task" \
      --seed "$seed" \
      --phase all; then
      touch "$queue_dir/${task}_seed${seed}.done"
      echo "===RUN_DONE ${task} seed=${seed} $(date -Is)==="
    else
      status=$?
      echo "$status" > "$queue_dir/FAILED"
      echo "${task} seed=${seed}" > "$queue_dir/failed_run.txt"
      echo "===RUN_FAILED ${task} seed=${seed} status=${status} $(date -Is)==="
      exit "$status"
    fi
  done
done

touch "$queue_dir/COMPLETE"
echo "===SSA_V2_REMAINING_QUEUE_COMPLETE $(date -Is)==="
