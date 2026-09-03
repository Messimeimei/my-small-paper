#!/usr/bin/env bash
set -uo pipefail

project_root=/root/my-small-paper
queue_dir=$project_root/outputs/queues/ssa_taskwise_3seeds
cd "$project_root" || exit 1
exec > >(tee -a "$queue_dir/queue.log") 2>&1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0

write_status() {
    local state=$1
    local current=$2
    printf '%s\nupdated_at=%s\ncurrent=%s\nsession=ssa_taskwise_3seeds\n' \
        "$state" "$(date -Is)" "$current" > "$queue_dir/status.txt"
}

fail_queue() {
    local code=$1
    local stage=$2
    echo "===QUEUE_FAILED stage=$stage status=$code $(date -Is)==="
    write_status failed "$stage status=$code"
    touch "$queue_dir/FAILED"
    exit "$code"
}

runs=(
    rev_util_actionability:43
    rev_util_actionability:44
    rev_util_grounding_specificity:43
    rev_util_grounding_specificity:44
    rev_util_helpfulness:42
    rev_util_helpfulness:43
    rev_util_helpfulness:44
    rev_util_verifiability:42
    rev_util_verifiability:43
    rev_util_verifiability:44
    rw_gen_coherence:42
    rw_gen_coherence:43
    rw_gen_coherence:44
    rw_gen_positioning_check:42
    rw_gen_positioning_check:43
    rw_gen_positioning_check:44
    rw_gen_positioning_type:42
    rw_gen_positioning_type:43
    rw_gen_positioning_type:44
)

echo "===QUEUE_START session=ssa_taskwise_3seeds runs=${#runs[@]} $(date -Is)==="
printf '===PLAN %s===\n' "${runs[*]}"
write_status running "initializing"

for spec in "${runs[@]}"; do
    task=${spec%:*}
    seed=${spec#*:}
    train_marker="$queue_dir/train_${task}_seed${seed}.done"
    run_marker="$queue_dir/run_${task}_seed${seed}.done"

    if [[ -f "$run_marker" ]]; then
        echo "===RUN_SKIP_COMPLETED task=$task seed=$seed $(date -Is)==="
        continue
    fi

    if [[ -f "$train_marker" ]]; then
        echo "===TRAIN_REUSE task=$task seed=$seed $(date -Is)==="
    else
        write_status running "$task seed$seed train"
        echo "===TRAIN_START task=$task seed=$seed $(date -Is)==="
        python scripts/train.py \
            --config "configs/training/$task/ssa.yaml" \
            --seed "$seed" \
            --fresh
        code=$?
        [[ "$code" -eq 0 ]] || fail_queue "$code" "$task seed$seed train"

        manifest=$(find -L "checkpoints/$task/ssa" \
            -path "*__seed${seed}__*" -type f -name manifest.json \
            -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')
        [[ -n "$manifest" ]] || fail_queue 1 "$task seed$seed missing_manifest"
        python -c 'import json,pathlib,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); s=json.load(open(pathlib.Path(sys.argv[1]).with_name("summary.json"),encoding="utf-8")); ok=m.get("status")=="completed" and s.get("task")==sys.argv[2] and int(s.get("seed"))==int(sys.argv[3]); raise SystemExit(0 if ok else 1)' \
            "$manifest" "$task" "$seed" \
            || fail_queue 1 "$task seed$seed invalid_training_metadata"
        run_dir=${manifest%/manifest.json}
        [[ -s "$run_dir/adapter/adapter_model.safetensors" ]] \
            || fail_queue 1 "$task seed$seed missing_adapter"

        touch "$train_marker"
        echo "===TRAIN_DONE task=$task seed=$seed manifest=$manifest $(date -Is)==="
    fi

    for mode in cot label_only; do
        eval_marker="$queue_dir/eval_${task}_seed${seed}_${mode}.done"
        if [[ -f "$eval_marker" ]]; then
            echo "===EVAL_REUSE task=$task seed=$seed mode=$mode $(date -Is)==="
            continue
        fi

        config="configs/evaluation/$task/ssa/greedy_on_${mode}.yaml"
        exp_name="${task}#qwen3_4b#ft#ssa#greedy#on_${mode}#seed_${seed}"
        eval_dir="outputs/evaluations/$task/$exp_name"
        if [[ "$mode" == cot ]]; then
            dataset="data/$task/cot/test_cot.jsonl"
        else
            dataset="data/$task/label_only/test_label_only.jsonl"
        fi

        write_status running "$task seed$seed eval_$mode"
        echo "===EVAL_START task=$task seed=$seed mode=$mode $(date -Is)==="
        python scripts/evaluate.py \
            --config "$config" \
            --exp_name "$exp_name" \
            --train_seed "$seed" \
            --seed "$seed"
        code=$?
        [[ "$code" -eq 0 ]] || fail_queue "$code" "$task seed$seed eval_$mode"

        [[ -s "$eval_dir/metrics.json" ]] \
            || fail_queue 1 "$task seed$seed eval_$mode missing_metrics"
        [[ -s "$eval_dir/predictions.jsonl" ]] \
            || fail_queue 1 "$task seed$seed eval_$mode missing_predictions"
        [[ -s "$eval_dir/resolved_config.json" ]] \
            || fail_queue 1 "$task seed$seed eval_$mode missing_resolved_config"

        expected=$(wc -l < "$dataset")
        actual=$(wc -l < "$eval_dir/predictions.jsonl")
        [[ "$actual" -eq "$expected" ]] \
            || fail_queue 1 "$task seed$seed eval_$mode count_${actual}_of_${expected}"
        python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); seed=int(sys.argv[2]); exp=sys.argv[3]; ok=d.get("train_seed")==seed and d.get("seed")==seed and d.get("exp_name")==exp and f"__seed{seed}__" in str(d.get("adapter")); raise SystemExit(0 if ok else 1)' \
            "$eval_dir/resolved_config.json" "$seed" "$exp_name" \
            || fail_queue 1 "$task seed$seed eval_$mode wrong_resolved_config"

        touch "$eval_marker"
        echo "===EVAL_DONE task=$task seed=$seed mode=$mode predictions=$actual $(date -Is)==="
    done

    touch "$run_marker"
    echo "===RUN_DONE task=$task seed=$seed $(date -Is)==="
done

touch "$queue_dir/COMPLETED"
write_status completed "all 19 train-and-two-eval runs completed"
echo "===QUEUE_COMPLETED session=ssa_taskwise_3seeds $(date -Is)==="
