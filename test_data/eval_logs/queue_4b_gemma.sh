#!/usr/bin/env bash
set -euo pipefail
cd /data01/public/yangxin/small-paper
echo "[queue] start $(date) gpu=2"
echo "============================================================"
echo "[run] rw_gen_gemma3_4b_it | model=Gemma-3-4B-It | n=1857 | rollout=1 | bs=64 | gpu=2"
echo "============================================================"
mkdir -p /data01/public/yangxin/small-paper/eval_data/eval_logs
set +e
CUDA_VISIBLE_DEVICES=2 /data01/public/yangxin/.conda/envs/small-paper/bin/python /data01/public/yangxin/small-paper/eval_data/inference.py \
  --exp_name rw_gen_gemma3_4b_it \
  --model_name /data01/public/yangxin/small-paper/model/Gemma-3-4B-It \
  --dataset_file /data01/public/yangxin/small-paper/eval_data/prompted_rw_gen_data.json \
  --output_path /data01/public/yangxin/small-paper/eval_data/eval_outputs \
  --max_model_len 16384 \
  --max_tokens 2048 \
  --temp 0 \
  --top_p 1.0 \
  --seed 42 \
  --rollout 1 \
  --batch_size 64 \
  2>&1 | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/rw_gen_gemma3_4b_it.log
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]]; then
  echo "[fail] rw_gen_gemma3_4b_it exit=$ec (continue next)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/rw_gen_gemma3_4b_it.log
else
  echo "[ok] rw_gen_gemma3_4b_it $(date)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/rw_gen_gemma3_4b_it.log
fi
set -e
echo "============================================================"
echo "[run] novelty_gemma3_4b_it | model=Gemma-3-4B-It | n=76 | rollout=3 | bs=4 | gpu=2"
echo "============================================================"
mkdir -p /data01/public/yangxin/small-paper/eval_data/eval_logs
set +e
CUDA_VISIBLE_DEVICES=2 /data01/public/yangxin/.conda/envs/small-paper/bin/python /data01/public/yangxin/small-paper/eval_data/inference.py \
  --exp_name novelty_gemma3_4b_it \
  --model_name /data01/public/yangxin/small-paper/model/Gemma-3-4B-It \
  --dataset_file /data01/public/yangxin/small-paper/eval_data/prompted_novelty_data.json \
  --output_path /data01/public/yangxin/small-paper/eval_data/eval_outputs \
  --max_model_len 16384 \
  --max_tokens 2048 \
  --temp 0 \
  --top_p 1.0 \
  --seed 42 \
  --rollout 3 \
  --batch_size 4 \
  2>&1 | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/novelty_gemma3_4b_it.log
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]]; then
  echo "[fail] novelty_gemma3_4b_it exit=$ec (continue next)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/novelty_gemma3_4b_it.log
else
  echo "[ok] novelty_gemma3_4b_it $(date)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/novelty_gemma3_4b_it.log
fi
set -e
echo "============================================================"
echo "[run] revision_gemma3_4b_it | model=Gemma-3-4B-It | n=6184 | rollout=1 | bs=64 | gpu=2"
echo "============================================================"
mkdir -p /data01/public/yangxin/small-paper/eval_data/eval_logs
set +e
CUDA_VISIBLE_DEVICES=2 /data01/public/yangxin/.conda/envs/small-paper/bin/python /data01/public/yangxin/small-paper/eval_data/inference.py \
  --exp_name revision_gemma3_4b_it \
  --model_name /data01/public/yangxin/small-paper/model/Gemma-3-4B-It \
  --dataset_file /data01/public/yangxin/small-paper/eval_data/prompted_revision_data.json \
  --output_path /data01/public/yangxin/small-paper/eval_data/eval_outputs \
  --max_model_len 16384 \
  --max_tokens 2048 \
  --temp 0 \
  --top_p 1.0 \
  --seed 42 \
  --rollout 1 \
  --batch_size 64 \
  2>&1 | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/revision_gemma3_4b_it.log
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]]; then
  echo "[fail] revision_gemma3_4b_it exit=$ec (continue next)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/revision_gemma3_4b_it.log
else
  echo "[ok] revision_gemma3_4b_it $(date)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/revision_gemma3_4b_it.log
fi
set -e
echo "============================================================"
echo "[run] rev_util_gemma3_4b_it | model=Gemma-3-4B-It | n=4788 | rollout=1 | bs=64 | gpu=2"
echo "============================================================"
mkdir -p /data01/public/yangxin/small-paper/eval_data/eval_logs
set +e
CUDA_VISIBLE_DEVICES=2 /data01/public/yangxin/.conda/envs/small-paper/bin/python /data01/public/yangxin/small-paper/eval_data/inference.py \
  --exp_name rev_util_gemma3_4b_it \
  --model_name /data01/public/yangxin/small-paper/model/Gemma-3-4B-It \
  --dataset_file /data01/public/yangxin/small-paper/eval_data/prompted_rev_util_data.json \
  --output_path /data01/public/yangxin/small-paper/eval_data/eval_outputs \
  --max_model_len 16384 \
  --max_tokens 2048 \
  --temp 0 \
  --top_p 1.0 \
  --seed 42 \
  --rollout 1 \
  --batch_size 64 \
  2>&1 | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/rev_util_gemma3_4b_it.log
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]]; then
  echo "[fail] rev_util_gemma3_4b_it exit=$ec (continue next)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/rev_util_gemma3_4b_it.log
else
  echo "[ok] rev_util_gemma3_4b_it $(date)" | tee -a /data01/public/yangxin/small-paper/eval_data/eval_logs/rev_util_gemma3_4b_it.log
fi
set -e
echo "[queue] all done $(date) gpu=2"
