#!/usr/bin/env bash
set -euo pipefail
cd /data01/public/yangxin/small-paper
echo "[queue] start $(date) gpu=0"
echo "[skip] rw_gen_qwen25_3b already has results"
echo "[skip] novelty_qwen25_3b already has results"
echo "[skip] revision_qwen25_3b already has results"
echo "[skip] rev_util_qwen25_3b already has results"
echo "[skip] rw_gen_qwen3_4b already has results"
echo "[skip] novelty_qwen3_4b already has results"
echo "[skip] revision_qwen3_4b already has results"
echo "[skip] rev_util_qwen3_4b already has results"
echo "[queue] all done $(date) gpu=0"
