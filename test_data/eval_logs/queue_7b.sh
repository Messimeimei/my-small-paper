#!/usr/bin/env bash
set -euo pipefail
cd /data01/public/yangxin/small-paper
echo "[queue] start $(date) gpu=2"
echo "[skip] rw_gen_qwen25_7b already has results"
echo "[skip] novelty_qwen25_7b already has results"
echo "[skip] revision_qwen25_7b already has results"
echo "[skip] rev_util_qwen25_7b already has results"
echo "[skip] rw_gen_scirm_7b already has results"
echo "[skip] novelty_scirm_7b already has results"
echo "[skip] revision_scirm_7b already has results"
echo "[skip] rev_util_scirm_7b already has results"
echo "[skip] rw_gen_scirm_ref_7b already has results"
echo "[skip] novelty_scirm_ref_7b already has results"
echo "[skip] revision_scirm_ref_7b already has results"
echo "[skip] rev_util_scirm_ref_7b already has results"
echo "[queue] all done $(date) gpu=2"
