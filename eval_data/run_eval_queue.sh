#!/usr/bin/env bash
# 并行评测队列：两个 tmux 分别跑小模型(≤4B) 与 7B 模型。
#
# 用法：
#   bash /data01/public/yangxin/small-paper/eval_data/run_eval_queue.sh
#   bash /data01/public/yangxin/small-paper/eval_data/run_eval_queue.sh --dry-run   # 只打印任务，不启动
#   bash /data01/public/yangxin/small-paper/eval_data/run_eval_queue.sh --force     # 已有结果也重跑
#   bash /data01/public/yangxin/small-paper/eval_data/run_eval_queue.sh --restart   # 杀掉旧 tmux 再启动
#
# 逻辑简述：
# 1) 枚举 3 个数据集 × 5 个模型 = 15 个任务
# 2) 按模型规模拆成两条队列：
#      small(GPU0): Qwen2.5-3B-Instruct, Qwen3-4B
#      large(GPU2): Qwen2.5-7B-Instruct, SciRM-7B, SciRM-Ref-7B
# 3) n<200 → rollout=3, bs=4；n≥200 → rollout=1, bs=64
# 4) exp_name = <任务短名>_<模型短名>；默认不开 thinking
# 5) 在两个 tmux 里各自按队列顺序串行执行；默认跳过已有 *_results.json 的任务

set -euo pipefail

ROOT="/data01/public/yangxin/small-paper"
PYTHON="/data01/public/yangxin/.conda/envs/small-paper/bin/python"
INFER="${ROOT}/eval_data/inference.py"
OUT_ROOT="${ROOT}/eval_data/eval_outputs"
MODEL_ROOT="${ROOT}/model"
LOG_DIR="${ROOT}/eval_data/eval_logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

DRY_RUN=0
FORCE=0
RESTART=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    --restart) RESTART=1 ;;  # 杀掉已有 eval_small / eval_7b 再重建
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

# ---------- 模型短名（用于 exp_name）----------
model_tag() {
  case "$1" in
    Qwen2.5-3B-Instruct) echo "qwen25_3b" ;;
    Qwen2.5-7B-Instruct) echo "qwen25_7b" ;;
    Qwen3-4B) echo "qwen3_4b" ;;
    SciRM-7B) echo "scirm_7b" ;;
    SciRM-Ref-7B) echo "scirm_ref_7b" ;;
    *) echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\+/_/g' ;;
  esac
}

# ---------- 数据集：短名 | json 路径 | test 条数 ----------
DATASETS=(
  "rw_gen|${ROOT}/eval_data/prompted_rw_gen_data.json|1857"
  "revision|${ROOT}/eval_data/prompted_revision_data.json|6184"
  "rev_util|${ROOT}/eval_data/prompted_rev_util_data.json|4788"
)

# 小模型（≤4B）：评测命令.txt 前半段模板，GPU0
SMALL_MODELS=(
  "Qwen2.5-3B-Instruct"
  "Qwen3-4B"
)
SMALL_GPU=0
SMALL_TMUX="eval_small"

# 7B：评测命令.txt 后半段模板，GPU2
LARGE_MODELS=(
  "Qwen2.5-7B-Instruct"
  "SciRM-7B"
  "SciRM-Ref-7B"
)
LARGE_GPU=2
LARGE_TMUX="eval_7b"

# 根据样本数选 rollout / batch_size
pick_runtime() {
  local n="$1"
  if (( n < 200 )); then
    echo "3 4"
  else
    echo "1 64"
  fi
}

is_done() {
  local exp_name="$1"
  local dir="${OUT_ROOT}/${exp_name}"
  [[ "${FORCE}" -eq 1 ]] && return 1
  compgen -G "${dir}/*_results.json" > /dev/null
}

build_cmd() {
  local gpu="$1"
  local model_dir="$2"
  local dataset="$3"
  local exp_name="$4"
  local rollout="$5"
  local bs="$6"
  # 不加 --enable_thinking（默认 False）；Qwen3 也不开 thinking
  cat <<CMD
CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${INFER} \\
  --exp_name ${exp_name} \\
  --model_name ${MODEL_ROOT}/${model_dir} \\
  --dataset_file ${dataset} \\
  --output_path ${OUT_ROOT} \\
  --max_model_len 16384 \\
  --max_tokens 2048 \\
  --temp 0 \\
  --top_p 1.0 \\
  --seed 42 \\
  --rollout ${rollout} \\
  --batch_size ${bs}
CMD
}

emit_queue_script() {
  local gpu="$1"
  shift
  local models=("$@")

  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd $(printf %q "${ROOT}")"
  echo "echo \"[queue] start \$(date) gpu=${gpu}\""

  local model ds_entry task_short dataset n rollout bs exp_name
  for model in "${models[@]}"; do
    for ds_entry in "${DATASETS[@]}"; do
      IFS='|' read -r task_short dataset n <<< "${ds_entry}"
      read -r rollout bs <<< "$(pick_runtime "${n}")"
      exp_name="${task_short}_$(model_tag "${model}")"

      if is_done "${exp_name}"; then
        echo "echo \"[skip] ${exp_name} already has results\""
        continue
      fi

      echo "echo \"============================================================\""
      echo "echo \"[run] ${exp_name} | model=${model} | n=${n} | rollout=${rollout} | bs=${bs} | gpu=${gpu}\""
      echo "echo \"============================================================\""
      echo "mkdir -p $(printf %q "${LOG_DIR}")"
      # 用 bash -lc 跑多行命令；失败不中断队列
      echo "set +e"
      echo "CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${INFER} \\"
      echo "  --exp_name ${exp_name} \\"
      echo "  --model_name ${MODEL_ROOT}/${model} \\"
      echo "  --dataset_file ${dataset} \\"
      echo "  --output_path ${OUT_ROOT} \\"
      echo "  --max_model_len 16384 \\"
      echo "  --max_tokens 2048 \\"
      echo "  --temp 0 \\"
      echo "  --top_p 1.0 \\"
      echo "  --seed 42 \\"
      echo "  --rollout ${rollout} \\"
      echo "  --batch_size ${bs} \\"
      echo "  2>&1 | tee -a $(printf %q "${LOG_DIR}/${exp_name}.log")"
      echo "ec=\${PIPESTATUS[0]}"
      echo "if [[ \$ec -ne 0 ]]; then"
      echo "  echo \"[fail] ${exp_name} exit=\$ec (continue next)\" | tee -a $(printf %q "${LOG_DIR}/${exp_name}.log")"
      echo "else"
      echo "  echo \"[ok] ${exp_name} \$(date)\" | tee -a $(printf %q "${LOG_DIR}/${exp_name}.log")"
      echo "fi"
      echo "set -e"
    done
  done

  echo "echo \"[queue] all done \$(date) gpu=${gpu}\""
}

start_tmux_queue() {
  local session="$1"
  local script_path="$2"

  if tmux has-session -t "${session}" 2>/dev/null; then
    if [[ "${RESTART}" -eq 1 ]]; then
      echo "[restart] killing existing tmux session '${session}'"
      tmux kill-session -t "${session}"
      sleep 1
    else
      echo "[warn] tmux session '${session}' already exists; skip create."
      echo "       attach: tmux attach -t ${session}"
      echo "       kill:   tmux kill-session -t ${session}"
      echo "       or re-run with: bash $0 --restart"
      return 0
    fi
  fi

  tmux new-session -d -s "${session}" -c "${ROOT}" \
    "bash '${script_path}'; echo; echo '[tmux] finished. press enter to close'; read"
  echo "[tmux] started session '${session}'"
  echo "       attach: tmux attach -t ${session}"
}

SMALL_SCRIPT="${LOG_DIR}/queue_small.sh"
LARGE_SCRIPT="${LOG_DIR}/queue_7b.sh"

emit_queue_script "${SMALL_GPU}" "${SMALL_MODELS[@]}" > "${SMALL_SCRIPT}"
emit_queue_script "${LARGE_GPU}" "${LARGE_MODELS[@]}" > "${LARGE_SCRIPT}"
chmod +x "${SMALL_SCRIPT}" "${LARGE_SCRIPT}"

echo "======= planned jobs ======="
echo "--- small (tmux=${SMALL_TMUX}, GPU=${SMALL_GPU}) ---"
grep -E '^echo "\[(run|skip)\]' "${SMALL_SCRIPT}" || true
echo "--- 7b (tmux=${LARGE_TMUX}, GPU=${LARGE_GPU}) ---"
grep -E '^echo "\[(run|skip)\]' "${LARGE_SCRIPT}" || true
echo "============================"
echo "queue scripts:"
echo "  ${SMALL_SCRIPT}"
echo "  ${LARGE_SCRIPT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] not starting tmux."
  exit 0
fi

start_tmux_queue "${SMALL_TMUX}" "${SMALL_SCRIPT}"
start_tmux_queue "${LARGE_TMUX}" "${LARGE_SCRIPT}"

echo
echo "Both queues launched (if sessions were free)."
echo "  tmux attach -t ${SMALL_TMUX}"
echo "  tmux attach -t ${LARGE_TMUX}"
echo "  tmux ls"
echo "  logs: ${LOG_DIR}"
