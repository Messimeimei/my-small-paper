#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

echo "Waiting for seed43 Paper Align w/o Loss Balance queue to finish..."
while tmux has-session -t paper_align_mix_seed43 2>/dev/null; do
  sleep 30
done

adapter_count="$(find /root/autodl-tmp/checkpoints -type f -path '*paper_align_without_loss_balance/*seed43*/adapter/adapter_model.safetensors' 2>/dev/null | wc -l)"
metrics_count="$(find outputs/evaluations -type f -path '*paper_align_without_loss_balance*seed_43/metrics.json' 2>/dev/null | wc -l)"
echo "seed43 verification: adapters=${adapter_count} metrics=${metrics_count}"
test "${adapter_count}" -eq 7
test "${metrics_count}" -eq 14

bash scripts/train_all_methods_seed44.sh 44
bash scripts/evaluate_all_methods_seed44.sh 44
python scripts/evaluate.py --refresh-analysis-only --output_path outputs/evaluations
