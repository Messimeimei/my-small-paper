#!/usr/bin/env bash

set -uo pipefail

cd /root/my-small-paper

output_dir="outputs/analysis/interface_switch_harmful_samples_v2_remaining"
log_path="${output_dir}/queue.log"
status_path="${output_dir}/queue_status.txt"

printf 'RUNNING\n' > "${status_path}"

python scripts/repair_harmful_cot_rationales.py \
  --input "${output_dir}/valid_harmful_samples.jsonl" \
  --output "${output_dir}/corrected_harmful_samples.jsonl" \
  --work-dir "${output_dir}/rationale_correction_api_v2" \
  2>&1 | tee -a "${log_path}"
run_status=${PIPESTATUS[0]}

if [[ ${run_status} -eq 0 ]]; then
  printf 'COMPLETED\n' > "${status_path}"
else
  printf 'FAILED exit_code=%s\n' "${run_status}" > "${status_path}"
fi

exit "${run_status}"
