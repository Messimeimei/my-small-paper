#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SCIRM_SESSION="scirm_ref7b_seed42_44"
export SCIRM_STATE_NAME="scirm_ref_multiseed"
export SCIRM_MODEL_DIR="SciRM-Ref-7B"
export SCIRM_HF_REPO="UKPLab/SciRM-Ref-7B"
export SCIRM_HF_REVISION="ba3656c87660440b8b4377055ccbe11f5fa60f5a"
export SCIRM_MODEL_TAG="scirm_ref_7b"
export SCIRM_MODEL_DISPLAY="SciRM-Ref-7B"
export SCIRM_SEED42_TAG="42"
export SCIRM_RUN_SEEDS="42,43,44"

exec bash "$SCRIPT_DIR/launch_scirm_multiseed_tmux.sh"
