#!/usr/bin/env bash

# Source this file from paper_workspace before running paper-analysis code.
PAPER_WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_PROJECT_ROOT="$(cd "$PAPER_WORKSPACE_ROOT/.." && pwd)"

source "$PAPER_PROJECT_ROOT/.venv-paper/bin/activate"
export UV_CACHE_DIR="$PAPER_PROJECT_ROOT/.cache/uv"
export MPLCONFIGDIR="$PAPER_PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$PAPER_PROJECT_ROOT/.cache"
export PAPER_TEXLIVE_BIN="$PAPER_PROJECT_ROOT/.tools/texlive/2026/bin/x86_64-linux"
export PAPER_POPPLER_BIN="$PAPER_PROJECT_ROOT/.tools/poppler/usr/bin"
export PAPER_POPPLER_LIB="$PAPER_PROJECT_ROOT/.tools/poppler/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$PAPER_POPPLER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$PAPER_TEXLIVE_BIN:$PAPER_POPPLER_BIN:$PATH"

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
echo "Paper workspace active: $PAPER_WORKSPACE_ROOT"
