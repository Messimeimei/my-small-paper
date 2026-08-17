#!/usr/bin/env python3
"""Stable CLI entry point for the modular evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluation.config import parse_args


def main() -> None:
    args = parse_args()
    if args.refresh_analysis_only:
        from utils.io import resolve_path
        from evaluation.reporting import update_evaluation_analysis

        analysis_path = update_evaluation_analysis(resolve_path(args.output_path))
        print(f"updated {analysis_path}")
        return

    from evaluation.runner import run_evaluation

    run_evaluation(args)


if __name__ == "__main__":
    main()
