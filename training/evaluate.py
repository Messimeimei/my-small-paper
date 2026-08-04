#!/usr/bin/env python3
"""Stable CLI entry point for the modular evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from evaluation.cli_config import parse_args


def main() -> None:
    args = parse_args()
    if args.refresh_analysis_only:
        from shared.project_io import resolve_path
        from evaluation.report_generation import update_evaluation_analysis

        analysis_path = update_evaluation_analysis(resolve_path(args.output_path))
        print(f"updated {analysis_path}")
        return

    from evaluation.evaluation_pipeline import run

    run(args)


if __name__ == "__main__":
    main()
