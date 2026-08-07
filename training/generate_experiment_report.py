#!/usr/bin/env python3
"""Regenerate eval_output/experiment_results.md from persisted results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from evaluation.experiment_report import update_experiment_report
from shared.project_io import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="eval_output")
    args = parser.parse_args()
    print(update_experiment_report(resolve_path(args.eval_root)))


if __name__ == "__main__":
    main()
