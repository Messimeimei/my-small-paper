#!/usr/bin/env python3
"""可复用多 Lora 专家训练脚本"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from pipeline import prepare_run_context, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        "--resume-from-checkpoint",
        dest="resume",
        type=Path,
        default=None,
        help=(
            "Resume an existing run in place. Accepts either a run directory "
            "(latest complete checkpoint is selected) or checkpoint-* directory."
        ),
    )
    mode.add_argument(
        "--fresh",
        action="store_true",
        help="Explicitly start from the base model in a new timestamped run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data and create or verify the fixed split without training.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    context = prepare_run_context(args)

    if args.dry_run:
        payload = {
            "mode": "resume" if context["resume_checkpoint"] else "fresh",
            "data_summary": context["data_summary"],
            "training_method": context["training_method"],
        }
        if context["resume_checkpoint"] is not None:
            payload["resume"] = {
                "run_directory": str(context["run_directory"]),
                "checkpoint": str(context["resume_checkpoint"]),
                **(context["resume_details"] or {}),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    run_training(context)


if __name__ == "__main__":
    main()
