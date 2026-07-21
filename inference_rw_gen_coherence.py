#!/usr/bin/env python3
"""Run eval_data/inference.py on the cleaned, test-only RW coherence data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from eval_data import inference as base_inference


DEFAULT_DATASET = (
    Path(__file__).resolve().parent
    / "rw_gen__coherence__exact_user_deduplicated__test__n1046.json"
)


def load_test_only_data(dataset_path: str) -> DatasetDict:
    path = Path(dataset_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("test") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty test list.")

    required = {"id", "task", "aspect", "labels", "score_sets", "prompt"}
    ids = []
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"test row {index} is missing fields: {sorted(missing)}")
        if row["task"] != "rw_gen" or row["aspect"] != "coherence":
            raise ValueError(
                f"test row {index} is {row['task']}/{row['aspect']}, "
                "expected rw_gen/coherence."
            )
        if not isinstance(row["prompt"], list):
            raise ValueError(f"test row {index} prompt must be a message list.")
        ids.append(str(row["id"]))

    if len(ids) != len(set(ids)):
        raise ValueError("test IDs must be unique.")

    return DatasetDict({"test": Dataset.from_list(rows)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility entry point for evaluating the cleaned, test-only "
            "RW coherence dataset with eval_data/inference.py."
        )
    )
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--dataset_file", default=str(DEFAULT_DATASET), type=str)
    parser.add_argument("--max_model_len", default=32768, type=int)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--temp", default=1, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument(
        "--rollout",
        default=5,
        type=int,
        help="Number of rollouts; with --recompute_from, <=0 uses all output columns.",
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable model-native thinking mode; disabled by default.",
    )
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument(
        "--recompute_from",
        default="",
        type=str,
        help="Existing *_outputs.parquet used to recompute metrics without inference.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.recompute_from and not args.model_name:
        parser.error("--model_name is required unless --recompute_from is set")

    base_inference.load_data = load_test_only_data
    base_inference.main(args)


if __name__ == "__main__":
    main()
