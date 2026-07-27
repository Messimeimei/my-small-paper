#!/usr/bin/env python3
"""Create the same deterministic stratified split used by training/train.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")

    content = args.input.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    ids = [row.get("id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError("Dataset contains an invalid id")
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset contains duplicate ids")
    if any(
        isinstance(row.get("label"), bool) or not isinstance(row.get("label"), int)
        for row in rows
    ):
        raise ValueError("Dataset contains an invalid label")

    labels = sorted({row["label"] for row in rows})
    counts = {label: sum(row["label"] == label for row in rows) for label in labels}
    target_total = max(len(labels), round(len(rows) * args.validation_ratio))
    raw_counts = {label: counts[label] * args.validation_ratio for label in labels}
    selected_counts = {label: math.floor(raw_counts[label]) for label in labels}
    remaining = target_total - sum(selected_counts.values())
    for label in sorted(
        labels,
        key=lambda value: raw_counts[value] - selected_counts[value],
        reverse=True,
    ):
        if remaining <= 0:
            break
        selected_counts[label] += 1
        remaining -= 1

    rng = random.Random(args.seed)
    ids_by_label = {
        label: [row["id"] for row in rows if row["label"] == label]
        for label in labels
    }
    for label_ids in ids_by_label.values():
        rng.shuffle(label_ids)
    validation_ids = {
        sample_id
        for label, label_ids in ids_by_label.items()
        for sample_id in label_ids[: selected_counts[label]]
    }

    payload = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(content).hexdigest(),
        "split_seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "train_ids": [sample_id for sample_id in ids if sample_id not in validation_ids],
        "validation_ids": [sample_id for sample_id in ids if sample_id in validation_ids],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "samples": len(rows),
                "labels": counts,
                "train_samples": len(payload["train_ids"]),
                "validation_samples": len(payload["validation_ids"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
