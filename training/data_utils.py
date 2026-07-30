"""Dataset loading and fixed train/validation splits."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from metrics_utils import SCORE_RE


def parse_train_label(row: dict[str, Any], line_number: int) -> int:
    raw = row.get("label", row.get("labels"))
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"Invalid label at line {line_number}: {raw!r}")
    return raw


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read training JSONL (data/*/cot|label_only/train_*.jsonl)."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    declared_score_sets: list[int] | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", "")).strip()
            prompt = row.get("prompt")
            completion = row.get("completion")
            label = parse_train_label(row, line_number)
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Invalid or duplicate id at {path}:{line_number}")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"Invalid prompt at {path}:{line_number}")
            if (
                not isinstance(completion, list)
                or len(completion) != 1
                or completion[0].get("role") != "assistant"
                or not SCORE_RE.search(str(completion[0].get("content", "")))
            ):
                raise ValueError(f"Invalid completion at {path}:{line_number}")
            if row.get("score_sets") is not None:
                score_sets = row["score_sets"]
                if (
                    not isinstance(score_sets, list)
                    or not score_sets
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in score_sets)
                ):
                    raise ValueError(f"Invalid score_sets at {path}:{line_number}")
                if declared_score_sets is None:
                    declared_score_sets = list(score_sets)
                elif list(score_sets) != declared_score_sets:
                    raise ValueError(f"Inconsistent score_sets at {path}:{line_number}")
                if label not in set(declared_score_sets):
                    raise ValueError(
                        f"Label {label} outside score_sets at {path}:{line_number}"
                    )
            seen_ids.add(sample_id)
            rows.append(
                {
                    **row,
                    "id": sample_id,
                    "label": label,
                    "prompt": prompt,
                    "completion": completion,
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def score_sets(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({row["label"] for row in rows})


def validation_counts(
    rows: list[dict[str, Any]], labels: list[int], ratio: float
) -> dict[int, int]:
    counts = {label: sum(row["label"] == label for row in rows) for label in labels}
    target_total = max(len(labels), round(len(rows) * ratio))
    raw = {label: counts[label] * ratio for label in counts}
    selected = {label: math.floor(raw[label]) for label in counts}
    remaining = target_total - sum(selected.values())
    for label in sorted(counts, key=lambda value: raw[value] - selected[value], reverse=True):
        if remaining <= 0:
            break
        selected[label] += 1
        remaining -= 1
    return selected


def load_or_create_split(
    rows: list[dict[str, Any]],
    labels: list[int],
    split_path: Path,
    dataset_hash: str,
    split_seed: int,
    validation_ratio: float,
    *,
    write_json,
) -> dict[str, Any]:
    all_ids = {row["id"] for row in rows}
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"Dataset hash no longer matches fixed split: {split_path}")
        train_ids = set(split.get("train_ids", []))
        validation_ids = set(split.get("validation_ids", []))
        if train_ids & validation_ids or train_ids | validation_ids != all_ids:
            raise ValueError(f"Invalid ID coverage in fixed split: {split_path}")
        return split

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    rng = random.Random(split_seed)
    per_label = {
        label: [row["id"] for row in rows if row["label"] == label] for label in labels
    }
    for ids in per_label.values():
        rng.shuffle(ids)
    selected_counts = validation_counts(rows, labels, validation_ratio)
    validation_ids = {
        sample_id
        for label, ids in per_label.items()
        for sample_id in ids[: selected_counts[label]]
    }
    train_ids = [row["id"] for row in rows if row["id"] not in validation_ids]
    ordered_validation_ids = [row["id"] for row in rows if row["id"] in validation_ids]
    split = {
        "schema_version": 1,
        "dataset_sha256": dataset_hash,
        "split_seed": split_seed,
        "validation_ratio": validation_ratio,
        "train_ids": train_ids,
        "validation_ids": ordered_validation_ids,
    }
    write_json(split_path, split)
    return split


def split_rows(
    rows: list[dict[str, Any]], split: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids = set(split["train_ids"])
    validation_ids = set(split["validation_ids"])
    return (
        [row for row in rows if row["id"] in train_ids],
        [row for row in rows if row["id"] in validation_ids],
    )


def label_counts(rows: list[dict[str, Any]], labels: list[int]) -> dict[str, int]:
    return {str(label): sum(row["label"] == label for row in rows) for label in labels}
