"""Evaluation dataset validation and rollout aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def normalize_score_sets(raw: Any, *, context: str) -> list[int]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError(f"{context} has invalid score_sets: {raw!r}")
    return list(raw)


def parse_label(row: dict[str, Any], index: int, allowed_scores: set[int]) -> int:
    raw = row.get("labels", row.get("label"))
    try:
        label = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Row {index} has invalid label: {raw!r}") from error
    if isinstance(raw, bool) or label not in allowed_scores:
        raise ValueError(
            f"Row {index} label {raw!r} is outside score_sets "
            f"{sorted(allowed_scores)}"
        )
    return label


def load_dataset_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                rows.append(row)
        return rows, None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("test", payload.get("train"))
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else None
    else:
        rows, metadata = payload, None
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list or a test/train list.")
    return rows, metadata


def _declared_scores(
    path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any] | None
) -> list[int] | None:
    declared: list[tuple[str, list[int]]] = []
    if metadata and metadata.get("score_sets") is not None:
        declared.append(
            (
                "metadata",
                normalize_score_sets(metadata["score_sets"], context=f"{path} metadata"),
            )
        )
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("score_sets") is not None:
            declared.append(
                (
                    f"row {index}",
                    normalize_score_sets(
                        row["score_sets"], context=f"{path} row {index}"
                    ),
                )
            )
    if not declared:
        return None
    score_sets = declared[0][1]
    for location, values in declared[1:]:
        if values != score_sets:
            raise ValueError(
                f"{path} {location} score_sets {values} does not match {score_sets}"
            )
    return score_sets


def _observed_scores(rows: list[dict[str, Any]]) -> list[int]:
    labels: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object.")
        raw = row.get("labels", row.get("label"))
        try:
            label = int(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {index} has invalid label: {raw!r}") from error
        if isinstance(raw, bool):
            raise ValueError(f"Row {index} has invalid label: {raw!r}")
        labels.append(label)
    return sorted(set(labels))


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows, metadata = load_dataset_rows(path)
    if not rows:
        raise ValueError(f"{path} must contain a non-empty test/train list.")
    score_sets = _declared_scores(path, rows, metadata) or _observed_scores(rows)
    allowed_scores = set(score_sets)

    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), list):
            raise ValueError(f"Row {index} missing a valid prompt list.")
        sample_id = str(row.get("id", "")).strip() or f"row_{index:04d}"
        if sample_id in seen:
            raise ValueError(f"Duplicate id: {sample_id}")
        seen.add(sample_id)
        cleaned.append(
            {
                "id": sample_id,
                "label": parse_label(row, index, allowed_scores),
                "prompt": row["prompt"],
                "task": row.get("task"),
                "aspect": row.get("aspect"),
                "evaluation_mode": row.get("evaluation_mode")
                or row.get("supervision_mode"),
                "prompt_version": row.get("prompt_version"),
                "score_sets": score_sets,
            }
        )
    return cleaned, score_sets


def mean_rollout_metrics(
    rollout_metrics: list[dict[str, Any]],
    score_sets: list[int],
    scalar_metrics: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    def summarize(values: list[float | int | None]) -> tuple[float | None, float | None]:
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            return None, None
        array = np.asarray(cleaned, dtype=float)
        return float(array.mean()), float(array.std())

    aggregate: dict[str, Any] = {
        "samples": rollout_metrics[0]["samples"],
        "score_sets": score_sets,
        "per_class": {},
    }
    default_scalar_metrics = (
        "accuracy",
        "macro_f1",
        "format_valid_rate",
        "mae",
        "qwk",
        "rail_mae",
        "rail_mse",
        "rail_rmse",
        "avg_score_probability_mass",
        "score_prefix_valid_rate",
        "reasoning_valid_rate",
    )
    for metric in scalar_metrics or default_scalar_metrics:
        if any(metric in rollout for rollout in rollout_metrics):
            mean, std = summarize([rollout.get(metric) for rollout in rollout_metrics])
            aggregate[metric] = mean
            aggregate[f"{metric}_std"] = std

    stable_fields = (
        "probability_normalization",
        "candidate_renormalization",
        "rail_implementation",
        "rail_expectation_formula",
        "discrete_decoding",
    )
    for field in stable_fields:
        values = {rollout.get(field) for rollout in rollout_metrics} - {None}
        if len(values) > 1:
            raise ValueError(f"Rollouts disagree on {field}: {sorted(values)}")
        if values:
            aggregate[field] = values.pop()

    for label in score_sets:
        key = str(label)
        per_class = {"support": rollout_metrics[0]["per_class"][key]["support"]}
        for metric in ("precision", "recall", "f1"):
            mean, std = summarize(
                [rollout["per_class"][key][metric] for rollout in rollout_metrics]
            )
            per_class[metric] = mean
            per_class[f"{metric}_std"] = std
        aggregate["per_class"][key] = per_class
    return aggregate
