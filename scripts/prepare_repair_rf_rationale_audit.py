#!/usr/bin/env python3
"""Attach 4B RePaIR RF predictions to the existing RS-harmful audit pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/analysis/rs_ds_correct_rf_wrong__scrs_rf_same_seed/all_samples.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/analysis/rs_ds_correct_rf_wrong__repair_rf_same_seed/all_samples.jsonl"
)
METHOD = "self_correct_align"
MODEL = "qwen3_4b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_path(task: str, seed: int) -> Path:
    run = f"{task}#{MODEL}#ft#{METHOD}#greedy#on_cot#seed_{seed}"
    return PROJECT_ROOT / "outputs/evaluations" / task / run / "predictions.jsonl"


def prediction_state(row: dict[str, Any], gold: int) -> dict[str, Any]:
    values = row.get("rollout_predictions")
    prediction = values[0] if isinstance(values, list) and values else row.get("prediction")
    if prediction is None:
        return {
            "strict_prediction": None,
            "strict_correct": False,
            "strict_status": "invalid",
            "absolute_error": None,
        }
    prediction = int(prediction)
    error = abs(prediction - gold)
    return {
        "strict_prediction": prediction,
        "strict_correct": error == 0,
        "strict_status": (
            "correct"
            if error == 0
            else "adjacent_error"
            if error == 1
            else "severe_error"
        ),
        "absolute_error": error,
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    source_rows = read_jsonl(input_path)
    cache: dict[tuple[str, int], tuple[Path, dict[str, dict[str, Any]]]] = {}
    output_rows: list[dict[str, Any]] = []

    for source in source_rows:
        task = str(source["task"])
        seed = int(source["seed"])
        sample_id = str(source["id"])
        cache_key = (task, seed)
        if cache_key not in cache:
            path = prediction_path(task, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            indexed = {str(row["id"]): row for row in read_jsonl(path)}
            cache[cache_key] = (path, indexed)
        path, indexed = cache[cache_key]
        candidate = indexed.get(sample_id)
        if candidate is None:
            raise KeyError(f"{path}: missing {sample_id}")
        gold = int(source["gold_label"])
        if int(candidate["label"]) != gold:
            raise ValueError(f"{path}:{sample_id}: label mismatch")

        value = dict(source)
        value["selection_states"] = dict(source["selection_states"])
        value["selection_states"]["scrs_rf"] = prediction_state(candidate, gold)
        value["evaluation_records"] = dict(source["evaluation_records"])
        value["evaluation_records"]["scrs_rf"] = candidate
        value["source_paths"] = dict(source["source_paths"])
        value["source_paths"]["scrs_rf_predictions"] = str(
            path.relative_to(PROJECT_ROOT)
        )
        value["candidate_method"] = METHOD
        value["candidate_display_name"] = "RePaIR"
        output_rows.append(value)

    write_jsonl(output_path, output_rows)
    write_json(
        output_path.with_name("manifest.json"),
        {
            "schema_version": 1,
            "source_input": str(input_path.relative_to(PROJECT_ROOT)),
            "source_input_sha256": sha256(input_path),
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "output_sha256": sha256(output_path),
            "candidate_method": METHOD,
            "candidate_display_name": "RePaIR",
            "model": MODEL,
            "samples": len(output_rows),
            "prediction_files": len(cache),
        },
    )
    print(
        f"prepared {len(output_rows)} RePaIR samples from {len(cache)} prediction files: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
