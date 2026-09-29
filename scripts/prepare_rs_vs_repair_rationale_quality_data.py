#!/usr/bin/env python3
"""Build RS-vs-RePaIR pairs on the exact 350 SO-vs-RS audit samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = (
    PROJECT_ROOT / "outputs/analysis/rationale_comparation_all_tasks_v2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/analysis/rs_vs_repair_rationale_quality_bundle"
    / "rs_vs_repair_matched_350.jsonl"
)
TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.I | re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = str(row.get("id", ""))
        if not row_id or row_id in indexed:
            raise ValueError(f"{path}: missing or duplicate id {row_id!r}")
        indexed[row_id] = row
    return indexed


def first_value(row: dict[str, Any], plural: str, singular: str) -> Any:
    values = row.get(plural)
    return values[0] if isinstance(values, list) and values else row.get(singular)


def parse_repair(row: dict[str, Any], gold: int) -> dict[str, Any]:
    prediction = first_value(row, "rollout_predictions", "prediction")
    output = str(first_value(row, "outputs", "output") or "")
    match = REASONING_RE.search(output)
    rationale = match.group(1).strip() if match else ""
    if prediction is None or not rationale:
        raise ValueError(f"{row.get('id')}: invalid RePaIR RF output")
    prediction = int(prediction)
    return {
        "prediction": prediction,
        "correct": prediction == gold,
        "reasoning": rationale,
        "raw_output": output,
    }


def outcome_stratum(rs: dict[str, Any], repair: dict[str, Any]) -> str:
    if rs["correct"] and repair["correct"]:
        return "both_correct"
    if rs["correct"]:
        return "rs_only_correct"
    if repair["correct"]:
        return "repair_only_correct"
    return "both_wrong"


def repair_prediction_path(task: str) -> Path:
    run = f"{task}#qwen3_4b#ft#self_correct_align#greedy#on_cot#seed_42"
    return PROJECT_ROOT / "outputs/evaluations" / task / run / "predictions.jsonl"


def main() -> None:
    args = parse_args()
    reference_root = resolve(args.reference_root)
    output_path = resolve(args.output)
    prepared: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    inherited_config: dict[str, Any] | None = None

    for task in TASKS:
        task_root = reference_root / task
        sampled_path = task_root / "sampled_items.jsonl"
        reference_manifest_path = task_root / "manifest.json"
        repair_path = repair_prediction_path(task)
        sampled = read_jsonl(sampled_path)
        reference_manifest = read_json(reference_manifest_path)
        repair_rows = index_rows(repair_path)
        if len(sampled) != 50:
            raise ValueError(f"{task}: expected 50 reference samples, found {len(sampled)}")

        config = {
            "prompt_version": reference_manifest["prompt_version"],
            "prediction_seed": reference_manifest["prediction_seed"],
            "sample_seed": reference_manifest["sample_seed"],
            "num": reference_manifest["num"],
            "sampling": reference_manifest["sampling"],
            "judge_models": reference_manifest["judge_models"],
            "tiebreaker_model": reference_manifest["tiebreaker_model"],
            "judge_max_tokens": reference_manifest["judge_max_tokens"],
            "base_url": reference_manifest["base_url"],
        }
        if inherited_config is None:
            inherited_config = config
        elif config != inherited_config:
            raise ValueError(f"{task}: SO/RS audit configuration differs")

        for reference in sampled:
            source_id = str(reference["source_id"])
            repair_row = repair_rows.get(source_id)
            if repair_row is None:
                raise KeyError(f"{repair_path}: missing {source_id}")
            gold = int(reference["gold_label"])
            if int(repair_row["label"]) != gold:
                raise ValueError(f"{task}:{source_id}: RePaIR label mismatch")
            rs = dict(reference["cc"])
            repair = parse_repair(repair_row, gold)

            # RS remains on the old CC side; RePaIR replaces SO on the old LC side.
            assignment = {
                side: "rs" if condition == "cc" else "repair"
                for side, condition in reference["blind_assignment"].items()
            }
            values = {"rs": rs, "repair": repair}
            prepared.append(
                {
                    "item_id": f"{task}__{reference['item_id']}",
                    "reference_item_id": reference["item_id"],
                    "source_id": source_id,
                    "task": task,
                    "query": reference["query"],
                    "criteria": reference["criteria"],
                    "evaluated_text": reference["evaluated_text"],
                    "gold_label": gold,
                    "outcome_stratum": outcome_stratum(rs, repair),
                    "blind_assignment": assignment,
                    "A": {
                        "reasoning": values[assignment["A"]]["reasoning"],
                        "prediction": values[assignment["A"]]["prediction"],
                    },
                    "B": {
                        "reasoning": values[assignment["B"]]["reasoning"],
                        "prediction": values[assignment["B"]]["prediction"],
                    },
                    "rs": rs,
                    "repair": repair,
                    "so_reference": reference["lc"],
                    "original_so_rs_blind_assignment": reference["blind_assignment"],
                }
            )

        sources[task] = {
            "reference_manifest": portable(reference_manifest_path),
            "reference_manifest_sha256": sha256(reference_manifest_path),
            "reference_samples": portable(sampled_path),
            "reference_samples_sha256": sha256(sampled_path),
            "repair_predictions": portable(repair_path),
            "repair_predictions_sha256": sha256(repair_path),
        }

    if inherited_config is None or len(prepared) != 350:
        raise ValueError(f"expected 350 matched samples, found {len(prepared)}")
    write_jsonl(output_path, prepared)
    write_json(
        output_path.with_name("data_manifest.json"),
        {
            "schema_version": 2,
            "description": (
                "RS-vs-RePaIR pairs on the exact seed-42, 50-per-task sample "
                "IDs and A/B sides used by the final SO-vs-RS rationale audit."
            ),
            "output": portable(output_path),
            "output_sha256": sha256(output_path),
            "samples": len(prepared),
            "by_task": dict(Counter(item["task"] for item in prepared)),
            "by_outcome_stratum": dict(
                Counter(item["outcome_stratum"] for item in prepared)
            ),
            "conditions": {
                "rs": "qwen3_4b cot training, RF inference, seed 42",
                "repair": (
                    "qwen3_4b self_correct_align (RePaIR) training, RF inference, "
                    "seed 42"
                ),
            },
            "inherited_evaluation_config": inherited_config,
            "sources": sources,
        },
    )
    print(f"prepared {len(prepared)} matched RS-vs-RePaIR pairs: {output_path}")


if __name__ == "__main__":
    main()
