#!/usr/bin/env python3
"""Build multitask gold score-only LoRA data from train_data/cleaned_data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_ROOT = PROJECT_ROOT / "train_data" / "cleaned_data"
OUTPUT_ROOT = PROJECT_ROOT / "train_data" / "lora_data" / "multitask_cleaned"
SPLIT_SEED = 20260720
VALIDATION_RATIO = 0.1

SOURCES = (
    "rw_gen_coherence_4811.json",
    "rw_gen_positioning_check_2822.json",
    "rw_gen_positioning_type_954.json",
    "rev_util_actionability_4800.json",
    "rev_util_grounding_specificity_4800.json",
    "rev_util_helpfulness_4800.json",
    "rev_util_verifiability_4800.json",
    "rev_util_verifiability_extraction_4800.json",
)


def load_cleaned(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("train"), list):
        return payload["train"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected cleaned format: {path}")


def to_score_only_row(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(row.get("id", "")).strip()
    task = str(row.get("task", "")).strip()
    aspect = str(row.get("aspect", "")).strip()
    label = row.get("labels", row.get("label"))
    prompt = row.get("prompt")
    if not sample_id or not task or not aspect:
        raise ValueError(f"Missing id/task/aspect: {row.get('id')}")
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(f"Invalid label for {sample_id}")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Invalid prompt for {sample_id}")
    score_sets = row.get("score_sets")
    if isinstance(score_sets, list) and label not in score_sets:
        raise ValueError(f"Label {label} outside score_sets for {sample_id}")

    return {
        "id": f"{task}__{aspect}__{sample_id}",
        "task": task,
        "aspect": aspect,
        "label": label,
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": f"<score>{label}</score>"}],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return digest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def filter_split_ids(ids: list[str], prefix: str) -> list[str]:
    return [sample_id for sample_id in ids if sample_id.startswith(prefix)]


def main() -> None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for filename in SOURCES:
        source_rows = load_cleaned(CLEANED_ROOT / filename)
        for source in source_rows:
            row = to_score_only_row(source)
            if row["id"] in seen:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen.add(row["id"])
            rows.append(row)

    expected_n = 32587
    if len(rows) != expected_n:
        raise ValueError(f"Expected {expected_n} rows, got {len(rows)}")

    # Stable order matching existing multitask split enumeration.
    rows.sort(key=lambda row: row["id"])

    output_path = OUTPUT_ROOT / f"multitask_{len(rows)}_cleaned_score_only.jsonl"
    dataset_sha256 = write_jsonl(output_path, rows)

    split_path = (
        OUTPUT_ROOT / "splits" / f"multitask_{len(rows)}_cleaned_score_only_seed{SPLIT_SEED}.json"
    )
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing fixed split {split_path}; refusing to invent a new stratified split."
        )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    all_ids = {row["id"] for row in rows}
    train_ids = split.get("train_ids", [])
    validation_ids = split.get("validation_ids", [])
    if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != all_ids:
        raise ValueError("Existing multitask split does not cover rebuilt dataset IDs")

    split["dataset_sha256"] = dataset_sha256
    split["seed"] = SPLIT_SEED
    split["validation_ratio"] = VALIDATION_RATIO
    split["stratify_by"] = ["task", "aspect", "label"]
    split["n_train"] = len(train_ids)
    split["n_validation"] = len(validation_ids)
    write_json(split_path, split)

    # Refresh per-task views derived from the parent multitask split.
    by_task_aspect: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_task_aspect.setdefault((row["task"], row["aspect"]), []).append(row)

    for (task, aspect), task_rows in sorted(by_task_aspect.items()):
        prefix = f"{task}__{aspect}__"
        task_train = filter_split_ids(train_ids, prefix)
        task_val = filter_split_ids(validation_ids, prefix)
        task_sha = hashlib.sha256(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_rows).encode("utf-8")
        ).hexdigest()
        task_split = {
            "seed": SPLIT_SEED,
            "validation_ratio": VALIDATION_RATIO,
            "dataset_sha256": task_sha,
            "stratify_by": ["task", "aspect", "label"],
            "parent_split": f"splits/multitask_{len(rows)}_cleaned_score_only_seed{SPLIT_SEED}.json",
            "n_train": len(task_train),
            "n_validation": len(task_val),
            "train_ids": task_train,
            "validation_ids": task_val,
        }
        write_json(
            OUTPUT_ROOT / "splits" / f"{task}_{aspect}_cleaned_score_only_seed{SPLIT_SEED}.json",
            task_split,
        )

    label_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["label"])
        label_counts[key] = label_counts.get(key, 0) + 1

    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(PROJECT_ROOT)),
                "samples": len(rows),
                "dataset_sha256": dataset_sha256,
                "train_samples": len(train_ids),
                "validation_samples": len(validation_ids),
                "labels": dict(sorted(label_counts.items(), key=lambda item: int(item[0]))),
                "split": str(split_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
