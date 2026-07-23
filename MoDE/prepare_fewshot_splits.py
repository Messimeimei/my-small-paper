#!/usr/bin/env python3
"""Build nested MoDE calibration sets from each expert's validation split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODE_ROOT.parent
DEFAULT_OUTPUT_DIR = MODE_ROOT / "data"

# These sources match the expert runs used by the factor-mix experiments. The mutable
# training/configs YAML files may point at other variants, so each dataset/split pair is
# pinned here and recorded by hash.
DEFAULT_TASK_SOURCES = {
    "coherence": {
        "task": "rw_gen",
        "dataset": PROJECT_ROOT
        / "train_data/lora_data/rw_gen_coherence_3629_distill_deepseek-v4-pro.jsonl",
        "split": PROJECT_ROOT
        / "train_data/lora_data/splits/rw_gen_coherence_deepseek-v4-pro_seed20260720.json",
        "test": PROJECT_ROOT / "test_data/prompted_rw_gen_coherence_data.json",
    },
    "positioning_check": {
        "task": "rw_gen",
        "dataset": PROJECT_ROOT
        / "train_data/lora_data/rw_gen_positioning_check_2666_distill_deepseek-v4-pro.jsonl",
        "split": PROJECT_ROOT
        / "train_data/lora_data/splits/rw_gen_positioning_check_deepseek-v4-pro_seed20260720.json",
        "test": PROJECT_ROOT
        / "test_data/prompted_rw_gen_positioning_check_data.json",
    },
    "positioning_type": {
        "task": "rw_gen",
        "dataset": PROJECT_ROOT
        / "train_data/lora_data/rw_gen_positioning_type_953_distill_glm-5.2.jsonl",
        "split": PROJECT_ROOT
        / "train_data/lora_data/splits/rw_gen_positioning_type_glm-5.2_seed20260720.json",
        "test": PROJECT_ROOT / "test_data/prompted_rw_gen_positioning_type_data.json",
    },
    "actionability": {
        "task": "rev_util",
        "dataset": PROJECT_ROOT
        / "train_data/lora_data/rev_util_actionability_1788_distill_deepseek-v4-pro.jsonl",
        "split": PROJECT_ROOT
        / "train_data/lora_data/splits/rev_util_actionability_deepseek-v4-pro_seed20260720.json",
        # This file is already aspect-filtered, prompt-unique, and test-clean. Reuse it
        # byte-for-byte so changing calibration provenance cannot change the final test.
        "test": MODE_ROOT / "data/actionability/clean_test1000.json",
        "reuse_clean_test": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(DEFAULT_TASK_SOURCES),
        default=list(DEFAULT_TASK_SOURCES),
        help="Only rebuild the selected expert-validation calibration tasks.",
    )
    parser.add_argument(
        "--shots-per-class",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Nested calibration sizes selected independently for every task label.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def relative_to_mode(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(MODE_ROOT))
    except ValueError:
        return str(resolved)


def raw_label(row: dict[str, Any], *, source: Path, location: str) -> int:
    value = row.get("labels", row.get("label"))
    if isinstance(value, bool):
        raise ValueError(f"Invalid integer label at {source}:{location}: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid integer label at {source}:{location}: {value!r}"
        ) from error


def validate_prompt(row: dict[str, Any], *, source: Path, location: str) -> None:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"Missing prompt at {source}:{location}")


def load_jsonl_dataset(path: Path, aspect: str) -> list[dict[str, Any]]:
    rows = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL row at {path}:{line_number}")
            sample_id = str(row.get("id", "")).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Missing or duplicate id at {path}:{line_number}")
            seen_ids.add(sample_id)
            if row.get("aspect") != aspect:
                raise ValueError(
                    f"Aspect mismatch at {path}:{line_number}: {row.get('aspect')!r}"
                )
            raw_label(row, source=path, location=str(line_number))
            validate_prompt(row, source=path, location=str(line_number))
            completion = row.get("completion")
            if (
                not isinstance(completion, list)
                or not completion
                or any(
                    not isinstance(message, dict)
                    or message.get("role") != "assistant"
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                    for message in completion
                )
            ):
                raise ValueError(
                    f"Missing assistant teacher completion at {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def load_split(path: Path, dataset_path: Path, row_ids: set[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Split must be a JSON object: {path}")
    if payload.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError(f"Split dataset hash does not match {dataset_path}")
    train_ids = payload.get("train_ids")
    validation_ids = payload.get("validation_ids")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise ValueError(f"Split is missing train_ids/validation_ids: {path}")
    train_set = {str(value) for value in train_ids}
    validation_set = {str(value) for value in validation_ids}
    if len(train_set) != len(train_ids) or len(validation_set) != len(validation_ids):
        raise ValueError(f"Split contains duplicate IDs: {path}")
    if train_set & validation_set:
        raise ValueError(f"Train and validation IDs overlap: {path}")
    if train_set | validation_set != row_ids:
        raise ValueError(f"Split IDs do not exactly cover the dataset: {path}")
    return payload


def load_test_rows(path: Path, aspect: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("test") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty top-level test list")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid test row at {path}:{index}")
        sample_id = str(row.get("id", "")).strip()
        if sample_id:
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate test id at {path}:{index}: {sample_id}")
            seen_ids.add(sample_id)
        if row.get("aspect") != aspect:
            raise ValueError(f"Test aspect mismatch at {path}:{index}")
        raw_label(row, source=path, location=str(index))
        validate_prompt(row, source=path, location=str(index))
    return rows


def prompt_fingerprint(row: dict[str, Any]) -> str:
    canonical = json.dumps(
        row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def row_reference(row: dict[str, Any]) -> str:
    sample_id = str(row.get("id", "")).strip()
    return sample_id or f"prompt_sha256:{prompt_fingerprint(row)}"


def prompt_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[prompt_fingerprint(row)].append(row)
    return dict(groups)


def overlap_groups(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    right = prompt_groups(right_rows)
    overlaps = []
    for fingerprint, group in prompt_groups(left_rows).items():
        if fingerprint in right:
            overlaps.append(
                {
                    "prompt_sha256": fingerprint,
                    "left_ids": [row_reference(row) for row in group],
                    "right_ids": [row_reference(row) for row in right[fingerprint]],
                }
            )
    return sorted(overlaps, key=lambda group: tuple(group["left_ids"]))


def duplicate_groups(rows: list[dict[str, Any]]) -> list[list[str]]:
    return sorted(
        (
            [row_reference(row) for row in group]
            for group in prompt_groups(rows).values()
            if len(group) > 1
        ),
        key=tuple,
    )


def clean_test_rows(
    test_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_fingerprints = set(prompt_groups(train_rows))
    seen_test_fingerprints: set[str] = set()
    removed_training_overlap_ids = []
    removed_duplicate_ids = []
    clean_rows = []
    for row in test_rows:
        fingerprint = prompt_fingerprint(row)
        sample_id = row_reference(row)
        if fingerprint in train_fingerprints:
            removed_training_overlap_ids.append(sample_id)
        elif fingerprint in seen_test_fingerprints:
            removed_duplicate_ids.append(sample_id)
        else:
            seen_test_fingerprints.add(fingerprint)
            clean_rows.append(row)
    return clean_rows, {
        "removed_training_overlap_ids": sorted(removed_training_overlap_ids),
        "removed_duplicate_test_ids": sorted(removed_duplicate_ids),
    }


def stable_rank(seed: int, aspect: str, label: int, sample_id: str) -> bytes:
    value = f"{seed}\0{aspect}\0{label}\0{sample_id}".encode()
    return hashlib.sha256(value).digest()


def label_counts(rows: list[dict[str, Any]], score_sets: list[int]) -> dict[str, int]:
    counts = Counter(int(row.get("labels", row.get("label"))) for row in rows)
    return {str(label): counts[label] for label in score_sets}


def select_nested_ids(
    rows: list[dict[str, Any]],
    *,
    aspect: str,
    score_sets: list[int],
    seed: int,
    max_shots: int,
) -> dict[int, list[str]]:
    selected = {}
    for label in score_sets:
        candidates = [
            str(row["id"])
            for row in rows
            if int(row.get("labels", row.get("label"))) == label
        ]
        candidates.sort(
            key=lambda sample_id: (
                stable_rank(seed, aspect, label, sample_id),
                sample_id,
            )
        )
        if len(candidates) < max_shots:
            raise ValueError(
                f"{aspect} label {label} has {len(candidates)} eligible validation "
                f"rows; need {max_shots}"
            )
        selected[label] = candidates[:max_shots]
    return selected


def create_splits(
    *,
    task_sources: dict[str, dict[str, Any]],
    output_dir: Path,
    seed: int,
    shots_per_class: list[int],
    base_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shots = sorted(set(shots_per_class))
    if not shots or any(value < 1 for value in shots):
        raise ValueError("shots_per_class must contain positive integers")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = copy.deepcopy(base_manifest) if base_manifest else {}
    manifest["schema_version"] = max(int(manifest.get("schema_version", 0)), 3)
    manifest["selection"] = {
        "source": "lora_training_validation_split",
        "strategy": "nested_sha256_rank_per_class",
        "seed": seed,
        "shots_per_class": shots,
        "calibration_policy": (
            "exclude exact prompt overlaps with gradient-train and final-test"
        ),
        "test_policy": (
            "independent of k; remove gradient-train overlaps and duplicate prompts"
        ),
    }
    manifest.setdefault("tasks", {})
    if not isinstance(manifest["tasks"], dict):
        raise ValueError("base_manifest.tasks must be a JSON object")

    for aspect, raw_sources in task_sources.items():
        dataset_path = raw_sources["dataset"].resolve()
        split_path = raw_sources["split"].resolve()
        test_source = raw_sources["test"].resolve()
        task_name = str(raw_sources.get("task", "rw_gen"))
        reuse_clean_test = bool(raw_sources.get("reuse_clean_test", False))
        dataset_rows = load_jsonl_dataset(dataset_path, aspect)
        score_sets = sorted(
            {
                raw_label(row, source=dataset_path, location=str(index))
                for index, row in enumerate(dataset_rows, start=1)
            }
        )
        rows_by_id = {str(row["id"]): row for row in dataset_rows}
        split = load_split(split_path, dataset_path, set(rows_by_id))
        train_rows = [rows_by_id[str(sample_id)] for sample_id in split["train_ids"]]
        validation_rows = [
            rows_by_id[str(sample_id)] for sample_id in split["validation_ids"]
        ]
        test_rows = load_test_rows(test_source, aspect)
        test_score_sets = sorted(
            {
                raw_label(row, source=test_source, location=str(index))
                for index, row in enumerate(test_rows)
            }
        )
        declared_test_score_sets = {
            tuple(row.get("score_sets", [])) for row in test_rows if row.get("score_sets")
        }
        if len(declared_test_score_sets) > 1:
            raise ValueError(f"Inconsistent test score_sets: {test_source}")
        if declared_test_score_sets:
            declared_scores = list(next(iter(declared_test_score_sets)))
            if declared_scores != score_sets:
                raise ValueError(
                    f"Dataset labels {score_sets} differ from test score_sets "
                    f"{declared_scores}: {aspect}"
                )
        elif test_score_sets != score_sets:
            raise ValueError(
                f"Dataset labels {score_sets} differ from observed test labels "
                f"{test_score_sets}: {aspect}"
            )

        validation_train_overlaps = overlap_groups(validation_rows, train_rows)
        validation_test_overlaps = overlap_groups(validation_rows, test_rows)
        validation_duplicate_groups = duplicate_groups(validation_rows)
        excluded_validation_ids = {
            sample_id
            for group in validation_train_overlaps + validation_test_overlaps
            for sample_id in group["left_ids"]
        }
        excluded_validation_ids.update(
            sample_id
            for group in validation_duplicate_groups
            for sample_id in group
        )
        eligible_validation = [
            row
            for row in validation_rows
            if str(row["id"]) not in excluded_validation_ids
        ]
        ranked_ids = select_nested_ids(
            eligible_validation,
            aspect=aspect,
            score_sets=score_sets,
            seed=seed,
            max_shots=max(shots),
        )

        clean_test, test_audit = clean_test_rows(test_rows, train_rows)
        if reuse_clean_test and (clean_test != test_rows or any(test_audit.values())):
            raise ValueError(f"Configured reusable test is not clean: {test_source}")
        task_output_dir = output_dir / aspect
        task_output_dir.mkdir(parents=True, exist_ok=True)
        clean_test_filename = f"clean_test{len(clean_test)}.json"
        clean_test_path = (
            test_source if reuse_clean_test else task_output_dir / clean_test_filename
        )
        clean_test_metadata = {
            "schema_version": 3,
            "purpose": "mode_final_test_shared_by_all_shot_settings",
            "task": task_name,
            "aspect": aspect,
            "source_file": relative_to_project(test_source),
            "source_sha256": sha256_file(test_source),
            "source_test_count": len(test_rows),
            "test_count": len(clean_test),
            "score_sets": score_sets,
            "test_label_counts": label_counts(clean_test, score_sets),
            "gradient_train_source": relative_to_project(dataset_path),
            "gradient_train_split": relative_to_project(split_path),
            **test_audit,
        }
        if not reuse_clean_test:
            write_json(
                clean_test_path,
                {"metadata": clean_test_metadata, "train": [], "test": clean_test},
            )

        task_manifest: dict[str, Any] = {
            "task": task_name,
            "aspect": aspect,
            "score_sets": score_sets,
            "calibration_source_kind": "lora_training_validation_split",
            "selection": {
                "strategy": "nested_sha256_rank_per_class",
                "seed": seed,
                "shots_per_class": shots,
                "calibration_policy": manifest["selection"]["calibration_policy"],
                "test_policy": manifest["selection"]["test_policy"],
            },
            "dataset_file": relative_to_project(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_count": len(dataset_rows),
            "split_file": relative_to_project(split_path),
            "split_sha256": sha256_file(split_path),
            "split_seed": split.get("split_seed"),
            "validation_ratio": split.get("validation_ratio"),
            "gradient_train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "validation_label_counts": label_counts(validation_rows, score_sets),
            "eligible_validation_count": len(eligible_validation),
            "eligible_validation_label_counts": label_counts(
                eligible_validation, score_sets
            ),
            "validation_train_overlap_groups": validation_train_overlaps,
            "validation_test_overlap_groups": validation_test_overlaps,
            "validation_duplicate_prompt_groups": validation_duplicate_groups,
            "excluded_validation_ids": sorted(excluded_validation_ids),
            "test_source_file": relative_to_project(test_source),
            "test_source_sha256": sha256_file(test_source),
            "source_test_count": len(test_rows),
            "clean_test": {
                "file": relative_to_mode(clean_test_path),
                "sha256": sha256_file(clean_test_path),
                "test_count": len(clean_test),
                "test_label_counts": label_counts(clean_test, score_sets),
                "reused_existing_file": reuse_clean_test,
                **test_audit,
            },
            "splits": {},
        }

        for shot_count in shots:
            selected_by_label = {
                str(label): ranked_ids[label][:shot_count] for label in score_sets
            }
            selected_ids = {
                sample_id for ids in selected_by_label.values() for sample_id in ids
            }
            calibration = [
                row for row in eligible_validation if str(row["id"]) in selected_ids
            ]
            calibration_count = len(calibration)
            if calibration_count != len(score_sets) * shot_count:
                raise AssertionError("Unexpected calibration size")
            filename = (
                f"validation_k{shot_count}_per_class_"
                f"cal{calibration_count}_seed{seed}.json"
            )
            calibration_path = task_output_dir / filename
            metadata = {
                "schema_version": 3,
                "purpose": "mode_validation_calibration_with_teacher_cot",
                "task": task_name,
                "aspect": aspect,
                "score_sets": score_sets,
                "dataset_file": relative_to_project(dataset_path),
                "dataset_sha256": task_manifest["dataset_sha256"],
                "split_file": relative_to_project(split_path),
                "split_sha256": task_manifest["split_sha256"],
                "split_seed": split.get("split_seed"),
                "selection_seed": seed,
                "selection_strategy": manifest["selection"]["strategy"],
                "shots_per_class": shot_count,
                "calibration_count": calibration_count,
                "calibration_label_counts": label_counts(calibration, score_sets),
                "selected_ids_by_label": selected_by_label,
                "completion_source": "teacher_completion_from_training_jsonl",
                "test_file": relative_to_mode(clean_test_path),
                "test_sha256": task_manifest["clean_test"]["sha256"],
                "test_count": len(clean_test),
            }
            write_json(calibration_path, {"metadata": metadata, "train": calibration})
            task_manifest["splits"][str(shot_count)] = {
                "file": relative_to_mode(calibration_path),
                "sha256": sha256_file(calibration_path),
                "calibration_count": calibration_count,
                "calibration_label_counts": label_counts(calibration, score_sets),
                "selected_ids_by_label": selected_by_label,
                "test_file": relative_to_mode(clean_test_path),
                "test_sha256": task_manifest["clean_test"]["sha256"],
                "test_count": len(clean_test),
            }
        manifest["tasks"][aspect] = task_manifest

    api_extension = manifest.get("extensions", {}).get("rev_util_api_calibration")
    if isinstance(api_extension, dict) and isinstance(api_extension.get("aspects"), list):
        api_extension["aspects"] = [
            aspect for aspect in api_extension["aspects"] if aspect not in task_sources
        ]

    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir.resolve() / "split_manifest.json"
    base_manifest = None
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid existing manifest: {manifest_path}")
        base_manifest = loaded
    manifest = create_splits(
        task_sources={task: DEFAULT_TASK_SOURCES[task] for task in args.tasks},
        output_dir=args.output_dir,
        seed=args.seed,
        shots_per_class=args.shots_per_class,
        base_manifest=base_manifest,
    )
    for aspect in args.tasks:
        task = manifest["tasks"][aspect]
        for shots, split in task["splits"].items():
            print(
                f"{aspect}: validation k={shots}/class "
                f"calibration={split['calibration_count']} -> {split['file']}"
            )
        print(
            f"{aspect}: shared clean test={task['clean_test']['test_count']} "
            f"-> {task['clean_test']['file']}"
        )


if __name__ == "__main__":
    main()
