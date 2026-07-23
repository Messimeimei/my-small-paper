#!/usr/bin/env python3
"""Build nested RevUtil MoDE calibration sets with API-generated teacher CoT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODE_ROOT.parent
DISTILL_DIR = PROJECT_ROOT / "train_data" / "distill_data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_data.distill_data.generate_mode_cot import (  # noqa: E402
    append_jsonl,
    build_distillation_record,
    create_completion_with_rate_limit_retry,
    load_env,
)


ASPECTS = (
    "actionability",
    "grounding_specificity",
    "helpfulness",
    "verifiability",
    "verifiability_extraction",
)
SOURCE_FILES = {
    aspect: PROJECT_ROOT / "train_data" / "cleaned_data" / f"rev_util_{aspect}_4800.json"
    for aspect in ASPECTS
}
DEFAULT_TEST_SOURCE = PROJECT_ROOT / "test_data" / "prompted_rev_util_data.json"
DEFAULT_MANIFEST = MODE_ROOT / "data" / "split_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aspects", nargs="+", choices=ASPECTS, default=list(ASPECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shots-per-class", type=int, nargs="+", default=[1, 3, 5]
    )
    parser.add_argument("--output-dir", type=Path, default=MODE_ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--test-source", type=Path, default=DEFAULT_TEST_SOURCE)
    parser.add_argument("--api-config", type=Path, default=DISTILL_DIR / ".env")
    parser.add_argument("--model", help="Override OPENBITFUN_MODEL from the API config.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-rate-limit-retries", type=int, default=-1)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    parser.add_argument("--rate-limit-max-backoff", type=float, default=300.0)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument(
        "--max-candidates-per-label",
        type=int,
        default=200,
        help="Stop instead of spending unbounded API calls when teacher/gold agreement is low.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only reuse existing trajectories; never call the API.",
    )
    parser.add_argument(
        "--tests-only",
        action="store_true",
        help="Only split and validate evaluator-ready test files; do not prepare calibration.",
    )
    parser.add_argument(
        "--no-reuse-project-distill",
        action="store_true",
        help="Do not reuse matching records from train_data/distill_data.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def relative_to_mode(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODE_ROOT))
    except ValueError:
        return str(path.resolve())


def prompt_fingerprint(row: dict[str, Any]) -> str:
    canonical = json.dumps(
        row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_rank(seed: int, aspect: str, label: int, sample_id: str) -> bytes:
    value = f"{seed}\0rev_util\0{aspect}\0{label}\0{sample_id}".encode()
    return hashlib.sha256(value).digest()


def read_train_rows(path: Path, aspect: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("train") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing non-empty train list: {path}")
    seen_ids: set[str] = set()
    expected_scores: tuple[int, ...] | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("aspect") != aspect:
            raise ValueError(f"Invalid {aspect} row at {path}:{index}")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Missing or duplicate ID at {path}:{index}")
        seen_ids.add(sample_id)
        if not isinstance(row.get("prompt"), list) or not row["prompt"]:
            raise ValueError(f"Missing prompt at {path}:{index}")
        score_sets = row.get("score_sets")
        if (
            not isinstance(score_sets, list)
            or not score_sets
            or any(isinstance(value, bool) or not isinstance(value, int) for value in score_sets)
        ):
            raise ValueError(f"Invalid score_sets at {path}:{index}")
        current_scores = tuple(score_sets)
        if expected_scores is None:
            expected_scores = current_scores
        elif current_scores != expected_scores:
            raise ValueError(f"Inconsistent score_sets at {path}:{index}")
        if row.get("labels") not in score_sets:
            raise ValueError(f"Label outside score_sets at {path}:{index}")
    return rows


def read_test_rows(path: Path, aspect: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_rows = payload.get("test") if isinstance(payload, dict) else None
    if not isinstance(all_rows, list):
        raise ValueError(f"Missing test list: {path}")
    rows = [row for row in all_rows if row.get("aspect") == aspect]
    if not rows:
        raise ValueError(f"No test rows for aspect={aspect}: {path}")
    return rows


def deduplicate_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    seen: set[str] = set()
    clean = []
    removed = []
    for row in rows:
        fingerprint = prompt_fingerprint(row)
        if fingerprint in seen:
            removed.append(str(row["id"]))
        else:
            seen.add(fingerprint)
            clean.append(row)
    return clean, sorted(removed)


def eligible_candidates(
    rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_fingerprints = {prompt_fingerprint(row) for row in test_rows}
    seen: set[str] = set()
    eligible = []
    test_overlaps = []
    duplicates = []
    for row in rows:
        fingerprint = prompt_fingerprint(row)
        sample_id = str(row["id"])
        if fingerprint in test_fingerprints:
            test_overlaps.append(sample_id)
        elif fingerprint in seen:
            duplicates.append(sample_id)
        else:
            seen.add(fingerprint)
            eligible.append(row)
    return eligible, {
        "excluded_test_overlap_ids": sorted(test_overlaps),
        "excluded_duplicate_source_ids": sorted(duplicates),
    }


def scan_trajectory_file(
    path: Path, *, aspect: str, teacher_model: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "distillation":
                continue
            if record.get("aspect") != aspect or record.get("teacher_model") != teacher_model:
                continue
            sample_id = str(record.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Trajectory lacks ID at {path}:{line_number}")
            if sample_id in records:
                raise ValueError(
                    f"Duplicate {teacher_model} trajectory for {sample_id} in {path}"
                )
            records[sample_id] = record
    return records


def load_known_trajectories(
    *,
    aspect: str,
    teacher_model: str,
    local_path: Path,
    reuse_project_distill: bool,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if reuse_project_distill:
        project_path = DISTILL_DIR / f"rev_util_{aspect}_4800_distill.jsonl"
        records.update(
            scan_trajectory_file(
                project_path, aspect=aspect, teacher_model=teacher_model
            )
        )
    records.update(
        scan_trajectory_file(local_path, aspect=aspect, teacher_model=teacher_model)
    )
    return records


def calibration_row(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    completion = record.get("completion")
    if not record.get("accepted") or not isinstance(completion, str) or not completion:
        raise ValueError(f"Cannot materialize rejected trajectory for {row['id']}")
    result = dict(row)
    result["completion"] = [{"role": "assistant", "content": completion}]
    result["mode_teacher"] = {
        "model": record["teacher_model"],
        "generated_at_utc": record.get("generated_at_utc"),
        "response_id": (record.get("response") or {}).get("response_id"),
        "trajectory_quality_status": record.get("quality_status"),
    }
    return result


def label_counts(rows: list[dict[str, Any]], score_sets: list[int]) -> dict[str, int]:
    counts = Counter(int(row["labels"]) for row in rows)
    return {str(label): counts[label] for label in score_sets}


def materialize_clean_test(
    *, aspect: str, test_source: Path, output_dir: Path
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    raw_test_rows = read_test_rows(test_source, aspect)
    clean_test_rows, duplicate_test_ids = deduplicate_rows(raw_test_rows)
    score_sets = list(clean_test_rows[0]["score_sets"])
    aspect_dir = output_dir / aspect
    aspect_dir.mkdir(parents=True, exist_ok=True)
    test_path = aspect_dir / f"clean_test{len(clean_test_rows)}.json"
    test_metadata = {
        "schema_version": 1,
        "purpose": "mode_unseen_task_final_test_shared_by_all_shot_settings",
        "task": "rev_util",
        "aspect": aspect,
        "score_sets": score_sets,
        "source_file": relative_to_project(test_source),
        "source_sha256": sha256_file(test_source),
        "source_test_count": len(raw_test_rows),
        "test_count": len(clean_test_rows),
        "test_label_counts": label_counts(clean_test_rows, score_sets),
        "removed_duplicate_test_ids": duplicate_test_ids,
    }
    write_json(
        test_path,
        {"metadata": test_metadata, "train": [], "test": clean_test_rows},
    )
    return test_path, clean_test_rows, duplicate_test_ids


def select_with_teacher(
    *,
    rows: list[dict[str, Any]],
    aspect: str,
    score_sets: list[int],
    seed: int,
    max_shots: int,
    max_candidates_per_label: int,
    teacher_model: str,
    base_url: str,
    api_key: str,
    trajectory_path: Path,
    known: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    client = None
    selected: dict[int, list[dict[str, Any]]] = {}
    attempted_by_label: dict[str, int] = {}
    reused_count = 0
    api_count = 0
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    append_jsonl(
        trajectory_path,
        {
            "record_type": "run_start",
            "schema_version": 1,
            "run_id": run_id,
            "started_at_utc": utc_now(),
            "task": "rev_util",
            "aspect": aspect,
            "teacher_model": teacher_model,
            "api_base_url": base_url,
            "selection_seed": seed,
            "required_per_label": max_shots,
            "offline": args.offline,
        },
    )
    status = "completed"
    error_text = None
    try:
        for label in score_sets:
            candidates = [row for row in rows if int(row["labels"]) == label]
            candidates.sort(
                key=lambda row: (
                    stable_rank(seed, aspect, label, str(row["id"])),
                    str(row["id"]),
                )
            )
            accepted: list[dict[str, Any]] = []
            considered = 0
            for source_index, row in enumerate(candidates):
                if considered >= max_candidates_per_label or len(accepted) >= max_shots:
                    break
                considered += 1
                sample_id = str(row["id"])
                record = known.get(sample_id)
                if record is not None:
                    reused_count += 1
                else:
                    if args.offline:
                        continue
                    if client is None:
                        from openai import OpenAI

                        client = OpenAI(
                            api_key=api_key,
                            base_url=base_url,
                            timeout=args.timeout,
                            max_retries=2,
                        )
                    print(
                        f"[{aspect}] label={label} accepted={len(accepted)}/{max_shots} "
                        f"calling {teacher_model} for id={sample_id}",
                        flush=True,
                    )
                    started = time.monotonic()
                    teacher_call = create_completion_with_rate_limit_retry(
                        client,
                        teacher_model,
                        row["prompt"],
                        args.max_tokens,
                        args.progress_interval,
                        args.max_rate_limit_retries,
                        args.rate_limit_backoff,
                        args.rate_limit_max_backoff,
                    )
                    record = build_distillation_record(
                        row=row,
                        source_index=source_index,
                        input_path=SOURCE_FILES[aspect].resolve(),
                        run_id=run_id,
                        teacher_model=teacher_model,
                        base_url=base_url,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        elapsed=time.monotonic() - started,
                        teacher_call=teacher_call,
                    )
                    append_jsonl(trajectory_path, record)
                    known[sample_id] = record
                    api_count += 1
                    print(
                        f"[{aspect}] id={sample_id} teacher={record.get('teacher_label')} "
                        f"gold={label} accepted={record.get('accepted')}",
                        flush=True,
                    )
                if record.get("accepted"):
                    accepted.append(calibration_row(row, record))
            attempted_by_label[str(label)] = considered
            if len(accepted) < max_shots:
                raise RuntimeError(
                    f"{aspect} label {label}: found {len(accepted)} accepted teacher "
                    f"trajectories after {considered} candidates; need {max_shots}. "
                    "Rerun with a larger --max-candidates-per-label or without --offline."
                )
            selected[label] = accepted
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        append_jsonl(
            trajectory_path,
            {
                "record_type": "run_end",
                "schema_version": 1,
                "run_id": run_id,
                "finished_at_utc": utc_now(),
                "task": "rev_util",
                "aspect": aspect,
                "teacher_model": teacher_model,
                "status": status,
                "error": error_text,
                "reused_trajectory_count": reused_count,
                "new_api_call_count": api_count,
                "considered_by_label": attempted_by_label,
            },
        )
    return selected, {
        "reused_trajectory_count": reused_count,
        "new_api_call_count": api_count,
        "considered_by_label": attempted_by_label,
    }


def update_manifest(
    path: Path, aspect: str, task_manifest: dict[str, Any]
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        raise ValueError(f"Invalid split manifest: {path}")
    payload["tasks"][aspect] = task_manifest
    api_aspects = sorted(
        name
        for name, task in payload["tasks"].items()
        if isinstance(task, dict)
        and task.get("calibration_source_kind")
        == "cleaned_human_labeled_data_with_api_teacher_cot"
    )
    payload.setdefault("extensions", {})["rev_util_api_calibration"] = {
        "schema_version": 1,
        "task": "rev_util",
        "aspects": api_aspects,
        "policy": (
            "nested SHA256-ranked samples per class from cleaned human-labeled data; "
            "only strict teacher CoT outputs whose score matches the human label are eligible"
        ),
    }
    write_json(path, payload)


def build_aspect(
    *,
    aspect: str,
    teacher_model: str,
    base_url: str,
    api_key: str,
    shots: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_path = SOURCE_FILES[aspect].resolve()
    test_source = args.test_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_train_rows(source_path, aspect)
    score_sets = list(source_rows[0]["score_sets"])
    test_path, clean_test_rows, duplicate_test_ids = materialize_clean_test(
        aspect=aspect, test_source=test_source, output_dir=output_dir
    )
    candidates, candidate_audit = eligible_candidates(source_rows, clean_test_rows)

    aspect_dir = output_dir / aspect
    aspect_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = aspect_dir / "api_trajectories.jsonl"
    known = load_known_trajectories(
        aspect=aspect,
        teacher_model=teacher_model,
        local_path=trajectory_path,
        reuse_project_distill=not args.no_reuse_project_distill,
    )
    selected, selection_audit = select_with_teacher(
        rows=candidates,
        aspect=aspect,
        score_sets=score_sets,
        seed=args.seed,
        max_shots=max(shots),
        max_candidates_per_label=args.max_candidates_per_label,
        teacher_model=teacher_model,
        base_url=base_url,
        api_key=api_key,
        trajectory_path=trajectory_path,
        known=known,
        args=args,
    )

    task_manifest: dict[str, Any] = {
        "task": "rev_util",
        "aspect": aspect,
        "score_sets": score_sets,
        "calibration_source_kind": "cleaned_human_labeled_data_with_api_teacher_cot",
        "candidate_source_file": relative_to_project(source_path),
        "candidate_source_sha256": sha256_file(source_path),
        "candidate_source_count": len(source_rows),
        "eligible_candidate_count": len(candidates),
        "eligible_candidate_label_counts": label_counts(candidates, score_sets),
        "candidate_audit": candidate_audit,
        "teacher_model": teacher_model,
        "api_base_url": base_url,
        "trajectory_file": relative_to_mode(trajectory_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "selection_seed": args.seed,
        "selection_strategy": "nested_sha256_rank_per_class_then_teacher_gold_agreement",
        "selection_audit": selection_audit,
        "clean_test": {
            "file": relative_to_mode(test_path),
            "sha256": sha256_file(test_path),
            "test_count": len(clean_test_rows),
            "test_label_counts": label_counts(clean_test_rows, score_sets),
            "removed_duplicate_test_ids": duplicate_test_ids,
        },
        "splits": {},
    }

    for shot_count in shots:
        selected_by_label = {
            str(label): [str(row["id"]) for row in selected[label][:shot_count]]
            for label in score_sets
        }
        calibration = [
            row
            for label in score_sets
            for row in selected[label][:shot_count]
        ]
        calibration_count = len(calibration)
        filename = (
            f"api_validation_k{shot_count}_per_class_"
            f"cal{calibration_count}_seed{args.seed}.json"
        )
        calibration_path = aspect_dir / filename
        metadata = {
            "schema_version": 1,
            "purpose": "mode_unseen_task_api_teacher_calibration",
            "task": "rev_util",
            "aspect": aspect,
            "score_sets": score_sets,
            "shots_per_class": shot_count,
            "calibration_count": calibration_count,
            "calibration_label_counts": label_counts(calibration, score_sets),
            "selected_ids_by_label": selected_by_label,
            "selection_seed": args.seed,
            "selection_strategy": task_manifest["selection_strategy"],
            "completion_source": "api_teacher_cot_gold_label_agreement",
            "teacher_model": teacher_model,
            "candidate_source_file": relative_to_project(source_path),
            "candidate_source_sha256": task_manifest["candidate_source_sha256"],
            "test_file": relative_to_mode(test_path),
            "test_sha256": task_manifest["clean_test"]["sha256"],
            "test_count": len(clean_test_rows),
        }
        write_json(calibration_path, {"metadata": metadata, "train": calibration})
        task_manifest["splits"][str(shot_count)] = {
            "file": relative_to_mode(calibration_path),
            "sha256": sha256_file(calibration_path),
            "calibration_count": calibration_count,
            "calibration_label_counts": label_counts(calibration, score_sets),
            "selected_ids_by_label": selected_by_label,
            "test_file": relative_to_mode(test_path),
            "test_sha256": task_manifest["clean_test"]["sha256"],
            "test_count": len(clean_test_rows),
        }
    return task_manifest


def main() -> None:
    args = parse_args()
    shots = sorted(set(args.shots_per_class))
    if not shots or any(value not in {1, 3, 5} for value in shots):
        raise SystemExit("--shots-per-class must contain values from 1, 3, and 5")
    if args.max_candidates_per_label < max(shots):
        raise SystemExit("--max-candidates-per-label must be at least the largest shot count")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_source = args.test_source.expanduser().resolve()
    print("Materializing evaluator-ready RevUtil test files", flush=True)
    for aspect in args.aspects:
        test_path, test_rows, _ = materialize_clean_test(
            aspect=aspect, test_source=test_source, output_dir=output_dir
        )
        print(f"  {aspect}: {len(test_rows)} -> {test_path}", flush=True)
    if args.tests_only:
        return

    load_env(args.api_config.expanduser().resolve())
    api_key = os.environ.get("OPENBITFUN_API_KEY", "").strip()
    base_url = os.environ.get("OPENBITFUN_BASE_URL", "").strip().rstrip("/")
    teacher_model = (args.model or os.environ.get("OPENBITFUN_MODEL", "")).strip()
    if not teacher_model:
        raise SystemExit("A teacher model is required via --model or OPENBITFUN_MODEL")
    if not args.offline and (not api_key or not base_url):
        raise SystemExit("OPENBITFUN_API_KEY and OPENBITFUN_BASE_URL are required")

    manifest_path = args.manifest.expanduser().resolve()
    for aspect in args.aspects:
        print(f"\nPreparing RevUtil aspect: {aspect}", flush=True)
        task_manifest = build_aspect(
            aspect=aspect,
            teacher_model=teacher_model,
            base_url=base_url,
            api_key=api_key,
            shots=shots,
            args=args,
        )
        update_manifest(manifest_path, aspect, task_manifest)
        print(
            f"Completed {aspect}: score_sets={task_manifest['score_sets']} "
            f"test={task_manifest['clean_test']['test_count']}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed API trajectory records were preserved.", file=sys.stderr)
        raise SystemExit(130)
