#!/usr/bin/env python3
"""Build API-teacher few-shot calibration and held-out tests for unseen tasks."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
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
    create_completion_with_rate_limit_retry,
    load_env,
)


TARGETS: dict[str, dict[str, Any]] = {
    "novelty": {
        "source": PROJECT_ROOT / "test_data" / "prompted_novelty_data.json",
        "source_task": "novelty_eval_updated",
        "source_aspect": "coherence",
        "shots": [1, 3, 5],
        "historical_outputs": "test_data/eval_outputs/novelty_*/novelty_eval_updated_outputs.parquet",
    },
    "revision_relatedness": {
        "source": PROJECT_ROOT / "test_data" / "prompted_revision_data.json",
        "source_task": "revision_eval",
        "source_aspect": "relatedness",
        "shots": [1, 3, 5],
        "historical_outputs": "test_data/eval_outputs/revision_*/revision_eval_outputs.parquet",
    },
    "revision_correctness": {
        "source": PROJECT_ROOT / "test_data" / "prompted_revision_data.json",
        "source_task": "revision_eval",
        "source_aspect": "correctness",
        "shots": [1, 3, 5],
        "historical_outputs": "test_data/eval_outputs/revision_*/revision_eval_outputs.parquet",
    },
    "meta_reviewer": {
        "source": PROJECT_ROOT / "test_data" / "prompted_meta_reviewer_data.json",
        "source_task": "meta_reviewer_eval",
        "source_aspect": "cascade_10way",
        # Rare labels have few unique rows (2 has 2, 5 has 3); each shot file uses
        # min(k, available_per_label) so k=3/5 remain nested and valid.
        "shots": [1, 3, 5],
        "historical_outputs": "test_data/eval_outputs/meta_reviewer_*/meta_reviewer_eval_outputs.parquet",
    },
}

REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.I | re.S)
SCORE_RE = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.I)


def force_gold_completion(record: dict[str, Any], gold_label: int) -> dict[str, Any]:
    """Build an accepted gold-aligned trajectory after teachers never match a rare label."""
    original_teacher_label = record.get("teacher_label")
    teacher_model = record.get("teacher_model") or "gold_score_fallback"
    note = (
        f"No teacher trajectory matched gold label {gold_label} after exhausting unique "
        f"candidates for this rare class (best teacher_label="
        f"{original_teacher_label!r} from {teacher_model}). "
        "This completion keeps the gold score so MoDE calibration covers every label."
    )
    completion = f"<reasoning>\n{note}\n</reasoning>\n<score>{gold_label}</score>"
    forced = dict(record)
    forced.update(
        {
            "accepted": True,
            "format_valid": True,
            "rejection_reason": None,
            "teacher_label": gold_label,
            "original_teacher_label": original_teacher_label,
            "teacher_model": teacher_model,
            "quality_status": "gold_score_forced_after_teacher_exhaustion",
            "completion": completion,
            "raw_output": completion,
            "teacher_reasoning": note,
            "forced_at_utc": utc_now(),
        }
    )
    return forced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODE_ROOT / "data")
    parser.add_argument("--api-config", type=Path, default=DISTILL_DIR / ".env")
    parser.add_argument("--model", help="Override OPENBITFUN_MODEL from the API config.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-rate-limit-retries", type=int, default=-1)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    parser.add_argument("--rate-limit-max-backoff", type=float, default=300.0)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--max-candidates-per-label", type=int, default=500)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only reuse existing trajectory JSONL records; never call the API.",
    )
    parser.add_argument(
        "--allow-gold-score-fallback",
        action="store_true",
        help=(
            "If every candidate for a label fails teacher-gold agreement, emit a short "
            "gold-aligned completion so rare classes (e.g. meta_reviewer label 5) can "
            "still form calibration."
        ),
    )
    parser.add_argument(
        "--api-attempts-per-candidate",
        type=int,
        default=1,
        help="API samples per candidate. Attempt 1 uses temperature 0; later attempts use --api-retry-temperature.",
    )
    parser.add_argument(
        "--api-retry-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for API attempts after the first greedy call.",
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


def stable_rank(seed: int, target: str, label: int, sample_id: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0unseen\0{target}\0{label}\0{sample_id}".encode()
    ).digest()


def read_target_rows(target: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    definition = TARGETS[target]
    source = Path(definition["source"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_rows = payload.get("test") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        raise ValueError(f"Missing test list: {source}")
    matching = [
        row
        for row in raw_rows
        if row.get("task") == definition["source_task"]
        and row.get("aspect") == definition["source_aspect"]
    ]
    if not matching:
        raise ValueError(f"No rows for target={target}: {source}")

    score_sets = matching[0].get("score_sets")
    if (
        not isinstance(score_sets, list)
        or not score_sets
        or any(isinstance(value, bool) or not isinstance(value, int) for value in score_sets)
        or len(set(score_sets)) != len(score_sets)
    ):
        raise ValueError(f"Invalid score_sets for target={target}: {score_sets!r}")

    rows = []
    seen_prompts: set[str] = set()
    duplicate_ids = []
    for source_index, raw in enumerate(matching):
        if raw.get("score_sets") != score_sets or raw.get("labels") not in score_sets:
            raise ValueError(f"Inconsistent labels for {target} row {source_index}")
        if not isinstance(raw.get("prompt"), list) or not raw["prompt"]:
            raise ValueError(f"Missing prompt for {target} row {source_index}")
        row = dict(raw)
        row["id"] = f"{target}_{source_index:04d}"
        row["source_task"] = raw.get("task")
        row["source_aspect"] = raw.get("aspect")
        row["task"] = "unseen_task"
        row["aspect"] = target
        fingerprint = prompt_fingerprint(row)
        if fingerprint in seen_prompts:
            duplicate_ids.append(row["id"])
            continue
        seen_prompts.add(fingerprint)
        rows.append(row)
    return rows, {
        "source_file": source,
        "source_count": len(matching),
        "score_sets": list(score_sets),
        "duplicate_source_ids": duplicate_ids,
    }


def scan_trajectories(
    path: Path, *, target: str, teacher_model: str
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
            if record.get("target") != target:
                continue
            sample_id = str(record.get("id", ""))
            if not sample_id:
                raise ValueError(f"Trajectory lacks an ID at {path}:{line_number}")
            existing = records.get(sample_id)
            if record.get("accepted"):
                if existing is None or not existing.get("accepted"):
                    records[sample_id] = record
            elif record.get("teacher_model") == teacher_model and existing is None:
                records[sample_id] = record
    return records


def import_historical_trajectories(
    *,
    target: str,
    rows: list[dict[str, Any]],
    source_path: Path,
    existing_ids: set[str],
    trajectory_path: Path,
) -> dict[str, dict[str, Any]]:
    import pandas as pd

    definition = TARGETS[target]
    rows_by_fingerprint = {prompt_fingerprint(row): row for row in rows}
    imported: dict[str, dict[str, Any]] = {}
    pattern = str(PROJECT_ROOT / definition["historical_outputs"])
    for raw_path in sorted(glob.glob(pattern)):
        history_path = Path(raw_path)
        model_tag = history_path.parent.name
        frame = pd.read_parquet(history_path)
        output_columns = sorted(
            (column for column in frame.columns if column.startswith("output_")),
            key=lambda value: int(value.rsplit("_", 1)[-1]),
        )
        for _, history_row in frame.iterrows():
            if (
                history_row.get("task") != definition["source_task"]
                or history_row.get("aspect") != definition["source_aspect"]
            ):
                continue
            prompt = history_row.get("prompt")
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            if not isinstance(prompt, list):
                continue
            fingerprint = prompt_fingerprint({"prompt": prompt})
            row = rows_by_fingerprint.get(fingerprint)
            if row is None or row["id"] in existing_ids or row["id"] in imported:
                continue
            for column in output_columns:
                output = history_row.get(column)
                if not isinstance(output, str):
                    continue
                normalized_output = output
                if (
                    not REASONING_RE.search(normalized_output)
                    and re.search(r"<reasoning>", normalized_output, re.I)
                    and re.search(r"<correctness>", normalized_output, re.I)
                ):
                    normalized_output = re.sub(
                        r"\s*(<correctness>)",
                        r"\n</reasoning>\n\1",
                        normalized_output,
                        count=1,
                        flags=re.I,
                    )
                record = build_record(
                    row=row,
                    target=target,
                    source_path=source_path,
                    source_index=int(history_row.name),
                    run_id="historical_eval_import",
                    teacher_model=f"historical:{model_tag}",
                    base_url="local-eval-output://test_data",
                    max_tokens=0,
                    timeout=0,
                    elapsed=0,
                    call={
                        "content": normalized_output,
                        "response_id": None,
                        "model": model_tag,
                        "finish_reason": None,
                        "usage": None,
                    },
                )
                if record["accepted"]:
                    record["quality_status"] = "imported_eval_output_gold_label_match"
                    record["historical_source_file"] = relative_to_project(history_path)
                    record["historical_output_normalized"] = normalized_output != output
                    record["imported_at_utc"] = utc_now()
                    append_jsonl(trajectory_path, record)
                    imported[row["id"]] = record
                    break
    return imported


def build_record(
    *,
    row: dict[str, Any],
    target: str,
    source_path: Path,
    source_index: int,
    run_id: str,
    teacher_model: str,
    base_url: str,
    max_tokens: int,
    timeout: float,
    elapsed: float,
    call: dict[str, Any],
) -> dict[str, Any]:
    content = str(call.get("content") or "").strip()
    reasoning_match = REASONING_RE.search(content)
    score_matches = SCORE_RE.findall(content)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    teacher_label = int(score_matches[-1]) if score_matches else None
    allowed_scores = set(row["score_sets"])
    if not reasoning_match or not reasoning:
        rejection_reason = "missing_reasoning"
    elif teacher_label not in allowed_scores:
        rejection_reason = "teacher_label_out_of_score_set"
    elif teacher_label != int(row["labels"]):
        rejection_reason = "teacher_label_mismatch"
    else:
        rejection_reason = None
    accepted = rejection_reason is None
    return {
        "record_type": "distillation",
        "schema_version": 1,
        "id": row["id"],
        "target": target,
        "source_index": source_index,
        "source_file": str(source_path),
        "task": row["task"],
        "aspect": row["aspect"],
        "score_sets": row["score_sets"],
        "gold_label": int(row["labels"]),
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "teacher_model": teacher_model,
        "api_base_url": base_url,
        "decoding": {
            "n": 1,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout,
        },
        "elapsed_seconds": round(elapsed, 3),
        "teacher_label": teacher_label,
        "accepted": accepted,
        "format_valid": reasoning_match is not None and bool(score_matches),
        "rejection_reason": rejection_reason,
        "quality_status": (
            "auto_accepted_label_match"
            if accepted
            else f"rejected_{rejection_reason}"
        ),
        "teacher_reasoning": reasoning,
        "internal_reasoning": call.get("reasoning_content") or "",
        "raw_output": content,
        "completion": content if reasoning_match and score_matches else None,
        "response": {
            "response_id": call.get("response_id"),
            "model": call.get("model"),
            "finish_reason": call.get("finish_reason"),
            "usage": call.get("usage"),
        },
    }


def calibration_row(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    completion = record.get("completion")
    if not record.get("accepted") or not completion:
        raise ValueError(f"Cannot materialize rejected trajectory: {row['id']}")
    result = dict(row)
    result["completion"] = [{"role": "assistant", "content": completion}]
    result["mode_teacher"] = {
        "model": record["teacher_model"],
        "generated_at_utc": record.get("generated_at_utc"),
        "response_id": (record.get("response") or {}).get("response_id"),
        "trajectory_quality_status": record.get("quality_status"),
    }
    return result


def label_counts(rows: list[dict[str, Any]], scores: list[int]) -> dict[str, int]:
    counts = Counter(int(row["labels"]) for row in rows)
    return {str(label): counts[label] for label in scores}


def select_calibration(
    *,
    target: str,
    rows: list[dict[str, Any]],
    source_path: Path,
    score_sets: list[int],
    required: int,
    trajectory_path: Path,
    teacher_model: str,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    known = scan_trajectories(
        trajectory_path, target=target, teacher_model=teacher_model
    )
    imported = import_historical_trajectories(
        target=target,
        rows=rows,
        source_path=source_path,
        existing_ids={
            sample_id for sample_id, record in known.items() if record.get("accepted")
        },
        trajectory_path=trajectory_path,
    )
    known.update(imported)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    append_jsonl(
        trajectory_path,
        {
            "record_type": "run_start",
            "schema_version": 1,
            "run_id": run_id,
            "target": target,
            "started_at_utc": utc_now(),
            "teacher_model": teacher_model,
            "api_base_url": base_url,
            "required_per_label": required,
            "offline": args.offline,
        },
    )
    client = None
    selected: dict[int, list[dict[str, Any]]] = {}
    reused_count = 0
    imported_count = len(imported)
    api_count = 0
    considered_by_label: dict[str, int] = {}
    status = "completed"
    error_text = None
    try:
        for label in score_sets:
            candidates = [row for row in rows if int(row["labels"]) == label]
            candidates.sort(
                key=lambda row: (
                    stable_rank(args.seed, target, label, row["id"]),
                    row["id"],
                )
            )
            # Cap by unique source rows so rare classes can still participate in k=3/5.
            label_required = min(required, len(candidates))
            accepted = []
            considered = 0
            for source_index, row in enumerate(candidates):
                if (
                    considered >= args.max_candidates_per_label
                    or len(accepted) >= label_required
                ):
                    break
                considered += 1
                record = known.get(row["id"])
                # Reuse only accepted trajectories. Rejected/empty API attempts should
                # not permanently block retries on a later run with larger max_tokens.
                if record is not None and record.get("accepted"):
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
                    attempts = max(1, int(args.api_attempts_per_candidate))
                    # Multi-sample only for rare classes; abundant labels can move on
                    # to the next candidate after one greedy miss.
                    if len(candidates) > 5:
                        attempts = 1
                    record = None
                    for attempt_index in range(1, attempts + 1):
                        temperature = (
                            0.0
                            if attempt_index == 1
                            else float(args.api_retry_temperature)
                        )
                        top_p = 1.0 if attempt_index == 1 else 0.95
                        print(
                            f"[{target}] label={label} "
                            f"accepted={len(accepted)}/{label_required} "
                            f"(global_k={required}) "
                            f"calling {teacher_model} for {row['id']} "
                            f"(attempt {attempt_index}/{attempts}, temp={temperature})",
                            flush=True,
                        )
                        started = time.monotonic()
                        call = create_completion_with_rate_limit_retry(
                            client,
                            teacher_model,
                            row["prompt"],
                            args.max_tokens,
                            args.progress_interval,
                            args.max_rate_limit_retries,
                            args.rate_limit_backoff,
                            args.rate_limit_max_backoff,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        record = build_record(
                            row=row,
                            target=target,
                            source_path=source_path,
                            source_index=source_index,
                            run_id=run_id,
                            teacher_model=teacher_model,
                            base_url=base_url,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            elapsed=time.monotonic() - started,
                            call=call,
                        )
                        record["api_attempt"] = attempt_index
                        record["decoding"] = {
                            **record.get("decoding", {}),
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                        append_jsonl(trajectory_path, record)
                        known[row["id"]] = record
                        api_count += 1
                        print(
                            f"[{target}] {row['id']} teacher={record['teacher_label']} "
                            f"gold={label} accepted={record['accepted']} "
                            f"attempt={attempt_index}",
                            flush=True,
                        )
                        if record.get("accepted"):
                            break
                    if record is None:
                        continue
                if record.get("accepted"):
                    accepted.append(calibration_row(row, record))
            considered_by_label[str(label)] = considered
            if len(accepted) < label_required:
                if not args.allow_gold_score_fallback:
                    raise RuntimeError(
                        f"{target} label {label}: only {len(accepted)}/{label_required} "
                        f"accepted after {considered} unique candidates "
                        f"(requested global_k={required})"
                    )
                for row in candidates[:considered]:
                    if len(accepted) >= label_required:
                        break
                    if any(item["id"] == row["id"] for item in accepted):
                        continue
                    base = known.get(row["id"]) or {
                        "id": row["id"],
                        "target": target,
                        "gold_label": label,
                        "teacher_model": teacher_model,
                        "record_type": "distillation",
                        "schema_version": 1,
                    }
                    forced = force_gold_completion(base, label)
                    append_jsonl(trajectory_path, forced)
                    known[row["id"]] = forced
                    accepted.append(calibration_row(row, forced))
                    print(
                        f"[{target}] {row['id']} gold-score fallback for label={label} "
                        f"(original_teacher_label={forced.get('original_teacher_label')})",
                        flush=True,
                    )
                if len(accepted) < label_required:
                    raise RuntimeError(
                        f"{target} label {label}: only {len(accepted)}/{label_required} "
                        f"after gold-score fallback over {considered} candidates"
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
                "target": target,
                "finished_at_utc": utc_now(),
                "teacher_model": teacher_model,
                "status": status,
                "error": error_text,
                "reused_trajectory_count": reused_count,
                "imported_historical_trajectory_count": imported_count,
                "new_api_call_count": api_count,
                "considered_by_label": considered_by_label,
            },
        )
    return selected, {
        "reused_trajectory_count": reused_count,
        "imported_historical_trajectory_count": imported_count,
        "new_api_call_count": api_count,
        "considered_by_label": considered_by_label,
    }


def build_target(
    *,
    target: str,
    teacher_model: str,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows, source_info = read_target_rows(target)
    definition = TARGETS[target]
    shots = list(definition["shots"])
    score_sets = source_info["score_sets"]
    output_dir = args.output_dir.resolve()
    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = target_dir / "api_trajectories.jsonl"
    selected, selection_audit = select_calibration(
        target=target,
        rows=rows,
        source_path=source_info["source_file"],
        score_sets=score_sets,
        required=max(shots),
        trajectory_path=trajectory_path,
        teacher_model=teacher_model,
        base_url=base_url,
        api_key=api_key,
        args=args,
    )

    selected_ids = {
        row["id"] for label in score_sets for row in selected[label][: max(shots)]
    }
    test_rows = [row for row in rows if row["id"] not in selected_ids]
    test_path = target_dir / f"clean_test{len(test_rows)}.json"
    test_metadata = {
        "schema_version": 1,
        "purpose": "mode_unseen_task_final_test_shared_by_all_shot_settings",
        "task": "unseen_task",
        "aspect": target,
        "score_sets": score_sets,
        "source_file": relative_to_project(source_info["source_file"]),
        "source_sha256": sha256_file(source_info["source_file"]),
        "source_target_count": source_info["source_count"],
        "deduplicated_source_count": len(rows),
        "removed_duplicate_source_ids": source_info["duplicate_source_ids"],
        "removed_calibration_ids": sorted(selected_ids),
        "test_count": len(test_rows),
        "test_label_counts": label_counts(test_rows, score_sets),
    }
    write_json(test_path, {"metadata": test_metadata, "train": [], "test": test_rows})

    result: dict[str, Any] = {
        "target": target,
        "score_sets": score_sets,
        "supported_shots_per_class": shots,
        "teacher_model": teacher_model,
        "trajectory_file": relative_to_mode(trajectory_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "selection_audit": selection_audit,
        "test_file": relative_to_mode(test_path),
        "test_sha256": sha256_file(test_path),
        "test_count": len(test_rows),
        "calibration_files": {},
    }
    for shot_count in shots:
        calibration = [
            row for label in score_sets for row in selected[label][:shot_count]
        ]
        calibration_count = len(calibration)
        calibration_path = target_dir / (
            f"api_validation_k{shot_count}_per_class_"
            f"cal{calibration_count}_seed{args.seed}.json"
        )
        metadata = {
            "schema_version": 1,
            "purpose": "mode_unseen_task_api_teacher_calibration",
            "task": "unseen_task",
            "aspect": target,
            "score_sets": score_sets,
            "shots_per_class": shot_count,
            "shots_per_class_policy": (
                "nested_min_k_and_available_unique_rows_per_label"
            ),
            "calibration_count": calibration_count,
            "calibration_label_counts": label_counts(calibration, score_sets),
            "selected_ids_by_label": {
                str(label): [row["id"] for row in selected[label][:shot_count]]
                for label in score_sets
            },
            "available_per_label": {
                str(label): len(selected[label]) for label in score_sets
            },
            "selection_seed": args.seed,
            "selection_strategy": "nested_sha256_rank_per_class_then_teacher_gold_agreement",
            "completion_source": (
                "historical_eval_or_api_teacher_cot_gold_label_agreement"
            ),
            "teacher_model": teacher_model,
            "source_file": relative_to_project(source_info["source_file"]),
            "source_sha256": sha256_file(source_info["source_file"]),
            "test_file": relative_to_mode(test_path),
            "test_sha256": result["test_sha256"],
            "test_count": len(test_rows),
        }
        write_json(calibration_path, {"metadata": metadata, "train": calibration})
        result["calibration_files"][str(shot_count)] = {
            "file": relative_to_mode(calibration_path),
            "sha256": sha256_file(calibration_path),
            "count": calibration_count,
        }
    return result


def main() -> None:
    args = parse_args()
    if args.max_candidates_per_label < 1:
        raise SystemExit("--max-candidates-per-label must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_env(args.api_config.expanduser().resolve())
    api_key = os.environ.get("OPENBITFUN_API_KEY", "").strip()
    base_url = os.environ.get("OPENBITFUN_BASE_URL", "").strip().rstrip("/")
    teacher_model = (args.model or os.environ.get("OPENBITFUN_MODEL", "")).strip()
    if not teacher_model:
        raise SystemExit("A teacher model is required via --model or OPENBITFUN_MODEL")
    if not args.offline and (not api_key or not base_url):
        raise SystemExit("OPENBITFUN_API_KEY and OPENBITFUN_BASE_URL are required")

    summary_path = args.output_dir / "unseen_generation_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not isinstance(summary.get("targets"), dict):
            raise ValueError(f"Invalid existing unseen summary: {summary_path}")
    else:
        summary = {"schema_version": 1, "targets": {}}
    summary.update(
        {
            "generated_at_utc": utc_now(),
            "seed": args.seed,
            "teacher_model": teacher_model,
        }
    )
    for target in args.targets:
        print(f"\nPreparing unseen target: {target}", flush=True)
        summary["targets"][target] = build_target(
            target=target,
            teacher_model=teacher_model,
            base_url=base_url,
            api_key=api_key,
            args=args,
        )
        write_json(summary_path, summary)
        print(
            f"Completed {target}: test={summary['targets'][target]['test_count']}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed API trajectories were preserved.", file=sys.stderr)
        raise SystemExit(130)
