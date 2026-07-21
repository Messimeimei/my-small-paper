#!/usr/bin/env python3
"""对清洗数据进行 CoT 蒸馏，并按样本 ID 与教师模型安全续跑。"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "cleaned_data"
    / "preview"
    / "rw_gen_coherence_4811_preview.json"
)
DEFAULT_CONFIG = SCRIPT_DIR / ".env"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "preview"

OUTPUT_RE = re.compile(
    r"\A\s*<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>\s*"
    r"<score>\s*(?P<score>[01])\s*</score>\s*\Z",
    re.I | re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        help=(
            "Teacher model for this run. Overrides OPENBITFUN_MODEL; samples already "
            "processed by this model are skipped."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens per call; MoDE-CoTD does not report this value.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between waiting messages during an API call; 0 disables them.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=180,
        help="Reasoning preview length printed after each sample; 0 disables it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the single distillation file before this run.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(
            f"Missing API config: {path}\n"
            "Create it with OPENBITFUN_API_KEY, OPENBITFUN_BASE_URL, and "
            "optionally OPENBITFUN_MODEL."
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("train") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list or an object containing a train list.")
    if not rows:
        raise ValueError("Input JSON contains no training rows.")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), list):
            raise ValueError(f"Row {index} has no valid prompt message list.")
        if str(row.get("labels")) not in {"0", "1"}:
            raise ValueError(f"Row {index} has a non-binary gold label: {row.get('labels')!r}")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"Row {index} has no non-empty id.")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate input id: {sample_id}")
        seen_ids.add(sample_id)
    return rows


def response_record(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    message = choice.message
    extra = getattr(message, "model_extra", None) or {}
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is None:
        reasoning_content = extra.get("reasoning_content")
    usage = response.usage.model_dump() if response.usage is not None else None
    return {
        "response_id": response.id,
        "model": response.model,
        "finish_reason": choice.finish_reason,
        "content": message.content or "",
        "reasoning_content": reasoning_content or "",
        "usage": usage,
    }


def create_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    progress_interval: float,
) -> dict[str, Any]:
    def request() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
        )

    if progress_interval <= 0:
        return response_record(request())

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("response", request()))
        except BaseException as error:
            result_queue.put(("error", error))

    started_at = time.monotonic()
    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            kind, payload = result_queue.get(timeout=progress_interval)
        except queue.Empty:
            elapsed = time.monotonic() - started_at
            print(f"    Waiting for teacher response... {elapsed:.0f}s", flush=True)
            continue
        if kind == "error":
            raise payload
        return response_record(payload)


def parse_teacher_output(
    call: dict[str, Any],
) -> tuple[str | None, int | None, str | None]:
    content = call["content"].strip()
    match = OUTPUT_RE.fullmatch(content)
    if not match:
        return None, None, "invalid_output_format"

    reasoning = match.group("reasoning").strip()
    if not reasoning:
        return None, None, "missing_reasoning"
    return reasoning, int(match.group("score")), None


def safe_filename_part(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text or fallback


def output_path(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    tasks = {str(row.get("task", "")).strip() for row in rows}
    aspects = {str(row.get("aspect", "")).strip() for row in rows}
    if len(tasks) != 1 or len(aspects) != 1:
        raise ValueError("All input rows must have the same task and aspect.")
    task = safe_filename_part(next(iter(tasks)), "task")
    aspect = safe_filename_part(next(iter(aspects)), "aspect")
    return output_dir / f"{task}_{aspect}_{len(rows)}_distill.jsonl"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def scan_existing_output(
    path: Path, input_path: Path, input_ids: set[str]
) -> tuple[set[tuple[str, str]], Counter[str], int]:
    used_pairs: set[tuple[str, str]] = set()
    model_counts: Counter[str] = Counter()
    line_count = 0
    if not path.is_file():
        return used_pairs, model_counts, line_count

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error.msg}. "
                    "Do not resume until the incomplete line is repaired."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Record at {path}:{line_number} is not an object.")

            source_file = str(record.get("source_file", "")).strip()
            if source_file and Path(source_file).expanduser().resolve() != input_path:
                raise ValueError(
                    f"Record at {path}:{line_number} belongs to another input: "
                    f"{source_file}"
                )
            if record.get("record_type") != "distillation":
                continue

            sample_id = str(record.get("id", "")).strip()
            teacher_model = str(record.get("teacher_model", "")).strip()
            if not sample_id or not teacher_model:
                raise ValueError(
                    f"Distillation record at {path}:{line_number} lacks id or teacher_model."
                )
            if sample_id not in input_ids:
                raise ValueError(
                    f"Distillation record at {path}:{line_number} has unknown id {sample_id!r}."
                )
            key = (sample_id, teacher_model)
            if key in used_pairs:
                raise ValueError(
                    f"Duplicate sample/model pair at {path}:{line_number}: {key!r}."
                )
            used_pairs.add(key)
            model_counts[teacher_model] += 1
    return used_pairs, model_counts, line_count


def build_distillation_record(
    *,
    row: dict[str, Any],
    source_index: int,
    input_path: Path,
    run_id: str,
    teacher_model: str,
    base_url: str,
    max_tokens: int,
    timeout: float,
    elapsed: float,
    teacher_call: dict[str, Any],
) -> dict[str, Any]:
    gold_label = int(row["labels"])
    reasoning, teacher_label, rejection_reason = parse_teacher_output(teacher_call)
    if rejection_reason is None and teacher_label != gold_label:
        rejection_reason = "teacher_label_mismatch"

    accepted = rejection_reason is None
    completion = None
    if reasoning is not None and teacher_label is not None:
        completion = (
            f"<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<score>{teacher_label}</score>"
        )
    messages = list(row["prompt"])
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})

    return {
        "record_type": "distillation",
        "schema_version": 1,
        "id": str(row["id"]),
        "source_index": source_index,
        "source_file": str(input_path),
        "task": row.get("task"),
        "aspect": row.get("aspect"),
        "score_sets": row.get("score_sets"),
        "gold_label": gold_label,
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
        "format_valid": rejection_reason
        not in {"invalid_output_format", "missing_reasoning"},
        "rejection_reason": rejection_reason,
        "quality_status": (
            "auto_accepted_label_match"
            if accepted
            else f"rejected_{rejection_reason}"
        ),
        "teacher_reasoning": reasoning,
        "internal_reasoning": teacher_call.get("reasoning_content") or "",
        "raw_output": teacher_call.get("content") or "",
        "completion": completion,
        "messages": messages,
        "response": {
            "response_id": teacher_call.get("response_id"),
            "model": teacher_call.get("model"),
            "finish_reason": teacher_call.get("finish_reason"),
            "usage": teacher_call.get("usage"),
        },
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    load_env(config_path)

    api_key = os.environ.get("OPENBITFUN_API_KEY", "").strip()
    base_url = os.environ.get("OPENBITFUN_BASE_URL", "").strip().rstrip("/")
    teacher_model = (args.model or os.environ.get("OPENBITFUN_MODEL", "")).strip()
    if not api_key or not base_url or not teacher_model:
        raise SystemExit(
            "OPENBITFUN_API_KEY and OPENBITFUN_BASE_URL are required; specify the "
            "teacher with --model or OPENBITFUN_MODEL."
        )

    rows = read_rows(input_path)
    distill_path = output_path(rows, output_dir)
    if args.overwrite:
        distill_path.parent.mkdir(parents=True, exist_ok=True)
        distill_path.write_text("", encoding="utf-8")

    input_ids = {str(row["id"]) for row in rows}
    used_pairs, model_counts, existing_lines = scan_existing_output(
        distill_path, input_path, input_ids
    )
    pending_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if (str(row["id"]), teacher_model) not in used_pairs
    ]
    skipped_count = len(rows) - len(pending_rows)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    decoding = {
        "n": 1,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout,
    }

    append_jsonl(
        distill_path,
        {
            "record_type": "run_start",
            "schema_version": 1,
            "run_id": run_id,
            "source_file": str(input_path),
            "started_at_utc": utc_now(),
            "teacher_model": teacher_model,
            "api_base_url": base_url,
            "input_count": len(rows),
            "existing_line_count": existing_lines,
            "existing_model_counts": dict(sorted(model_counts.items())),
            "skipped_same_model_count": skipped_count,
            "pending_count": len(pending_rows),
            "decoding": decoding,
        },
    )

    print("Teacher CoT distillation", flush=True)
    print(f"  Input: {input_path}", flush=True)
    print(f"  Teacher model: {teacher_model}", flush=True)
    print(f"  Output: {distill_path}", flush=True)
    print(f"  Input samples: {len(rows)}", flush=True)
    print(
        f"  Resume for this model: skipped={skipped_count} "
        f"pending={len(pending_rows)}",
        flush=True,
    )
    if model_counts:
        print(f"  Existing trajectories by model: {dict(model_counts)}", flush=True)
    print("  Decoding: n=1, temperature=0, top_p=1", flush=True)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=2,
    )
    run_started_at = time.monotonic()
    attempted = 0
    accepted_count = 0
    rejected_count = 0
    run_status = "completed"
    run_error = None

    try:
        for pending_index, (source_index, row) in enumerate(pending_rows, start=1):
            sample_id = str(row["id"])
            gold_label = int(row["labels"])
            sample_started_at = time.monotonic()
            print(
                f"\n[{pending_index}/{len(pending_rows)}] id={sample_id} "
                f"input={source_index + 1}/{len(rows)} gold={gold_label}",
                flush=True,
            )
            print(f"    Sending prompt to {teacher_model}...", flush=True)
            teacher_call = create_completion(
                client,
                teacher_model,
                row["prompt"],
                args.max_tokens,
                args.progress_interval,
            )
            elapsed = time.monotonic() - sample_started_at
            record = build_distillation_record(
                row=row,
                source_index=source_index,
                input_path=input_path,
                run_id=run_id,
                teacher_model=teacher_model,
                base_url=base_url,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                elapsed=elapsed,
                teacher_call=teacher_call,
            )
            append_jsonl(distill_path, record)
            attempted += 1
            if record["accepted"]:
                accepted_count += 1
                status = "ACCEPTED"
            else:
                rejected_count += 1
                status = f"REJECTED ({record['rejection_reason']})"

            usage = record["response"].get("usage") or {}
            print(
                f"    Result: teacher={record['teacher_label']} "
                f"gold={gold_label} -> {status}",
                flush=True,
            )
            print(
                f"    Time: {elapsed:.1f}s | tokens: "
                f"prompt={usage.get('prompt_tokens', '?')} "
                f"completion={usage.get('completion_tokens', '?')} "
                f"total={usage.get('total_tokens', '?')}",
                flush=True,
            )
            reasoning = record["teacher_reasoning"]
            if reasoning and args.preview_chars > 0:
                preview = " ".join(reasoning.split())
                if len(preview) > args.preview_chars:
                    preview = preview[: args.preview_chars].rstrip() + "..."
                print(f"    Reasoning: {preview}", flush=True)
            print(
                f"    This run: attempted={attempted}/{len(pending_rows)} "
                f"accepted={accepted_count} rejected={rejected_count}",
                flush=True,
            )
    except KeyboardInterrupt:
        run_status = "interrupted"
        run_error = "KeyboardInterrupt"
        raise
    except BaseException as error:
        run_status = "failed"
        run_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        append_jsonl(
            distill_path,
            {
                "record_type": "run_end",
                "schema_version": 1,
                "run_id": run_id,
                "source_file": str(input_path),
                "finished_at_utc": utc_now(),
                "teacher_model": teacher_model,
                "status": run_status,
                "error": run_error,
                "skipped_same_model_count": skipped_count,
                "attempted_count": attempted,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "remaining_for_model": len(pending_rows) - attempted,
                "elapsed_seconds": round(time.monotonic() - run_started_at, 3),
            },
        )

    print("\nDistillation run complete", flush=True)
    print(f"  Output: {distill_path}", flush=True)
    print(f"  Skipped for {teacher_model}: {skipped_count}", flush=True)
    print(
        f"  New trajectories: {attempted} "
        f"(accepted={accepted_count}, rejected={rejected_count})",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed sample/model records were preserved.", file=sys.stderr)
        raise SystemExit(130)
