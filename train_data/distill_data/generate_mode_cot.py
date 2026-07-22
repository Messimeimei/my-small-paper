#!/usr/bin/env python3
"""对清洗数据进行 CoT 蒸馏，并按样本 ID 与教师模型安全续跑。"""

from __future__ import annotations

import argparse
import email.utils
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

from openai import OpenAI, RateLimitError


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
    r"<score>\s*(?P<score>-?\d+)\s*</score>\s*\Z",
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
        "--max-rate-limit-retries",
        type=int,
        default=-1,
        help=(
            "Retries after HTTP 429; -1 retries indefinitely (default), 0 disables "
            "the extra retry loop. Ctrl-C still stops safely."
        ),
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=30.0,
        help="Initial seconds to wait after HTTP 429 (default: 30).",
    )
    parser.add_argument(
        "--rate-limit-max-backoff",
        type=float,
        default=300.0,
        help="Maximum seconds between HTTP 429 retries (default: 300).",
    )
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
        score_sets = row.get("score_sets")
        if (
            not isinstance(score_sets, list)
            or not score_sets
            or any(isinstance(score, bool) or not isinstance(score, int) for score in score_sets)
            or len(set(score_sets)) != len(score_sets)
        ):
            raise ValueError(
                f"Row {index} has no valid unique integer score_sets: {score_sets!r}"
            )
        gold_label = row.get("labels")
        if isinstance(gold_label, bool) or not isinstance(gold_label, int):
            raise ValueError(f"Row {index} has a non-integer gold label: {gold_label!r}")
        if gold_label not in score_sets:
            raise ValueError(
                f"Row {index} gold label {gold_label!r} is not in score_sets "
                f"{score_sets!r}"
            )
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


def retry_after_seconds(error: RateLimitError) -> float | None:
    """Read Retry-After as either seconds or an HTTP date when the API provides it."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def wait_with_progress(seconds: float, progress_interval: float) -> None:
    deadline = time.monotonic() + seconds
    report_interval = progress_interval if progress_interval > 0 else seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, report_interval))
        remaining = deadline - time.monotonic()
        if remaining > 0 and progress_interval > 0:
            print(f"    Rate-limit wait remaining: {remaining:.0f}s", flush=True)


def create_completion_with_rate_limit_retry(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    progress_interval: float,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
) -> dict[str, Any]:
    retry_count = 0
    while True:
        try:
            return create_completion(
                client, model, messages, max_tokens, progress_interval
            )
        except RateLimitError as error:
            if max_retries >= 0 and retry_count >= max_retries:
                raise
            retry_count += 1
            exponential_delay = min(
                max_backoff, initial_backoff * (2 ** min(retry_count - 1, 10))
            )
            delay = retry_after_seconds(error)
            if delay is None:
                delay = exponential_delay
            else:
                delay = min(max_backoff, max(delay, initial_backoff))
            retry_limit = "unlimited" if max_retries < 0 else str(max_retries)
            print(
                f"    HTTP 429 rate limit; waiting {delay:.0f}s before retry "
                f"{retry_count} (limit: {retry_limit}).",
                flush=True,
            )
            wait_with_progress(delay, progress_interval)


def parse_teacher_output(
    call: dict[str, Any], allowed_scores: set[int]
) -> tuple[str | None, int | None, str | None]:
    content = call["content"].strip()
    match = OUTPUT_RE.fullmatch(content)
    if not match:
        return None, None, "invalid_output_format"

    reasoning = match.group("reasoning").strip()
    if not reasoning:
        return None, None, "missing_reasoning"
    teacher_label = int(match.group("score"))
    if teacher_label not in allowed_scores:
        return reasoning, teacher_label, "teacher_label_out_of_score_set"
    return reasoning, teacher_label, None


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
    reasoning, teacher_label, rejection_reason = parse_teacher_output(
        teacher_call, set(row["score_sets"])
    )
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
    if args.max_rate_limit_retries < -1:
        raise SystemExit("--max-rate-limit-retries must be -1 or greater.")
    if args.rate_limit_backoff <= 0:
        raise SystemExit("--rate-limit-backoff must be greater than 0.")
    if args.rate_limit_max_backoff < args.rate_limit_backoff:
        raise SystemExit(
            "--rate-limit-max-backoff must be at least --rate-limit-backoff."
        )
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
        "max_rate_limit_retries": args.max_rate_limit_retries,
        "rate_limit_backoff_seconds": args.rate_limit_backoff,
        "rate_limit_max_backoff_seconds": args.rate_limit_max_backoff,
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
