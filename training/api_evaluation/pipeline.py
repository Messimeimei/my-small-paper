"""Run OpenAI-compatible API baselines on the existing evaluation datasets."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from evaluation.dataset_loading import load_rows
from evaluation.result_records import pearson_coefficient
from shared.metrics import classification_metrics, extract_score
from shared.project_io import resolve_path, utc_now, write_json, write_jsonl


@dataclass(frozen=True)
class ApiRun:
    model: str
    model_slug: str
    task: str
    mode: str
    dataset_file: Path
    response_model_aliases: tuple[str, ...] = ()


def load_env(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and (override or key not in os.environ):
            os.environ[key] = value.strip().strip('"').strip("'")


def load_matrix_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"API evaluation config must be an object: {path}")
    required = {"models", "tasks", "api_base_url", "api_key_env", "output_path"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"API evaluation config missing fields: {sorted(missing)}")
    if not isinstance(payload["models"], list) or not payload["models"]:
        raise ValueError("models must be a non-empty list")
    if not isinstance(payload["tasks"], list) or not payload["tasks"]:
        raise ValueError("tasks must be a non-empty list")
    return payload


def expand_runs(
    config: dict[str, Any],
    *,
    model_filters: set[str] | None = None,
    task_filters: set[str] | None = None,
    mode_filters: set[str] | None = None,
) -> list[ApiRun]:
    runs: list[ApiRun] = []
    seen_slugs: set[str] = set()
    models: list[tuple[str, str, tuple[str, ...]]] = []
    for entry in config["models"]:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("slug"):
            raise ValueError("each model requires name and slug")
        name, slug = str(entry["name"]), str(entry["slug"])
        if slug in seen_slugs:
            raise ValueError(f"duplicate model slug: {slug}")
        seen_slugs.add(slug)
        aliases = entry.get("response_model_aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias for alias in aliases
        ):
            raise ValueError("response_model_aliases must be a list of strings")
        if model_filters and name not in model_filters and slug not in model_filters:
            continue
        models.append((name, slug, tuple(aliases)))

    for task_entry in config["tasks"]:
        if not isinstance(task_entry, dict) or not task_entry.get("name"):
            raise ValueError("each task requires a name")
        task = str(task_entry["name"])
        if task_filters and task not in task_filters:
            continue
        datasets = task_entry.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError(f"task {task} requires a datasets mapping")
        for mode in ("label_only", "cot"):
            if mode_filters and mode not in mode_filters:
                continue
            if mode not in datasets:
                raise ValueError(f"task {task} is missing the {mode} dataset")
            dataset_file = resolve_path(str(datasets[mode]))
            for model, slug, aliases in models:
                runs.append(ApiRun(model, slug, task, mode, dataset_file, aliases))
    return runs


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_directory(output_root: Path, run: ApiRun, *, limit: int | None = None) -> Path:
    exp_name = (
        f"{run.task}#{run.model_slug}#api#greedy#on_{run.mode}#snapshot"
    )
    if limit is not None:
        exp_name += f"#limit_{limit}"
    return output_root / run.task / exp_name


def response_record(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    message = choice.message
    extra = getattr(message, "model_extra", None) or {}
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        reasoning = extra.get("reasoning_content")
    usage = response.usage.model_dump() if response.usage is not None else None
    return {
        "response_id": response.id,
        "request_id": getattr(response, "_request_id", None),
        "response_model": response.model,
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": choice.finish_reason,
        "content": message.content or "",
        "reasoning_content": reasoning or "",
        "usage": usage,
    }


def read_completed_responses(
    path: Path, *, request_hash: str
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if (
                row.get("record_type") == "response"
                and row.get("request_hash") == request_hash
                and row.get("response_model_match", True)
                and row.get("sample_id")
            ):
                completed[str(row["sample_id"])] = row
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _usage_value(usage: dict[str, Any] | None, *path: str) -> int | None:
    value: Any = usage
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def build_artifacts(
    *,
    run: ApiRun,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    responses: dict[str, dict[str, Any]],
    resolved: dict[str, Any],
    output_dir: Path,
    elapsed_sec: float,
    complete_dataset: bool,
) -> None:
    allowed_scores = set(score_sets)
    predictions: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    completion_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    prompt_tokens: list[int] = []

    for row in rows:
        response = responses[row["id"]]
        output = str(response.get("content") or "")
        prediction = extract_score(output, allowed_scores)
        usage = response.get("usage")
        out_tokens = _usage_value(usage, "completion_tokens")
        in_tokens = _usage_value(usage, "prompt_tokens")
        reason_tokens = _usage_value(
            usage, "completion_tokens_details", "reasoning_tokens"
        )
        if out_tokens is not None:
            completion_tokens.append(out_tokens)
        if in_tokens is not None:
            prompt_tokens.append(in_tokens)
        if reason_tokens is not None:
            reasoning_tokens.append(reason_tokens)
        metric_rows.append(
            {
                "label": row["label"],
                "prediction": prediction,
                "correct": prediction == row["label"],
            }
        )
        predictions.append(
            {
                "id": row["id"],
                "label": row["label"],
                "rollout_predictions": [prediction],
                "rollout_correct": [prediction == row["label"]],
                "mean_correct": float(prediction == row["label"]),
                "outputs": [output],
                "raw_outputs": [output],
                "reasoning_outputs": [response.get("reasoning_content") or ""],
                "response_ids": [response.get("response_id")],
                "response_models": [response.get("response_model")],
                "usage": [usage],
                "task": row.get("task"),
                "aspect": row.get("aspect"),
            }
        )

    aggregate = classification_metrics(metric_rows, score_sets)
    valid_pairs = [
        (float(row["label"]), float(row["prediction"]))
        for row in metric_rows
        if row["prediction"] is not None
    ]
    aggregate["pearson"] = pearson_coefficient(
        [label for label, _ in valid_pairs],
        [prediction for _, prediction in valid_pairs],
    )
    aggregate["elapsed_sec"] = round(elapsed_sec, 3)
    aggregate["samples_per_sec"] = round(len(rows) / max(elapsed_sec, 1e-9), 3)
    aggregate["tokens"] = {
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "avg_output_tokens": (
            sum(completion_tokens) / len(completion_tokens) if completion_tokens else None
        ),
        "avg_reasoning_tokens": (
            sum(reasoning_tokens) / len(reasoning_tokens) if reasoning_tokens else None
        ),
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_output_tokens": sum(completion_tokens) if completion_tokens else None,
        "total_reasoning_tokens": sum(reasoning_tokens) if reasoning_tokens else None,
        "samples": len(rows),
    }
    aspect = rows[0].get("aspect") or run.task
    exp_name = output_dir.name
    summary = {
        "schema_version": 1,
        "exp_name": exp_name,
        "task": run.task,
        "aspect": aspect,
        "supervision_mode": run.mode,
        "evaluation_mode": run.mode,
        "eval_condition": "API",
        "output_dir": str(output_dir),
        "backend": "openai-compatible-api",
        "model_name": run.model,
        "model_slug": run.model_slug,
        "adapter": None,
        "dataset_file": str(run.dataset_file),
        "score_sets": score_sets,
        "decoding": "greedy",
        "temp": resolved["temperature"],
        "top_p": resolved["top_p"],
        "max_tokens": resolved["max_tokens"],
        "rollout": 1,
        "finished_at_utc": utc_now(),
        "complete_dataset": complete_dataset,
        "test_accuracy": aggregate.get("accuracy"),
        "test_macro_f1": aggregate.get("macro_f1"),
        "test_mae": aggregate.get("mae"),
        "test_qwk": aggregate.get("qwk"),
        "test_pearson": aggregate.get("pearson"),
        "format_valid_rate": aggregate.get("format_valid_rate"),
        "avg_output_tokens": aggregate["tokens"]["avg_output_tokens"],
        "avg_reasoning_tokens": aggregate["tokens"]["avg_reasoning_tokens"],
        "response_models": sorted(
            {
                str(response.get("response_model"))
                for response in responses.values()
                if response.get("response_model")
            }
        ),
        "system_fingerprints": sorted(
            {
                str(response.get("system_fingerprint"))
                for response in responses.values()
                if response.get("system_fingerprint")
            }
        ),
        "wall_time_sec": round(elapsed_sec, 3),
        "full_config": resolved,
        "aggregate": aggregate,
        "rollouts": [aggregate],
    }
    write_json(output_dir / "metrics.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", predictions)


def execute_run(
    *,
    client: Any,
    run: ApiRun,
    config: dict[str, Any],
    output_root: Path,
    limit: int | None,
) -> Path:
    rows, score_sets = load_rows(run.dataset_file)
    full_count = len(rows)
    if limit is not None:
        rows = rows[:limit]
    dataset_hash = canonical_hash(
        [{"id": row["id"], "label": row["label"], "prompt": row["prompt"]} for row in rows]
    )
    extra_body = config.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be an object")
    max_tokens_by_mode = config.get("max_tokens_by_mode", {})
    if not isinstance(max_tokens_by_mode, dict):
        raise ValueError("max_tokens_by_mode must be an object")
    max_tokens = int(max_tokens_by_mode.get(run.mode, config.get("max_tokens", 512)))
    request_spec = {
        "model": run.model,
        "accepted_response_models": [run.model, *run.response_model_aliases],
        "temperature": float(config.get("temperature", 0)),
        "top_p": float(config.get("top_p", 1)),
        "max_tokens": max_tokens,
        "dataset_hash": dataset_hash,
    }
    if extra_body is not None:
        request_spec["extra_body"] = extra_body
    request_hash = canonical_hash(request_spec)
    output_dir = run_directory(output_root, run, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schema_version": 1,
        "model_name": run.model,
        "model_slug": run.model_slug,
        "task": run.task,
        "mode": run.mode,
        "dataset_file": str(run.dataset_file),
        "dataset_hash": dataset_hash,
        "dataset_samples": len(rows),
        "full_dataset_samples": full_count,
        "complete_dataset": len(rows) == full_count,
        "api_base_url": str(config["api_base_url"]).rstrip("/"),
        "temperature": request_spec["temperature"],
        "top_p": request_spec["top_p"],
        "max_tokens": request_spec["max_tokens"],
        "require_response_model_match": bool(
            config.get("require_response_model_match", True)
        ),
        "accepted_response_models": request_spec["accepted_response_models"],
        "request_hash": request_hash,
    }
    if extra_body is not None:
        resolved["extra_body"] = extra_body
    config_path = output_dir / "resolved_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise ValueError(f"refusing to mix incompatible API runs in {output_dir}")
    else:
        write_json(config_path, resolved)

    trajectory_path = output_dir / "api_responses.jsonl"
    completed = read_completed_responses(trajectory_path, request_hash=request_hash)
    pending = [row for row in rows if row["id"] not in completed]
    print(
        f"[{run.model}][{run.task}/{run.mode}] samples={len(rows)} "
        f"completed={len(rows) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    started = time.perf_counter()
    for index, row in enumerate(pending, start=1):
        sample_started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=run.model,
                messages=row["prompt"],
                n=1,
                temperature=request_spec["temperature"],
                top_p=request_spec["top_p"],
                max_tokens=request_spec["max_tokens"],
                extra_body=extra_body,
            )
            record = {
                "record_type": "response",
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "sample_id": row["id"],
                "gold_label": row["label"],
                "requested_model": run.model,
                "request_hash": request_hash,
                "elapsed_sec": round(time.perf_counter() - sample_started, 3),
                **response_record(response),
            }
            record["response_model_match"] = record.get("response_model") in {
                run.model,
                *run.response_model_aliases,
            }
            append_jsonl(trajectory_path, record)
            if (
                config.get("require_response_model_match", True)
                and not record["response_model_match"]
            ):
                raise RuntimeError(
                    f"requested model {run.model!r}, but the API reported "
                    f"{record.get('response_model')!r}; expected one of "
                    f"{request_spec['accepted_response_models']!r}; response was archived "
                    "but rejected"
                )
            completed[row["id"]] = record
        except BaseException as error:
            append_jsonl(
                trajectory_path,
                {
                    "record_type": "error",
                    "schema_version": 1,
                    "created_at_utc": utc_now(),
                    "sample_id": row["id"],
                    "requested_model": run.model,
                    "request_hash": request_hash,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        done = len(rows) - len(pending) + index
        if done % int(config.get("progress_every", 10)) == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}", flush=True)

    audited_elapsed = sum(
        float(response.get("elapsed_sec") or 0.0) for response in completed.values()
    )
    build_artifacts(
        run=run,
        rows=rows,
        score_sets=score_sets,
        responses=completed,
        resolved=resolved,
        output_dir=output_dir,
        elapsed_sec=audited_elapsed or (time.perf_counter() - started),
        complete_dataset=len(rows) == full_count,
    )
    return output_dir


def total_samples(runs: Iterable[ApiRun], *, limit: int | None = None) -> int:
    count = 0
    for run in runs:
        rows, _ = load_rows(run.dataset_file)
        count += min(len(rows), limit) if limit is not None else len(rows)
    return count
