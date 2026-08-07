"""Generate a complete, reproducible experiment ledger under eval_output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.condition_labels import infer_eval_condition
from evaluation.result_records import (
    CONDITION_DATA,
    CONDITION_INFERENCE,
    CONDITION_META,
    TRAINABLE_TASKS,
    collect_eval_records,
    normalize_condition,
)
from shared.metrics import criterion_title
from shared.project_io import utc_now


MISSING = "—"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return MISSING
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return MISSING


def _pct(value: Any) -> str:
    if value is None:
        return MISSING
    try:
        return f"{100 * float(value):.1f}"
    except (TypeError, ValueError):
        return MISSING


def _load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _local_rows(results_root: Path) -> list[dict[str, Any]]:
    records = collect_eval_records(results_root)
    rows = []
    for key, record in records.items():
        path = Path(str(record.get("metrics_path") or ""))
        metrics = _load(path) or {}
        condition = str(record.get("condition") or "")
        aggregate = metrics.get("aggregate") or {}
        tokens = aggregate.get("tokens") or {}
        rows.append({
            "source": "Local",
            "task": record.get("task"),
            "criterion": criterion_title(str(record.get("task") or "")),
            "condition": condition,
            "model": metrics.get("model_name") or CONDITION_META.get(condition, (condition,))[0],
            "train": CONDITION_META.get(condition, (condition,))[0],
            "prompt": CONDITION_DATA.get(condition, MISSING),
            "inference": CONDITION_INFERENCE.get(condition, MISSING),
            "seed": record.get("train_seed"),
            "samples": record.get("n_samples"),
            "accuracy": record.get("accuracy"),
            "qwk": record.get("qwk"),
            "pearson": record.get("pearson"),
            "macro_f1": record.get("macro_f1"),
            "mae": record.get("mae"),
            "format_valid": record.get("format_valid_rate"),
            "avg_output_tokens": record.get("avg_output_tokens") or tokens.get("avg_output_tokens"),
            "avg_reasoning_tokens": record.get("avg_reasoning_tokens") or tokens.get("avg_reasoning_tokens"),
            "max_tokens": metrics.get("max_tokens"),
            "status": "complete",
            "path": str(path),
        })
    return rows


def _api_model_label(metrics: dict[str, Any]) -> str:
    """Return a readable model label that distinguishes long-CoT API runs."""
    model = str(metrics.get("model_name") or metrics.get("model_slug") or MISSING)
    slug = str(metrics.get("model_slug") or "").lower()
    max_tokens = metrics.get("max_tokens")
    if max_tokens is None:
        max_tokens = (metrics.get("full_config") or {}).get("max_tokens")
    if "cot2048" in slug or (isinstance(max_tokens, (int, float)) and max_tokens >= 2048):
        return f"{model} (CoT 2048)"
    return model


def _api_rows(api_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(api_root.rglob("metrics.json")):
        metrics = _load(path)
        if not metrics or metrics.get("backend") != "openai-compatible-api":
            continue
        aggregate = metrics.get("aggregate") or {}
        rows.append({
            "source": "API",
            "task": metrics.get("task"),
            "criterion": criterion_title(str(metrics.get("task") or "")),
            "condition": "API",
            "model": _api_model_label(metrics),
            "train": "API baseline",
            "prompt": metrics.get("supervision_mode") or metrics.get("mode"),
            "inference": metrics.get("decoding") or "Greedy",
            "seed": MISSING,
            "samples": aggregate.get("samples"),
            "accuracy": metrics.get("test_accuracy", aggregate.get("accuracy")),
            "qwk": metrics.get("test_qwk", aggregate.get("qwk")),
            "pearson": metrics.get("test_pearson", aggregate.get("pearson")),
            "macro_f1": metrics.get("test_macro_f1", aggregate.get("macro_f1")),
            "mae": metrics.get("test_mae", aggregate.get("mae")),
            "format_valid": metrics.get("format_valid_rate", aggregate.get("format_valid_rate")),
            "avg_output_tokens": metrics.get("avg_output_tokens"),
            "avg_reasoning_tokens": metrics.get("avg_reasoning_tokens"),
            "max_tokens": metrics.get("max_tokens") or (metrics.get("full_config") or {}).get("max_tokens"),
            "status": "complete" if metrics.get("complete_dataset", True) else "incomplete",
            "path": str(path),
        })
    return rows


def _summary_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["source"]), str(row["task"])), []).append(row)
    lines = [
        "| 来源 | 任务 | 条件/运行数 | 平均 Accuracy (%) | 平均 QWK | 平均 Pearson | 平均 Macro-F1 | 平均 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (source, task), items in sorted(grouped.items()):
        def mean(field: str) -> str:
            values = [float(item[field]) for item in items if item.get(field) is not None]
            return _num(sum(values) / len(values)) if values else MISSING
        lines.append(
            f"| {source} | {criterion_title(task)} | {len(items)} | {_pct(sum(float(item['accuracy']) for item in items if item.get('accuracy') is not None) / max(1, sum(item.get('accuracy') is not None for item in items)) if any(item.get('accuracy') is not None for item in items) else None)} | {mean('qwk')} | {mean('pearson')} | {mean('macro_f1')} | {mean('mae')} |"
        )
    return "\n".join(lines)


def _ledger_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "Id", "来源", "任务", "条件", "模型", "训练/来源", "Prompt", "Inf.",
        "Seed", "N", "Accuracy (%)", "QWK", "Pearson", "Macro-F1", "MAE",
        "Valid (%)", "Avg output tok.", "Avg reasoning tok.", "Max tokens", "Status", "Metrics path",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 9 + ["---:"] * 10 + ["---"] * 2) + " |",
    ]
    previous_source = None
    for index, row in enumerate(rows, start=1):
        if row["source"] != previous_source:
            lines.append("| **" + str(row["source"]) + " experiments** | " + " | ".join([""] * (len(headers) - 1)) + " |")
            previous_source = row["source"]
        cells = [
            str(index), str(row["source"]), criterion_title(str(row["task"] or "")), str(row.get("condition") or MISSING),
            str(row.get("model") or MISSING), str(row.get("train") or MISSING), str(row.get("prompt") or MISSING), str(row.get("inference") or MISSING),
            str(row.get("seed") or MISSING), str(row.get("samples") or MISSING), _pct(row.get("accuracy")), _num(row.get("qwk")), _num(row.get("pearson")), _num(row.get("macro_f1")), _num(row.get("mae")), _pct(row.get("format_valid")),
            _num(row.get("avg_output_tokens"), 1), _num(row.get("avg_reasoning_tokens"), 1), str(row.get("max_tokens") or MISSING), str(row.get("status") or MISSING), str(row.get("path") or MISSING),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_experiment_report(eval_root: Path) -> str:
    results_root = eval_root / "results"
    api_root = eval_root / "api_results"
    rows = _local_rows(results_root) + _api_rows(api_root)
    rows.sort(key=lambda row: (row["source"], str(row.get("task")), str(row.get("condition")), str(row.get("model")), str(row.get("seed")), str(row.get("path"))))
    local_count = sum(row["source"] == "Local" for row in rows)
    api_count = sum(row["source"] == "API" for row in rows)
    return f"""# Complete Experiment Results Ledger

> Generated at {utc_now()}. This ledger includes every discovered complete `metrics.json` result under `eval_output/results` and `eval_output/api_results`.
> Local records: {local_count}; API records: {api_count}; total: {len(rows)}.
> `CoT 2048` runs are stored in the shared API directory and remain distinguishable through their model slug/configuration.

## Coverage summary

{_summary_table(rows)}

## Complete experiment records

The table below is the authoritative row-level experiment ledger. Each row corresponds to one persisted result record; no best-result filtering is applied. `—` means unavailable or not applicable.

{_ledger_table(rows)}
"""


def update_experiment_report(eval_root: Path) -> Path:
    eval_root.mkdir(parents=True, exist_ok=True)
    path = eval_root / "experiment_results.md"
    path.write_text(render_experiment_report(eval_root), encoding="utf-8")
    return path
