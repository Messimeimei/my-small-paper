"""Build an API-only report and a read-only comparison with local results."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.result_records import (
    CONDITION_DATA,
    CONDITION_META,
    TRAINABLE_TASKS,
    collect_eval_records,
    metric_specs,
    primary_metric,
    records_for,
)
from shared.metrics import criterion_title
from shared.project_io import utc_now


def _load_api_records(api_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(api_root.rglob("metrics.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("backend") != "openai-compatible-api" or not row.get(
            "complete_dataset", False
        ):
            continue
        aggregate = row.get("aggregate") or {}
        records.append(
            {
                "task": row.get("task"),
                "mode": row.get("supervision_mode"),
                "model": row.get("model_name"),
                "n_samples": aggregate.get("samples"),
                "accuracy": aggregate.get("accuracy"),
                "macro_f1": aggregate.get("macro_f1"),
                "qwk": aggregate.get("qwk"),
                "mae": aggregate.get("mae"),
                "pearson": aggregate.get("pearson"),
                "format_valid_rate": aggregate.get("format_valid_rate"),
                "metrics_path": str(path),
            }
        )
    return records


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{100 * number:.1f}" if percent else f"{number:.3f}"


def _task_table(
    task: str,
    api_records: list[dict[str, Any]],
    local_records: dict[tuple[str, str, str], dict[str, Any]],
) -> str:
    specs = metric_specs(task)
    lines = [
        "| 来源 | 模型/条件 | Prompt | N | "
        + " | ".join(display for _, display, _, _ in specs)
        + " |",
        "| --- | --- | --- | ---: | " + " | ".join("---:" for _ in specs) + " |",
    ]
    for condition in sorted({key[1] for key in local_records if key[0] == task}):
        grouped = records_for(local_records, task, condition)
        if not grouped:
            continue
        values = []
        for metric, _, percent, _ in specs:
            present = [float(row[metric]) for row in grouped if row.get(metric) is not None]
            values.append(_fmt(statistics.fmean(present) if present else None, percent=percent))
        counts = sorted({str(row.get("n_samples") or "—") for row in grouped})
        lines.append(
            f"| Local | {condition}: {CONDITION_META[condition][0]} | "
            f"{CONDITION_DATA[condition]} | {', '.join(counts)} | "
            + " | ".join(values)
            + " |"
        )
    for row in sorted(
        (record for record in api_records if record["task"] == task),
        key=lambda record: (str(record["model"]), str(record["mode"])),
    ):
        values = [
            _fmt(row.get(metric), percent=percent)
            for metric, _, percent, _ in specs
        ]
        lines.append(
            f"| API | {row['model']} | {row['mode']} | {row['n_samples']} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines)


def _api_average_table(api_records: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in api_records:
        grouped[(str(row["model"]), str(row["mode"]))].append(row)
    lines = [
        "| 模型 | Prompt | 任务覆盖 | 平均主指标 | 平均 Accuracy (%) | 平均 Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (model, mode), rows in sorted(grouped.items()):
        primary_values = [
            float(row[primary_metric(str(row["task"]))])
            for row in rows
            if row.get(primary_metric(str(row["task"]))) is not None
        ]
        accuracies = [float(row["accuracy"]) for row in rows if row.get("accuracy") is not None]
        macro_f1 = [float(row["macro_f1"]) for row in rows if row.get("macro_f1") is not None]
        lines.append(
            f"| {model} | {mode} | {len(rows)}/{len(TRAINABLE_TASKS)} | "
            f"{_fmt(statistics.fmean(primary_values) if primary_values else None)} | "
            f"{_fmt(statistics.fmean(accuracies) if accuracies else None, percent=True)} | "
            f"{_fmt(statistics.fmean(macro_f1) if macro_f1 else None)} |"
        )
    return "\n".join(lines)


def update_api_reports(api_root: Path, local_root: Path) -> tuple[Path, Path]:
    api_root.mkdir(parents=True, exist_ok=True)
    api_records = _load_api_records(api_root)
    local_records = collect_eval_records(local_root)
    api_report = api_root / "api_baseline_analysis.md"
    comparison_report = api_root / "comparison_with_existing.md"
    api_report.write_text(
        "# 固定版本 API 基线评测\n\n"
        f"> 自动生成于 {utc_now()}；仅纳入完整测试集结果，当前 {len(api_records)} 条。\n\n"
        "## 跨任务汇总\n\n"
        + _api_average_table(api_records)
        + "\n",
        encoding="utf-8",
    )
    sections = []
    for task in TRAINABLE_TASKS:
        sections.append(
            f"## {criterion_title(task)}\n\n"
            + _task_table(task, api_records, local_records)
        )
    comparison_report.write_text(
        "# 固定版本 API 基线与已有结果对照\n\n"
        f"> 自动生成于 {utc_now()}。现有 `eval_output/results` 只读，未被修改。\n\n"
        + "\n\n".join(sections)
        + "\n",
        encoding="utf-8",
    )
    return api_report, comparison_report
