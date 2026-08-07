"""Build an API-only report and a read-only comparison with local results."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.experiment_report import update_experiment_report
from evaluation.paper_table import PaperMetricColumn, PaperMetricRow, PaperResultRow, render_grouped_metric_table, render_result_matrix
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


def _display_model(model: Any, label_suffix: str) -> str:
    return f"{model}{label_suffix}"


def _result_label_suffix(row: dict[str, Any]) -> str:
    """Disambiguate long-CoT API runs stored beside regular API runs."""
    slug = str(row.get("model_slug") or "").lower()
    max_tokens = row.get("max_tokens")
    if "cot2048" in slug or (isinstance(max_tokens, (int, float)) and max_tokens >= 2048):
        return " (CoT 2048)"
    return ""


def _load_api_records(
    api_root: Path,
    *,
    label_suffix: str = "",
) -> list[dict[str, Any]]:
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
                "model": _display_model(row.get("model_name"), label_suffix + _result_label_suffix(row)),
                "n_samples": aggregate.get("samples"),
                "accuracy": aggregate.get("accuracy"),
                "macro_f1": aggregate.get("macro_f1"),
                "qwk": aggregate.get("qwk"),
                "mae": aggregate.get("mae"),
                "pearson": aggregate.get("pearson"),
                "format_valid_rate": aggregate.get("format_valid_rate"),
                "max_tokens": row.get("max_tokens") or (row.get("full_config") or {}).get("max_tokens"),
                "model_slug": row.get("model_slug"),
                "metrics_path": str(path),
            }
        )
    return records


def _load_incomplete_run_statuses(
    api_root: Path,
    *,
    label_suffix: str = "",
) -> list[dict[str, Any]]:
    statuses = []
    for config_path in sorted(api_root.rglob("resolved_config.json")):
        if "#limit_" in config_path.parent.name:
            continue
        try:
            resolved = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected = resolved.get("dataset_samples")
        request_hash = resolved.get("request_hash")
        if not isinstance(expected, int) or expected < 1 or not request_hash:
            continue

        completed_ids: set[str] = set()
        last_error_type = None
        response_path = config_path.with_name("api_responses.jsonl")
        if response_path.is_file():
            try:
                with response_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("request_hash") != request_hash:
                            continue
                        if (
                            row.get("record_type") == "response"
                            and row.get("response_model_match", True)
                            and row.get("sample_id")
                        ):
                            completed_ids.add(str(row["sample_id"]))
                            last_error_type = None
                        elif row.get("record_type") == "error":
                            last_error_type = str(row.get("error_type") or "Error")
            except (OSError, json.JSONDecodeError):
                continue

        metrics_path = config_path.with_name("metrics.json")
        metrics = None
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if isinstance(metrics, dict) and metrics.get("complete_dataset", False):
            continue

        completed = len(completed_ids)
        if completed >= expected:
            state = "响应完成，待汇总"
        elif last_error_type:
            state = f"中断（{last_error_type}）"
        else:
            state = "进行中或待续跑"
        statuses.append(
            {
                "model": _display_model(resolved.get("model_name"), label_suffix + _result_label_suffix(resolved)),
                "mode": resolved.get("mode"),
                "task": resolved.get("task"),
                "completed": completed,
                "expected": expected,
                "state": state,
            }
        )
    return statuses


def _incomplete_run_table(statuses: list[dict[str, Any]]) -> str:
    if not statuses:
        return "当前没有已开始但未完成的 API 评测。"
    lines = [
        "| 模型 | Prompt | 当前任务 | 有效响应 | 状态 |",
        "| --- | --- | --- | ---: | --- |",
        "| **Incomplete runs** |  |  |  |  |",
    ]
    for row in sorted(
        statuses,
        key=lambda item: (
            str(item["model"]),
            str(item["mode"]),
            str(item["task"]),
        ),
    ):
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['task']} | "
            f"{row['completed']}/{row['expected']} | {row['state']} |"
        )
    return "\n".join(lines)


def _fmt(value: Any, *, percent: bool = False, best: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value)
    rendered = f"{100 * number:.1f}" if percent else f"{number:.3f}"
    return f"<u>{rendered}</u>" if best else rendered


def _is_best(value: Any, best: float | None) -> bool:
    return (
        value is not None
        and best is not None
        and math.isclose(float(value), best, rel_tol=0.0, abs_tol=1e-12)
    )


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    present = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return statistics.fmean(present) if present else None


def _task_table(
    task: str,
    api_records: list[dict[str, Any]],
    local_records: dict[tuple[str, str, str], dict[str, Any]],
) -> str:
    """Render one API/local task table in the TRACT grouped layout."""
    specs = metric_specs(task)
    rows: list[PaperMetricRow] = []
    for condition in sorted({key[1] for key in local_records if key[0] == task}):
        grouped = records_for(local_records, task, condition)
        if not grouped:
            continue
        counts = sorted({str(row.get("n_samples") or "—") for row in grouped})
        rows.append(PaperMetricRow(
            group="Local results",
            config=(condition, f"{condition}: {CONDITION_META[condition][0]}", CONDITION_DATA[condition], ", ".join(counts)),
            values={metric: _mean_metric(grouped, metric) for metric, _, _, _ in specs},
        ))
    for record in sorted((row for row in api_records if row["task"] == task), key=lambda row: (str(row["model"]), str(row["mode"]))):
        rows.append(PaperMetricRow(
            group="API results",
            config=("API", str(record["model"]), str(record["mode"]), str(record.get("n_samples") or "—")),
            values={metric: record.get(metric) for metric, _, _, _ in specs},
        ))
    columns = [PaperMetricColumn(display, metric, (lambda value, percent=percent: f"{100 * value:.1f}" if percent else f"{value:.3f}"), higher) for metric, display, percent, higher in specs]
    return render_grouped_metric_table(rows, config_headers=("Source", "Model/condition", "Prompt", "N"), metric_columns=columns)


SUMMARY_METRICS = (
    ("qwk", "QWK", False, True),
    ("accuracy", "Accuracy (%)", True, True),
    ("pearson", "Pearson", False, True),
    ("macro_f1", "Macro-F1", False, True),
    ("mae", "MAE", False, False),
)


def _task_supports_metric(task: str, metric: str) -> bool:
    return any(spec_metric == metric for spec_metric, _, _, _ in metric_specs(task))


def _comparison_summary_table(
    api_records: list[dict[str, Any]],
    local_records: dict[tuple[str, str, str], dict[str, Any]],
    *, metric: str, percent: bool, higher_is_better: bool, expected_api_rows: set[tuple[str, str]] | None = None,
) -> str:
    summary: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for condition in sorted({key[1] for key in local_records}):
        key = ("Local", f"{condition}: {CONDITION_META[condition][0]}", CONDITION_DATA[condition])
        summary[key] = {task: (_mean_metric(records_for(local_records, task, condition), metric) if _task_supports_metric(task, metric) else None) for task in TRAINABLE_TASKS}
    for row in api_records:
        key = ("API", str(row["model"]), str(row["mode"]))
        summary.setdefault(key, {})[str(row["task"])] = row.get(metric) if _task_supports_metric(str(row["task"]), metric) else None
    for model, mode in expected_api_rows or set():
        summary.setdefault(("API", model, mode), {})
    applicable = [task for task in TRAINABLE_TASKS if _task_supports_metric(task, metric)]
    averages = {key: (statistics.fmean([float(values[task]) for task in applicable if values.get(task) is not None]) if all(values.get(task) is not None for task in applicable) else None) for key, values in summary.items()}
    rows: list[PaperResultRow] = []
    for (source, model, mode), values in sorted(summary.items(), key=lambda item: (item[0][0] != "Local", item[0][1], item[0][2])):
        rows.append(PaperResultRow(group="Local results" if source == "Local" else "API results", config=(source, model, mode), values={**{task: values.get(task) for task in TRAINABLE_TASKS}, "Average": averages[(source, model, mode)]}))
    labels = {task: criterion_title(task) for task in TRAINABLE_TASKS}
    return render_result_matrix(rows, config_headers=("Source", "Model/condition", "Prompt"), task_keys=TRAINABLE_TASKS, task_labels=labels, applicable_tasks=applicable, value_formatter=lambda value: f"{100 * value:.1f}" if percent else f"{value:.3f}", higher_is_better=higher_is_better, average_label="Average")


def _comparison_summary_sections(
    api_records: list[dict[str, Any]],
    local_records: dict[tuple[str, str, str], dict[str, Any]],
    *,
    expected_api_rows: set[tuple[str, str]],
) -> str:
    sections = []
    for metric, display, percent, higher_is_better in SUMMARY_METRICS:
        sections.append(
            f"### {display} 汇总\n\n"
            + _comparison_summary_table(
                api_records,
                local_records,
                metric=metric,
                percent=percent,
                higher_is_better=higher_is_better,
                expected_api_rows=expected_api_rows,
            )
        )
    return "\n\n".join(sections)


def _api_average_table(api_records: list[dict[str, Any]]) -> str:
    """Render API primary metrics as the task matrix used in the TRACT paper."""
    grouped: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in api_records:
        grouped.setdefault((str(row["model"]), str(row["mode"])), {})[str(row["task"])] = row.get(primary_metric(str(row["task"])))
    rows = [
        PaperResultRow(
            group="API baselines",
            config=("API", model, "✓" if mode == "cot" else "✗", mode),
            values={task: values.get(task) for task in TRAINABLE_TASKS},
        )
        for (model, mode), values in sorted(grouped.items())
    ]
    return render_result_matrix(
        rows,
        config_headers=("Id", "Model", "CoT", "Prompt"),
        task_keys=TRAINABLE_TASKS,
        task_labels={task: criterion_title(task) for task in TRAINABLE_TASKS},
        applicable_tasks=TRAINABLE_TASKS,
        value_formatter=lambda value: f"{value:.3f}",
        higher_is_better=True,
    )


def update_api_reports(
    api_root: Path,
    local_root: Path,
    *,
    supplemental_api_roots: tuple[tuple[Path, str], ...] = (),
) -> tuple[Path, Path]:
    api_root.mkdir(parents=True, exist_ok=True)
    api_records = _load_api_records(api_root)
    incomplete_statuses = _load_incomplete_run_statuses(api_root)
    supplemental_records = [
        row
        for root, label_suffix in supplemental_api_roots
        for row in _load_api_records(root, label_suffix=label_suffix)
    ]
    supplemental_statuses = [
        row
        for root, label_suffix in supplemental_api_roots
        for row in _load_incomplete_run_statuses(root, label_suffix=label_suffix)
    ]
    comparison_records = api_records + supplemental_records
    comparison_statuses = incomplete_statuses + supplemental_statuses
    local_records = collect_eval_records(local_root)
    api_report = api_root / "api_baseline_analysis.md"
    comparison_report = api_root / "comparison_with_existing.md"
    api_status_section = (
        "## 未完成运行\n\n"
        + _incomplete_run_table(incomplete_statuses)
        + "\n\n"
    )
    comparison_status_section = (
        "## 未完成运行\n\n"
        + _incomplete_run_table(comparison_statuses)
        + "\n\n"
    )
    api_report.write_text(
        "# 固定版本 API 基线评测\n\n"
        f"> 自动生成于 {utc_now()}；仅纳入完整测试集结果，当前 {len(api_records)} 条。\n\n"
        + api_status_section
        + "## 跨任务汇总\n\n"
        + _api_average_table(api_records)
        + "\n",
        encoding="utf-8",
    )
    sections = []
    for task in TRAINABLE_TASKS:
        sections.append(
            f"## {criterion_title(task)}\n\n"
            + _task_table(task, comparison_records, local_records)
        )
    expected_api_rows = {
        (str(row["model"]), str(row["mode"])) for row in supplemental_statuses
    }
    summary_section = (
        "## 分指标七任务汇总\n\n"
        "> 每张表均保留七个任务列；**粗体**标出最优值，<u>下划线</u>标出次优值。"
        "MAE 越低越好，其余指标越高越好；只有覆盖全部适用任务时才计算平均；"
        "`—` 表示不适用或尚无完整结果。\n\n"
        + _comparison_summary_sections(
            comparison_records,
            local_records,
            expected_api_rows=expected_api_rows,
        )
    )
    comparison_report.write_text(
        "# 固定版本 API 基线与已有结果对照\n\n"
        f"> 自动生成于 {utc_now()}；纳入标准 API 完整结果 {len(api_records)} 条，"
        f"独立补充结果 {len(supplemental_records)} 条。"
        "现有 `eval_output/results` 只读，未被修改。\n\n"
        + comparison_status_section
        + "## 完整结果对照\n\n"
        "> 各指标列中，**粗体**标出最优值，<u>下划线</u>标出次优值；MAE 取最低值，其他指标取最高值。\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + summary_section
        + "\n",
        encoding="utf-8",
    )
    update_experiment_report(api_root.parent)
    return api_report, comparison_report
