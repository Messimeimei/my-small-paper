"""Render aggregate evaluation reports in the TRACT paper-table style."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable

from evaluation.reporting.comparison import render_comparison_table
from evaluation.reporting.tables import (
    PaperMetricColumn,
    PaperMetricRow,
    PaperResultRow,
    render_grouped_metric_table,
    render_result_matrix,
)
from evaluation.reporting.records import (
    CONDITION_DATA,
    CONDITION_INFERENCE,
    CONDITION_META,
    EVAL_CONDITIONS,
    ORDINAL_TASKS,
    RECORD_CACHE_NAME,
    TRAINABLE_TASKS,
    Records,
    collect_eval_records,
    mean_of,
    metric_specs,
    present_values,
    primary_metric,
    records_for,
    seed_sort_key,
    utc_now,
)
from utils.metrics import criterion_title


METRIC_SECTIONS = (
    ("qwk", "QWK", False, True),
    ("accuracy", "Accuracy (%)", True, True),
    ("pearson", "Pearson", False, True),
    ("macro_f1", "Macro-F1", False, True),
    ("mae", "MAE", False, False),
)

CONDITION_GROUPS = (
    ("Baselines", ("B-L", "B-C", "SciRM-L", "SciRM-C")),
    ("Standard fine-tuning", ("LL", "LC", "CL", "CC")),
    ("Paper Align", ("PAL", "PAC")),
    ("Self-correct Align", ("SCAL", "SCAC")),
    (
        "Regression-aware methods",
        ("LL-R", "RAFT-G", "RAFT-R", "CC-R", "COT-RAFT-G", "COT-RAFT-R"),
    ),
)

CONDITION_MODEL = {
    "B-L": "Qwen3-4B Base",
    "B-C": "Qwen3-4B Base",
    "SciRM-L": "SciRM-7B RL",
    "SciRM-C": "SciRM-7B RL",
}

CONDITION_TRAIN = {
    "B-L": "Base",
    "B-C": "Base",
    "SciRM-L": "RL",
    "SciRM-C": "RL",
    "LL": "Label-only SFT",
    "LC": "Label-only SFT",
    "CL": "CoT SFT",
    "CC": "CoT SFT",
    "PAL": "Paper Align SFT",
    "PAC": "Paper Align SFT",
    "SCAL": "Self-correct Align SFT",
    "SCAC": "Self-correct Align SFT",
    "LL-R": "Label-only CE",
    "RAFT-G": "RAFT without CoT",
    "RAFT-R": "RAFT without CoT",
    "CC-R": "CoT CE",
    "COT-RAFT-G": "CoT-RAFT",
    "COT-RAFT-R": "CoT-RAFT",
}

CONFIG_HEADERS = ("Id", "Model", "CoT", "Train", "Prompt", "Inf.", "Seed")
TASK_LABELS = {task: criterion_title(task) for task in TRAINABLE_TASKS}


def _condition_group(condition: str) -> str:
    for group, conditions in CONDITION_GROUPS:
        if condition in conditions:
            return group
    return "Other configurations"


def _condition_model(condition: str) -> str:
    return CONDITION_MODEL.get(condition, "Qwen3-4B")


def _format_value(value: float, *, percent: bool) -> str:
    return f"{100 * value:.1f}" if percent else f"{value:.3f}"


def _format_stats(values: list[float], *, percent: bool) -> str:
    if not values:
        return "\u2014"
    mean = statistics.fmean(values)
    rendered = _format_value(mean, percent=percent)
    if len(values) > 1:
        std = statistics.stdev(values)
        rendered += f" +/- {_format_value(std, percent=percent)}"
    return rendered


def _task_supports_metric(task: str, metric: str) -> bool:
    return any(spec_metric == metric for spec_metric, _, _, _ in metric_specs(task))


def _present_conditions(records: Records) -> list[str]:
    return [
        condition
        for condition in EVAL_CONDITIONS
        if any(records_for(records, task, condition) for task in TRAINABLE_TASKS)
    ]


def _paper_rows(
    records: Records,
    *,
    metric_for_task: Callable[[str], str | None],
    percent: bool,
) -> list[PaperResultRow]:
    rows = []
    for condition in _present_conditions(records):
        values: dict[str, float | None] = {}
        display_values: dict[str, str] = {}
        seeds: set[str] = set()
        for task in TRAINABLE_TASKS:
            metric = metric_for_task(task)
            grouped = records_for(records, task, condition)
            seeds.update(str(row["train_seed"]) for row in grouped)
            metric_values = present_values(grouped, metric) if metric else []
            values[task] = mean_of(metric_values)
            display_values[task] = _format_stats(metric_values, percent=percent)
        ordered_seeds = sorted(seeds, key=seed_sort_key)
        prompt = CONDITION_DATA[condition]
        rows.append(
            PaperResultRow(
                group=_condition_group(condition),
                config=(
                    condition,
                    _condition_model(condition),
                    "\u2713" if prompt == "CoT" else "\u2717",
                    CONDITION_TRAIN[condition],
                    prompt,
                    CONDITION_INFERENCE[condition],
                    ", ".join(ordered_seeds) or "\u2014",
                ),
                values=values,
                display_values=display_values,
            )
        )
    return rows


def render_metric_table(
    records: Records,
    *,
    metric: str,
    percent: bool,
    higher_is_better: bool,
) -> str:
    applicable_tasks = [
        task for task in TRAINABLE_TASKS if _task_supports_metric(task, metric)
    ]
    rows = _paper_rows(
        records,
        metric_for_task=lambda task: metric if task in applicable_tasks else None,
        percent=percent,
    )
    return render_result_matrix(
        rows,
        config_headers=CONFIG_HEADERS,
        task_keys=TRAINABLE_TASKS,
        task_labels=TASK_LABELS,
        applicable_tasks=applicable_tasks,
        value_formatter=lambda value: _format_value(value, percent=percent),
        higher_is_better=higher_is_better,
    )


def render_primary_metric_table(records: Records) -> str:
    rows = _paper_rows(
        records,
        metric_for_task=primary_metric,
        percent=False,
    )
    return render_result_matrix(
        rows,
        config_headers=CONFIG_HEADERS,
        task_keys=TRAINABLE_TASKS,
        task_labels=TASK_LABELS,
        applicable_tasks=TRAINABLE_TASKS,
        value_formatter=lambda value: _format_value(value, percent=False),
    )


def _sample_size_note(records: Records) -> str:
    entries = []
    for task in TRAINABLE_TASKS:
        counts = sorted(
            {
                int(record["n_samples"])
                for (record_task, _, _), record in records.items()
                if record_task == task and record.get("n_samples") is not None
            }
        )
        rendered = "/".join(map(str, counts)) if counts else "\u2014"
        entries.append(f"{criterion_title(task)}={rendered}")
    return "; ".join(entries)


def render_evaluation_analysis(records: Records) -> str:
    present_conditions = _present_conditions(records)
    condition_text = ", ".join(present_conditions) if present_conditions else "none"
    metric_sections = []
    for section_number, (metric, display, percent, higher_is_better) in enumerate(
        METRIC_SECTIONS,
        start=3,
    ):
        direction = "higher is better" if higher_is_better else "lower is better"
        applicable = "four ordinal tasks" if metric in {"qwk", "mae"} else "all seven tasks"
        metric_sections.append(
            f"## {section_number}. {display}\n\n"
            f"{display} applies to {applicable}; {direction}.\n\n"
            f"{render_metric_table(records, metric=metric, percent=percent, higher_is_better=higher_is_better)}\n\n"
            f"*Table {section_number - 1}. {display} by training/inference configuration and task.*"
        )

    return f"""# Qwen3-4B and SciRM-7B evaluation results

> Generated at {utc_now()}; {len(records)} deduplicated task/configuration/seed records are included.
> This file is rebuilt by `scripts/evaluate.py` after evaluation.

## 1. Reporting protocol

Included configurations: {condition_text}.

Tables follow the main-result layout of TRACT: training and inference configurations are listed on the left, tasks are expanded across columns, and the strict macro average is reported last. **Bold** marks the best result and <u>underline</u> marks the second-best distinct result in each column. Ties share a rank. MAE is ranked in ascending order; all other metrics are ranked in descending order.

`CoT` indicates whether the evaluation prompt requests an explicit rationale; it does not indicate hidden/internal model reasoning. Multi-seed cells report `mean +/- sample standard deviation`. Variances remain available in `evaluation_analysis_records.json`. An average is shown only when a configuration covers every task to which that metric applies; `\u2014` means not applicable or unavailable.

Task sample counts: {_sample_size_note(records)}.

## 2. Main results by primary metric

The primary metric is QWK for the four ordinal review-utility tasks and Macro-F1 for the three binary writing-quality tasks.

{render_primary_metric_table(records)}

*Table 1. Primary-metric results. Average requires complete coverage of all seven tasks.*

{chr(10).join(metric_sections)}

## 8. Cross-task method averages

{_render_tract_method_average_table(records)}

## 9. Rebuild report

```bash
python scripts/evaluate.py --refresh-analysis-only --output_path outputs/evaluations
```
"""


def refresh_comparison_tables(output_root: Path) -> list[Path]:
    """Rebuild the seven compact task tables from their persisted JSON rows."""

    written = []
    for json_path in sorted(output_root.glob("*/comparison_table.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = list((payload.get("rows") or {}).values())
        if not rows:
            continue
        markdown_path = json_path.with_suffix(".md")
        markdown_path.write_text(render_comparison_table(rows), encoding="utf-8")
        written.append(markdown_path)
    return written


METHOD_AVERAGE_METRICS = (
    ("qwk", "Average QWK", False, True),
    ("accuracy", "Average Accuracy (%)", True, True),
    ("pearson", "Average Pearson", False, True),
    ("macro_f1", "Average Macro-F1", False, True),
    ("mae", "Average MAE", False, False),
)

def method_metric_task_means(records: Records, condition: str, metric: str) -> list[float]:
    applicable_tasks = ORDINAL_TASKS if metric in {"qwk", "mae"} else TRAINABLE_TASKS
    return [
        value
        for task in applicable_tasks
        if (value := mean_of(present_values(records_for(records, task, condition), metric))) is not None
    ]

def _format_average(values: list[float], *, percent: bool) -> str:
    if not values:
        return "—"
    value = statistics.fmean(values)
    rendered = f"{100 * value:.1f}" if percent else f"{value:.3f}"
    return f"{rendered} (n={len(values)})"

def _render_tract_method_average_table(records: Records) -> str:
    rows: list[PaperMetricRow] = []
    for condition in _present_conditions(records):
        prompt = CONDITION_DATA[condition]
        covered = sum(bool(records_for(records, task, condition)) for task in TRAINABLE_TASKS)
        values = {}
        display_values = {}
        for metric, _display, percent, _higher in METHOD_AVERAGE_METRICS:
            metric_values = method_metric_task_means(records, condition, metric)
            values[metric] = statistics.fmean(metric_values) if metric_values else None
            display_values[metric] = _format_average(metric_values, percent=percent)
        rows.append(PaperMetricRow(
            group=_condition_group(condition),
            config=(condition, _condition_model(condition), "✓" if prompt == "CoT" else "✗", CONDITION_TRAIN[condition], prompt, CONDITION_INFERENCE[condition], f"{covered}/{len(TRAINABLE_TASKS)}"),
            values=values,
            display_values=display_values,
        ))
    columns = [
        PaperMetricColumn(display, metric, (lambda value, percent=percent: f"{100 * value:.1f}" if percent else f"{value:.3f}"), higher)
        for metric, display, percent, higher in METHOD_AVERAGE_METRICS
    ]
    return render_grouped_metric_table(rows, config_headers=("Id", "Model", "CoT", "Train", "Prompt", "Inf.", "Coverage"), metric_columns=columns) + "\n"

def render_method_average_table(records: Records) -> str:
    """Backward-compatible compact helper; the report itself uses TRACT formatting."""
    headers = ("条件", "训练方式", "推理方式", "测试数据", "任务覆盖", "平均 QWK", "平均 Accuracy (%)", "平均 Pearson", "平均 Macro-F1", "平均 MAE")
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition in _present_conditions(records):
        covered = sum(bool(records_for(records, task, condition)) for task in TRAINABLE_TASKS)
        cells = [condition, CONDITION_META[condition][0], CONDITION_INFERENCE[condition], CONDITION_DATA[condition], f"{covered}/{len(TRAINABLE_TASKS)}"]
        cells.extend(_format_average(method_metric_task_means(records, condition, metric), percent=percent) for metric, _display, percent, _higher in (("qwk", "", False, True), ("accuracy", "", True, True), ("pearson", "", False, True), ("macro_f1", "", False, True), ("mae", "", False, False)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def update_evaluation_analysis(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    records = collect_eval_records(output_root)
    ordered_records = [
        records[key]
        for key in sorted(
            records,
            key=lambda key: (
                TRAINABLE_TASKS.index(key[0]),
                EVAL_CONDITIONS.index(key[1]),
                seed_sort_key(key[2]),
            ),
        )
    ]
    cache_path = output_root / RECORD_CACHE_NAME
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "updated_at_utc": utc_now(),
                "records": ordered_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    analysis_path = output_root / "evaluation_analysis.md"
    analysis_path.write_text(render_evaluation_analysis(records), encoding="utf-8")
    refresh_comparison_tables(output_root)
    return analysis_path
