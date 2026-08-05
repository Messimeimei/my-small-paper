"""Render the aggregate Markdown evaluation report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from shared.metrics import criterion_title
from evaluation.result_records import (
    CONDITION_DATA,
    CONDITION_INFERENCE,
    CONDITION_META,
    EVAL_CONDITIONS,
    ORDINAL_TASKS,
    RECORD_CACHE_NAME,
    TRAINABLE_TASKS,
    Records,
    best_metric_value,
    collect_eval_records,
    format_stats_cell,
    is_best_value,
    mean_of,
    metric_means,
    metric_specs,
    present_values,
    primary_metric,
    records_for,
    seed_sort_key,
    utc_now,
)


METHOD_AVERAGE_METRICS = (
    ("qwk", "平均 QWK", False),
    ("accuracy", "平均 Accuracy (%)", True),
    ("pearson", "平均 Pearson", False),
    ("macro_f1", "平均 Macro-F1", False),
    ("mae", "平均 MAE", False),
)


def render_single_task_table(records: Records, task: str) -> str:
    grouped_conditions = [
        (condition, grouped)
        for condition in EVAL_CONDITIONS
        if (grouped := records_for(records, task, condition))
    ]
    if not grouped_conditions:
        return "当前没有可用结果。\n"

    specs = metric_specs(task)
    means = {
        metric: metric_means(grouped_conditions, metric)
        for metric, _, _, _ in specs
    }
    best = {
        metric: best_metric_value(means[metric], higher_is_better=higher_is_better)
        for metric, _, _, higher_is_better in specs
    }
    headers = [
        "条件",
        "训练方式",
        "推理方式",
        "测试数据",
        "训练 seed",
        "N",
        *(display for _, display, _, _ in specs),
    ]
    separator_cells = [
        "---", "---", "---", "---", "---", "---:",
        *(["---:"] * len(specs)),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator_cells) + " |",
    ]
    for condition, grouped in grouped_conditions:
        sample_counts = sorted({
            int(record["n_samples"])
            for record in grouped
            if record.get("n_samples") is not None
        })
        sample_text = ", ".join(map(str, sample_counts)) if sample_counts else "—"
        values = [
            condition,
            CONDITION_META[condition][0],
            CONDITION_INFERENCE[condition],
            CONDITION_DATA[condition],
            ", ".join(str(record["train_seed"]) for record in grouped),
            sample_text,
        ]
        for metric, _, percent, _ in specs:
            value = means[metric][condition]
            values.append(
                format_stats_cell(
                    present_values(grouped, metric),
                    percent=percent,
                    underline=is_best_value(value, best[metric]),
                )
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def best_task_condition(
    records: Records,
    task: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    metric = primary_metric(task)
    candidates = [
        (condition, grouped, mean_of(present_values(grouped, metric)))
        for condition in EVAL_CONDITIONS
        if (grouped := records_for(records, task, condition))
    ]
    present = [candidate for candidate in candidates if candidate[2] is not None]
    if not present:
        return None
    best_mean = max(value for _, _, value in present)
    condition, grouped, _ = next(
        candidate
        for candidate in present
        if math.isclose(candidate[2], best_mean, rel_tol=0.0, abs_tol=1e-12)
    )
    return condition, grouped


def render_summary_table(records: Records) -> str:
    headers = [
        "任务",
        "主指标",
        "最优条件",
        "训练方式",
        "推理方式",
        "测试数据",
        "训练 seed",
        "主指标结果",
        "Accuracy (%)",
        "Pearson",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 7 + ["---:"] * 3) + " |",
    ]
    for task in TRAINABLE_TASKS:
        selected = best_task_condition(records, task)
        primary = primary_metric(task)
        primary_name = "QWK" if primary == "qwk" else "Macro-F1"
        if selected is None:
            lines.append(
                f"| {criterion_title(task)} | {primary_name} | — | — | — | — | — | — | — | — |"
            )
            continue
        condition, grouped = selected
        lines.append(
            "| "
            + " | ".join(
                [
                    criterion_title(task),
                    primary_name,
                    condition,
                    CONDITION_META[condition][0],
                    CONDITION_INFERENCE[condition],
                    CONDITION_DATA[condition],
                    ", ".join(str(record["train_seed"]) for record in grouped),
                    format_stats_cell(present_values(grouped, primary), underline=True),
                    format_stats_cell(present_values(grouped, "accuracy"), percent=True),
                    format_stats_cell(present_values(grouped, "pearson")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def method_metric_task_means(
    records: Records,
    condition: str,
    metric: str,
) -> list[float]:
    """Return per-task means so every task has equal weight across seeds."""
    applicable_tasks = (
        ORDINAL_TASKS if metric in {"qwk", "mae"} else TRAINABLE_TASKS
    )
    task_means = []
    for task in TRAINABLE_TASKS:
        if task not in applicable_tasks:
            continue
        grouped = records_for(records, task, condition)
        task_mean = mean_of(present_values(grouped, metric))
        if task_mean is not None:
            task_means.append(task_mean)
    return task_means


def format_method_average(values: list[float], *, percent: bool = False) -> str:
    mean = mean_of(values)
    if mean is None:
        return "—"
    rendered = f"{100 * mean:.1f}" if percent else f"{mean:.3f}"
    return f"{rendered} (n={len(values)})"


def render_method_average_table(records: Records) -> str:
    conditions = [
        condition
        for condition in EVAL_CONDITIONS
        if any(records_for(records, task, condition) for task in TRAINABLE_TASKS)
    ]
    if not conditions:
        return "当前没有可用结果。\n"

    headers = [
        "条件",
        "训练方式",
        "推理方式",
        "测试数据",
        "任务覆盖",
        *(display for _, display, _ in METHOD_AVERAGE_METRICS),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 4 + ["---:"] * 6) + " |",
    ]
    for condition in conditions:
        covered_tasks = sum(
            bool(records_for(records, task, condition)) for task in TRAINABLE_TASKS
        )
        values = [
            condition,
            CONDITION_META[condition][0],
            CONDITION_INFERENCE[condition],
            CONDITION_DATA[condition],
            f"{covered_tasks}/{len(TRAINABLE_TASKS)}",
        ]
        for metric, _, percent in METHOD_AVERAGE_METRICS:
            values.append(
                format_method_average(
                    method_metric_task_means(records, condition, metric),
                    percent=percent,
                )
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_evaluation_analysis(records: Records) -> str:
    present_conditions = [
        condition
        for condition in EVAL_CONDITIONS
        if any(records_for(records, task, condition) for task in TRAINABLE_TASKS)
    ]
    condition_text = "、".join(present_conditions) if present_conditions else "（尚无结果）"
    task_sections = []
    for section_number, task in enumerate(TRAINABLE_TASKS, start=2):
        primary_name = "QWK" if task in ORDINAL_TASKS else "Macro-F1"
        metric_note = (
            "QWK、Accuracy、Pearson、Macro-F1 和 MAE"
            if task in ORDINAL_TASKS
            else "Accuracy、Macro-F1 和 Pearson"
        )
        task_sections.append(
            f"## {section_number}. {criterion_title(task)}\n\n"
            f"主指标为 {primary_name}；本任务展示 {metric_note}。\n\n"
            f"{render_single_task_table(records, task)}"
        )

    return f"""# Qwen3-4B 与 SciRM-7B 评测结果分析

> 自动生成于 {utc_now()}；当前纳入 {len(records)} 条按任务、条件和训练 seed 去重后的有效记录。
> 本文件由 `training/evaluate.py` 在每次评测后重建。

## 1. 统计口径

当前覆盖条件：{condition_text}。

其中 B-L/B-C 为 Qwen3-4B 基座模型，SciRM-L/SciRM-C 为经过强化学习训练的
SciRM-7B；后缀 L/C 分别表示 Label-only/CoT 测试数据与 prompt。

有序任务以 QWK 为主指标，并同时展示 Accuracy、Pearson、Macro-F1 和 MAE；
二分类任务以 Macro-F1 为主指标，并展示 Accuracy 与 Pearson。Pearson 使用离散预测与
真实标签计算：先分别计算每个 rollout 的 Pearson 系数，再按照现有评测口径对 rollout 取均值。

同一任务和方法存在多个训练 seed 时，指标在对应方法行内展示为
`均值 ± 样本标准差 (var=样本方差)`，标准差和方差均使用 `n-1`；单 seed 只展示
该次结果。Accuracy 使用百分数，方差单位为百分点平方。

每个任务表内的最优值使用下划线标出；QWK、Accuracy、Pearson、Macro-F1 取最高值，
MAE 取最低值。RAIL 结果仅纳入 `probability_normalization=full_vocab_raw` 的官方口径。

{chr(10).join(task_sections)}
## 9. 七任务汇总

每个任务按其主指标选择最优条件；下表汇总该条件的主指标、Accuracy 和 Pearson。

{render_summary_table(records)}
## 10. 各方法跨任务平均指标

先对同一任务、同一方法的多个训练 seed 求均值，再对任务等权求宏平均，避免多 seed 的
任务获得更高权重。QWK 和 MAE 仅适用于 4 个有序任务，其他指标适用于全部 7 个任务；
每个指标后的 `n` 是实际参与该指标平均的任务数，`任务覆盖` 则表示该方法已有结果的任务数。

{render_method_average_table(records)}
## 11. 重建报告

```bash
python training/evaluate.py --refresh-analysis-only --output_path eval_output/results
```
"""


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
    return analysis_path
