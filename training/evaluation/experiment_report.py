"""生成 eval_output 下的实验结果汇总报告。"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

from evaluation.condition_labels import infer_eval_condition
from evaluation.result_records import (
    CONDITION_DATA,
    CONDITION_INFERENCE,
    CONDITION_META,
    ORDINAL_TASKS,
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
            "model": ("SciRM-7B" if "scirm" in str(metrics.get("model_name") or "").lower() else "Qwen3-4B" if "qwen3" in str(metrics.get("model_name") or "").lower() else metrics.get("model_name")) or CONDITION_META.get(condition, (condition,))[0],
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


METRICS = ("accuracy", "qwk", "pearson", "macro_f1", "mae")
METRIC_LABELS = {"accuracy": "准确率 (%)", "qwk": "QWK", "pearson": "Pearson", "macro_f1": "Macro-F1", "mae": "MAE"}
ConfigKey = tuple[str, str, str, str, str, str]


def _task_label(task: str) -> str:
    """数据集名称保持英文。"""
    return criterion_title(task)


def _config_key(row: dict[str, Any]) -> ConfigKey:
    return (
        str(row.get("source") or MISSING),
        str(row.get("model") or MISSING),
        str(row.get("condition") or MISSING),
        str(row.get("train") or MISSING),
        str(row.get("prompt") or MISSING),
        str(row.get("inference") or MISSING),
    )


def _group_by_config(rows: list[dict[str, Any]]) -> dict[ConfigKey, list[dict[str, Any]]]:
    grouped: dict[ConfigKey, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_config_key(row), []).append(row)
    return grouped


def _config_cells(key: ConfigKey) -> list[str]:
    source, model, condition, train, prompt, inference = key
    model_config = model if source == "API" else f"{model} / {condition}"
    return ["本地" if source == "Local" else source, model_config, train, prompt, inference]


def _seed_sort_key(seed: str) -> tuple[int, int | str]:
    if seed.isdigit():
        return (0, int(seed))
    return (1, seed)


def _seed_text(items: list[dict[str, Any]]) -> str:
    seeds = sorted(
        {str(item.get("seed")) for item in items if item.get("seed") not in {None, MISSING}},
        key=_seed_sort_key,
    )
    return ", ".join(seeds) if seeds else MISSING


def _values(items: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(item[metric]) for item in items if item.get(metric) is not None]


def _mean(items: list[dict[str, Any]], metric: str) -> float | None:
    values = _values(items, metric)
    return statistics.fmean(values) if values else None


def _render_value(metric: str, value: float) -> str:
    return _pct(value) if metric == "accuracy" else _num(value)


def _format_stats(metric: str, values: list[float], *, best: bool = False, coverage: str | None = None) -> str:
    if not values:
        return MISSING
    rendered = _render_value(metric, statistics.fmean(values))
    if len(values) > 1:
        rendered += f" ± {_render_value(metric, statistics.stdev(values))}"
    if best:
        rendered = f"<u>{rendered}</u>"
    return f"{rendered} ({coverage})" if coverage else rendered


def _best_value(values: list[float], metric: str) -> float | None:
    if not values:
        return None
    return min(values) if metric == "mae" else max(values)


def _task_table(rows: list[dict[str, Any]], task: str) -> str:
    grouped = _group_by_config([row for row in rows if str(row.get("task")) == task])
    keys = sorted(grouped)
    means = {metric: [_mean(grouped[key], metric) for key in keys] for metric in METRICS}
    best = {
        metric: _best_value([value for value in means[metric] if value is not None], metric)
        for metric in METRICS
    }
    headers = ("来源", "模型 / 配置", "训练", "Prompt", "推理", "Seed", "运行数", *(METRIC_LABELS[m] for m in METRICS))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * 6 + ["---:"] * (len(METRICS) + 1)) + " |"]
    for index, key in enumerate(keys):
        items = grouped[key]
        metric_cells = [
            _format_stats(metric, _values(items, metric), best=means[metric][index] is not None and means[metric][index] == best[metric])
            for metric in METRICS
        ]
        lines.append("| " + " | ".join([*_config_cells(key), _seed_text(items), str(len(items)), *metric_cells]) + " |")
    return "\n".join(lines)


def _applicable_tasks(metric: str) -> tuple[str, ...]:
    return tuple(ORDINAL_TASKS) if metric in {"qwk", "mae"} else tuple(TRAINABLE_TASKS)


def _seed_macro_values(items: list[dict[str, Any]], metric: str) -> list[float]:
    by_seed: dict[str, list[float]] = {}
    applicable = set(_applicable_tasks(metric))
    for item in items:
        if str(item.get("task")) not in applicable or item.get(metric) is None:
            continue
        seed = str(item.get("seed") or MISSING)
        by_seed.setdefault(seed, []).append(float(item[metric]))
    return [statistics.fmean(values) for values in by_seed.values() if values]


def _coverage(items: list[dict[str, Any]], metric: str) -> tuple[int, int]:
    applicable = _applicable_tasks(metric)
    covered = {str(item.get("task")) for item in items if str(item.get("task")) in applicable and item.get(metric) is not None}
    return len(covered), len(applicable)


def _metric_table(rows: list[dict[str, Any]], metric: str) -> str:
    grouped = _group_by_config(rows)
    keys = sorted(grouped)
    task_means = {
        task: [_mean([item for item in grouped[key] if str(item.get("task")) == task], metric) for key in keys]
        for task in TRAINABLE_TASKS
    }
    task_best = {
        task: _best_value([value for value in task_means[task] if value is not None], metric)
        for task in TRAINABLE_TASKS
    }
    averages = [_seed_macro_values(grouped[key], metric) for key in keys]
    coverage = [_coverage(grouped[key], metric) for key in keys]
    complete_means = [statistics.fmean(values) for values, (covered, total) in zip(averages, coverage, strict=True) if values and covered == total]
    average_best = _best_value(complete_means, metric)
    headers = ("来源", "模型 / 配置", "训练", "Prompt", "推理", "Seed", *(_task_label(task) for task in TRAINABLE_TASKS), "平均值（覆盖）")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * 6 + ["---:"] * (len(TRAINABLE_TASKS) + 1)) + " |"]
    for index, key in enumerate(keys):
        items = grouped[key]
        task_cells = []
        for task in TRAINABLE_TASKS:
            task_items = [item for item in items if str(item.get("task")) == task]
            value = task_means[task][index]
            task_cells.append(_format_stats(metric, _values(task_items, metric), best=value is not None and value == task_best[task]))
        covered, total = coverage[index]
        average_mean = statistics.fmean(averages[index]) if averages[index] else None
        average_cell = _format_stats(
            metric,
            averages[index],
            best=covered == total and average_mean is not None and average_mean == average_best,
            coverage=f"{covered}/{total}",
        )
        lines.append("| " + " | ".join([*_config_cells(key), _seed_text(items), *task_cells, average_cell]) + " |")
    return "\n".join(lines)


def _model_average_table(rows: list[dict[str, Any]]) -> str:
    grouped = _group_by_config(rows)
    keys = sorted(grouped)
    averages = {metric: [_seed_macro_values(grouped[key], metric) for key in keys] for metric in METRICS}
    coverage = {metric: [_coverage(grouped[key], metric) for key in keys] for metric in METRICS}
    best: dict[str, float | None] = {}
    for metric in METRICS:
        complete = [
            statistics.fmean(values)
            for values, (covered, total) in zip(averages[metric], coverage[metric], strict=True)
            if values and covered == total
        ]
        best[metric] = _best_value(complete, metric)
    headers = ("来源", "模型 / 配置", "训练", "Prompt", "推理", "Seed", *(METRIC_LABELS[m] for m in METRICS))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * 6 + ["---:"] * len(METRICS)) + " |"]
    for index, key in enumerate(keys):
        items = grouped[key]
        cells = []
        for metric in METRICS:
            values = averages[metric][index]
            covered, total = coverage[metric][index]
            mean = statistics.fmean(values) if values else None
            cells.append(_format_stats(metric, values, best=covered == total and mean is not None and mean == best[metric], coverage=f"{covered}/{total}"))
        lines.append("| " + " | ".join([*_config_cells(key), _seed_text(items), *cells]) + " |")
    return "\n".join(lines)


def _report_summaries(rows: list[dict[str, Any]]) -> str:
    tasks = [task for task in TRAINABLE_TASKS if any(str(row.get("task")) == task for row in rows)]
    task_tables = "\n\n".join(f"### {_task_label(task)}\n\n{_task_table(rows, task)}" for task in tasks)
    metric_tables = "\n\n".join(f"### {METRIC_LABELS[metric]}\n\n{_metric_table(rows, metric)}" for metric in METRICS)
    return f"""## 各数据集的模型结果

数据集名称保持英文。同一模型配置的多个 seed 报告为 `均值 ± 样本标准差`；单 seed 仅报告单值。下划线表示该数据集、该指标的最优均值。

{task_tables}

## 按指标比较所有数据集

每张表固定一个指标，列出全部七个数据集及模型配置。`平均值（覆盖）`先对每个 seed 做跨数据集宏平均，再报告 seed 间的均值与样本标准差。

{metric_tables}

## 模型 / 配置平均结果汇总

括号内为数据集覆盖数。最佳值只在完整覆盖适用数据集的配置中比较；MAE 越低越好，其余指标越高越好。`—` 表示缺失或不适用。

{_model_average_table(rows)}"""


def render_experiment_report(eval_root: Path) -> str:
    results_root = eval_root / "results"
    api_root = eval_root / "api_results"
    rows = _local_rows(results_root) + _api_rows(api_root)
    rows.sort(key=lambda row: (row["source"], str(row.get("task")), str(row.get("condition")), str(row.get("model")), str(row.get("seed")), str(row.get("path"))))
    local_count = sum(row["source"] == "Local" for row in rows)
    api_count = sum(row["source"] == "API" for row in rows)
    return f"""# 实验结果汇总

> 生成时间：{utc_now()}。统计范围为 `eval_output/results` 和 `eval_output/api_results` 中已发现的完整 `metrics.json`。
> 本地实验 {local_count} 条，API 实验 {api_count} 条，共 {len(rows)} 条。
> 仅聚合同一模型配置的不同 seed，并报告 `均值 ± 样本标准差`；单 seed 报告单值。`CoT 2048` 作为单独模型配置展示。

{_report_summaries(rows)}
"""


def update_experiment_report(eval_root: Path) -> Path:
    eval_root.mkdir(parents=True, exist_ok=True)
    path = eval_root / "experiment_results.md"
    path.write_text(render_experiment_report(eval_root), encoding="utf-8")
    return path
