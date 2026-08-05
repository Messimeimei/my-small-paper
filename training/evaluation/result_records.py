"""Discover, normalize, and aggregate persisted evaluation records."""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from shared.metrics import criterion_title
from evaluation.condition_labels import infer_eval_condition

EVAL_CONDITIONS = (
    "B-L", "B-C", "SciRM-L", "SciRM-C", "LL", "LC", "CL", "CC",
    "PAL", "PAC", "LL-R", "RAFT-G", "RAFT-R", "CC-R", "COT-RAFT-G", "COT-RAFT-R",
)
# Only conditions included in the current report are normalized here.
LEGACY_CONDITION_ALIASES = {
    "B-S": "B-L", "S-S": "LL", "L-L": "LL", "C→S": "CL", "C→L": "CL",
    "S→C": "LC", "L→C": "LC", "C-C": "CC",
}

CONDITION_META = {
    "B-L": ("Qwen3-4B Base", "Label-only", "Qwen3-4B 基座模型直接输出标签"),
    "B-C": ("Qwen3-4B Base", "CoT", "Qwen3-4B 基座模型先输出推理再输出标签"),
    "SciRM-L": ("SciRM-7B RL", "Label-only", "SciRM-7B 强化学习模型直接输出标签"),
    "SciRM-C": ("SciRM-7B RL", "CoT", "SciRM-7B 强化学习模型先输出推理再输出标签"),
    "LL": ("Label-only SFT", "Label-only", "同格式 Label-only 微调与测试"),
    "LC": ("Label-only SFT", "CoT", "Label-only adapter 交叉测试 CoT prompt"),
    "CL": ("CoT SFT", "Label-only", "CoT adapter 交叉测试 Label-only prompt"),
    "CC": ("CoT SFT", "CoT", "同格式 CoT 微调与测试"),
    "PAL": ("Paper Align SFT", "Label-only", "Paper Align adapter 交叉测试 Label-only prompt"),
    "PAC": ("Paper Align SFT", "CoT", "Paper Align adapter 在 CoT 测试 prompt 上评测"),
    "LL-R": ("Label-only CE", "RAIL", "Label-only adapter 使用官方完整词表概率加权和"),
    "RAFT-G": ("RAFT without CoT", "Greedy", "RAFT adapter 使用原自由生成"),
    "RAFT-R": ("RAFT without CoT", "RAIL", "RAFT adapter 使用官方完整词表概率加权和"),
    "CC-R": ("CoT CE", "CoT-RAIL", "CoT CE adapter 先生成解释再使用官方完整词表概率加权和"),
    "COT-RAFT-G": ("CoT-RAFT", "Greedy", "CoT-RAFT adapter 使用原自由生成"),
    "COT-RAFT-R": (
        "CoT-RAFT",
        "CoT-RAIL",
        "CoT-RAFT adapter 先生成解释再使用官方完整词表概率加权和",
    ),
}

TRAINABLE_TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
ORDINAL_TASKS = frozenset(TRAINABLE_TASKS[:4])

RecordKey = tuple[str, str, str]
Records = dict[RecordKey, dict[str, Any]]
SEED_PATTERN = re.compile(r"(?:^|[_#])seed_?(\d+)(?:[_#]|$)", re.IGNORECASE)
RECORD_CACHE_NAME = "evaluation_analysis_records.json"
RAIL_CONDITIONS = frozenset({"LL-R", "RAFT-R", "CC-R", "COT-RAFT-R"})
OFFICIAL_RAIL_NORMALIZATION = "full_vocab_raw"
CONDITION_INFERENCE = {
    "B-L": "Greedy", "B-C": "Greedy", "SciRM-L": "Greedy", "SciRM-C": "Greedy",
    "LL": "Greedy", "LC": "Greedy",
    "CL": "Greedy", "CC": "Greedy", "PAL": "Greedy", "PAC": "Greedy",
    "LL-R": "RAIL", "RAFT-G": "Greedy", "RAFT-R": "RAIL",
    "CC-R": "CoT-RAIL", "COT-RAFT-G": "Greedy", "COT-RAFT-R": "CoT-RAIL",
}
CONDITION_DATA = {
    "B-L": "Label-only", "B-C": "CoT", "SciRM-L": "Label-only", "SciRM-C": "CoT",
    "LL": "Label-only", "LC": "CoT",
    "CL": "Label-only", "CC": "CoT", "PAL": "Label-only", "PAC": "CoT",
    "LL-R": "Label-only", "RAFT-G": "Label-only", "RAFT-R": "Label-only",
    "CC-R": "CoT", "COT-RAFT-G": "CoT", "COT-RAFT-R": "CoT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_condition(code: str | None) -> str | None:
    if code is None:
        return None
    normalized = LEGACY_CONDITION_ALIASES.get(code, code)
    return normalized if normalized in EVAL_CONDITIONS else None


def exp_name_with_seed(exp_name: str, train_seed: str) -> str:
    base_name = re.sub(r"#seed_(?:\d+|base)$", "", exp_name)
    return f"{base_name}#seed_{train_seed}"


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_train_seed(metrics: dict[str, Any], condition: str) -> str:
    """Extract the training seed without confusing it with the eval seed."""
    if condition.startswith("B-") or condition.startswith("SciRM-"):
        return "base"
    full_config = metrics.get("full_config") or {}
    train_run = full_config.get("train_run") or {}
    for candidate in (
        metrics.get("train_seed"),
        train_run.get("train_seed"),
        (train_run.get("train_resolved_config") or {}).get("seed"),
    ):
        if candidate is not None:
            return str(candidate)
    for value in (
        metrics.get("adapter"),
        train_run.get("train_run_id"),
        train_run.get("train_run_directory"),
        metrics.get("exp_name"),
    ):
        match = SEED_PATTERN.search(str(value or ""))
        if match:
            return match.group(1)
    return "unknown"


def load_metrics(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def pearson_coefficient(labels: list[float], predictions: list[float]) -> float | None:
    """Return Pearson correlation, or None when fewer than two varying pairs exist."""
    if len(labels) != len(predictions) or len(labels) < 2:
        return None
    label_mean = statistics.fmean(labels)
    prediction_mean = statistics.fmean(predictions)
    label_ss = sum((value - label_mean) ** 2 for value in labels)
    prediction_ss = sum((value - prediction_mean) ** 2 for value in predictions)
    if label_ss == 0 or prediction_ss == 0:
        return None
    covariance = sum(
        (label - label_mean) * (prediction - prediction_mean)
        for label, prediction in zip(labels, predictions, strict=True)
    )
    return covariance / math.sqrt(label_ss * prediction_ss)


def prediction_pearson(prediction_path: Path) -> float | None:
    """Calculate mean discrete-prediction Pearson correlation across rollouts."""
    try:
        rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    rollout_count = max(
        (len(row.get("rollout_predictions") or []) for row in rows),
        default=0,
    )
    correlations: list[float] = []
    for rollout_index in range(rollout_count):
        pairs = [
            (as_float(row.get("label")), as_float(row["rollout_predictions"][rollout_index]))
            for row in rows
            if rollout_index < len(row.get("rollout_predictions") or [])
        ]
        valid_pairs = [
            (label, prediction)
            for label, prediction in pairs
            if label is not None and prediction is not None
        ]
        labels = [label for label, _ in valid_pairs]
        predictions = [prediction for _, prediction in valid_pairs]
        correlation = pearson_coefficient(labels, predictions)
        if correlation is not None:
            correlations.append(correlation)
    return statistics.fmean(correlations) if correlations else None


def extract_run_record(metrics: dict[str, Any], metrics_path: Path) -> dict[str, Any] | None:
    exp_name = str(metrics.get("exp_name") or metrics_path.parent.name)
    task = str(metrics.get("task") or metrics_path.parent.parent.name)
    mode = str(metrics.get("supervision_mode") or metrics.get("evaluation_mode") or "cot")
    adapter = metrics.get("adapter")
    full_config = metrics.get("full_config") or {}
    train_config = full_config.get("train_config")
    if isinstance(train_config, dict):
        train_config = train_config.get("experiment_name") or json.dumps(train_config)
    condition = normalize_condition(metrics.get("eval_condition"))
    inferred_condition = normalize_condition(
        infer_eval_condition(
            exp_name=exp_name,
            supervision_mode=mode,
            adapter=str(adapter) if adapter is not None else None,
            train_config=str(train_config) if train_config is not None else None,
        )
    )
    # SciRM results created before the dedicated conditions existed were persisted
    # as B-L/B-C. Prefer the model-specific inference for those stale records.
    if inferred_condition in {"SciRM-L", "SciRM-C"}:
        condition = inferred_condition
    elif condition is None:
        condition = inferred_condition
    if condition is None or task not in TRAINABLE_TASKS:
        return None

    aggregate = metrics.get("aggregate") or {}
    probability_normalization = metrics.get(
        "probability_normalization",
        aggregate.get("probability_normalization"),
    )
    if (
        condition in RAIL_CONDITIONS
        and probability_normalization != OFFICIAL_RAIL_NORMALIZATION
    ):
        return None
    accuracy = metrics.get("test_accuracy", aggregate.get("accuracy"))
    macro_f1 = metrics.get("test_macro_f1", aggregate.get("macro_f1"))
    qwk = metrics.get("test_qwk", aggregate.get("qwk"))
    primary = qwk if task in ORDINAL_TASKS else macro_f1
    if accuracy is None and primary is None:
        return None

    pearson = metrics.get("test_pearson", aggregate.get("pearson"))
    if pearson is None:
        pearson = prediction_pearson(metrics_path.with_name("predictions.jsonl"))
    tokens = aggregate.get("tokens") or {}
    return {
        "task": task,
        "condition": condition,
        "train_seed": extract_train_seed(metrics, condition),
        "exp_name": exp_name,
        "metrics_path": str(metrics_path),
        "finished_at_utc": metrics.get("finished_at_utc") or "",
        "n_samples": aggregate.get("samples"),
        "accuracy": as_float(accuracy),
        "macro_f1": as_float(macro_f1),
        "mae": as_float(metrics.get("test_mae", aggregate.get("mae"))),
        "qwk": as_float(qwk),
        "pearson": as_float(pearson),
        "rail_mae": as_float(metrics.get("test_rail_mae", aggregate.get("rail_mae"))),
        "rail_rmse": as_float(metrics.get("test_rail_rmse", aggregate.get("rail_rmse"))),
        "probability_normalization": probability_normalization,
        "score_prefix_valid_rate": as_float(
            metrics.get(
                "score_prefix_valid_rate",
                aggregate.get("score_prefix_valid_rate"),
            )
        ),
        "reasoning_valid_rate": as_float(
            metrics.get("reasoning_valid_rate", aggregate.get("reasoning_valid_rate"))
        ),
        "avg_score_probability_mass": as_float(
            metrics.get(
                "avg_score_probability_mass",
                aggregate.get("avg_score_probability_mass"),
            )
        ),
        "format_valid_rate": as_float(
            metrics.get("format_valid_rate", aggregate.get("format_valid_rate"))
        ),
        "avg_output_tokens": as_float(
            metrics.get("avg_output_tokens", tokens.get("avg_output_tokens"))
        ),
        "avg_reasoning_tokens": as_float(
            metrics.get("avg_reasoning_tokens", tokens.get("avg_reasoning_tokens"))
        ),
        "samples_per_sec": as_float(aggregate.get("samples_per_sec")),
        "gpu_time_sec": as_float(metrics.get("gpu_time_sec", aggregate.get("gpu_time_sec"))),
    }


def load_record_cache(output_root: Path) -> Records:
    """Load historical records so overwritten eval directories do not lose seeds."""
    payload = load_metrics(output_root / RECORD_CACHE_NAME) or {}
    records: Records = {}
    for record in payload.get("records") or []:
        if not isinstance(record, dict):
            continue
        task = str(record.get("task") or "")
        condition = normalize_condition(record.get("condition"))
        train_seed = str(record.get("train_seed") or "unknown")
        old_exp_name = str(record.get("exp_name") or "")
        inferred_condition = normalize_condition(
            infer_eval_condition(
                exp_name=old_exp_name,
                supervision_mode="cot",
                adapter=None,
            )
        )
        if inferred_condition in {"SciRM-L", "SciRM-C"}:
            condition = inferred_condition
        seeded_exp_name = exp_name_with_seed(old_exp_name, train_seed)
        metrics_path = Path(str(record.get("metrics_path") or ""))
        if old_exp_name and metrics_path.parent.name == old_exp_name:
            record["metrics_path"] = str(
                metrics_path.parent.parent / seeded_exp_name / metrics_path.name
            )
        record["exp_name"] = seeded_exp_name
        if task not in TRAINABLE_TASKS or condition is None:
            continue
        if (
            condition in RAIL_CONDITIONS
            and record.get("probability_normalization")
            != OFFICIAL_RAIL_NORMALIZATION
        ):
            continue
        record["condition"] = condition
        record["train_seed"] = train_seed
        records[(task, condition, train_seed)] = record
    return records


def collect_eval_records(output_root: Path) -> Records:
    """Return the latest record per (task, condition, training seed)."""
    if not output_root.is_dir():
        return {}
    records = load_record_cache(output_root)
    for metrics_path in sorted(output_root.rglob("metrics.json")):
        if "configs" in metrics_path.parts:
            continue
        metrics = load_metrics(metrics_path)
        record = extract_run_record(metrics, metrics_path) if metrics is not None else None
        if record is None:
            continue
        key = (record["task"], record["condition"], record["train_seed"])
        existing = records.get(key)
        if existing is None or str(record["finished_at_utc"]) >= str(
            existing.get("finished_at_utc") or ""
        ):
            records[key] = record
    return records


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    if seed == "base":
        return (0, 0)
    try:
        return (1, int(seed))
    except ValueError:
        return (2, seed)


def records_for(records: Records, task: str, condition: str) -> list[dict[str, Any]]:
    grouped = (
        record for (record_task, record_condition, _), record in records.items()
        if record_task == task and record_condition == condition
    )
    return sorted(grouped, key=lambda record: seed_sort_key(str(record["train_seed"])))


def present_values(records: Iterable[dict[str, Any]], metric: str) -> list[float]:
    return [float(record[metric]) for record in records if record.get(metric) is not None]


def mean_of(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def metric_stats(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    present = list(values)
    if not present:
        return None, None, None
    if len(present) == 1:
        return present[0], None, None
    return statistics.fmean(present), statistics.stdev(present), statistics.variance(present)


def primary_metric(task: str) -> str:
    return "qwk" if task in ORDINAL_TASKS else "macro_f1"


def metric_specs(task: str) -> tuple[tuple[str, str, bool, bool], ...]:
    """Return metric key, display name, percentage flag, and direction."""
    if task in ORDINAL_TASKS:
        return (
            ("qwk", "QWK", False, True),
            ("accuracy", "Accuracy (%)", True, True),
            ("pearson", "Pearson", False, True),
            ("macro_f1", "Macro-F1", False, True),
            ("mae", "MAE", False, False),
        )
    return (
        ("accuracy", "Accuracy (%)", True, True),
        ("macro_f1", "Macro-F1", False, True),
        ("pearson", "Pearson", False, True),
    )


def format_stats_cell(
    values: Iterable[float],
    *,
    percent: bool = False,
    underline: bool = False,
) -> str:
    mean, std, variance = metric_stats(values)
    if mean is None:
        return "—"
    if percent:
        rendered = f"{100 * mean:.1f}"
        if std is not None and variance is not None:
            rendered += f" ± {100 * std:.1f} (var={10000 * variance:.4f})"
    else:
        rendered = f"{mean:.3f}"
        if std is not None and variance is not None:
            rendered += f" ± {std:.3f} (var={variance:.6f})"
    return f"<u>{rendered}</u>" if underline else rendered


def metric_means(
    grouped_conditions: list[tuple[str, list[dict[str, Any]]]],
    metric: str,
) -> dict[str, float | None]:
    return {
        condition: mean_of(present_values(grouped, metric))
        for condition, grouped in grouped_conditions
    }


def best_metric_value(
    means: dict[str, float | None],
    *,
    higher_is_better: bool,
) -> float | None:
    present = [value for value in means.values() if value is not None]
    if not present:
        return None
    return max(present) if higher_is_better else min(present)


def is_best_value(value: float | None, best: float | None) -> bool:
    return (
        value is not None
        and best is not None
        and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
    )
