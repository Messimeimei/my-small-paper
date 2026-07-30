"""Rebuild eval_output/evaluation_analysis.md from per-run metrics.json files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics_utils import criterion_title, infer_eval_condition

EVAL_CONDITIONS: tuple[str, ...] = (
    "B-L",
    "B-C",
    "L-L",
    "C-C",
    "C→L",
    "L→C",
    "A-C",
    "A→L",
)

# Legacy condition codes from older runs (Label-only wording).
LEGACY_CONDITION_ALIASES = {
    "B-S": "B-L",
    "S-S": "L-L",
    "C→S": "C→L",
    "S→C": "L→C",
    "A-A": "A-C",
    "A→S": "A→L",
}

CONDITION_META: dict[str, tuple[str, str, str]] = {
    "B-L": ("Base", "Label-only", "基座模型直接输出标签"),
    "B-C": ("Base", "CoT", "基座模型先输出推理再输出标签"),
    "L-L": ("Label-only SFT", "Label-only", "同格式 Label-only 微调与测试"),
    "C-C": ("CoT SFT", "CoT", "同格式 CoT 微调与测试"),
    "C→L": ("CoT SFT", "Label-only", "CoT adapter 交叉测试 Label-only prompt"),
    "L→C": ("Label-only SFT", "CoT", "Label-only adapter 交叉测试 CoT prompt"),
    "A-C": ("Align SFT", "CoT", "Align adapter 在 CoT 测试 prompt 上评测"),
    "A→L": ("Align SFT", "Label-only", "Align adapter 交叉测试 Label-only prompt"),
}

TRAINABLE_TASKS: tuple[str, ...] = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)

ORDINAL_TASKS: frozenset[str] = frozenset(
    {
        "rev_util_actionability",
        "rev_util_grounding_specificity",
        "rev_util_helpfulness",
        "rev_util_verifiability",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_condition(code: str | None) -> str | None:
    if code is None:
        return None
    return LEGACY_CONDITION_ALIASES.get(code, code)


def format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}"


def format_cell(acc: float | None, f1: float | None) -> str:
    if acc is None and f1 is None:
        return "—"
    if acc is None:
        return f"— / {format_pct(f1)}"
    if f1 is None:
        return f"{format_pct(acc)} / —"
    return f"{format_pct(acc)} / {format_pct(f1)}"


def format_mae_qwk(mae: float | None, qwk: float | None) -> str:
    if mae is None and qwk is None:
        return "—"
    if mae is None:
        return f"— / {qwk:.3f}" if qwk is not None else "—"
    if qwk is None:
        return f"{mae:.3f} / —"
    return f"{mae:.3f} / {qwk:.3f}"


def load_metrics(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def extract_run_record(metrics: dict[str, Any], metrics_path: Path) -> dict[str, Any] | None:
    exp_name = str(metrics.get("exp_name") or metrics_path.parent.name)
    task = str(metrics.get("task") or metrics_path.parent.parent.name)
    supervision_mode = str(
        metrics.get("supervision_mode")
        or metrics.get("evaluation_mode")
        or "cot"
    )
    adapter = metrics.get("adapter")
    full_config = metrics.get("full_config") or {}
    train_config = full_config.get("train_config")
    if isinstance(train_config, dict):
        train_config = train_config.get("experiment_name") or json.dumps(train_config)
    condition = normalize_condition(
        infer_eval_condition(
            exp_name=exp_name,
            supervision_mode=supervision_mode,
            adapter=str(adapter) if adapter is not None else None,
            train_config=str(train_config) if train_config is not None else None,
        )
    )
    if condition is None:
        return None

    aggregate = metrics.get("aggregate") or {}
    accuracy = metrics.get("test_accuracy", aggregate.get("accuracy"))
    macro_f1 = metrics.get("test_macro_f1", aggregate.get("macro_f1"))
    if accuracy is None and macro_f1 is None:
        return None

    tokens = aggregate.get("tokens") or {}
    finished_at = metrics.get("finished_at_utc") or ""
    return {
        "task": task,
        "condition": condition,
        "exp_name": exp_name,
        "metrics_path": str(metrics_path),
        "finished_at_utc": finished_at,
        "n_samples": aggregate.get("samples"),
        "accuracy": float(accuracy) if accuracy is not None else None,
        "macro_f1": float(macro_f1) if macro_f1 is not None else None,
        "mae": metrics.get("test_mae", aggregate.get("mae")),
        "qwk": metrics.get("test_qwk", aggregate.get("qwk")),
        "format_valid_rate": metrics.get("format_valid_rate", aggregate.get("format_valid_rate")),
        "avg_output_tokens": metrics.get("avg_output_tokens", tokens.get("avg_output_tokens")),
        "avg_reasoning_tokens": metrics.get(
            "avg_reasoning_tokens", tokens.get("avg_reasoning_tokens")
        ),
        "samples_per_sec": aggregate.get("samples_per_sec"),
        "gpu_time_sec": metrics.get("gpu_time_sec", aggregate.get("gpu_time_sec")),
    }


def collect_eval_records(output_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return latest metrics record per (task, condition)."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not output_root.is_dir():
        return records

    for metrics_path in sorted(output_root.glob("*/*/metrics.json")):
        if metrics_path.parent.parent.name == "configs":
            continue
        metrics = load_metrics(metrics_path)
        if metrics is None:
            continue
        record = extract_run_record(metrics, metrics_path)
        if record is None:
            continue
        key = (record["task"], record["condition"])
        existing = records.get(key)
        if existing is None or str(record["finished_at_utc"]) >= str(
            existing.get("finished_at_utc") or ""
        ):
            records[key] = record
    return records


def mean_of(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def render_condition_legend() -> str:
    lines = [
        "| 记号 | 训练方式 | 测试 prompt | 含义 |",
        "| --- | --- | --- | --- |",
    ]
    for code in EVAL_CONDITIONS:
        train, test, desc = CONDITION_META[code]
        lines.append(f"| {code} | {train} | {test} | {desc} |")
    return "\n".join(lines) + "\n"


def render_main_table(records: dict[tuple[str, str], dict[str, Any]]) -> str:
    headers = ["任务", "N", *EVAL_CONDITIONS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    macro_acc: dict[str, list[float | None]] = {code: [] for code in EVAL_CONDITIONS}
    macro_f1: dict[str, list[float | None]] = {code: [] for code in EVAL_CONDITIONS}

    for task in TRAINABLE_TASKS:
        title = criterion_title(task)
        n_value = "—"
        cells = [title, n_value]
        for code in EVAL_CONDITIONS:
            record = records.get((task, code))
            if record is not None and n_value == "—" and record.get("n_samples") is not None:
                n_value = str(int(record["n_samples"]))
            if record is None:
                cells.append("—")
                continue
            cells.append(format_cell(record.get("accuracy"), record.get("macro_f1")))
            macro_acc[code].append(record.get("accuracy"))
            macro_f1[code].append(record.get("macro_f1"))
        cells[1] = n_value
        lines.append("| " + " | ".join(cells) + " |")

    macro_cells = ["**任务宏平均**", ""]
    for code in EVAL_CONDITIONS:
        macro_cells.append(
            format_cell(mean_of(macro_acc[code]), mean_of(macro_f1[code]))
        )
    lines.append("| " + " | ".join(macro_cells) + " |")
    return "\n".join(lines) + "\n"


def render_migration_table(records: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = [
        "| 任务 | C→L 相对 L-L | L→C 相对 C-C | A→L 相对 L-L |",
        "| --- | ---: | ---: | ---: |",
    ]
    deltas_cl: list[float] = []
    deltas_lc: list[float] = []
    deltas_al: list[float] = []

    def delta(left: str, right: str) -> tuple[str, float | None]:
        left_rec = records.get((task, left))
        right_rec = records.get((task, right))
        if (
            left_rec is None
            or right_rec is None
            or left_rec.get("accuracy") is None
            or right_rec.get("accuracy") is None
        ):
            return "—", None
        value = 100 * (left_rec["accuracy"] - right_rec["accuracy"])
        return f"{value:+.1f} pp", value

    for task in TRAINABLE_TASKS:
        title = criterion_title(task)
        d_cl, v_cl = delta("C→L", "L-L")
        d_lc, v_lc = delta("L→C", "C-C")
        d_al, v_al = delta("A→L", "L-L")
        if v_cl is not None:
            deltas_cl.append(v_cl)
        if v_lc is not None:
            deltas_lc.append(v_lc)
        if v_al is not None:
            deltas_al.append(v_al)
        lines.append(f"| {title} | {d_cl} | {d_lc} | {d_al} |")

    avg_cl = mean_of(deltas_cl)
    avg_lc = mean_of(deltas_lc)
    avg_al = mean_of(deltas_al)
    lines.append(
        "| **平均** | "
        f"{f'{avg_cl:+.1f} pp' if avg_cl is not None else '—'} | "
        f"{f'{avg_lc:+.1f} pp' if avg_lc is not None else '—'} | "
        f"{f'{avg_al:+.1f} pp' if avg_al is not None else '—'} |"
    )
    return "\n".join(lines) + "\n"


def render_ordinal_table(records: dict[tuple[str, str], dict[str, Any]]) -> str:
    headers = [
        "任务",
        "L-L MAE / QWK",
        "C-C MAE / QWK",
        "C→L MAE / QWK",
        "L→C MAE / QWK",
        "A-C MAE / QWK",
        "A→L MAE / QWK",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for task in TRAINABLE_TASKS:
        if task not in ORDINAL_TASKS:
            continue
        title = criterion_title(task)
        cells = [title]
        for code in ("L-L", "C-C", "C→L", "L→C", "A-C", "A→L"):
            record = records.get((task, code))
            if record is None:
                cells.append("—")
            else:
                cells.append(format_mae_qwk(record.get("mae"), record.get("qwk")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_efficiency_table(records: dict[tuple[str, str], dict[str, Any]]) -> str:
    headers = ["条件", "格式有效率", "平均输出 token", "平均 reasoning token", "平均 samples/s"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for code in EVAL_CONDITIONS:
        valid_rates: list[float] = []
        output_tokens: list[float] = []
        reasoning_tokens: list[float] = []
        samples_per_sec: list[float] = []
        for task in TRAINABLE_TASKS:
            record = records.get((task, code))
            if record is None:
                continue
            if record.get("format_valid_rate") is not None:
                valid_rates.append(float(record["format_valid_rate"]))
            if record.get("avg_output_tokens") is not None:
                output_tokens.append(float(record["avg_output_tokens"]))
            if record.get("avg_reasoning_tokens") is not None:
                reasoning_tokens.append(float(record["avg_reasoning_tokens"]))
            if record.get("samples_per_sec") is not None:
                samples_per_sec.append(float(record["samples_per_sec"]))
            elif (
                record.get("gpu_time_sec")
                and record.get("n_samples")
                and float(record["gpu_time_sec"]) > 0
            ):
                samples_per_sec.append(
                    float(record["n_samples"]) / float(record["gpu_time_sec"])
                )
        lines.append(
            "| "
            + " | ".join(
                [
                    code,
                    format_pct(mean_of(valid_rates)),
                    f"{mean_of(output_tokens):.1f}" if mean_of(output_tokens) is not None else "—",
                    f"{mean_of(reasoning_tokens):.1f}"
                    if mean_of(reasoning_tokens) is not None
                    else "—",
                    f"{mean_of(samples_per_sec):.1f}"
                    if mean_of(samples_per_sec) is not None
                    else "—",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_evaluation_analysis(records: dict[tuple[str, str], dict[str, Any]]) -> str:
    run_count = len(records)
    present_conditions = [
        code for code in EVAL_CONDITIONS if any(records.get((task, code)) for task in TRAINABLE_TASKS)
    ]
    condition_text = "、".join(present_conditions) if present_conditions else "（尚无结果）"
    align_note = ""
    if not any(code.startswith("A") for code in present_conditions):
        align_note = (
            "\n> Align 列（A-C / A→L）会在运行 "
            "`eval_output/configs/<task>/ft_align_on_*.yaml` 后自动填入。\n"
        )

    return f"""# Qwen3-4B 评测结果分析

> 自动生成于 {utc_now()}；扫描到 {run_count} 条有效 `metrics.json` 记录。
> 本文件由 `training/evaluate.py` 在每次评测后重建。

## 1. 分析范围与记号

本报告直接读取 `eval_output/<task>/<exp_name>/metrics.json`，按实验目录归档，避免不同训练方式互相覆盖。

当前覆盖条件：{condition_text}。

配置文件命名规则：`{{train}}_on_{{test}}.yaml`，例如 `base_on_cot.yaml`、`ft_cot_on_label_only.yaml`。

{render_condition_legend()}
Label-only 与 CoT 测试集按任务逐 ID、逐标签配对，因此交叉评测差值不受测试样本变化影响。{align_note}
## 2. 完整结果

下表单元格均为 `Accuracy / Macro-F1`，单位为 `%`。最后一行为 7 个任务的非加权宏平均。

{render_main_table(records)}
## 3. 跨格式迁移

这里比较同一个测试 prompt 下不同训练格式 adapter 的差异（单位：Accuracy 百分点）。

{render_migration_table(records)}
## 4. 有序评分指标

四个评审意见任务是 1–5 分有序分类。除 Accuracy 和 Macro-F1 外，使用 MAE 与 QWK 衡量距离和顺序质量。MAE 越低越好，QWK 越高越好。

{render_ordinal_table(records)}
## 5. 格式稳定性与效率

下表为 7 个任务的非加权宏平均。`samples/s` 基于各结果中的 GPU 推理时间计算。

{render_efficiency_table(records)}
## 6. 使用说明

| 配置文件 | 含义 |
| --- | --- |
| `base_on_cot.yaml` | Base × CoT 测试 |
| `base_on_label_only.yaml` | Base × Label-only 测试 |
| `ft_cot_on_cot.yaml` | CoT SFT × CoT 测试 |
| `ft_label_only_on_label_only.yaml` | Label-only SFT × Label-only 测试 |
| `ft_cot_on_label_only.yaml` | CoT SFT × Label-only 测试（交叉） |
| `ft_label_only_on_cot.yaml` | Label-only SFT × CoT 测试（交叉） |
| `ft_align_on_cot.yaml` | Align SFT × CoT 测试 |
| `ft_align_on_label_only.yaml` | Align SFT × Label-only 测试（交叉） |

```bash
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \\
  --config eval_output/configs/rev_util_actionability/ft_align_on_cot.yaml
```

每次评测完成后，本文件会自动刷新；详细逐样本结果仍在对应实验目录的 `metrics.json` 与 `predictions.jsonl`。
"""


def update_evaluation_analysis(output_root: Path) -> Path:
    records = collect_eval_records(output_root)
    analysis_path = output_root / "evaluation_analysis.md"
    analysis_path.write_text(
        render_evaluation_analysis(records),
        encoding="utf-8",
    )
    return analysis_path
