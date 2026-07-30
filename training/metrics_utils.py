"""Shared classification / ordinal / token metrics for train and eval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SCORE_RE = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.I)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.I | re.S)

CRITERION_TITLES = {
    "actionability": "Actionability",
    "grounding_specificity": "Grounding Specificity",
    "helpfulness": "Helpfulness",
    "verifiability": "Verifiability",
    "coherence": "Coherence",
    "positioning_check": "Positioning Check",
    "positioning_type": "Positioning Type",
    "novelty": "Novelty",
    "revision_correctness": "Revision Correctness",
    "revision_relatedness": "Revision Relatedness",
}


def criterion_title(aspect_or_task: str | None) -> str:
    if not aspect_or_task:
        return "Unknown"
    key = aspect_or_task.strip().lower()
    if key in CRITERION_TITLES:
        return CRITERION_TITLES[key]
    for suffix, title in CRITERION_TITLES.items():
        if key.endswith(suffix):
            return title
    return aspect_or_task.replace("_", " ").title()


def extract_score(text: str, allowed_scores: set[int]) -> int | None:
    matches = SCORE_RE.findall(text or "")
    if not matches:
        return None
    score = int(matches[-1])
    return score if score in allowed_scores else None


def extract_reasoning(text: str) -> str:
    match = REASONING_RE.search(text or "")
    return match.group(1).strip() if match else ""


def is_ordinal_scores(score_sets: list[int]) -> bool:
    return sorted(score_sets) == list(range(min(score_sets), max(score_sets) + 1)) and len(
        score_sets
    ) >= 3


def mean_absolute_error(
    predictions: list[dict[str, Any]], *, only_valid: bool = True
) -> float | None:
    pairs = []
    for row in predictions:
        pred = row.get("prediction")
        if pred is None:
            if only_valid:
                continue
            continue
        pairs.append(abs(int(pred) - int(row["label"])))
    if not pairs:
        return None
    return sum(pairs) / len(pairs)


def quadratic_weighted_kappa(
    predictions: list[dict[str, Any]], score_sets: list[int]
) -> float | None:
    """QWK over format-valid predictions; returns None if fewer than 2 valid rows."""
    labels = sorted(score_sets)
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    if n < 2:
        return None
    conf = [[0 for _ in range(n)] for _ in range(n)]
    valid = 0
    for row in predictions:
        pred = row.get("prediction")
        if pred not in index:
            continue
        conf[index[row["label"]]][index[pred]] += 1
        valid += 1
    if valid < 2:
        return None

    hist_true = [sum(conf[i][j] for j in range(n)) for i in range(n)]
    hist_pred = [sum(conf[i][j] for i in range(n)) for j in range(n)]
    weights = [[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)]
    observed = sum(weights[i][j] * conf[i][j] for i in range(n) for j in range(n))
    expected = sum(
        weights[i][j] * hist_true[i] * hist_pred[j] / valid
        for i in range(n)
        for j in range(n)
    )
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def classification_metrics(
    predictions: list[dict[str, Any]], score_sets: list[int]
) -> dict[str, Any]:
    total = len(predictions)
    allowed_scores = set(score_sets)
    valid = [row for row in predictions if row.get("prediction") in allowed_scores]
    f1_values: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for label in score_sets:
        tp = sum(row["label"] == label and row.get("prediction") == label for row in predictions)
        fp = sum(row["label"] != label and row.get("prediction") == label for row in predictions)
        fn = sum(row["label"] == label and row.get("prediction") != label for row in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(row["label"] == label for row in predictions),
        }
    confusion_matrix = {
        str(gold): {
            ("invalid" if predicted is None else str(predicted)): sum(
                row["label"] == gold and row.get("prediction") == predicted
                for row in predictions
            )
            for predicted in (*score_sets, None)
        }
        for gold in score_sets
    }
    metrics: dict[str, Any] = {
        "samples": total,
        "score_sets": score_sets,
        "accuracy": (
            sum(bool(row.get("correct")) for row in predictions) / total if total else 0.0
        ),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "format_valid_rate": len(valid) / total if total else 0.0,
        "invalid_outputs": total - len(valid),
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
    }
    if is_ordinal_scores(score_sets):
        metrics["mae"] = mean_absolute_error(predictions)
        metrics["qwk"] = quadratic_weighted_kappa(predictions, score_sets)
    return metrics


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_stats(
    predictions: list[dict[str, Any]],
    tokenizer: Any,
    *,
    output_key: str = "output",
) -> dict[str, float | int | None]:
    output_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    for row in predictions:
        output = str(row.get(output_key) or "")
        n_out = count_tokens(tokenizer, output)
        n_reason = count_tokens(tokenizer, extract_reasoning(output))
        output_tokens.append(n_out)
        reasoning_tokens.append(n_reason)
    n = len(output_tokens)
    if n == 0:
        return {
            "avg_output_tokens": None,
            "avg_reasoning_tokens": None,
            "samples": 0,
        }
    return {
        "samples": n,
        "avg_output_tokens": sum(output_tokens) / n,
        "avg_reasoning_tokens": sum(reasoning_tokens) / n,
        "total_output_tokens": sum(output_tokens),
        "total_reasoning_tokens": sum(reasoning_tokens),
    }


def infer_supervision_mode(path: Path | str, rows: list[dict[str, Any]] | None = None) -> str:
    text = str(path).replace("\\", "/").lower()
    if "label_only" in text:
        return "label_only"
    if "/cot/" in text or text.endswith("_cot.jsonl") or "test_cot" in text:
        return "cot"
    if rows:
        mode = rows[0].get("supervision_mode") or rows[0].get("evaluation_mode")
        if mode:
            mode = str(mode)
            if mode in {"score_only", "label_only"}:
                return "label_only"
            return mode
    return "unknown"


def infer_task_name(path: Path | str, rows: list[dict[str, Any]] | None = None) -> str:
    if rows:
        aspect = rows[0].get("aspect")
        task = rows[0].get("task")
        if aspect:
            if task and not str(task).endswith(str(aspect)):
                return f"{task}_{aspect}" if task != "unseen_task" else str(aspect)
            if task and task != "unseen_task":
                return str(task)
            return str(aspect)
        if task:
            return str(task)
    parts = Path(path).resolve().parts
    for index, part in enumerate(parts):
        if part == "data" and index + 1 < len(parts):
            return parts[index + 1]
    stem = Path(path).stem
    for suffix in (
        "_train_cot",
        "_test_cot",
        "_train_label_only",
        "_test_label_only",
        "_cot",
        "_label_only",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def short_model_name(model_path: Path | str) -> str:
    return Path(model_path).name


def format_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{100 * value:.{digits}f}"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


COMPARISON_MODES = (
    "base_label_only",
    "base_cot",
    "ft_label_only",
    "ft_cot",
)

# Markdown column titles (ft modes keep short names as requested).
COMPARISON_MODE_LABELS = {
    "base_label_only": "base-Label-only",
    "base_cot": "base-cot",
    "ft_label_only": "Label-only",
    "ft_cot": "cot",
}


def comparison_field_prefix(mode: str) -> str:
    """Map comparison mode -> JSON field prefix (base_label / base_cot / ft_label / ft_cot)."""
    mapping = {
        "base_label_only": "base_label",
        "base_cot": "base_cot",
        "ft_label_only": "ft_label",
        "ft_cot": "ft_cot",
    }
    if mode not in mapping:
        raise ValueError(f"Unknown comparison mode: {mode}")
    return mapping[mode]


def infer_comparison_mode(
    *,
    supervision_mode: str,
    adapter: Path | str | None,
    exp_name: str | None = None,
) -> str:
    """Return one of base_label_only / base_cot / ft_label_only / ft_cot."""
    mode = supervision_mode if supervision_mode in {"cot", "label_only", "score_only"} else "cot"
    if mode == "score_only":
        mode = "label_only"
    text = (exp_name or "").lower()
    if "#ft#" in text or text.endswith("#ft") or "/ft#" in text:
        setting = "ft"
    elif "#base#" in text or text.endswith("#base"):
        setting = "base"
    elif adapter is None or str(adapter).strip().lower() in {"", "none"}:
        setting = "base"
    else:
        setting = "ft"
    return f"{setting}_{mode}"


def infer_eval_condition(
    *,
    exp_name: str,
    supervision_mode: str,
    adapter: str | None,
    train_config: str | None = None,
) -> str | None:
    """Map one eval run to a stable matrix code such as B-L / C-C / A→L."""
    text = (exp_name or "").lower()
    cfg = (train_config or "").lower().replace("\\", "/")
    is_base = (
        adapter is None
        or str(adapter).strip().lower() in {"", "none"}
        or "#base#" in text
    )

    if "#on_label_only" in text or "#on_label_only" in text:
        test = "label_only"
    elif "#on_cot" in text:
        test = "cot"
    else:
        test = "label_only" if supervision_mode in {"label_only", "label_only"} else "cot"

    if is_base:
        train = "B"
    elif "align" in text or "align.yaml" in cfg or "/align/" in cfg:
        train = "A"
    elif "#ft#label_only" in text or "#ft#label_only" in text or text.endswith(
        ("#label_only", "#label_only")
    ):
        train = "L"
    elif "#ft#cot" in text or text.endswith("#cot"):
        train = "C"
    elif "#ft#" in text:
        if "align" in text:
            train = "A"
        elif "label_only" in text or "label_only" in text:
            train = "L"
        elif "cot" in text:
            train = "C"
        else:
            return None
    else:
        return None

    mapping = {
        ("B", "label_only"): "B-L",
        ("B", "cot"): "B-C",
        ("L", "label_only"): "L-L",
        ("C", "cot"): "C-C",
        ("C", "label_only"): "C→L",
        ("L", "cot"): "L→C",
        ("A", "cot"): "A-C",
        ("A", "label_only"): "A→L",
    }
    return mapping.get((train, test))


def render_comparison_table(rows: list[dict[str, Any]]) -> str:
    """Plain GFM Markdown table (Cursor preview strips HTML styles).

    Only fine-tuned columns are marked:
      🟦 Label-only   🟧 cot
    Base columns stay unmarked.
    """
    metric_specs = (
        ("Acc", "acc", format_pct),
        ("F1", "f1", format_pct),
        ("samp/s", "samples_per_sec", lambda v: format_float(v, digits=1)),
    )

    def header_for(mode: str, suffix: str) -> str:
        label = COMPARISON_MODE_LABELS[mode]
        if mode == "ft_label_only":
            return f"🟦 {label} {suffix}"
        if mode == "ft_cot":
            return f"🟧 {label} {suffix}"
        return f"{label} {suffix}"

    def cell_for(mode: str, text: str) -> str:
        if not text:
            return ""
        if mode == "ft_label_only":
            return f"**🟦 {text}**"
        if mode == "ft_cot":
            return f"**🟧 {text}**"
        return text

    headers = ["Criterion"]
    for suffix, _field, _fmt in metric_specs:
        for mode in COMPARISON_MODES:
            headers.append(header_for(mode, suffix))
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "> **🟦 Label-only**　**🟧 cot**　｜　base 列不加标记\n\n",
        "| " + " | ".join(headers) + " |\n",
        "| " + " | ".join(align) + " |\n",
    ]
    for row in rows:
        cells = [str(row.get("criterion", ""))]
        for _suffix, field, fmt in metric_specs:
            for mode in COMPARISON_MODES:
                prefix = comparison_field_prefix(mode)
                cells.append(cell_for(mode, fmt(row.get(f"{prefix}_{field}"))))
        lines.append("| " + " | ".join(cells) + " |\n")
    return "".join(lines)


def _refresh_token_ratios(row: dict[str, Any]) -> None:
    base_label_tok = row.get("base_label_avg_output_tokens")
    base_cot_tok = row.get("base_cot_avg_output_tokens")
    ft_label_tok = row.get("ft_label_avg_output_tokens")
    ft_cot_tok = row.get("ft_cot_avg_output_tokens")
    if base_label_tok and base_cot_tok:
        row["token_ratio_base"] = base_cot_tok / base_label_tok
    if ft_label_tok and ft_cot_tok:
        row["token_ratio_ft"] = ft_cot_tok / ft_label_tok


def compute_efficiency_metrics(
    *,
    n_samples: int | None,
    avg_output_tokens: float | None,
    gpu_time_sec: float | None,
    samples_per_sec: float | None = None,
) -> dict[str, float | None]:
    """Derive throughput metrics from eval timing (prefer GPU generate time)."""
    sps = samples_per_sec
    if sps is None and n_samples and gpu_time_sec and gpu_time_sec > 0:
        sps = n_samples / gpu_time_sec
    tps = None
    if (
        n_samples
        and avg_output_tokens is not None
        and gpu_time_sec
        and gpu_time_sec > 0
    ):
        tps = (n_samples * avg_output_tokens) / gpu_time_sec
    return {
        "gpu_time_sec": gpu_time_sec,
        "samples_per_sec": sps,
        "tokens_per_sec": tps,
    }


def migrate_legacy_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    """Upgrade old score_*/cot_* fields into base_*/ft_* using source paths when possible."""
    out = dict(row)
    sources = out.get("sources") or {}
    # Prefer explicit new fields; only migrate when missing.
    for source_mode, legacy_prefix, new_mode in (
        ("score_only", "score", "label_only"),
        ("label_only", "label", "label_only"),
        ("cot", "cot", "cot"),
    ):
        source = str(sources.get(source_mode) or sources.get(new_mode) or "")
        if "#base#" in source.lower():
            target_mode = f"base_{new_mode}"
        elif "#ft#" in source.lower():
            target_mode = f"ft_{new_mode}"
        else:
            # Historical tables mostly stored fine-tuned runs under score/cot.
            target_mode = f"ft_{new_mode}"
        target_prefix = comparison_field_prefix(target_mode)
        for metric in ("acc", "f1", "avg_output_tokens"):
            legacy_key = f"{legacy_prefix}_{metric}"
            new_key = f"{target_prefix}_{metric}"
            if legacy_key in out and new_key not in out:
                out[new_key] = out[legacy_key]
        for mode_key in (source_mode, new_mode):
            if mode_key in sources and target_mode not in sources:
                sources = {**sources, target_mode: sources[mode_key]}
    out["sources"] = sources
    _refresh_token_ratios(out)
    return out


def merge_comparison_row(
    existing: dict[str, Any] | None,
    *,
    criterion: str,
    mode: str,
    accuracy: float,
    macro_f1: float,
    avg_output_tokens: float | None,
    gpu_time_sec: float | None = None,
    samples_per_sec: float | None = None,
    tokens_per_sec: float | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Merge one of the four eval settings into a criterion row."""
    if mode in {"label_only", "cot"}:
        # Backward-compatible callers: treat bare modes as fine-tuned.
        mode = f"ft_{mode}"
    if mode not in COMPARISON_MODES:
        raise ValueError(
            f"mode must be one of {COMPARISON_MODES} (or legacy label_only/cot), got {mode!r}"
        )
    row = migrate_legacy_comparison_row(dict(existing or {"criterion": criterion}))
    row["criterion"] = criterion
    prefix = comparison_field_prefix(mode)
    row[f"{prefix}_acc"] = accuracy
    row[f"{prefix}_f1"] = macro_f1
    row[f"{prefix}_avg_output_tokens"] = avg_output_tokens
    eff = compute_efficiency_metrics(
        n_samples=n_samples,
        avg_output_tokens=avg_output_tokens,
        gpu_time_sec=gpu_time_sec,
        samples_per_sec=samples_per_sec,
    )
    if tokens_per_sec is not None:
        eff["tokens_per_sec"] = tokens_per_sec
    if eff["gpu_time_sec"] is not None:
        row[f"{prefix}_gpu_time_sec"] = eff["gpu_time_sec"]
    if eff["samples_per_sec"] is not None:
        row[f"{prefix}_samples_per_sec"] = eff["samples_per_sec"]
    if eff["tokens_per_sec"] is not None:
        row[f"{prefix}_tokens_per_sec"] = eff["tokens_per_sec"]
    if n_samples is not None:
        row[f"{prefix}_n_samples"] = n_samples
    _refresh_token_ratios(row)
    return row
