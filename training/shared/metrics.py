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
