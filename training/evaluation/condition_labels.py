"""Evaluation-condition naming shared by runners and reports."""

from __future__ import annotations


def infer_eval_condition(
    *,
    exp_name: str,
    supervision_mode: str,
    adapter: str | None,
    train_config: str | None = None,
) -> str | None:
    """Map one evaluation run to a stable matrix code such as B-L or PAC."""
    text = (exp_name or "").lower()
    config_path = (train_config or "").lower().replace("\\", "/")
    is_scirm = "scirm" in text or "scirm" in config_path
    is_base = (
        adapter is None
        or str(adapter).strip().lower() in {"", "none"}
        or "#base#" in text
    )

    if "#on_label_only" in text:
        test_mode = "label_only"
    elif "#on_cot" in text:
        test_mode = "cot"
    else:
        test_mode = "label_only" if supervision_mode == "label_only" else "cot"

    if is_scirm:
        train_mode = "SciRM"
    elif is_base:
        train_mode = "B"
    elif (
        "paper_align" in text
        or "paper_align.yaml" in config_path
        or "/paper_align/" in config_path
    ):
        train_mode = "PA"
    elif (
        "legacy_align" in text
        or "legacy_align.yaml" in config_path
        or "/legacy_align/" in config_path
        or "#ft#align" in text
    ):
        train_mode = "A"
    elif "#ft#label_only" in text or text.endswith("#label_only"):
        train_mode = "L"
    elif "#ft#cot" in text or text.endswith("#cot"):
        train_mode = "C"
    elif "#ft#" in text:
        if "label_only" in text:
            train_mode = "L"
        elif "cot" in text:
            train_mode = "C"
        else:
            return None
    else:
        return None

    return {
        ("B", "label_only"): "B-L",
        ("B", "cot"): "B-C",
        ("SciRM", "label_only"): "SciRM-L",
        ("SciRM", "cot"): "SciRM-C",
        ("L", "label_only"): "LL",
        ("L", "cot"): "LC",
        ("C", "label_only"): "CL",
        ("C", "cot"): "CC",
        ("A", "label_only"): "AL",
        ("A", "cot"): "AC",
        ("PA", "label_only"): "PAL",
        ("PA", "cot"): "PAC",
    }.get((train_mode, test_mode))


def resolve_eval_condition(
    *,
    exp_name: str,
    supervision_mode: str,
    adapter: str | None,
    train_config: str | None,
    training_method: str | None,
    inference_mode: str,
) -> str | None:
    """Apply inference-method variants to the base train/test condition."""
    condition = infer_eval_condition(
        exp_name=exp_name,
        supervision_mode=supervision_mode,
        adapter=adapter,
        train_config=train_config,
    )
    text = exp_name.lower()
    is_raft_without_cot = (
        training_method == "raft_without_cot" or "raft_without_cot" in text
    )
    is_cot_raft = training_method == "cot_raft" or "cot_raft" in text
    if is_cot_raft:
        return "COT-RAFT-R" if inference_mode == "cot_rail" else "COT-RAFT-G"
    if is_raft_without_cot:
        return "RAFT-R" if inference_mode == "rail" else "RAFT-G"
    if inference_mode in {"rail", "cot_rail"}:
        return f"{condition}-R" if condition else "RAIL"
    return condition
