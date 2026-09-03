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

    adapter_path = (adapter or "").lower().replace("\\", "/")

    if is_scirm:
        train_mode = "SciRM"
    elif is_base:
        train_mode = "B"
    elif (
        "self_correct_align" in text
        or "self_correct_align.yaml" in config_path
        or "/self_correct_align/" in config_path
        or "/self_correct_align/" in adapter_path
    ):
        # Must precede paper_align / generic #ft# fallbacks: training still uses
        # supervision.method=paper_align, but exp/adapter paths differ.
        train_mode = "SCA"
    elif (
        "self_correct_cot" in text
        or "self_correct_cot.yaml" in config_path
        or "/self_correct_cot/" in config_path
        or "/self_correct_cot/" in adapter_path
    ):
        train_mode = "SC"
    elif (
        "paper_align_without_loss_balance" in text
        or "paper_align_without_loss_balance.yaml" in config_path
        or "/paper_align_without_loss_balance/" in config_path
        or "/paper_align_without_loss_balance/" in adapter_path
    ):
        train_mode = "MIX"
    elif (
        "paper_align" in text
        or "paper_align.yaml" in config_path
        or "/paper_align/" in config_path
        or "/paper_align/" in adapter_path
    ):
        train_mode = "PA"
    elif (
        "#ssa_v3#" in text
        or "ssa_v3.yaml" in config_path
        or "/ssa_v3/" in config_path
        or "/ssa_v3/" in adapter_path
    ):
        train_mode = "SSA3"
    elif (
        "#ssa_v2#" in text
        or "ssa_v2.yaml" in config_path
        or "/ssa_v2/" in config_path
        or "/ssa_v2/" in adapter_path
    ):
        train_mode = "SSA2"
    elif (
        "#ssa#" in text
        or "ssa.yaml" in config_path
        or "/ssa/" in config_path
        or "/ssa/" in adapter_path
    ):
        train_mode = "SSA"
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
        ("MIX", "label_only"): "MIX-L",
        ("MIX", "cot"): "MIX-C",
        ("SSA", "label_only"): "SSAL",
        ("SSA", "cot"): "SSAC",
        ("SSA2", "label_only"): "SSA2L",
        ("SSA2", "cot"): "SSA2C",
        ("SSA3", "label_only"): "SSA3L",
        ("SSA3", "cot"): "SSA3C",
        ("SC", "label_only"): "SCL",
        ("SC", "cot"): "SCC",
        ("SCA", "label_only"): "SCAL",
        ("SCA", "cot"): "SCAC",
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
