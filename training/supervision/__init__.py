"""Supervision strategy registry for training."""

from __future__ import annotations

from typing import Any

from supervision.align import AlignStrategy
from supervision.standard import StandardStrategy

STRATEGIES = {
    "standard": StandardStrategy,
    "align": AlignStrategy,
}


def resolve_training_method(config: dict[str, Any], dataset_supervision_mode: str) -> str:
    supervision = config.get("supervision") or {}
    method = supervision.get("method")
    if method is None:
        return "standard"
    if method not in STRATEGIES:
        raise ValueError(
            f"Unknown supervision.method={method!r}; expected one of {sorted(STRATEGIES)}"
        )
    if method == "align" and dataset_supervision_mode != "cot":
        raise ValueError(
            "supervision.method=align requires a CoT training dataset with "
            "<reasoning> and <score> blocks."
        )
    return method


def get_supervision_strategy(method: str):
    return STRATEGIES[method]()
