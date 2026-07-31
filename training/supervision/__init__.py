"""Supervision strategy registry for training."""

from __future__ import annotations

from typing import Any

from supervision.align import LegacyAlignStrategy
from supervision.paper_align import PaperAlignStrategy
from supervision.standard import StandardStrategy

STRATEGIES = {
    "standard": StandardStrategy,
    "legacy_align": LegacyAlignStrategy,
    "paper_align": PaperAlignStrategy,
}


def resolve_training_method(
    config: dict[str, Any],
    dataset_supervision_mode: str,
) -> str:
    supervision = config.get("supervision") or {}
    method = supervision.get("method")
    if method is None:
        return "standard"
    if method not in STRATEGIES:
        raise ValueError(
            f"Unknown supervision.method={method!r}; expected one of {sorted(STRATEGIES)}"
        )
    if (
        method in {"legacy_align", "paper_align"}
        and dataset_supervision_mode != "cot"
    ):
        raise ValueError(
            f"supervision.method={method} requires a CoT training dataset with "
            "<reasoning> and <score> blocks."
        )
    return method


def get_supervision_strategy(method: str):
    return STRATEGIES[method]()
