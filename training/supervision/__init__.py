"""注册不同的训练方法，如标准 sft、对齐训练、RAFT 等。"""

from __future__ import annotations

from typing import Any

from supervision.align import LegacyAlignStrategy
from supervision.cot_raft import CotRaftStrategy
from supervision.paper_align import PaperAlignStrategy
from supervision.raft_without_cot import RaftWithoutCotStrategy
from supervision.standard import StandardStrategy

STRATEGIES = {
    "standard": StandardStrategy,
    "legacy_align": LegacyAlignStrategy,
    "paper_align": PaperAlignStrategy,
    "raft_without_cot": RaftWithoutCotStrategy,
    "cot_raft": CotRaftStrategy,
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
        method in {"legacy_align", "paper_align", "cot_raft"}
        and dataset_supervision_mode != "cot"
    ):
        raise ValueError(
            f"supervision.method={method} requires a CoT training dataset with "
            "<reasoning> and <score> blocks."
        )
    if method == "raft_without_cot" and dataset_supervision_mode != "label_only":
        raise ValueError(
            "supervision.method=raft_without_cot requires a label-only training dataset."
        )
    return method


def get_supervision_strategy(method: str):
    return STRATEGIES[method]()
