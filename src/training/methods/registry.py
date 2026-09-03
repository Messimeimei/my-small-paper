"""Lazy registry for pluggable training methods."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategySpec:
    name: str
    target: str
    dataset_modes: frozenset[str] | None = None

    def load(self):
        module_name, class_name = self.target.rsplit(":", 1)
        strategy_class = getattr(importlib.import_module(module_name), class_name)
        strategy = strategy_class()
        if getattr(strategy, "training_method", None) != self.name:
            raise RuntimeError(
                f"Strategy {self.target} declares training_method="
                f"{getattr(strategy, 'training_method', None)!r}, expected {self.name!r}."
            )
        return strategy


_STRATEGIES: dict[str, StrategySpec] = {}


def register_supervision_strategy(
    spec: StrategySpec,
    *,
    replace: bool = False,
) -> None:
    if not spec.name:
        raise ValueError("A supervision strategy requires a non-empty name.")
    if spec.name in _STRATEGIES and not replace:
        raise ValueError(f"Supervision strategy already registered: {spec.name}")
    _STRATEGIES[spec.name] = spec


def available_training_methods() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def resolve_training_method(
    config: dict[str, Any],
    dataset_supervision_mode: str,
) -> str:
    supervision = config.get("supervision") or {}
    method = str(supervision.get("method") or "standard")
    spec = _STRATEGIES.get(method)
    if spec is None:
        raise ValueError(
            f"Unknown supervision.method={method!r}; "
            f"expected one of {list(available_training_methods())}"
        )
    if (
        spec.dataset_modes is not None
        and dataset_supervision_mode not in spec.dataset_modes
    ):
        expected = ", ".join(sorted(spec.dataset_modes))
        raise ValueError(
            f"supervision.method={method} requires dataset supervision mode "
            f"in {{{expected}}}, got {dataset_supervision_mode!r}."
        )
    return method


def get_supervision_strategy(method: str):
    try:
        spec = _STRATEGIES[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training method {method!r}; "
            f"expected one of {list(available_training_methods())}"
        ) from exc
    return spec.load()


for _spec in (
    StrategySpec("standard", "training.methods.standard_sft:StandardStrategy"),
    StrategySpec(
        "paper_align",
        "training.methods.paper_align:PaperAlignStrategy",
        frozenset({"cot"}),
    ),
    StrategySpec(
        "paper_align_without_loss_balance",
        "training.methods.paper_align_without_loss_balance:PaperAlignWithoutLossBalanceStrategy",
        frozenset({"cot"}),
    ),
    StrategySpec(
        "ssa",
        "training.methods.ssa:SsaStrategy",
        frozenset({"cot"}),
    ),
    StrategySpec(
        "ssa_v2",
        "training.methods.ssa_v2:SsaV2Strategy",
        frozenset({"cot"}),
    ),
    StrategySpec(
        "ssa_v3",
        "training.methods.ssa_v3:SsaV3Strategy",
        frozenset({"cot"}),
    ),
    StrategySpec(
        "raft_without_cot",
        "training.methods.raft_without_cot:RaftWithoutCotStrategy",
        frozenset({"label_only"}),
    ),
    StrategySpec(
        "cot_raft",
        "training.methods.cot_raft:CotRaftStrategy",
        frozenset({"cot"}),
    ),
):
    register_supervision_strategy(_spec)

del _spec
