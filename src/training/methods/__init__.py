"""Public API for pluggable training methods."""

from __future__ import annotations

from training.methods.registry import (
    StrategySpec,
    available_training_methods,
    get_supervision_strategy,
    register_supervision_strategy,
    resolve_training_method,
)

__all__ = [
    "StrategySpec",
    "available_training_methods",
    "get_supervision_strategy",
    "register_supervision_strategy",
    "resolve_training_method",
]
