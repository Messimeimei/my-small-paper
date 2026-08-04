"""Stable public API for extending evaluation inference methods."""

from evaluation.methods.base import EvaluationMethod, MethodRuntime
from evaluation.methods.registry import (
    available_inference_modes,
    get_evaluation_method,
    register_evaluation_method,
)

__all__ = [
    "EvaluationMethod",
    "MethodRuntime",
    "available_inference_modes",
    "get_evaluation_method",
    "register_evaluation_method",
]
