"""Stable public API for extending evaluation inference methods."""

from evaluation.inference.base import InferenceMethod, InferenceRuntime
from evaluation.inference.registry import (
    available_inference_modes,
    get_inference_method,
    register_inference_method,
)

__all__ = [
    "InferenceMethod",
    "InferenceRuntime",
    "available_inference_modes",
    "get_inference_method",
    "register_inference_method",
]
