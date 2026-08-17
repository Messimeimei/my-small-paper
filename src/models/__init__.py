"""Model adapters and pluggable inference backends."""

from models.base import ModelBackend
from models.registry import (
    BackendSpec,
    available_model_backends,
    get_model_backend,
    register_model_backend,
)

__all__ = [
    "BackendSpec",
    "ModelBackend",
    "available_model_backends",
    "get_model_backend",
    "register_model_backend",
]
