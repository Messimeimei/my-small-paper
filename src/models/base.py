"""Stable interface implemented by evaluation model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelBackend(ABC):
    name: str

    @abstractmethod
    def initialize(
        self,
        model_path: Path,
        *,
        max_model_len: int,
        max_tokens: int,
        seed: int,
        gpu_memory_utilization: float,
    ) -> tuple[Any, Any]:
        """Load a model runtime and its default sampling parameters."""

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Seed all random number generators used by the backend."""

    @abstractmethod
    def gpu_snapshot(self) -> dict[str, Any]:
        """Return lightweight runtime device metadata."""
