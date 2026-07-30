"""Shared supervision strategy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from generative_trainer import GenerativeEvalSFTTrainer


@dataclass
class TrainingBuildContext:
    config: dict[str, Any]
    seed: int
    run_id: str
    run_directory: Any
    run_tag: str
    model_path: Any
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    labels: list[int]
    logger: Any
    manifest: dict[str, Any]


class SupervisionStrategy(Protocol):
    training_method: str

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config,
        context: TrainingBuildContext,
    ) -> GenerativeEvalSFTTrainer: ...
