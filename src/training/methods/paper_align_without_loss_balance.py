"""Matched Mix ablation for Paper Align without view-wise loss balancing."""

from __future__ import annotations

from typing import Any

from training.methods.paper_align import PaperAlignStrategy
from training.trainers.paper_align_trainer import (
    PaperAlignWithoutLossBalanceGenerativeEvalSFTTrainer,
)


class PaperAlignWithoutLossBalanceStrategy(PaperAlignStrategy):
    """Reuse paired views while averaging all supervised tokens jointly."""

    training_method = "paper_align_without_loss_balance"
    trainer_class = PaperAlignWithoutLossBalanceGenerativeEvalSFTTrainer

    def trainer_kwargs(self, config: dict[str, Any]) -> dict[str, float]:
        return {}
