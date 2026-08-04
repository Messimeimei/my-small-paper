"""Trainer for the legacy unpaired split-view Align implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from trl import SFTTrainer

from training_workflow.generation_validation import GenerativeEvalSFTTrainer


class LegacyAlignGenerativeEvalSFTTrainer(GenerativeEvalSFTTrainer):
    """Legacy trainer retained so historical training behavior stays reproducible."""

    def __init__(
        self,
        *args,
        label_coeff: float = 0.5,
        rationale_coeff: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label_coeff = float(label_coeff)
        self.rationale_coeff = float(rationale_coeff)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training or "label_loss_mask" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        label_loss_mask = inputs.pop("label_loss_mask")
        rationale_loss_mask = inputs.pop("rationale_loss_mask")
        # Align computes its own token-level objective below. Do not pass labels to
        # the model: TRL's chunked_nll forward intentionally returns logits=None
        # when labels are present, and the native LM loss would be redundant here.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_label_mask = label_loss_mask[..., 1:].contiguous().bool()
        shift_rationale_mask = rationale_loss_mask[..., 1:].contiguous().bool()

        vocab = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        supervised = shift_labels != -100
        label_tokens = token_loss[shift_label_mask & supervised]
        rationale_tokens = token_loss[shift_rationale_mask & supervised]

        label_loss = (
            label_tokens.mean()
            if label_tokens.numel() > 0
            else token_loss.new_zeros(())
        )
        rationale_loss = (
            rationale_tokens.mean()
            if rationale_tokens.numel() > 0
            else token_loss.new_zeros(())
        )

        if label_tokens.numel() == 0 and rationale_tokens.numel() == 0:
            loss = token_loss[supervised].mean()
        else:
            loss = self.label_coeff * label_loss + self.rationale_coeff * rationale_loss

        self.log(
            {
                "legacy_align_label_loss": float(label_loss.item()),
                "legacy_align_rationale_loss": float(rationale_loss.item()),
            }
        )

        return (loss, outputs) if return_outputs else loss
