"""Trainer for the Single Sample Align objective."""

from __future__ import annotations

import torch.nn.functional as F

from training.validation import GenerativeEvalSFTTrainer


class SsaGenerativeEvalSFTTrainer(GenerativeEvalSFTTrainer):
    """Normalize rationale and score token losses separately, then combine."""

    def __init__(
        self,
        *args,
        rationale_coeff: float = 0.5,
        score_coeff: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rationale_coeff = float(rationale_coeff)
        self.score_coeff = float(score_coeff)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if "rationale_loss_mask" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        rationale_loss_mask = inputs.pop("rationale_loss_mask")
        score_loss_mask = inputs.pop("score_loss_mask")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_rationale_mask = rationale_loss_mask[..., 1:].contiguous().bool()
        shift_score_mask = score_loss_mask[..., 1:].contiguous().bool()

        vocab_size = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        supervised = shift_labels != -100
        rationale_tokens = token_loss[shift_rationale_mask & supervised]
        score_tokens = token_loss[shift_score_mask & supervised]
        if rationale_tokens.numel() == 0 or score_tokens.numel() == 0:
            raise RuntimeError(
                "SSA requires every micro-batch to contain rationale and score tokens."
            )

        rationale_loss = rationale_tokens.mean()
        score_loss = score_tokens.mean()
        loss = (
            self.rationale_coeff * rationale_loss
            + self.score_coeff * score_loss
        )
        self.log(
            {
                "ssa_rationale_loss": float(rationale_loss.item()),
                "ssa_score_loss": float(score_loss.item()),
            }
        )
        return (loss, outputs) if return_outputs else loss
