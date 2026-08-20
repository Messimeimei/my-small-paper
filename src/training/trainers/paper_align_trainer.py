"""Trainer for the paper-faithful paired-view Align objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.validation import GenerativeEvalSFTTrainer


class PaperAlignGenerativeEvalSFTTrainer(GenerativeEvalSFTTrainer):
    """Average Direct and Reason view losses separately, then combine them."""

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

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if not model.training or "label_loss_mask" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        label_loss_mask = inputs.pop("label_loss_mask")
        rationale_loss_mask = inputs.pop("rationale_loss_mask")
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
        if label_tokens.numel() == 0 or rationale_tokens.numel() == 0:
            raise RuntimeError(
                "paper_align requires every micro-batch to contain both paired views"
            )

        label_loss = label_tokens.mean()
        rationale_loss = rationale_tokens.mean()
        loss = self.label_coeff * label_loss + self.rationale_coeff * rationale_loss

        self.log(
            {
                "paper_align_label_loss": float(label_loss.item()),
                "paper_align_reason_loss": float(rationale_loss.item()),
            }
        )
        return (loss, outputs) if return_outputs else loss


class PaperAlignWithoutLossBalanceGenerativeEvalSFTTrainer(
    GenerativeEvalSFTTrainer
):
    """Apply ordinary token-averaged CE to the paired Direct/Reason batch."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if not model.training or "label_loss_mask" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        label_loss_mask = inputs.pop("label_loss_mask")
        rationale_loss_mask = inputs.pop("rationale_loss_mask")
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
        if label_tokens.numel() == 0 or rationale_tokens.numel() == 0:
            raise RuntimeError(
                "paper_align_without_loss_balance requires every micro-batch "
                "to contain both paired views"
            )

        # This is the matched Mix control: unlike Paper Align, Direct and
        # Reason tokens share one denominator, so longer Reason sequences
        # naturally contribute more to the gradient.
        loss = token_loss[supervised].mean()
        supervised_count = label_tokens.numel() + rationale_tokens.numel()
        self.log(
            {
                "mix_token_average_loss": float(loss.item()),
                "mix_label_token_loss": float(label_tokens.mean().item()),
                "mix_reason_token_loss": float(rationale_tokens.mean().item()),
                "mix_label_token_share": label_tokens.numel() / supervised_count,
                "mix_reason_token_share": rationale_tokens.numel()
                / supervised_count,
            }
        )
        return (loss, outputs) if return_outputs else loss
