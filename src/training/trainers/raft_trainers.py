"""RAFT-without-CoT loss and matching RAIL validation."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from training.validation import GenerativeEvalSFTTrainer
from utils.metrics import classification_metrics, token_stats


def resolve_score_token_ids(tokenizer: Any, score_values: list[int]) -> list[int]:
    """Resolve numeric score tokens instead of hard-coding model-specific IDs."""
    token_ids: list[int] = []
    for score in score_values:
        encoded = tokenizer.encode(str(score), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"RAFT requires score {score!r} to be exactly one tokenizer token; "
                f"got token IDs {encoded}."
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            f"RAFT score tokens must be unique; scores={score_values}, token_ids={token_ids}"
        )
    return token_ids


def raft_score_statistics(
    score_logits: torch.Tensor,
    score_token_ids: list[int],
    score_values: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the RAIL expected score and each score token's full-vocab probability."""
    probabilities = torch.softmax(score_logits.float(), dim=-1)
    token_ids = torch.tensor(score_token_ids, device=score_logits.device, dtype=torch.long)
    values = torch.tensor(score_values, device=score_logits.device, dtype=torch.float32)
    score_probabilities = probabilities.index_select(-1, token_ids)
    expected_scores = (score_probabilities * values).sum(dim=-1)
    return expected_scores, score_probabilities


def locate_score_targets(
    labels: torch.Tensor,
    score_token_ids: list[int],
    score_values: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate the one supervised numeric score token in every completion."""
    candidate_ids = torch.tensor(
        score_token_ids, device=labels.device, dtype=labels.dtype
    )
    matches = labels.unsqueeze(-1).eq(candidate_ids)
    matches_per_row = matches.sum(dim=(1, 2))
    if not torch.all(matches_per_row == 1):
        counts = matches_per_row.detach().cpu().tolist()
        raise RuntimeError(
            "RAFT requires exactly one valid numeric score token in each unmasked "
            f"completion; found counts={counts}. The completion may have been truncated."
        )

    locations = matches.nonzero(as_tuple=False)
    score_positions = locations[:, 1]
    if torch.any(score_positions == 0):
        raise RuntimeError("RAFT score token cannot be the first token in a sequence.")
    values = torch.tensor(score_values, device=labels.device, dtype=torch.float32)
    targets = values.index_select(0, locations[:, 2])
    return score_positions, targets


@torch.inference_mode()
def rail_validation(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    score_values: list[int],
    score_token_ids: list[int],
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate by RAIL expected scores at the token after the <score> prefix."""
    was_training = model.training
    original_padding_side = tokenizer.padding_side
    device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    inputs = None

    try:
        model.eval()
        tokenizer.padding_side = "left"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if logger is not None:
            logger.info(
                "Starting RAIL validation on %s: samples=%d batch_size=%d",
                device,
                len(rows),
                batch_size,
            )

        total_batches = math.ceil(len(rows) / batch_size)
        progress_interval = max(1, total_batches // 20)
        for batch_index, start in enumerate(
            range(0, len(rows), batch_size), start=1
        ):
            batch = rows[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [*row["prompt"], {"role": "assistant", "content": "<score>"}],
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            score_logits = model(
                **inputs, use_cache=False, logits_to_keep=1
            ).logits[:, -1, :]
            expected_scores, score_probabilities = raft_score_statistics(
                score_logits, score_token_ids, score_values
            )
            score_grid = torch.tensor(
                score_values, device=device, dtype=expected_scores.dtype
            )
            nearest_indices = torch.abs(
                expected_scores.unsqueeze(-1) - score_grid
            ).argmin(dim=-1)

            for row_index, row in enumerate(batch):
                prediction = score_values[int(nearest_indices[row_index].item())]
                expected_score = float(expected_scores[row_index].item())
                per_score = score_probabilities[row_index]
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": prediction,
                        "correct": prediction == row["label"],
                        "output": f"<score>{prediction}</score>",
                        "expected_score": expected_score,
                        "score_probabilities": {
                            str(score): float(probability.item())
                            for score, probability in zip(
                                score_values, per_score, strict=True
                            )
                        },
                        "score_probability_mass": float(per_score.sum().item()),
                    }
                )
            if logger is not None and (
                batch_index % progress_interval == 0 or batch_index == total_batches
            ):
                logger.info(
                    "RAIL validation progress: %d/%d",
                    min(start + batch_size, len(rows)),
                    len(rows),
                )

        metrics = classification_metrics(predictions, score_values)
        if predictions:
            metrics["rail_mse"] = sum(
                (row["expected_score"] - row["label"]) ** 2 for row in predictions
            ) / len(predictions)
            metrics["rail_mae"] = sum(
                abs(row["expected_score"] - row["label"]) for row in predictions
            ) / len(predictions)
            metrics["avg_score_probability_mass"] = sum(
                row["score_probability_mass"] for row in predictions
            ) / len(predictions)
        metrics["tokens"] = token_stats(predictions, tokenizer)
        return metrics, predictions
    finally:
        inputs = None
        tokenizer.padding_side = original_padding_side
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()


class RaftWithoutCotTrainer(GenerativeEvalSFTTrainer):
    """Optimize only RAFT MSE and validate using the corresponding RAIL estimator."""

    def __init__(self, *args, score_token_ids: list[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_token_ids = list(score_token_ids)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]
        score_positions, targets = locate_score_targets(
            labels, self.score_token_ids, self.score_sets
        )
        logit_positions = score_positions - 1
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs, logits_to_keep=logit_positions)
        batch_indices = torch.arange(labels.size(0), device=labels.device)
        # Tensor indices are shared across rows, so select the row-wise diagonal.
        score_logits = outputs.logits[batch_indices, batch_indices, :]
        expected_scores, _ = raft_score_statistics(
            score_logits, self.score_token_ids, self.score_sets
        )
        loss = F.mse_loss(expected_scores, targets, reduction="mean")
        return (loss, outputs) if return_outputs else loss

    def _run_generation_validation(self):
        return rail_validation(
            self.model,
            self.processing_class,
            self.validation_rows,
            batch_size=self.generation_batch_size,
            max_length=self.generation_max_length,
            score_values=self.score_sets,
            score_token_ids=self.score_token_ids,
            logger=self.logger,
        )


def locate_cot_score_targets(
    labels: torch.Tensor,
    score_mask: torch.Tensor,
    score_token_ids: list[int],
    score_values: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one structurally marked score token from every CoT completion."""
    if labels.shape != score_mask.shape:
        raise RuntimeError(
            "CoT-RAFT labels and score mask must have the same shape; "
            f"got labels={tuple(labels.shape)}, mask={tuple(score_mask.shape)}."
        )
    marked = score_mask.bool()
    counts = marked.sum(dim=1)
    if not torch.all(counts == 1):
        raise RuntimeError(
            "CoT-RAFT requires exactly one structurally marked score token per "
            f"completion; found counts={counts.detach().cpu().tolist()}."
        )

    locations = marked.nonzero(as_tuple=False)
    score_positions = locations[:, 1]
    if torch.any(score_positions == 0):
        raise RuntimeError("CoT-RAFT score token cannot be the first sequence token.")

    observed_ids = labels[locations[:, 0], score_positions]
    candidate_ids = torch.tensor(
        score_token_ids, device=labels.device, dtype=labels.dtype
    )
    matches = observed_ids.unsqueeze(-1).eq(candidate_ids)
    if not torch.all(matches.sum(dim=1) == 1):
        raise RuntimeError(
            "CoT-RAFT score mask does not point to a legal numeric score token; "
            f"observed token IDs={observed_ids.detach().cpu().tolist()}."
        )
    values = torch.tensor(score_values, device=labels.device, dtype=torch.float32)
    targets = values.index_select(0, matches.to(torch.long).argmax(dim=1))
    return score_positions, targets


class CotRaftTrainer(GenerativeEvalSFTTrainer):
    """Train CoT text with CE and its numeric score with RAFT regression."""

    def __init__(
        self,
        *args,
        score_token_ids: list[int],
        raft_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.score_token_ids = list(score_token_ids)
        self.raft_weight = float(raft_weight)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]
        score_mask = inputs["raft_score_mask"]
        score_positions, targets = locate_cot_score_targets(
            labels,
            score_mask,
            self.score_token_ids,
            self.score_sets,
        )

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"labels", "raft_score_mask"}
        }
        outputs = model(**model_inputs)
        logits = outputs.logits

        lm_labels = labels.masked_fill(score_mask.bool(), -100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        batch_indices = torch.arange(labels.size(0), device=labels.device)
        score_logits = logits[batch_indices, score_positions - 1, :]
        expected_scores, _ = raft_score_statistics(
            score_logits,
            self.score_token_ids,
            self.score_sets,
        )
        score_loss = F.mse_loss(expected_scores, targets, reduction="mean")
        loss = lm_loss + self.raft_weight * score_loss

        if model.training:
            self.log(
                {
                    "cot_raft_lm_loss": float(lm_loss.detach().item()),
                    "cot_raft_score_loss": float(score_loss.detach().item()),
                }
            )
        return (loss, outputs) if return_outputs else loss
