"""SSA v3 strategy with prompt-only attention for every assistant region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from training.methods.interfaces import TrainingBuildContext
from training.methods.sft_config import build_sft_config_kwargs
from training.methods.ssa_v3_data import tokenize_ssa_v3_row
from training.trainers.ssa_v3_trainer import SsaV3GenerativeEvalSFTTrainer
from training.validation import CompactTrainLogCallback, JsonlLogCallback


@dataclass(frozen=True)
class SsaV3Coefficients:
    rationale: float
    score: float


def resolve_ssa_v3_coefficients(config: dict[str, Any]) -> SsaV3Coefficients:
    supervision = config.get("supervision") or {}
    rationale = float(supervision.get("rationale_coeff", 0.5))
    score = float(supervision.get("score_coeff", 0.5))
    if rationale < 0 or score < 0:
        raise ValueError("SSA v3 coefficients must be non-negative.")
    total = rationale + score
    if total <= 0:
        raise ValueError("SSA v3 coefficients must sum to a positive value.")
    return SsaV3Coefficients(
        rationale=rationale / total,
        score=score / total,
    )


def build_ssa_v3_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    include_branch_masks: bool = True,
) -> Dataset:
    return Dataset.from_list(
        [
            tokenize_ssa_v3_row(
                tokenizer,
                row,
                max_length=max_length,
                include_branch_masks=include_branch_masks,
            )
            for row in rows
        ]
    )


def build_ssa_v3_attention_mask(
    *,
    sequence_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Allow assistant queries to attend only to formatted prompt keys."""
    indexes = torch.arange(max_length)
    query = indexes.view(1, max_length, 1)
    key = indexes.view(1, 1, max_length)
    sequence_lengths = sequence_lengths.view(-1, 1, 1)
    prompt_lengths = prompt_lengths.view(-1, 1, 1)
    valid_query = query < sequence_lengths
    valid_key = key < sequence_lengths
    causal = key <= query
    prompt_query = query < prompt_lengths
    prompt_key = key < prompt_lengths
    allowed = valid_query & valid_key & causal & (prompt_query | prompt_key)

    # Query-token embeddings still flow through residual connections; this mask
    # prevents attention to generated assistant keys.
    # Fully masked padding queries can produce NaNs in SDPA even though their
    # labels are ignored. Let them attend to the first prompt token.
    allowed = allowed | ((~valid_query) & (key == 0))
    return allowed.unsqueeze(1)


class SsaV3DataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    @staticmethod
    def _pad(
        values: list[int],
        *,
        max_length: int,
        pad_value: int,
    ) -> list[int]:
        return list(values) + [pad_value] * (max_length - len(values))

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        training_batch = "rationale_loss_mask" in features[0]
        common_keys = ("input_ids", "attention_mask", "labels")

        if not training_batch:
            pad_values = {
                "input_ids": self.pad_token_id,
                "attention_mask": 0,
                "labels": -100,
            }
            return {
                key: torch.tensor(
                    [
                        self._pad(
                            feature[key],
                            max_length=max_length,
                            pad_value=pad_values[key],
                        )
                        for feature in features
                    ],
                    dtype=torch.long,
                )
                for key in common_keys
            }

        sequence_lengths = torch.tensor(
            [len(feature["input_ids"]) for feature in features],
            dtype=torch.long,
        )
        prompt_lengths = torch.tensor(
            [int(feature["prompt_length"]) for feature in features],
            dtype=torch.long,
        )
        pad_values = {
            "input_ids": self.pad_token_id,
            "labels": -100,
            "position_ids": 0,
            "rationale_loss_mask": 0,
            "score_loss_mask": 0,
            "score_value_loss_mask": 0,
        }
        keys = (
            "input_ids",
            "labels",
            "position_ids",
            "rationale_loss_mask",
            "score_loss_mask",
            "score_value_loss_mask",
        )
        batch = {
            key: torch.tensor(
                [
                    self._pad(
                        feature[key],
                        max_length=max_length,
                        pad_value=pad_values[key],
                    )
                    for feature in features
                ],
                dtype=torch.long,
            )
            for key in keys
        }
        batch["attention_mask"] = build_ssa_v3_attention_mask(
            sequence_lengths=sequence_lengths,
            prompt_lengths=prompt_lengths,
            max_length=max_length,
        )
        return batch


class SsaV3Strategy:
    training_method = "ssa_v3"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> SsaV3GenerativeEvalSFTTrainer:
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        if str(training.get("attn_implementation", "sdpa")) != "sdpa":
            raise ValueError(
                "SSA v3 requires training.attn_implementation=sdpa for its "
                "custom boolean 4D attention mask."
            )

        max_length = int(training.get("max_length", 8192))
        coefficients = resolve_ssa_v3_coefficients(config)
        context.logger.info(
            "SSA v3 coefficients: rationale=%s score=%s; all assistant attention "
            "is prompt-only; complete <score> block belongs to score loss",
            coefficients.rationale,
            coefficients.score,
        )

        train_dataset = build_ssa_v3_dataset(
            context.train_rows,
            tokenizer,
            max_length=max_length,
        )
        eval_dataset = build_ssa_v3_dataset(
            context.validation_rows,
            tokenizer,
            max_length=max_length,
        )
        sft_config = SFTConfig(
            **build_sft_config_kwargs(
                context,
                max_length=max_length,
                pretokenized=True,
            )
        )
        return SsaV3GenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=SsaV3DataCollator(tokenizer.pad_token_id),
            rationale_coeff=coefficients.rationale,
            score_coeff=coefficients.score,
            validation_rows=context.validation_rows,
            score_sets=context.labels,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=max_length,
            generation_max_new_tokens=int(
                generation.get("max_new_tokens", 512)
            ),
            run_directory=context.run_directory,
            logger=context.logger,
            callbacks=[
                JsonlLogCallback(
                    context.run_directory / "train_history.jsonl",
                    run_tag=context.run_tag,
                ),
                CompactTrainLogCallback(
                    run_tag=context.run_tag,
                    logger=context.logger,
                ),
            ],
        )
