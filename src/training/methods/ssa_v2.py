"""SSA v2 strategy with prompt-only attention for score prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from training.methods.interfaces import TrainingBuildContext
from training.methods.sft_config import build_sft_config_kwargs
from training.methods.ssa_v2_data import tokenize_ssa_v2_row
from training.trainers.ssa_v2_trainer import SsaV2GenerativeEvalSFTTrainer
from training.validation import CompactTrainLogCallback, JsonlLogCallback


@dataclass(frozen=True)
class SsaV2Coefficients:
    rationale: float
    score: float


def resolve_ssa_v2_coefficients(config: dict[str, Any]) -> SsaV2Coefficients:
    supervision = config.get("supervision") or {}
    rationale = float(supervision.get("rationale_coeff", 0.5))
    score = float(supervision.get("score_coeff", 0.5))
    if rationale < 0 or score < 0:
        raise ValueError("SSA v2 coefficients must be non-negative.")
    total = rationale + score
    if total <= 0:
        raise ValueError("SSA v2 coefficients must sum to a positive value.")
    return SsaV2Coefficients(
        rationale=rationale / total,
        score=score / total,
    )


def build_ssa_v2_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    include_branch_masks: bool = True,
) -> Dataset:
    return Dataset.from_list(
        [
            tokenize_ssa_v2_row(
                tokenizer,
                row,
                max_length=max_length,
                include_branch_masks=include_branch_masks,
            )
            for row in rows
        ]
    )


def build_ssa_v2_attention_mask(
    *,
    sequence_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    score_attention_starts: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Return [batch, 1, query, key] bool mask; True means attend."""
    indexes = torch.arange(max_length)
    query = indexes.view(1, max_length, 1)
    key = indexes.view(1, 1, max_length)
    sequence_lengths = sequence_lengths.view(-1, 1, 1)
    prompt_lengths = prompt_lengths.view(-1, 1, 1)
    score_attention_starts = score_attention_starts.view(-1, 1, 1)

    valid_query = query < sequence_lengths
    valid_key = key < sequence_lengths
    causal = key <= query
    rationale_query = query < score_attention_starts
    score_key = (key < prompt_lengths) | (key >= score_attention_starts)
    allowed = valid_query & valid_key & causal & (rationale_query | score_key)

    # Fully masked padding queries can produce NaNs in SDPA even though their
    # labels are ignored. Let them attend to the first prompt token.
    allowed = allowed | ((~valid_query) & (key == 0))
    return allowed.unsqueeze(1)


class SsaV2DataCollator:
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
        score_attention_starts = torch.tensor(
            [int(feature["score_attention_start"]) for feature in features],
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
        batch["attention_mask"] = build_ssa_v2_attention_mask(
            sequence_lengths=sequence_lengths,
            prompt_lengths=prompt_lengths,
            score_attention_starts=score_attention_starts,
            max_length=max_length,
        )
        return batch


class SsaV2Strategy:
    training_method = "ssa_v2"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> SsaV2GenerativeEvalSFTTrainer:
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        if str(training.get("attn_implementation", "sdpa")) != "sdpa":
            raise ValueError(
                "SSA v2 requires training.attn_implementation=sdpa for its "
                "custom boolean 4D attention mask."
            )

        max_length = int(training.get("max_length", 8192))
        coefficients = resolve_ssa_v2_coefficients(config)
        context.logger.info(
            "SSA v2 coefficients: rationale=%s score=%s; score context excludes "
            "rationale; <score> opening tag belongs to rationale loss",
            coefficients.rationale,
            coefficients.score,
        )

        train_dataset = build_ssa_v2_dataset(
            context.train_rows,
            tokenizer,
            max_length=max_length,
        )
        eval_dataset = build_ssa_v2_dataset(
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
        return SsaV2GenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=SsaV2DataCollator(tokenizer.pad_token_id),
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
