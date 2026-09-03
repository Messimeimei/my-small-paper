"""Single Sample Align supervision with rationale/score region balancing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from training.methods.interfaces import TrainingBuildContext
from training.methods.sft_config import build_sft_config_kwargs
from training.methods.ssa_data import tokenize_ssa_row
from training.trainers.ssa_trainer import SsaGenerativeEvalSFTTrainer
from training.validation import CompactTrainLogCallback, JsonlLogCallback


@dataclass(frozen=True)
class SsaCoefficients:
    rationale: float
    score: float


def resolve_ssa_coefficients(config: dict[str, Any]) -> SsaCoefficients:
    supervision = config.get("supervision") or {}
    rationale = float(supervision.get("rationale_coeff", 0.5))
    score = float(supervision.get("score_coeff", 0.5))
    if rationale < 0 or score < 0:
        raise ValueError("SSA coefficients must be non-negative.")
    total = rationale + score
    if total <= 0:
        raise ValueError("SSA coefficients must sum to a positive value.")
    return SsaCoefficients(rationale=rationale / total, score=score / total)


def build_ssa_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    include_region_masks: bool = True,
) -> Dataset:
    return Dataset.from_list(
        [
            tokenize_ssa_row(
                tokenizer,
                row,
                max_length=max_length,
                include_region_masks=include_region_masks,
            )
            for row in rows
        ]
    )


class SsaDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        keys = ["input_ids", "attention_mask", "labels"]
        if "rationale_loss_mask" in features[0]:
            keys.extend(("rationale_loss_mask", "score_loss_mask"))
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {key: [] for key in keys}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            for key in keys:
                pad_value = (
                    self.pad_token_id
                    if key == "input_ids"
                    else -100
                    if key == "labels"
                    else 0
                )
                batch[key].append(feature[key] + [pad_value] * pad_len)
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


class SsaStrategy:
    training_method = "ssa"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> SsaGenerativeEvalSFTTrainer:
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        max_length = int(training.get("max_length", 8192))
        coefficients = resolve_ssa_coefficients(config)
        context.logger.info(
            "SSA loss coefficients: rationale=%s score=%s",
            coefficients.rationale,
            coefficients.score,
        )
        train_dataset = build_ssa_dataset(
            context.train_rows,
            tokenizer,
            max_length=max_length,
        )
        eval_dataset = build_ssa_dataset(
            context.validation_rows,
            tokenizer,
            max_length=max_length,
            include_region_masks=False,
        )
        sft_config = SFTConfig(
            **build_sft_config_kwargs(
                context,
                max_length=max_length,
                pretokenized=True,
            )
        )
        return SsaGenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=SsaDataCollator(tokenizer.pad_token_id),
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
