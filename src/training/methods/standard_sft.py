"""Existing label_only / cot SFT path (unchanged behavior)."""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from training.validation import (
    CompactTrainLogCallback,
    GenerativeEvalSFTTrainer,
    JsonlLogCallback,
)
from training.methods.interfaces import SupervisionStrategy, TrainingBuildContext
from training.methods.sft_config import build_sft_config_kwargs


class StandardStrategy:
    training_method = "standard"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> GenerativeEvalSFTTrainer:
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        sft_config = SFTConfig(**build_sft_config_kwargs(context))
        return GenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=Dataset.from_list(context.train_rows),
            eval_dataset=Dataset.from_list(context.validation_rows),
            processing_class=tokenizer,
            peft_config=peft_config,
            validation_rows=context.validation_rows,
            score_sets=context.labels,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=int(training.get("max_length", 8192)),
            generation_max_new_tokens=int(generation.get("max_new_tokens", 512)),
            run_directory=context.run_directory,
            logger=context.logger,
            callbacks=[
                JsonlLogCallback(
                    context.run_directory / "train_history.jsonl",
                    run_tag=context.run_tag,
                ),
                CompactTrainLogCallback(run_tag=context.run_tag, logger=context.logger),
            ],
        )
