"""Existing label_only / cot SFT path (unchanged behavior)."""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from generative_trainer import (
    CompactTrainLogCallback,
    GenerativeEvalSFTTrainer,
    JsonlLogCallback,
)
from supervision.base import SupervisionStrategy, TrainingBuildContext


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
        report_to = training.get("report_to", "none")
        if report_to == "none":
            report_to = []

        sft_config = SFTConfig(
            output_dir=str(context.run_directory / "checkpoints"),
            logging_dir=str(context.run_directory / "tensorboard"),
            run_name=context.run_id,
            seed=context.seed,
            data_seed=context.seed,
            max_length=int(training.get("max_length", 8192)),
            completion_only_loss=True,
            packing=False,
            per_device_train_batch_size=int(
                training.get("per_device_train_batch_size", 1)
            ),
            per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 16)
            ),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(training.get("warmup_ratio", 0.03)),
            num_train_epochs=float(training.get("num_train_epochs", 3)),
            max_grad_norm=float(training.get("max_grad_norm", 0.3)),
            bf16=bool(training.get("bf16", True)),
            fp16=not bool(training.get("bf16", True)),
            gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_strategy="steps",
            logging_steps=int(training.get("logging_steps", 10)),
            eval_strategy="epoch",
            save_strategy="best",
            load_best_model_at_end=True,
            metric_for_best_model="eval_generation_accuracy",
            greater_is_better=True,
            save_total_limit=int(training.get("save_total_limit", 2)),
            report_to=report_to,
        )
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
