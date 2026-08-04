"""RAFT without CoT: label-only input with regression-aware score loss."""

from __future__ import annotations

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from generative_trainer import CompactTrainLogCallback, JsonlLogCallback
from supervision.base import TrainingBuildContext
from trainers.raft_trainer import RaftWithoutCotTrainer, resolve_score_token_ids


class RaftWithoutCotStrategy:
    training_method = "raft_without_cot"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> RaftWithoutCotTrainer:
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
            save_strategy="epoch",
            load_best_model_at_end=False,
            save_total_limit=1,
            report_to=report_to,
        )
        score_token_ids = resolve_score_token_ids(tokenizer, context.labels)
        context.logger.info(
            "RAFT score mapping: %s",
            dict(zip(context.labels, score_token_ids, strict=True)),
        )
        return RaftWithoutCotTrainer(
            model=model,
            args=sft_config,
            train_dataset=Dataset.from_list(context.train_rows),
            eval_dataset=Dataset.from_list(context.validation_rows),
            processing_class=tokenizer,
            peft_config=peft_config,
            validation_rows=context.validation_rows,
            score_sets=context.labels,
            score_token_ids=score_token_ids,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=int(training.get("max_length", 8192)),
            generation_max_new_tokens=int(generation.get("max_new_tokens", 32)),
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
