"""RAFT without CoT: label-only input with regression-aware score loss."""

from __future__ import annotations

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from training_workflow.generation_validation import CompactTrainLogCallback, JsonlLogCallback
from training_methods.interfaces import TrainingBuildContext
from training_methods.sft_config import build_sft_config_kwargs
from custom_trainers.raft_trainers import RaftWithoutCotTrainer, resolve_score_token_ids


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
        sft_config = SFTConfig(**build_sft_config_kwargs(context))
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
