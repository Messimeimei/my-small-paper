"""Shared SFT arguments; method modules only define their real differences."""

from __future__ import annotations

from typing import Any

from training_methods.interfaces import TrainingBuildContext


def build_sft_config_kwargs(
    context: TrainingBuildContext,
    *,
    max_length: int | None = None,
    pretokenized: bool = False,
) -> dict[str, Any]:
    training = context.config.get("training", {})
    use_bf16 = bool(training.get("bf16", True))
    report_to = training.get("report_to", "none")
    if report_to == "none":
        report_to = []

    kwargs: dict[str, Any] = {
        "output_dir": str(context.run_directory / "checkpoints"),
        "logging_dir": str(context.run_directory / "tensorboard"),
        "run_name": context.run_id,
        "seed": context.seed,
        "data_seed": context.seed,
        "max_length": int(
            max_length
            if max_length is not None
            else training.get("max_length", 8192)
        ),
        "completion_only_loss": not pretokenized,
        "packing": False,
        "per_device_train_batch_size": int(
            training.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            training.get("per_device_eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 16)
        ),
        "learning_rate": float(training.get("learning_rate", 1e-4)),
        "lr_scheduler_type": str(training.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
        "num_train_epochs": float(training.get("num_train_epochs", 3)),
        "max_grad_norm": float(training.get("max_grad_norm", 0.3)),
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_strategy": "steps",
        "logging_steps": int(training.get("logging_steps", 10)),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": False,
        "save_total_limit": 1,
        "report_to": report_to,
    }
    if pretokenized:
        kwargs.update(
            completion_only_loss=False,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
    return kwargs
