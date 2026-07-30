"""Align-style supervision: separate label / rationale token losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from generative_trainer import (
    CompactTrainLogCallback,
    JsonlLogCallback,
)
from metrics_utils import REASONING_RE, SCORE_RE
from supervision.base import TrainingBuildContext
from trainers.align_trainer import AlignGenerativeEvalSFTTrainer


@dataclass(frozen=True)
class AlignCoefficients:
    label: float
    rationale: float


def resolve_align_coefficients(config: dict[str, Any]) -> AlignCoefficients:
    supervision = config.get("supervision") or {}
    label = float(supervision.get("label_coeff", 0.5))
    rationale = float(supervision.get("rationale_coeff", 0.5))
    if label < 0 or rationale < 0:
        raise ValueError("Align coefficients must be non-negative.")
    total = label + rationale
    if total <= 0:
        raise ValueError("Align coefficients must sum to a positive value.")
    if abs(total - 1.0) > 1e-6:
        label /= total
        rationale /= total
    return AlignCoefficients(label=label, rationale=rationale)


def split_cot_completion(content: str) -> tuple[str, str]:
    reasoning_match = REASONING_RE.search(content or "")
    score_match = SCORE_RE.search(content or "")
    if reasoning_match is None or score_match is None:
        raise ValueError("CoT completion must contain both <reasoning> and <score> blocks.")
    reasoning_block = content[reasoning_match.start() : reasoning_match.end()]
    score_block = content[score_match.start() : score_match.end()]
    return reasoning_block, score_block


def expand_align_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One CoT sample -> label-only + rationale-only training views."""
    expanded: list[dict[str, Any]] = []
    for row in rows:
        content = str(row["completion"][0]["content"])
        reasoning_block, score_block = split_cot_completion(content)
        base = {key: value for key, value in row.items() if key not in {"completion", "id"}}
        expanded.append(
            {
                **base,
                "id": f"{row['id']}__align_label",
                "align_source_id": row["id"],
                "align_part": "label",
                "prompt": row["prompt"],
                "completion": [{"role": "assistant", "content": score_block}],
            }
        )
        expanded.append(
            {
                **base,
                "id": f"{row['id']}__align_rationale",
                "align_source_id": row["id"],
                "align_part": "rationale",
                "prompt": row["prompt"],
                "completion": [{"role": "assistant", "content": reasoning_block}],
            }
        )
    return expanded


def _truncate_left(
    input_ids: list[int],
    labels: list[int],
    align_part_ids: list[int],
    max_length: int,
) -> tuple[list[int], list[int], list[int]]:
    if len(input_ids) <= max_length:
        return input_ids, labels, align_part_ids
    overflow = len(input_ids) - max_length
    return (
        input_ids[overflow:],
        labels[overflow:],
        align_part_ids[overflow:],
    )


def tokenize_align_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    completion_text = row["completion"][0]["content"]
    part = row["align_part"]
    if part not in {"label", "rationale"}:
        raise ValueError(f"Invalid align_part: {part!r}")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    part_value = 1 if part == "label" else 2
    align_part_ids = [0] * len(prompt_ids) + [part_value] * len(completion_ids)

    input_ids, labels, align_part_ids = _truncate_left(
        input_ids, labels, align_part_ids, max_length
    )
    attention_mask = [1] * len(input_ids)
    label_loss_mask = [1 if value == 1 else 0 for value in align_part_ids]
    rationale_loss_mask = [1 if value == 2 else 0 for value in align_part_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_loss_mask": label_loss_mask,
        "rationale_loss_mask": rationale_loss_mask,
    }


def tokenize_sft_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Standard prompt + full completion tokenization for eval loss."""
    prompt_text = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    completion_text = row["completion"][0]["content"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    input_ids, labels, _ = _truncate_left(
        input_ids,
        labels,
        [0] * len(input_ids),
        max_length,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_standard_eval_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
) -> Dataset:
    tokenized = [
        tokenize_sft_row(tokenizer, row, max_length=max_length) for row in rows
    ]
    return Dataset.from_list(tokenized)


def build_align_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
) -> Dataset:
    expanded = expand_align_rows(rows)
    tokenized = [
        tokenize_align_row(tokenizer, row, max_length=max_length) for row in expanded
    ]
    return Dataset.from_list(tokenized)


class AlignDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "label_loss_mask": [],
            "rationale_loss_mask": [],
        }
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            label_loss_mask = feature.get("label_loss_mask")
            rationale_loss_mask = feature.get("rationale_loss_mask")
            if label_loss_mask is None:
                label_loss_mask = [0] * len(feature["input_ids"])
            if rationale_loss_mask is None:
                rationale_loss_mask = [0] * len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
            batch["label_loss_mask"].append(label_loss_mask + [0] * pad_len)
            batch["rationale_loss_mask"].append(rationale_loss_mask + [0] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class AlignStrategy:
    training_method = "align"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> AlignGenerativeEvalSFTTrainer:
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        coeffs = resolve_align_coefficients(config)
        max_length = int(training.get("max_length", 8192))
        report_to = training.get("report_to", "none")
        if report_to == "none":
            report_to = []

        train_dataset = build_align_dataset(
            context.train_rows,
            tokenizer,
            max_length=max_length,
        )
        eval_dataset = build_standard_eval_dataset(
            context.validation_rows,
            tokenizer,
            max_length=max_length,
        )
        sft_config = SFTConfig(
            output_dir=str(context.run_directory / "checkpoints"),
            logging_dir=str(context.run_directory / "tensorboard"),
            run_name=context.run_id,
            seed=context.seed,
            data_seed=context.seed,
            max_length=max_length,
            completion_only_loss=False,
            packing=False,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
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
        return AlignGenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=AlignDataCollator(pad_token_id=tokenizer.pad_token_id),
            validation_rows=context.validation_rows,
            score_sets=context.labels,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=max_length,
            generation_max_new_tokens=int(generation.get("max_new_tokens", 512)),
            run_directory=context.run_directory,
            logger=context.logger,
            label_coeff=coeffs.label,
            rationale_coeff=coeffs.rationale,
            callbacks=[
                JsonlLogCallback(
                    context.run_directory / "train_history.jsonl",
                    run_tag=context.run_tag,
                ),
                CompactTrainLogCallback(run_tag=context.run_tag, logger=context.logger),
            ],
        )
