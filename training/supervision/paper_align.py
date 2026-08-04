"""Paper-faithful Align supervision with paired Direct and Reason views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from generative_trainer import CompactTrainLogCallback, JsonlLogCallback
from metrics_utils import REASONING_RE, SCORE_RE
from supervision.align import build_standard_eval_dataset
from supervision.base import TrainingBuildContext
from trainers.paper_align_trainer import PaperAlignGenerativeEvalSFTTrainer


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
    return AlignCoefficients(label=label / total, rationale=rationale / total)


def _completion_content(row: dict[str, Any]) -> str:
    return str(row["completion"][0]["content"])


def validate_and_pair_rows(
    cot_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair CoT and label-only rows and enforce the paper's two-view contract."""
    labels_by_id = {row["id"]: row for row in label_rows}
    cot_ids = {row["id"] for row in cot_rows}
    if len(labels_by_id) != len(label_rows) or cot_ids != set(labels_by_id):
        missing = sorted(cot_ids - set(labels_by_id))[:5]
        extra = sorted(set(labels_by_id) - cot_ids)[:5]
        raise ValueError(
            "paper_align datasets must have identical unique IDs; "
            f"missing_label_ids={missing}, extra_label_ids={extra}"
        )

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cot_row in cot_rows:
        label_row = labels_by_id[cot_row["id"]]
        if cot_row["label"] != label_row["label"]:
            raise ValueError(f"Label mismatch for paper_align id={cot_row['id']}")
        if cot_row["prompt"][1:] != label_row["prompt"][1:]:
            raise ValueError(f"User/context prompt mismatch for id={cot_row['id']}")
        if cot_row["prompt"][0] == label_row["prompt"][0]:
            raise ValueError(
                f"Direct and Reason system prompts must differ for id={cot_row['id']}"
            )

        cot_completion = _completion_content(cot_row)
        label_completion = _completion_content(label_row)
        reasoning_match = REASONING_RE.search(cot_completion)
        cot_score_match = SCORE_RE.search(cot_completion)
        label_score_match = SCORE_RE.search(label_completion)
        if (
            reasoning_match is None
            or cot_score_match is None
            or reasoning_match.start() > cot_score_match.start()
        ):
            raise ValueError(
                f"Reason view must contain reasoning before score for id={cot_row['id']}"
            )
        if (
            REASONING_RE.search(label_completion) is not None
            or label_score_match is None
        ):
            raise ValueError(
                f"Direct view must contain score only for id={cot_row['id']}"
            )
        expected_score = cot_row["label"]
        if (
            int(cot_score_match.group(1)) != expected_score
            or int(label_score_match.group(1)) != expected_score
        ):
            raise ValueError(
                f"Completion score does not match label for id={cot_row['id']}"
            )
        pairs.append((label_row, cot_row))
    return pairs


def _tokenize_view(
    tokenizer,
    row: dict[str, Any],
    max_length: int,
) -> dict[str, list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        _completion_content(row), add_special_tokens=False
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_paired_align_dataset(
    cot_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
) -> Dataset:
    records: list[dict[str, Any]] = []
    for label_row, cot_row in validate_and_pair_rows(cot_rows, label_rows):
        label_view = _tokenize_view(tokenizer, label_row, max_length)
        reason_view = _tokenize_view(tokenizer, cot_row, max_length)
        records.append(
            {
                "source_id": cot_row["id"],
                **{f"label_{key}": value for key, value in label_view.items()},
                **{f"reason_{key}": value for key, value in reason_view.items()},
            }
        )
    return Dataset.from_list(records)


class PaperAlignPairCollator:
    """Expand every source record into an adjacent Direct/Reason sequence pair."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if "input_ids" in features[0]:
            return self._collate_eval(features)

        sequences: list[tuple[str, dict[str, list[int]]]] = []
        for feature in features:
            for part in ("label", "reason"):
                sequences.append(
                    (
                        part,
                        {
                            key: feature[f"{part}_{key}"]
                            for key in ("input_ids", "attention_mask", "labels")
                        },
                    )
                )

        max_len = max(len(sequence["input_ids"]) for _, sequence in sequences)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "label_loss_mask": [],
            "rationale_loss_mask": [],
        }
        for part, sequence in sequences:
            pad_len = max_len - len(sequence["input_ids"])
            supervised = [int(value != -100) for value in sequence["labels"]]
            batch["input_ids"].append(
                sequence["input_ids"] + [self.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(
                sequence["attention_mask"] + [0] * pad_len
            )
            batch["labels"].append(sequence["labels"] + [-100] * pad_len)
            batch["label_loss_mask"].append(
                (supervised if part == "label" else [0] * len(supervised))
                + [0] * pad_len
            )
            batch["rationale_loss_mask"].append(
                (supervised if part == "reason" else [0] * len(supervised))
                + [0] * pad_len
            )
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }

    def _collate_eval(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * pad_len
            )
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


class PaperAlignStrategy:
    training_method = "paper_align"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> PaperAlignGenerativeEvalSFTTrainer:
        if (
            context.label_train_rows is None
            or context.label_validation_rows is None
        ):
            raise ValueError(
                "paper_align requires paired label train and validation rows"
            )
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        coeffs = resolve_align_coefficients(config)
        max_length = int(training.get("max_length", 8192))
        report_to = training.get("report_to", "none")
        if report_to == "none":
            report_to = []

        train_dataset = build_paired_align_dataset(
            context.train_rows,
            context.label_train_rows,
            tokenizer,
            max_length=max_length,
        )
        eval_dataset = build_standard_eval_dataset(
            context.label_validation_rows,
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
            per_device_eval_batch_size=int(
                training.get("per_device_eval_batch_size", 1)
            ),
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
        return PaperAlignGenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=PaperAlignPairCollator(
                pad_token_id=tokenizer.pad_token_id
            ),
            validation_rows=context.label_validation_rows,
            score_sets=context.labels,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=max_length,
            generation_max_new_tokens=int(
                generation.get("max_new_tokens", 512)
            ),
            run_directory=context.run_directory,
            logger=context.logger,
            label_coeff=coeffs.label,
            rationale_coeff=coeffs.rationale,
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
