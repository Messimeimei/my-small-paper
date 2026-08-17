"""CoT-RAFT: CE for reasoning/format and RAFT MSE for the numeric score."""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from training.validation import CompactTrainLogCallback, JsonlLogCallback
from utils.metrics import SCORE_RE
from training.methods.interfaces import TrainingBuildContext
from training.methods.sft_config import build_sft_config_kwargs
from training.trainers.raft_trainers import CotRaftTrainer, resolve_score_token_ids


def tokenize_cot_raft_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
    score_token_ids: list[int],
    score_values: list[int],
) -> dict[str, list[int]]:
    """Tokenize one CoT row and mark only the number inside its score block."""
    row_id = str(row["id"])
    row_label = int(row["label"])
    completion_text = str(row["completion"][0]["content"])
    score_matches = list(SCORE_RE.finditer(completion_text))
    if len(score_matches) != 1:
        raise ValueError(
            f"CoT-RAFT row {row_id!r} requires exactly one <score> block; "
            f"found {len(score_matches)}."
        )
    score_match = score_matches[0]
    score = int(score_match.group(1))
    if score != row_label:
        raise ValueError(
            f"CoT-RAFT row {row_id!r} completion score {score} does not match "
            f"label {row_label}."
        )
    try:
        score_index = score_values.index(score)
    except ValueError as exc:
        raise ValueError(
            f"CoT-RAFT row {row_id!r} score {score} is outside {score_values}."
        ) from exc

    prompt_kwargs: dict[str, Any] = {}
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        prompt_kwargs["enable_thinking"] = False
    prompt_text = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **prompt_kwargs,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_encoding = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    completion_ids = list(completion_encoding["input_ids"])
    offsets = completion_encoding.get("offset_mapping")
    if offsets is None:
        raise ValueError("CoT-RAFT requires a fast tokenizer with offset mappings.")

    score_start, score_end = score_match.span(1)
    score_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < score_end and end > score_start
    ]
    if len(score_positions) != 1:
        raise ValueError(
            f"CoT-RAFT row {row_id!r} numeric score must occupy exactly one "
            f"token; found completion positions {score_positions}."
        )
    completion_score_position = score_positions[0]
    expected_token_id = score_token_ids[score_index]
    observed_token_id = completion_ids[completion_score_position]
    if observed_token_id != expected_token_id:
        raise ValueError(
            f"CoT-RAFT row {row_id!r} score token mismatch: expected "
            f"{expected_token_id}, got {observed_token_id}."
        )

    completion_score_mask = [0] * len(completion_ids)
    completion_score_mask[completion_score_position] = 1
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
        completion_score_mask.append(0)

    input_ids = list(prompt_ids) + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    raft_score_mask = [0] * len(prompt_ids) + completion_score_mask
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        raft_score_mask = raft_score_mask[overflow:]
    if sum(raft_score_mask) != 1:
        raise ValueError(
            f"CoT-RAFT row {row_id!r} score was truncated at max_length={max_length}."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "raft_score_mask": raft_score_mask,
    }


def build_cot_raft_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    score_token_ids: list[int],
    score_values: list[int],
) -> Dataset:
    return Dataset.from_list(
        [
            tokenize_cot_raft_row(
                tokenizer,
                row,
                max_length=max_length,
                score_token_ids=score_token_ids,
                score_values=score_values,
            )
            for row in rows
        ]
    )


class CotRaftDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "raft_score_mask": [],
        }
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
            batch["raft_score_mask"].append(
                feature["raft_score_mask"] + [0] * pad_len
            )
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


class CotRaftStrategy:
    training_method = "cot_raft"

    def build_trainer(
        self,
        *,
        model,
        tokenizer,
        peft_config: LoraConfig,
        context: TrainingBuildContext,
    ) -> CotRaftTrainer:
        config = context.config
        training = config.get("training", {})
        generation = config.get("generation", {})
        supervision = config.get("supervision", {})
        raft_weight = float(supervision.get("raft_weight", 1.0))
        if raft_weight < 0:
            raise ValueError("supervision.raft_weight must be non-negative.")
        max_length = int(training.get("max_length", 8192))
        score_token_ids = resolve_score_token_ids(tokenizer, context.labels)
        context.logger.info(
            "CoT-RAFT score mapping: %s; raft_weight=%s",
            dict(zip(context.labels, score_token_ids, strict=True)),
            raft_weight,
        )
        train_dataset = build_cot_raft_dataset(
            context.train_rows,
            tokenizer,
            max_length=max_length,
            score_token_ids=score_token_ids,
            score_values=context.labels,
        )
        eval_dataset = build_cot_raft_dataset(
            context.validation_rows,
            tokenizer,
            max_length=max_length,
            score_token_ids=score_token_ids,
            score_values=context.labels,
        )
        sft_config = SFTConfig(
            **build_sft_config_kwargs(
                context,
                max_length=max_length,
                pretokenized=True,
            )
        )
        return CotRaftTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=CotRaftDataCollator(tokenizer.pad_token_id),
            validation_rows=context.validation_rows,
            score_sets=context.labels,
            score_token_ids=score_token_ids,
            raft_weight=raft_weight,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=max_length,
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
