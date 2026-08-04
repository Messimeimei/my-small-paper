#!/usr/bin/env python3
"""Trainer callbacks and generative validation used by train.py."""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback
from trl import SFTTrainer

_TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from shared.metrics import classification_metrics, extract_score, token_stats


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


class JsonlLogCallback(TrainerCallback):
    """把 Trainer 的 log 追加写入 train_history.jsonl。"""

    def __init__(self, path: Path, run_tag: str) -> None:
        self.path = path
        self.run_tag = run_tag

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero or not logs:
            return
        record = {
            "time_utc": utc_now(),
            "step": state.global_step,
            "run_tag": self.run_tag,
            **logs,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class CompactTrainLogCallback(TrainerCallback):
    """控制台打印简短训练日志：任务/模式/模型 + step 指标。"""

    def __init__(self, run_tag: str, logger: logging.Logger) -> None:
        self.run_tag = run_tag
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero or not logs:
            return
        interesting = []
        for key in (
            "loss",
            "eval_loss",
            "eval_generation_accuracy",
            "eval_generation_macro_f1",
            "eval_generation_format_valid_rate",
            "eval_generation_mae",
            "eval_generation_qwk",
            "cot_raft_lm_loss",
            "cot_raft_score_loss",
            "learning_rate",
            "epoch",
        ):
            if key in logs:
                value = logs[key]
                if isinstance(value, float):
                    interesting.append(f"{key}={value:.4f}")
                else:
                    interesting.append(f"{key}={value}")
        if not interesting:
            return
        self.logger.info(
            "[%s] step=%d %s",
            self.run_tag,
            state.global_step,
            " ".join(interesting),
        )


@torch.inference_mode()
def generate_validation(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    score_sets: list[int],
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """用训练模型所在设备做 greedy 生成，并从输出抽取 <score>。"""
    was_training = model.training
    original_use_cache = model.config.use_cache
    original_padding_side = tokenizer.padding_side
    device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    allowed_scores = set(score_sets)
    inputs = None
    output_ids = None
    generated = None

    try:
        model.eval()
        # Checkpointing is inactive in eval mode; keep its flag for resumed training.
        model.config.use_cache = True
        tokenizer.padding_side = "left"  # 生成时左填充
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if logger is not None:
            logger.info(
                "Starting generation validation on training device %s: samples=%d batch_size=%d",
                device,
                len(rows),
                batch_size,
            )

        total_batches = math.ceil(len(rows) / batch_size)
        progress_interval = max(1, total_batches // 20)
        batches = range(0, len(rows), batch_size)
        for batch_index, start in enumerate(batches, start=1):
            batch = rows[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    row["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, output in zip(batch, outputs, strict=True):
                prediction = extract_score(output, allowed_scores)
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": prediction,
                        "correct": prediction == row["label"],
                        "output": output,
                    }
                )
            if logger is not None and (
                batch_index % progress_interval == 0 or batch_index == total_batches
            ):
                logger.info(
                    "Generation validation progress: %d/%d",
                    min(start + batch_size, len(rows)),
                    len(rows),
                )
        metrics = classification_metrics(predictions, score_sets)
        metrics["tokens"] = token_stats(predictions, tokenizer)
        return metrics, predictions
    finally:
        # Drop the final generation tensors before returning cached memory to CUDA.
        inputs = None
        output_ids = None
        generated = None
        model.config.use_cache = original_use_cache
        tokenizer.padding_side = original_padding_side
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_eval_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "label": row["label"],
            "prompt": row["prompt"],
            "task": row.get("task"),
            "aspect": row.get("aspect"),
        }
        for row in rows
    ]


def write_eval_dataset(
    path: Path, rows: list[dict[str, Any]], score_sets: list[int]
) -> None:
    write_json(path, {"metadata": {"score_sets": score_sets}, "test": build_eval_dataset(rows)})


class GenerativeEvalSFTTrainer(SFTTrainer):
    """每次 evaluate 后用训练模型和同一设备做生成式分类验证。"""

    def __init__(
        self,
        *args,
        validation_rows: list[dict[str, Any]],
        score_sets: list[int],
        generation_batch_size: int,
        generation_max_length: int,
        generation_max_new_tokens: int,
        run_directory: Path,
        logger: logging.Logger,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.validation_rows = validation_rows
        self.score_sets = score_sets
        self.generation_batch_size = generation_batch_size
        self.generation_max_length = generation_max_length
        self.generation_max_new_tokens = generation_max_new_tokens
        self.run_directory = run_directory
        self.logger = logger
        self.latest_generation_metrics: dict[str, Any] | None = None
        self.latest_generation_predictions: list[dict[str, Any]] | None = None
        self.validation_dataset_path = self.run_directory / "validation_dataset.json"
        write_eval_dataset(
            self.validation_dataset_path, self.validation_rows, self.score_sets
        )

    def _run_generation_validation(self):
        return generate_validation(
            self.model,
            self.processing_class,
            self.validation_rows,
            batch_size=self.generation_batch_size,
            max_length=self.generation_max_length,
            max_new_tokens=self.generation_max_new_tokens,
            score_sets=self.score_sets,
            logger=self.logger,
        )

    def evaluate(self, *args, metric_key_prefix: str = "eval", **kwargs):  # noqa: ANN002
        metrics = super().evaluate(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        validation_metrics, predictions = self._run_generation_validation()
        metric_names = {
            f"{metric_key_prefix}_generation_accuracy": validation_metrics["accuracy"],
            f"{metric_key_prefix}_generation_macro_f1": validation_metrics["macro_f1"],
            f"{metric_key_prefix}_generation_format_valid_rate": validation_metrics[
                "format_valid_rate"
            ],
            f"{metric_key_prefix}_generation_invalid_outputs": validation_metrics[
                "invalid_outputs"
            ],
        }
        if validation_metrics.get("mae") is not None:
            metric_names[f"{metric_key_prefix}_generation_mae"] = validation_metrics["mae"]
        if validation_metrics.get("qwk") is not None:
            metric_names[f"{metric_key_prefix}_generation_qwk"] = validation_metrics["qwk"]
        metrics.update(metric_names)
        self.latest_generation_metrics = validation_metrics
        self.latest_generation_predictions = predictions

        step = int(self.state.global_step)
        epoch_value = self.state.epoch
        epoch_tag = (
            f"{epoch_value:.4f}".replace(".", "p") if epoch_value is not None else "unknown"
        )
        eval_root = self.run_directory / "epoch_evals"
        payload = {
            "step": step,
            "epoch": epoch_value,
            "metrics": validation_metrics,
            "trainer_metrics": {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float, str, bool)) or value is None
            },
            "checkpoint_retention": "last",
        }
        write_json(eval_root / f"step_{step:06d}__epoch_{epoch_tag}.metrics.json", payload)
        write_json(eval_root / "latest.metrics.json", payload)
        write_jsonl(
            eval_root / f"step_{step:06d}__epoch_{epoch_tag}.predictions.jsonl",
            predictions,
        )
        write_jsonl(eval_root / "latest.predictions.jsonl", predictions)
        self.logger.info(
            "Generation val @ epoch=%s step=%d: acc=%.4f macro_f1=%.4f valid=%.4f "
            "(retaining last checkpoint)",
            epoch_value,
            step,
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            validation_metrics["format_valid_rate"],
        )
        self.log(metric_names)
        return metrics


