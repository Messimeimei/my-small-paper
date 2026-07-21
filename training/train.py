#!/usr/bin/env python3
"""可复用多 Lora 专家训练脚本"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import trl
import yaml
from datasets import Dataset, __version__ as datasets_version
from peft import LoraConfig, TaskType, __version__ as peft_version
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_RE = re.compile(r"<score>\s*([01])\s*</score>", re.I)  # 从生成文本中抽最终 0/1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint path used to resume an interrupted run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data and create or verify the fixed split without training.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    """相对路径按项目根目录解析。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML object.")
    required = {"experiment_name", "model_name_or_path", "dataset_path", "split_path"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing fields: {sorted(missing)}")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """原子写：先 .tmp 再 replace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """读 JSONL；校验 id / label / prompt / completion 格式。"""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", "")).strip()
            prompt = row.get("prompt")
            completion = row.get("completion")
            label = row.get("label")
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Invalid or duplicate id at {path}:{line_number}")
            if label not in {0, 1}:
                raise ValueError(f"Invalid label at {path}:{line_number}")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"Invalid prompt at {path}:{line_number}")
            if (
                not isinstance(completion, list)
                or len(completion) != 1
                or completion[0].get("role") != "assistant"
                or not SCORE_RE.search(str(completion[0].get("content", "")))
            ):
                raise ValueError(f"Invalid completion at {path}:{line_number}")
            seen_ids.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def validation_counts(rows: list[dict[str, Any]], ratio: float) -> dict[int, int]:
    """按标签分层，决定验证集每类抽多少条。"""
    counts = {label: sum(row["label"] == label for row in rows) for label in (0, 1)}
    target_total = max(2, round(len(rows) * ratio))
    raw = {label: counts[label] * ratio for label in counts}
    selected = {label: math.floor(raw[label]) for label in counts}
    remaining = target_total - sum(selected.values())
    for label in sorted(counts, key=lambda value: raw[value] - selected[value], reverse=True):
        if remaining <= 0:
            break
        selected[label] += 1
        remaining -= 1
    return selected


def load_or_create_split(
    rows: list[dict[str, Any]],
    split_path: Path,
    dataset_hash: str,
    split_seed: int,
    validation_ratio: float,
) -> dict[str, Any]:
    """已有固定划分则复用；否则按 seed 分层创建并落盘。"""
    all_ids = {row["id"] for row in rows}
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"Dataset hash no longer matches fixed split: {split_path}")
        train_ids = set(split.get("train_ids", []))
        validation_ids = set(split.get("validation_ids", []))
        if train_ids & validation_ids or train_ids | validation_ids != all_ids:
            raise ValueError(f"Invalid ID coverage in fixed split: {split_path}")
        return split

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    rng = random.Random(split_seed)
    per_label = {
        label: [row["id"] for row in rows if row["label"] == label]
        for label in (0, 1)
    }
    for ids in per_label.values():
        rng.shuffle(ids)
    selected_counts = validation_counts(rows, validation_ratio)
    validation_ids = {
        sample_id
        for label, ids in per_label.items()
        for sample_id in ids[: selected_counts[label]]
    }
    train_ids = [row["id"] for row in rows if row["id"] not in validation_ids]
    ordered_validation_ids = [row["id"] for row in rows if row["id"] in validation_ids]
    split = {
        "schema_version": 1,
        "dataset_sha256": dataset_hash,
        "split_seed": split_seed,
        "validation_ratio": validation_ratio,
        "train_ids": train_ids,
        "validation_ids": ordered_validation_ids,
    }
    write_json(split_path, split)
    return split


def split_rows(
    rows: list[dict[str, Any]], split: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids = set(split["train_ids"])
    validation_ids = set(split["validation_ids"])
    return (
        [row for row in rows if row["id"] in train_ids],
        [row for row in rows if row["id"] in validation_ids],
    )


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(label): sum(row["label"] == label for row in rows) for label in (0, 1)}


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def disable_incompatible_torchao() -> str | None:
    """Ignore an old optional TorchAO backend when training regular BF16 weights."""
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return None
    major, minor, *_ = (int(part) for part in torchao_version.split(".")[:2])
    if (major, minor) >= (0, 16):
        return None

    from peft.tuners.lora import torchao as peft_torchao_backend

    peft_torchao_backend.is_torchao_available = lambda: False
    return f"Disabled optional torchao {torchao_version}; PEFT requires >=0.16.0."


class JsonlLogCallback(TrainerCallback):
    """把 Trainer 的 log 追加写入 train_history.jsonl。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero or not logs:
            return
        record = {"time_utc": utc_now(), "step": state.global_step, **logs}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classification_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy / macro-F1 / 格式有效率 / 混淆矩阵。"""
    total = len(predictions)
    valid = [row for row in predictions if row["prediction"] in {0, 1}]
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label in (0, 1):
        tp = sum(row["label"] == label and row["prediction"] == label for row in predictions)
        fp = sum(row["label"] != label and row["prediction"] == label for row in predictions)
        fn = sum(row["label"] == label and row["prediction"] != label for row in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(row["label"] == label for row in predictions),
        }
    confusion_matrix = {
        str(gold): {
            ("invalid" if predicted is None else str(predicted)): sum(
                row["label"] == gold and row["prediction"] == predicted
                for row in predictions
            )
            for predicted in (0, 1, None)
        }
        for gold in (0, 1)
    }
    return {
        "samples": total,
        "accuracy": sum(row["correct"] for row in predictions) / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "format_valid_rate": len(valid) / total,
        "invalid_outputs": total - len(valid),
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
    }


@torch.inference_mode()
def generate_validation(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """训练后生成式验证：greedy 解码，从输出抽 <score>。"""
    was_training = model.training
    original_use_cache = model.config.use_cache
    original_padding_side = tokenizer.padding_side
    device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    inputs = None
    output_ids = None
    generated = None

    try:
        model.eval()
        # Checkpointing is inactive in eval mode; keep its flag for resumed training.
        model.config.use_cache = True
        tokenizer.padding_side = "left"  # 生成时左填充

        for start in range(0, len(rows), batch_size):
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
                scores = SCORE_RE.findall(output)
                prediction = int(scores[-1]) if scores else None
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": prediction,
                        "correct": prediction == row["label"],
                        "output": output,
                    }
                )
        return classification_metrics(predictions), predictions
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


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


def write_eval_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path, {"test": build_eval_dataset(rows)})


def detect_physical_training_gpu() -> int | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    first = raw.split(",")[0].strip()
    return int(first) if first.isdigit() else None


def query_gpu_stats() -> dict[int, dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    stats: dict[int, dict[str, float | int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, total, free, util = (int(value) for value in parts)
        stats[index] = {
            "index": index,
            "memory_total_mib": total,
            "memory_free_mib": free,
            "utilization_gpu": util,
        }
    return stats


def choose_vllm_gpu(
    evaluation: dict[str, Any],
    logger: logging.Logger,
) -> tuple[int | None, dict[str, Any]]:
    preferred_gpu = int(evaluation.get("vllm_preferred_gpu", 0))
    max_gpu_utilization = int(evaluation.get("vllm_max_gpu_utilization", 10))
    gpu_memory_utilization = float(evaluation.get("vllm_gpu_memory_utilization", 0.9))
    min_free_memory_gib = evaluation.get("vllm_min_free_memory_gib", 20)
    training_gpu = detect_physical_training_gpu()
    stats = query_gpu_stats()
    gpu = stats.get(preferred_gpu)
    decision = {
        "preferred_gpu": preferred_gpu,
        "training_gpu": training_gpu,
        "stats": stats,
        "selected_gpu": None,
        "reason": None,
    }
    if gpu is None:
        decision["reason"] = f"gpu{preferred_gpu} not found"
        return None, decision
    if training_gpu is not None and preferred_gpu == training_gpu:
        decision["reason"] = f"gpu{preferred_gpu} is the training GPU"
        return None, decision
    total_mib = int(gpu["memory_total_mib"])
    free_mib = int(gpu["memory_free_mib"])
    util = int(gpu["utilization_gpu"])
    if min_free_memory_gib is None:
        required_free_mib = math.ceil(total_mib * gpu_memory_utilization)
    else:
        required_free_mib = int(float(min_free_memory_gib) * 1024)
    decision["required_free_mib"] = required_free_mib
    if util > max_gpu_utilization:
        decision["reason"] = f"gpu{preferred_gpu} utilization {util}% > {max_gpu_utilization}%"
        return None, decision
    if free_mib < required_free_mib:
        decision["reason"] = (
            f"gpu{preferred_gpu} free {free_mib} MiB < required {required_free_mib} MiB"
        )
        return None, decision
    decision["selected_gpu"] = preferred_gpu
    decision["reason"] = f"gpu{preferred_gpu} accepted"
    logger.info(
        "Selected GPU%d for vLLM eval: free=%d/%d MiB util=%d%%",
        preferred_gpu,
        free_mib,
        total_mib,
        util,
    )
    return preferred_gpu, decision


def run_vllm_validation(
    *,
    model_path: Path,
    adapter_path: Path,
    dataset_path: Path,
    output_root: Path,
    eval_gpu: int,
    evaluation: dict[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    exp_name = safe_name(f"step_{adapter_path.name}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "training/evaluate.py"),
        "--exp_name",
        exp_name,
        "--model_name",
        str(model_path),
        "--adapter",
        str(adapter_path),
        "--dataset_file",
        str(dataset_path),
        "--output_path",
        str(output_root),
        "--max_model_len",
        str(int(evaluation.get("max_model_len", 8192))),
        "--max_tokens",
        str(int(evaluation.get("max_tokens", 512))),
        "--temp",
        str(float(evaluation.get("temp", 0.0))),
        "--top_p",
        str(float(evaluation.get("top_p", 1.0))),
        "--seed",
        str(int(evaluation.get("seed", 42))),
        "--rollout",
        str(int(evaluation.get("rollout", 1))),
        "--batch_size",
        str(int(evaluation.get("batch_size", 64))),
        "--gpu_memory_utilization",
        str(float(evaluation.get("vllm_gpu_memory_utilization", 0.9))),
        "--merge_cache",
        str(resolve_path(evaluation.get("merge_cache", "eval_data/.merged_models"))),
    ]
    if bool(evaluation.get("enable_thinking", False)):
        command.append("--enable_thinking")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    logger.info("Launching vLLM eval on physical GPU%d", eval_gpu)
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        cwd=PROJECT_ROOT,
    )
    if result.stdout.strip():
        logger.info("vLLM eval stdout:\n%s", result.stdout.strip())
    if result.stderr.strip():
        logger.warning("vLLM eval stderr:\n%s", result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"vLLM eval failed with exit code {result.returncode}")
    result_dir = output_root / exp_name
    metrics_payload = read_json(result_dir / "metrics.json")
    predictions: list[dict[str, Any]] = []
    with (result_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                predictions.append(json.loads(line))
    metrics = metrics_payload["aggregate"]
    meta = {
        "backend": "vllm",
        "eval_gpu": eval_gpu,
        "result_dir": str(result_dir),
    }
    return metrics, predictions, meta


class GenerativeEvalSFTTrainer(SFTTrainer):
    """在每次 evaluate 时追加生成式分类验证，并按 step 落盘结果。"""

    def __init__(
        self,
        *args,
        validation_rows: list[dict[str, Any]],
        generation_batch_size: int,
        generation_max_length: int,
        generation_max_new_tokens: int,
        run_directory: Path,
        model_path: Path,
        logger: logging.Logger,
        evaluation: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.validation_rows = validation_rows
        self.generation_batch_size = generation_batch_size
        self.generation_max_length = generation_max_length
        self.generation_max_new_tokens = generation_max_new_tokens
        self.run_directory = run_directory
        self.model_path = model_path
        self.logger = logger
        self.evaluation = evaluation
        self.latest_generation_metrics: dict[str, Any] | None = None
        self.latest_generation_predictions: list[dict[str, Any]] | None = None
        self.latest_generation_meta: dict[str, Any] | None = None
        self.validation_dataset_path = self.run_directory / "validation_dataset.json"
        write_eval_dataset(self.validation_dataset_path, self.validation_rows)

    def _run_native_generation_validation(self) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        validation_metrics, predictions = generate_validation(
            self.model,
            self.processing_class,
            self.validation_rows,
            batch_size=self.generation_batch_size,
            max_length=self.generation_max_length,
            max_new_tokens=self.generation_max_new_tokens,
        )
        return validation_metrics, predictions, {"backend": "native"}

    def _run_generation_validation(
        self,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        backend = str(self.evaluation.get("backend", "auto")).lower()
        if backend not in {"auto", "native", "vllm"}:
            raise ValueError(f"Unsupported evaluation backend: {backend}")
        if backend == "native":
            return self._run_native_generation_validation()

        if backend in {"auto", "vllm"}:
            try:
                eval_gpu, decision = choose_vllm_gpu(self.evaluation, self.logger)
            except Exception as error:
                if backend == "vllm":
                    raise
                self.logger.warning("vLLM GPU auto-check failed, fallback to native: %s", error)
                return self._run_native_generation_validation()

            if eval_gpu is not None:
                step = int(self.state.global_step)
                adapter_path = self.run_directory / "epoch_eval_adapters" / f"step_{step:06d}"
                output_root = self.run_directory / "epoch_evals" / "vllm_runs"
                if adapter_path.exists():
                    shutil.rmtree(adapter_path)
                adapter_path.mkdir(parents=True, exist_ok=True)
                self.save_model(adapter_path)
                self.processing_class.save_pretrained(adapter_path)
                try:
                    metrics, predictions, meta = run_vllm_validation(
                        model_path=self.model_path,
                        adapter_path=adapter_path,
                        dataset_path=self.validation_dataset_path,
                        output_root=output_root,
                        eval_gpu=eval_gpu,
                        evaluation=self.evaluation,
                        logger=self.logger,
                    )
                finally:
                    shutil.rmtree(adapter_path, ignore_errors=True)
                meta["auto_gpu_decision"] = decision
                return metrics, predictions, meta

            if backend == "vllm":
                raise RuntimeError(f"vLLM requested but unavailable: {decision['reason']}")
            self.logger.info("Fallback to native validation: %s", decision["reason"])
            native_metrics, native_predictions, native_meta = (
                self._run_native_generation_validation()
            )
            native_meta["auto_gpu_decision"] = decision
            return native_metrics, native_predictions, native_meta

        return self._run_native_generation_validation()

    def evaluate(self, *args, metric_key_prefix: str = "eval", **kwargs):  # noqa: ANN002
        metrics = super().evaluate(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        validation_metrics, predictions, meta = self._run_generation_validation()
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
        metrics.update(metric_names)
        self.latest_generation_metrics = validation_metrics
        self.latest_generation_predictions = predictions
        self.latest_generation_meta = meta

        step = int(self.state.global_step)
        epoch_value = self.state.epoch
        epoch_tag = (
            f"{epoch_value:.4f}".replace(".", "p") if epoch_value is not None else "unknown"
        )
        eval_root = self.run_directory / "epoch_evals"
        payload = {
            "step": step,
            "epoch": epoch_value,
            "generation_backend": meta,
            "metrics": validation_metrics,
            "trainer_metrics": metrics,
        }
        write_json(eval_root / f"step_{step:06d}__epoch_{epoch_tag}.metrics.json", payload)
        write_json(eval_root / "latest.metrics.json", payload)
        write_jsonl(
            eval_root / f"step_{step:06d}__epoch_{epoch_tag}.predictions.jsonl",
            predictions,
        )
        write_jsonl(eval_root / "latest.predictions.jsonl", predictions)
        self.log(metric_names)
        return metrics


def create_run_directory(config: dict[str, Any], seed: int) -> tuple[str, Path]:
    """新建独立 run 目录：experiment__seed__UTC。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config['experiment_name']}__seed{seed}__{timestamp}"
    output_root = resolve_path(config.get("output_root", "train_outputs/lora"))
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_id, run_directory


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    config_path = resolve_path(args.config)
    config = read_config(config_path)
    dataset_path = resolve_path(config["dataset_path"])
    split_path = resolve_path(config["split_path"])
    model_path = resolve_path(config["model_name_or_path"])

    # --- 数据与固定划分 ---
    dataset_hash = sha256_file(dataset_path)
    rows = load_rows(dataset_path)
    split = load_or_create_split(
        rows,
        split_path,
        dataset_hash,
        int(config.get("split_seed", 20260720)),
        float(config.get("validation_ratio", 0.1)),
    )
    train_rows, validation_rows = split_rows(rows, split)
    data_summary = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "split": str(split_path),
        "all": {"samples": len(rows), "labels": label_counts(rows)},
        "train": {"samples": len(train_rows), "labels": label_counts(train_rows)},
        "validation": {
            "samples": len(validation_rows),
            "labels": label_counts(validation_rows),
        },
    }

    if args.dry_run:
        print(json.dumps(data_summary, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run training on a GPU node.")

    # --- run 目录与复现元数据 ---
    run_id, run_directory = create_run_directory(config, args.seed)
    log_path = run_directory / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    logger = logging.getLogger(__name__)

    resolved_config = {
        **config,
        "model_name_or_path": str(model_path),
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "seed": args.seed,
    }
    write_json(run_directory / "resolved_config.json", resolved_config)
    write_json(run_directory / "data_summary.json", data_summary)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": sys.argv,
        "git": git_metadata(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft_version,
            "datasets": datasets_version,
            "tensorboard": importlib.metadata.version("tensorboard"),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    }
    write_json(run_directory / "manifest.json", manifest)

    try:
        training = config.get("training", {})
        lora = config.get("lora", {})
        generation = config.get("generation", {})
        evaluation = config.get("evaluation", {})
        if training.get("bf16", True) and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured bf16=true, but this GPU does not support BF16.")

        # --- 模型 / LoRA / SFT ---
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # 训练右填充

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if training.get("bf16", True) else torch.float16,
            attn_implementation=training.get("attn_implementation", "sdpa"),
        )
        model.config.use_cache = False  # 与 gradient checkpointing 兼容
        torchao_note = disable_incompatible_torchao()
        if torchao_note:
            logger.warning(torchao_note)
            manifest["torchao_compatibility"] = torchao_note
            write_json(run_directory / "manifest.json", manifest)
        target_modules = lora.get("target_modules", "all-linear")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias=str(lora.get("bias", "none")),
        )

        report_to = training.get("report_to", "none")
        if report_to == "none":
            report_to = []
        sft_config = SFTConfig(
            output_dir=str(run_directory / "checkpoints"),
            logging_dir=str(run_directory / "tensorboard"),
            run_name=run_id,
            seed=args.seed,
            data_seed=args.seed,
            max_length=int(training.get("max_length", 8192)),
            completion_only_loss=True,  # 只对 assistant completion 算 loss
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
            load_best_model_at_end=True,  # 按最高生成式验证 acc 保留最佳
            metric_for_best_model="eval_generation_accuracy",
            greater_is_better=True,
            save_total_limit=int(training.get("save_total_limit", 2)),
            report_to=report_to,
        )
        trainer = GenerativeEvalSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(validation_rows),
            processing_class=tokenizer,
            peft_config=peft_config,
            validation_rows=validation_rows,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=int(training.get("max_length", 8192)),
            generation_max_new_tokens=int(generation.get("max_new_tokens", 512)),
            run_directory=run_directory,
            model_path=model_path,
            logger=logger,
            evaluation={
                "backend": evaluation.get("backend", "auto"),
                "vllm_preferred_gpu": evaluation.get("vllm_preferred_gpu", 0),
                "vllm_max_gpu_utilization": evaluation.get(
                    "vllm_max_gpu_utilization", 10
                ),
                "vllm_min_free_memory_gib": evaluation.get(
                    "vllm_min_free_memory_gib", 20
                ),
                "vllm_gpu_memory_utilization": evaluation.get(
                    "vllm_gpu_memory_utilization", 0.9
                ),
                "max_model_len": evaluation.get(
                    "max_model_len", training.get("max_length", 8192)
                ),
                "max_tokens": evaluation.get(
                    "max_tokens", generation.get("max_new_tokens", 512)
                ),
                "temp": evaluation.get("temp", 0.0),
                "top_p": evaluation.get("top_p", 1.0),
                "seed": evaluation.get("seed", args.seed),
                "rollout": evaluation.get("rollout", 1),
                "batch_size": evaluation.get("batch_size", 64),
                "merge_cache": evaluation.get("merge_cache", "eval_data/.merged_models"),
                "enable_thinking": evaluation.get("enable_thinking", False),
            },
            callbacks=[JsonlLogCallback(run_directory / "train_history.jsonl")],
        )
        trainable_parameters = sum(
            parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
        manifest["parameters"] = {
            "trainable": trainable_parameters,
            "total": total_parameters,
            "trainable_percent": 100 * trainable_parameters / total_parameters,
        }
        write_json(run_directory / "manifest.json", manifest)

        logger.info("Starting run %s", run_id)
        logger.info("Train=%d validation=%d", len(train_rows), len(validation_rows))
        logger.info(
            "Trainable parameters=%d (%.4f%%)",
            trainable_parameters,
            manifest["parameters"]["trainable_percent"],
        )
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        language_model_metrics = trainer.evaluate()
        adapter_directory = run_directory / "adapter"
        trainer.save_model(adapter_directory)
        tokenizer.save_pretrained(adapter_directory)
        trainer.state.save_to_json(str(run_directory / "trainer_state.json"))

        validation_metrics = trainer.latest_generation_metrics
        predictions = trainer.latest_generation_predictions
        if validation_metrics is None or predictions is None:
            raise RuntimeError("Generation validation did not run during evaluation.")
        write_json(run_directory / "validation_metrics.json", validation_metrics)
        write_jsonl(run_directory / "validation_predictions.jsonl", predictions)

        summary = {
            "run_id": run_id,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "train_metrics": train_result.metrics,
            "language_model_validation": language_model_metrics,
            "generation_validation": validation_metrics,
            "generation_validation_meta": trainer.latest_generation_meta,
            "adapter_directory": str(adapter_directory),
        }
        write_json(run_directory / "summary.json", summary)
        manifest.update({"status": "completed", "finished_at_utc": utc_now()})
        write_json(run_directory / "manifest.json", manifest)
        logger.info("Completed run %s", run_id)
        logger.info("Validation metrics: %s", validation_metrics)
    except BaseException as error:
        # 失败也写回 manifest，便于事后排查
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(run_directory / "manifest.json", manifest)
        logger.exception("Run failed")
        raise


if __name__ == "__main__":
    main()
