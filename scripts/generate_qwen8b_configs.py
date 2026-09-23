#!/usr/bin/env python3
"""Generate the complete Qwen3-8B training and evaluation configuration trees."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TRAIN_ROOT = PROJECT_ROOT / "configs" / "training"
TRAIN_ROOT = SOURCE_TRAIN_ROOT / "training-8B"
EVAL_ROOT = PROJECT_ROOT / "configs" / "evaluation" / "evaluation-8b"
MODEL_NAME = "model/Qwen3-8B"
MODEL_TAG = "qwen3_8b"
TRAIN_SEED = 42
TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
METHODS = (
    "cot",
    "label_only",
    "paper_align",
    "self_correct_cot",
    "self_correct_align",
)
ALIGN_METHODS = frozenset({"paper_align", "self_correct_align"})
INTERFACES = ("cot", "label_only")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def transform_training_config(task: str, method: str) -> dict[str, Any]:
    source = SOURCE_TRAIN_ROOT / task / f"{method}.yaml"
    config = deepcopy(read_yaml(source))
    config["experiment_name"] = f"{task}#{MODEL_TAG}#{method}"
    config["model_name_or_path"] = MODEL_NAME
    config["output_root"] = f"checkpoints/training-8B/{task}/{method}"

    if method == "self_correct_cot":
        prefix = f"data/cot_self_generated_data-8B/{task}"
        config["dataset_path"] = f"{prefix}/cot/train_cot.jsonl"
        config["split_path"] = (
            f"{prefix}/cot/splits/train_cot_seed20260720.json"
        )
    elif method == "self_correct_align":
        prefix = f"data/align_self_generated_data-8B/{task}"
        config["dataset_path"] = f"{prefix}/cot/train_cot.jsonl"
        config["label_dataset_path"] = (
            f"{prefix}/label_only/train_label_only.jsonl"
        )
        config["split_path"] = (
            f"{prefix}/cot/splits/train_cot_seed20260720.json"
        )

    training = config.setdefault("training", {})
    training["per_device_train_batch_size"] = 1
    training["per_device_eval_batch_size"] = 1
    training["gradient_accumulation_steps"] = 8 if method in ALIGN_METHODS else 16
    training["save_total_limit"] = 1
    training["save_only_model"] = True

    generation = config.setdefault("generation", {})
    generation["batch_size"] = 4
    return config


def greedy_eval_config(task: str, method: str, interface: str) -> dict[str, Any]:
    max_tokens = 512 if interface == "cot" else 32
    return {
        "exp_name": (
            f"{task}#{MODEL_TAG}#ft#{method}#greedy"
            f"#on_{interface}#seed_{TRAIN_SEED}"
        ),
        "model_name": MODEL_NAME,
        "adapter": f"checkpoints/training-8B/{task}/{method}",
        "train_seed": TRAIN_SEED,
        "dataset_file": f"data/{task}/{interface}/test_{interface}.jsonl",
        "train_config": f"configs/training/training-8B/{task}/{method}.yaml",
        "inference_mode": "greedy",
        "output_path": "outputs/evaluations/evaluation-8b",
        "max_model_len": 8192,
        "max_tokens": max_tokens,
        "temp": 0,
        "top_p": 1.0,
        "seed": 42,
        "rollout": 1,
        "batch_size": 64,
        "gpu_memory_utilization": 0.9,
        "merge_cache": "/root/autodl-tmp/merged_qwen8b",
        "merge_retention_days": 0,
        "enable_thinking": False,
    }


def rail_eval_config(task: str, method: str) -> tuple[str, dict[str, Any]]:
    if method == "cot":
        interface = "cot"
        filename = "rail_on_cot.yaml"
        inference_mode = "cot_rail"
        max_tokens = 512
    elif method == "label_only":
        interface = "label_only"
        filename = "rail_on_label_only.yaml"
        inference_mode = "rail"
        max_tokens = 1
    else:
        raise ValueError(f"RAIL is not configured for method={method}")
    config = greedy_eval_config(task, method, interface)
    config["exp_name"] = (
        f"{task}#{MODEL_TAG}#ft#{method}#rail"
        f"#on_{interface}#seed_{TRAIN_SEED}"
    )
    config["inference_mode"] = inference_mode
    config["max_tokens"] = max_tokens
    return filename, config


def write_readmes() -> None:
    train_readme = """# Qwen3-8B training configs

Seven tasks and five methods mirror the Qwen3-4B and Qwen3-0.6B experiments.
Core hyperparameters are unchanged. Standard methods use effective batch 16;
paired Align methods use effective batch 8.

Self-correct configs consume data/cot_self_generated_data-8B and
data/align_self_generated_data-8B, generated from completed seed42 adapters.
"""
    eval_readme = """# Qwen3-8B evaluation configs

Each task and training method has greedy CoT and Label-only interface configs.
CoT and Label-only methods also include their RAIL evaluation configs.
Runtime queue commands override train_seed and exp_name for seeds 43 and 44.
"""
    for path, text in (
        (TRAIN_ROOT / "README.md", train_readme),
        (EVAL_ROOT / "README.md", eval_readme),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    data_readmes = {
        PROJECT_ROOT / "data" / "cot_self_generated_data-8B" / "README.md":
            "Qwen3-8B seed42 CoT-adapter self-generated training data.\n",
        PROJECT_ROOT / "data" / "align_self_generated_data-8B" / "README.md":
            "Qwen3-8B seed42 Paper-Align-adapter self-generated training data.\n",
    }
    for path, text in data_readmes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    training_count = 0
    evaluation_count = 0
    for task in TASKS:
        for method in METHODS:
            write_yaml(
                TRAIN_ROOT / task / f"{method}.yaml",
                transform_training_config(task, method),
            )
            training_count += 1
            for interface in INTERFACES:
                write_yaml(
                    EVAL_ROOT
                    / task
                    / method
                    / f"greedy_on_{interface}.yaml",
                    greedy_eval_config(task, method, interface),
                )
                evaluation_count += 1
        for method in ("cot", "label_only"):
            filename, config = rail_eval_config(task, method)
            write_yaml(EVAL_ROOT / task / method / filename, config)
            evaluation_count += 1
    write_readmes()
    print(f"generated training={training_count} evaluation={evaluation_count}")


if __name__ == "__main__":
    main()
