#!/usr/bin/env python3
"""Optimize task-level MoDE factor-mixing weights without model gradients."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    from .factor_mix import (
        adapter_weight_file,
        factor_mix_state_dict,
        l1_penalty,
        load_adapter_config,
        load_adapter_state,
        normalized_config,
        validate_adapter_compatibility,
    )
except ImportError:  # Direct execution: python MoDE/optimize_factor_mix.py
    from factor_mix import (
        adapter_weight_file,
        factor_mix_state_dict,
        l1_penalty,
        load_adapter_config,
        load_adapter_state,
        normalized_config,
        validate_adapter_compatibility,
    )


MODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODE_ROOT.parent
DEFAULT_CONFIG = MODE_ROOT / "configs" / "factor_mix-example.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--target",
        help=(
            "Override target.aspect from the YAML config. The target must have a "
            "matching datasets entry in that config."
        ),
    )
    parser.add_argument(
        "--shots-per-class",
        type=int,
        choices=(1, 3, 5),
        help="Override target.shots_per_class from the YAML config.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("nevergrad_ngopt", "scipy_differential_evolution"),
        help="Override method.optimizer from the YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, split metadata, and adapter configs without loading the model.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def expert_set_run_id(experts: Sequence[dict[str, Any]]) -> str:
    """Return a readable, bounded identifier for the ordered expert set."""
    names = [str(expert["name"]) for expert in experts]
    readable_parts = []
    for name in names:
        cleaned = "".join(
            character if character.isascii() and (character.isalnum() or character in "-_")
            else "-"
            for character in name
        ).strip("-_")
        readable_parts.append((cleaned or "expert")[:24])
    readable = "--".join(readable_parts)[:80].rstrip("-_")
    identity = [
        {
            "name": str(expert["name"]),
            "weight_identity": str(
                expert.get("weight_sha256") or expert.get("adapter", "")
            ),
        }
        for expert in experts
    ]
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"n{len(names)}-{readable}-{digest}"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_mode_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (MODE_ROOT / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return payload


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_calibration_rows(
    calibration_path: Path, *, target: str, shots_per_class: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    rows = payload.get("train") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"Calibration file must contain a non-empty train list: {calibration_path}"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"Calibration file is missing metadata: {calibration_path}")
    if metadata.get("aspect") != target:
        raise ValueError(
            f"Calibration aspect mismatch: expected {target}, got {metadata.get('aspect')}"
        )
    if int(metadata.get("shots_per_class", -1)) != shots_per_class:
        raise ValueError("Calibration shots_per_class does not match the requested split")

    raw_score_sets = metadata.get("score_sets")
    if raw_score_sets is None:
        row_score_sets = {
            tuple(row["score_sets"])
            for row in rows
            if isinstance(row, dict) and row.get("score_sets") is not None
        }
        if len(row_score_sets) > 1:
            raise ValueError(
                f"Calibration rows must share one score_sets definition: {calibration_path}"
            )
        if row_score_sets:
            raw_score_sets = list(next(iter(row_score_sets)))
        else:
            try:
                raw_score_sets = sorted(
                    {
                        int(row.get("labels", row.get("label")))
                        for row in rows
                        if isinstance(row, dict)
                    }
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot infer calibration score_sets: {calibration_path}"
                ) from error
    if (
        not isinstance(raw_score_sets, list)
        or not raw_score_sets
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_score_sets
        )
        or len(set(raw_score_sets)) != len(raw_score_sets)
    ):
        raise ValueError(f"Invalid calibration score_sets: {raw_score_sets!r}")
    score_sets = [int(value) for value in raw_score_sets]

    seen_ids: set[str] = set()
    counts = {label: 0 for label in score_sets}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), list):
            raise ValueError(f"Invalid calibration row {index} in {calibration_path}")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Missing or duplicate calibration id: {sample_id!r}")
        seen_ids.add(sample_id)
        raw_label = row.get("labels", row.get("label"))
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid calibration label for {sample_id}: {raw_label!r}")
        if isinstance(raw_label, bool) or label not in counts:
            raise ValueError(f"Invalid calibration label for {sample_id}: {raw_label!r}")
        row_scores = row.get("score_sets")
        if row_scores is not None and row_scores != score_sets:
            raise ValueError(
                f"Calibration row {sample_id} has score_sets {row_scores!r}; "
                f"expected {score_sets!r}"
            )
        counts[label] += 1
        if row.get("aspect") != target:
            raise ValueError(f"Calibration row {sample_id} has aspect {row.get('aspect')!r}")
        completion = row.get("completion")
        if (
            not isinstance(completion, list)
            or not completion
            or any(
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in completion
            )
        ):
            raise ValueError(
                f"Calibration row {sample_id} has no valid teacher completion"
            )
    expected_counts = {label: shots_per_class for label in score_sets}
    policy = str(metadata.get("shots_per_class_policy") or "")
    if policy == "nested_min_k_and_available_unique_rows_per_label":
        available = metadata.get("available_per_label") or {}
        expected_counts = {
            label: min(shots_per_class, int(available.get(str(label), shots_per_class)))
            for label in score_sets
        }
    if counts != expected_counts:
        raise ValueError(
            f"Calibration must contain {shots_per_class} rows per class; got {counts}"
        )
    metadata["score_sets"] = score_sets
    return rows, metadata


def normalized_expert_configs(experts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configs = []
    for expert in experts:
        adapter_path = Path(expert["adapter"])
        config = load_adapter_config(adapter_path)
        compatibility_config = normalized_config(config)
        run_manifest_path = adapter_path.parent / "manifest.json"
        run_manifest = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.is_file()
            else {}
        )
        configs.append(
            {
                "name": expert["name"],
                "adapter": str(adapter_path),
                "source_run_id": run_manifest.get("run_id"),
                "source_run_status": run_manifest.get("status"),
                **compatibility_config,
                "weight_sha256": sha256_file(adapter_weight_file(Path(expert["adapter"]))),
            }
        )
    hashes = [config["weight_sha256"] for config in configs]
    duplicate_hashes = sorted(
        weight_hash for weight_hash in set(hashes) if hashes.count(weight_hash) > 1
    )
    if duplicate_hashes:
        duplicate_names = [
            [config["name"] for config in configs if config["weight_sha256"] == weight_hash]
            for weight_hash in duplicate_hashes
        ]
        raise ValueError(
            "Experts must have distinct adapter weights; duplicate groups: "
            f"{duplicate_names}"
        )
    provenance_fields = {
        "name",
        "adapter",
        "source_run_id",
        "source_run_status",
        "weight_sha256",
    }
    first = {
        key: value
        for key, value in configs[0].items()
        if key not in provenance_fields
    }
    for config in configs[1:]:
        comparable = {
            key: value for key, value in config.items() if key not in provenance_fields
        }
        if comparable != first:
            raise ValueError(f"Adapter config differs for expert {config['name']}")
    return configs


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    target_config = config.setdefault("target", {})
    method_config = config.setdefault("method", {})
    if args.target:
        target_config["aspect"] = args.target
    if args.shots_per_class:
        target_config["shots_per_class"] = args.shots_per_class
    if args.optimizer:
        method_config["optimizer"] = args.optimizer

    target = str(target_config.get("aspect", "")).strip()
    if not target:
        raise ValueError("target.aspect must be a non-empty configured dataset name")
    shots_per_class = int(target_config.get("shots_per_class", 0))
    if shots_per_class not in {1, 3, 5}:
        raise ValueError("target.shots_per_class must be one of 1, 3, or 5")

    model_path = resolve_project_path(config["model_name_or_path"])
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Invalid base model directory: {model_path}")

    raw_experts = config.get("experts")
    if not isinstance(raw_experts, list) or not raw_experts:
        raise ValueError("Config must define at least one expert")
    experts: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_adapters: set[Path] = set()
    for raw_expert in raw_experts:
        if not isinstance(raw_expert, dict):
            raise ValueError("Each expert config must be an object")
        name = str(raw_expert.get("name", "")).strip()
        if not name or name in seen_names:
            raise ValueError(f"Missing or duplicate expert name: {name!r}")
        seen_names.add(name)
        adapter = resolve_project_path(raw_expert["adapter"])
        if adapter in seen_adapters:
            raise ValueError(f"Duplicate expert adapter path: {adapter}")
        seen_adapters.add(adapter)
        adapter_weight_file(adapter)
        load_adapter_config(adapter)
        experts.append({"name": name, "adapter": str(adapter)})

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or target not in datasets:
        available = sorted(datasets) if isinstance(datasets, dict) else []
        raise ValueError(
            f"Config has no datasets entry for target={target!r}; available={available}"
        )
    dataset_config = datasets[target]
    if not isinstance(dataset_config, dict):
        raise ValueError(f"datasets.{target} must be a YAML object")
    calibration_files = dataset_config.get("calibration_files")
    if not isinstance(calibration_files, dict):
        raise ValueError(f"datasets.{target}.calibration_files must be a YAML object")
    calibration_value = calibration_files.get(
        shots_per_class, calibration_files.get(str(shots_per_class))
    )
    if not calibration_value:
        raise ValueError(
            f"datasets.{target}.calibration_files has no {shots_per_class}-shot path"
        )
    calibration_path = resolve_mode_path(calibration_value)
    test_value = dataset_config.get("test_file")
    if not test_value:
        raise ValueError(f"datasets.{target}.test_file is required")
    test_path = resolve_mode_path(test_value)
    if not calibration_path.is_file():
        raise ValueError(f"Calibration file does not exist: {calibration_path}")
    if not test_path.is_file():
        raise ValueError(f"Final test file does not exist: {test_path}")
    calibration_rows, calibration_metadata = load_calibration_rows(
        calibration_path, target=target, shots_per_class=shots_per_class
    )
    metadata_test_file = calibration_metadata.get("test_file")
    if metadata_test_file and resolve_mode_path(metadata_test_file) != test_path:
        raise ValueError(
            f"Configured test file {test_path} differs from calibration metadata "
            f"{metadata_test_file}"
        )
    metadata_test_sha256 = calibration_metadata.get("test_sha256")
    test_sha256 = sha256_file(test_path)
    if metadata_test_sha256 and metadata_test_sha256 != test_sha256:
        raise ValueError(f"Final test file hash mismatch: {test_path}")

    optimizer = str(method_config.get("optimizer", "nevergrad_ngopt"))
    if optimizer not in {"nevergrad_ngopt", "scipy_differential_evolution"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    bounds = method_config.get("weight_bounds", [-3.0, 3.0])
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(value, (int, float)) for value in bounds)
        or not float(bounds[0]) < float(bounds[1])
    ):
        raise ValueError("method.weight_bounds must be [lower, upper]")

    resolved = json.loads(json.dumps(config))
    resolved["config_path"] = str(config_path)
    resolved["model_name_or_path"] = str(model_path)
    resolved["experts"] = experts
    resolved["target"].update(
        {
            "aspect": target,
            "shots_per_class": shots_per_class,
            "calibration_file": str(calibration_path),
            "calibration_sha256": sha256_file(calibration_path),
            "calibration_ids": [str(row["id"]) for row in calibration_rows],
            "test_file": str(test_path),
            "test_sha256": sha256_file(test_path),
            "test_count": int(calibration_metadata["test_count"]),
        }
    )
    resolved["method"]["optimizer"] = optimizer
    resolved["method"]["weight_bounds"] = [float(bounds[0]), float(bounds[1])]
    objective_max_length = dataset_config.get("objective_max_length")
    if objective_max_length is not None:
        objective_max_length = int(objective_max_length)
        if objective_max_length < 1:
            raise ValueError(f"datasets.{target}.objective_max_length must be positive")
        resolved.setdefault("objective", {})["max_length"] = objective_max_length
    resolved["output_root"] = str(resolve_mode_path(config.get("output_root", "outputs")))
    resolved["seed"] = int(config.get("seed", 42))
    return resolved


def chat_template_supports_thinking(tokenizer: Any) -> bool:
    return "enable_thinking" in (getattr(tokenizer, "chat_template", None) or "")


def prepare_score_records(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    completion_source: str,
    completion_template: str | None,
    max_length: int,
    enable_thinking: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    template_kwargs: dict[str, Any] = {}
    supports_thinking = chat_template_supports_thinking(tokenizer)
    if supports_thinking:
        template_kwargs["enable_thinking"] = enable_thinking
    elif enable_thinking:
        raise ValueError("Tokenizer chat template does not support enable_thinking")

    records = []
    lengths = []
    completion_lengths = []
    for row in rows:
        label = int(row.get("labels", row.get("label")))
        if completion_source == "teacher_completion":
            completion_messages = row.get("completion")
            if not isinstance(completion_messages, list) or not completion_messages:
                raise ValueError(f"Calibration row {row['id']} has no completion")
        elif completion_source == "canonical_score":
            if not completion_template:
                raise ValueError(
                    "completion_template is required for canonical_score"
                )
            completion_messages = [
                {
                    "role": "assistant",
                    "content": completion_template.format(label=label),
                }
            ]
        else:
            raise ValueError(f"Unsupported completion_source: {completion_source}")
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, **template_kwargs
        )
        full_ids = tokenizer.apply_chat_template(
            row["prompt"] + completion_messages,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat template prompt/full prefix mismatch for calibration row {row['id']}"
            )
        completion_length = len(full_ids) - len(prompt_ids)
        if completion_length < 1:
            raise ValueError(f"Calibration row {row['id']} has no completion tokens")
        if len(full_ids) > max_length:
            raise ValueError(
                f"Calibration sequence {row['id']} has {len(full_ids)} tokens, "
                f"above max_length={max_length}; no silent truncation is allowed"
            )
        lengths.append(len(full_ids))
        completion_lengths.append(completion_length)
        records.append(
            {
                "id": str(row["id"]),
                "label": label,
                "input_ids": full_ids,
                "prompt_length": len(prompt_ids),
                "completion_length": completion_length,
            }
        )
    return records, {
        "completion_source": completion_source,
        "completion_template": completion_template,
        "min_total_tokens": min(lengths),
        "max_total_tokens": max(lengths),
        "mean_total_tokens": sum(lengths) / len(lengths),
        "min_completion_tokens": min(completion_lengths),
        "max_completion_tokens": max(completion_lengths),
        "truncated_samples": 0,
    }


def disable_incompatible_torchao() -> str | None:
    version = package_version("torchao")
    if version is None:
        return None
    major, minor, *_ = (int(part) for part in version.split(".")[:2])
    if (major, minor) >= (0, 16):
        return None
    from peft.tuners.lora import torchao as peft_torchao_backend

    peft_torchao_backend.is_torchao_available = lambda: False
    return f"Disabled optional torchao {version}; PEFT requires >=0.16.0."


class FactorMixObjective:
    def __init__(
        self,
        *,
        model: Any,
        states: list[dict[str, torch.Tensor]],
        expert_names: list[str],
        score_records: list[dict[str, Any]],
        device: torch.device,
        l1_alpha: float,
        l1_reduction: str,
        history_path: Path,
    ) -> None:
        self.model = model
        self.states = states
        self.expert_names = expert_names
        self.score_records = score_records
        self.device = device
        self.l1_alpha = l1_alpha
        self.l1_reduction = l1_reduction
        self.history_path = history_path
        self.evaluation_count = 0
        self.best_record: dict[str, Any] | None = None

    @torch.inference_mode()
    def evaluate(self, weights: Sequence[float], *, stage: str = "search") -> dict[str, Any]:
        from peft.utils.save_and_load import set_peft_model_state_dict

        numeric_weights = [float(value) for value in weights]
        started = time.perf_counter()
        mixed_state = factor_mix_state_dict(self.states, numeric_weights)
        load_result = set_peft_model_state_dict(
            self.model, mixed_state, adapter_name="default"
        )
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading mixed adapter: {unexpected[:3]}")
        del mixed_state

        ce_sum = 0.0
        completion_token_count = 0
        for row in self.score_records:
            input_ids = torch.tensor(
                [row["input_ids"]], dtype=torch.long, device=self.device
            )
            prompt_length = int(row["prompt_length"])
            prediction_positions = torch.arange(
                prompt_length - 1,
                input_ids.shape[1] - 1,
                dtype=torch.long,
                device=self.device,
            )
            targets = input_ids[:, prompt_length:].reshape(-1)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=prediction_positions,
                return_dict=True,
            )
            logits = outputs.logits.reshape(-1, outputs.logits.shape[-1])
            if logits.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    f"Completion logits/target mismatch for {row['id']}: "
                    f"{logits.shape[0]} vs {targets.shape[0]}"
                )
            ce_sum += float(
                F.cross_entropy(logits.float(), targets, reduction="sum").item()
            )
            completion_token_count += targets.numel()

        completion_ce = ce_sum / completion_token_count
        regularization = l1_penalty(
            numeric_weights, alpha=self.l1_alpha, reduction=self.l1_reduction
        )
        objective = completion_ce + regularization
        self.evaluation_count += 1
        record = {
            "evaluation": self.evaluation_count,
            "stage": stage,
            "weights": {
                name: weight
                for name, weight in zip(self.expert_names, numeric_weights, strict=True)
            },
            "completion_ce": completion_ce,
            "completion_token_count": completion_token_count,
            "l1_penalty": regularization,
            "objective": objective,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
        append_jsonl(self.history_path, record)
        if self.best_record is None or objective < float(self.best_record["objective"]):
            self.best_record = record
        print(
            f"eval={self.evaluation_count} stage={stage} objective={objective:.8f} "
            f"ce={completion_ce:.8f} weights={numeric_weights}",
            flush=True,
        )
        return record

    def __call__(self, weights: Sequence[float]) -> float:
        return float(self.evaluate(weights)["objective"])


def optimize_nevergrad(
    objective: Callable[[Sequence[float]], float],
    *,
    dimensions: int,
    lower: float,
    upper: float,
    budget: int,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    try:
        import nevergrad as ng
    except ImportError as error:
        raise RuntimeError(
            "nevergrad is required for nevergrad_ngopt; install MoDE/requirements.txt "
            "or use --optimizer scipy_differential_evolution"
        ) from error

    parametrization = ng.p.Array(init=np.zeros(dimensions, dtype=float)).set_bounds(
        lower=np.full(dimensions, lower), upper=np.full(dimensions, upper)
    )
    parametrization.random_state.seed(seed)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    anchors = optimizer_anchors(dimensions)
    if budget < len(anchors):
        raise ValueError(
            f"Nevergrad budget {budget} is smaller than the {len(anchors)} required "
            "order-neutral anchor evaluations"
        )

    # NGOpt's default deterministic trajectory starts with coordinate 0 and can
    # spend a small budget around that first good point without trying the other
    # one-hot experts. Tell it every symmetric baseline before free search.
    for anchor in anchors:
        optimizer.suggest(anchor)
        candidate = optimizer.ask()
        optimizer.tell(candidate, objective(candidate.value))
    while optimizer.num_ask < budget:
        candidate = optimizer.ask()
        optimizer.tell(candidate, objective(candidate.value))

    recommendation = optimizer.provide_recommendation()
    weights = [float(value) for value in recommendation.value]
    return weights, {
        "name": "nevergrad_ngopt",
        "nevergrad_version": package_version("nevergrad"),
        "budget": budget,
        "anchor_evaluations": [anchor.tolist() for anchor in anchors],
        "anchor_count": len(anchors),
        "free_search_evaluations": budget - len(anchors),
        "reported_num_ask": int(getattr(optimizer, "num_ask", budget)),
        "reported_num_tell": int(getattr(optimizer, "num_tell", budget)),
    }


def optimizer_anchors(dimensions: int) -> list[np.ndarray]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    anchors = [np.zeros(dimensions)]
    anchors.extend(np.eye(dimensions))
    anchors.append(np.full(dimensions, 1.0 / dimensions))
    return anchors


def initial_scipy_population(
    *, dimensions: int, population_size: int, lower: float, upper: float, seed: int
) -> np.ndarray:
    minimum = max(5, dimensions + 2)
    population_size = max(population_size, minimum)
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, size=(population_size, dimensions))
    anchors = optimizer_anchors(dimensions)
    for index, anchor in enumerate(anchors[:population_size]):
        population[index] = np.clip(anchor, lower, upper)
    return population


def optimize_scipy(
    objective: Callable[[Sequence[float]], float],
    *,
    dimensions: int,
    lower: float,
    upper: float,
    generations: int,
    population_size: int,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    from scipy.optimize import differential_evolution

    population = initial_scipy_population(
        dimensions=dimensions,
        population_size=population_size,
        lower=lower,
        upper=upper,
        seed=seed,
    )
    actual_population_size = int(population.shape[0])
    result = differential_evolution(
        objective,
        bounds=[(lower, upper)] * dimensions,
        strategy="best1bin",
        maxiter=generations,
        init=population,
        rng=np.random.default_rng(seed),
        polish=False,
        updating="immediate",
        workers=1,
        disp=True,
    )
    return [float(value) for value in result.x], {
        "name": "scipy_differential_evolution",
        "scipy_version": package_version("scipy"),
        "generations": generations,
        "population_size": actual_population_size,
        "requested_population_size": population_size,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "message": str(result.message),
        "success": bool(result.success),
    }


def weights_from_record(
    record: dict[str, Any], expert_names: Sequence[str]
) -> list[float]:
    weights = record.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(expert_names):
        raise ValueError("Evaluation record weights do not match the configured experts")
    result = [float(weights[name]) for name in expert_names]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Evaluation record contains non-finite weights")
    return result


def save_mixed_adapter(
    *,
    model: Any,
    tokenizer: Any,
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
    adapter_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    from peft.utils.save_and_load import set_peft_model_state_dict

    mixed_state = factor_mix_state_dict(states, weights)
    load_result = set_peft_model_state_dict(model, mixed_state, adapter_name="default")
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading final adapter: {unexpected[:3]}")

    temporary = adapter_dir.with_name(adapter_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    write_json(temporary / "mode_meta.json", metadata)
    if adapter_dir.exists():
        raise FileExistsError(f"Refusing to overwrite adapter directory: {adapter_dir}")
    temporary.rename(adapter_dir)

    saved_state = load_adapter_state(adapter_dir, device=next(iter(mixed_state.values())).device)
    if set(saved_state) != set(mixed_state):
        raise RuntimeError("Saved mixed adapter has different tensor keys")
    max_abs_diff = 0.0
    for key in mixed_state:
        difference = float((saved_state[key] - mixed_state[key]).abs().max().item())
        max_abs_diff = max(max_abs_diff, difference)
    weight_path = adapter_weight_file(adapter_dir)
    return {
        "adapter_directory": str(adapter_dir),
        "weight_file": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "save_reload_max_abs_diff": max_abs_diff,
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    resolved = resolve_config(args)
    target = resolved["target"]["aspect"]
    shots_per_class = int(resolved["target"]["shots_per_class"])
    calibration_path = Path(resolved["target"]["calibration_file"])
    calibration_rows, calibration_metadata = load_calibration_rows(
        calibration_path, target=target, shots_per_class=shots_per_class
    )
    expert_configs = normalized_expert_configs(resolved["experts"])

    dry_run_summary = {
        "target": target,
        "shots_per_class": shots_per_class,
        "calibration_count": len(calibration_rows),
        "test_file": resolved["target"]["test_file"],
        "test_count": calibration_metadata["test_count"],
        "calibration_file": str(calibration_path),
        "optimizer": resolved["method"]["optimizer"],
        "expert_count": len(expert_configs),
        "expert_set_id": expert_set_run_id(expert_configs),
        "experts": expert_configs,
    }
    if args.dry_run:
        print(json.dumps(dry_run_summary, ensure_ascii=False, indent=2))
        return

    seed = int(resolved["seed"])
    set_seed(seed)
    output_root = Path(resolved["output_root"])
    expert_set_id = expert_set_run_id(expert_configs)
    run_id = (
        f"factor_mix__experts-{expert_set_id}__target-{target}__"
        f"k{shots_per_class}pc__seed{seed}__"
        f"{compact_utc_now()}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "resolved_config.json", resolved)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": list(sys.argv),
        "expert_count": len(expert_configs),
        "expert_set_id": expert_set_id,
        "experts": expert_configs,
        "git": git_metadata(),
        "versions": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "safetensors", "scipy", "nevergrad")
        },
        "implementation": {
            "mixing": "factor_mix: A_hat=sum(w_i*A_i), B_hat=sum(w_i*B_i)",
            "effective_delta": "(alpha/r) * B_hat @ A_hat",
            "objective": "completion-only token CE over validation teacher CoT targets",
            "objective_scope": (
                "Calibration is selected by the configured target and includes teacher "
                "reasoning plus score completion. Final test remains independent."
            ),
            "no_model_gradients": True,
            "paper_difference": (
                "Paper text also defines sum(w_i*delta_i), while Eq.(1) uses factor_mix; "
                "paper reports Shiwa, bounds +/-1.5, and L1 sum. The official code uses "
                "NGOpt, bounds +/-3, and L1 mean. This run follows resolved_config.json."
            ),
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime = resolved.get("runtime", {})
        device_name = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime requested, but torch.cuda.is_available() is false")
        device = torch.device(device_name)
        dtype_name = str(runtime.get("torch_dtype", "bfloat16"))
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported runtime.torch_dtype: {dtype_name}")
        torch_dtype = dtype_map[dtype_name]

        torchao_note = disable_incompatible_torchao()
        if torchao_note:
            manifest["torchao_compatibility"] = torchao_note
            write_json(run_dir / "manifest.json", manifest)

        model_path = Path(resolved["model_name_or_path"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            attn_implementation=str(runtime.get("attn_implementation", "sdpa")),
        ).to(device)
        first_adapter = Path(resolved["experts"][0]["adapter"])
        model = PeftModel.from_pretrained(
            base_model, first_adapter, adapter_name="default", is_trainable=False
        ).to(device)
        model.eval()
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        cache_device_name = str(runtime.get("expert_cache_device", device_name))
        cache_device = torch.device(cache_device_name)
        adapter_dirs = [Path(expert["adapter"]) for expert in resolved["experts"]]
        states = [load_adapter_state(path, device=cache_device) for path in adapter_dirs]
        compatibility = validate_adapter_compatibility(adapter_dirs, states)

        objective_config = resolved.get("objective", {})
        score_records, token_summary = prepare_score_records(
            tokenizer,
            calibration_rows,
            completion_source=str(
                objective_config.get("completion_source", "teacher_completion")
            ),
            completion_template=objective_config.get("completion_template"),
            max_length=int(objective_config.get("max_length", 8192)),
            enable_thinking=bool(objective_config.get("enable_thinking", False)),
        )
        write_json(
            run_dir / "calibration_summary.json",
            {
                "calibration_file": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
                "sample_ids": [row["id"] for row in score_records],
                "labels": [row["label"] for row in score_records],
                "tokenization": token_summary,
            },
        )

        method = resolved["method"]
        evaluator = FactorMixObjective(
            model=model,
            states=states,
            expert_names=[expert["name"] for expert in resolved["experts"]],
            score_records=score_records,
            device=device,
            l1_alpha=float(method.get("l1_alpha", 0.05)),
            l1_reduction=str(method.get("l1_reduction", "mean")),
            history_path=run_dir / "search_history.jsonl",
        )
        lower, upper = (float(value) for value in method["weight_bounds"])
        if method["optimizer"] == "nevergrad_ngopt":
            recommendation_weights, optimizer_summary = optimize_nevergrad(
                evaluator,
                dimensions=len(states),
                lower=lower,
                upper=upper,
                budget=int(method.get("budget", 40)),
                seed=seed,
            )
        else:
            recommendation_weights, optimizer_summary = optimize_scipy(
                evaluator,
                dimensions=len(states),
                lower=lower,
                upper=upper,
                generations=int(method.get("scipy_generations", 3)),
                population_size=int(method.get("scipy_population_size", 9)),
                seed=seed,
            )

        recommendation_record = evaluator.evaluate(
            recommendation_weights, stage="optimizer_recommendation_recheck"
        )
        if evaluator.best_record is None:
            raise RuntimeError("Optimizer completed without an evaluated candidate")
        selected_from_record = dict(evaluator.best_record)
        best_weights = weights_from_record(
            selected_from_record,
            [expert["name"] for expert in resolved["experts"]],
        )
        final_record = evaluator.evaluate(best_weights, stage="selected_best_recheck")
        best_payload = {
            "expert_order": [expert["name"] for expert in resolved["experts"]],
            "weights": best_weights,
            "weights_by_expert": final_record["weights"],
            "final_objective": final_record["objective"],
            "final_completion_ce": final_record["completion_ce"],
            "final_l1_penalty": final_record["l1_penalty"],
            "optimizer": optimizer_summary,
            "optimizer_recommendation_weights": recommendation_weights,
            "optimizer_recommendation_record": recommendation_record,
            "selection_rule": "lowest_evaluated_objective",
            "selected_from_record": selected_from_record,
            "best_observed_record": evaluator.best_record,
        }
        write_json(run_dir / "best_weights.json", best_payload)

        adapter_metadata = {
            "method": "mode_official_factor_mix",
            "target": target,
            "shots_per_class": shots_per_class,
            "expert_order": best_payload["expert_order"],
            "weights": best_weights,
            "source_adapters": expert_configs,
            "objective": resolved["objective"],
            "method_config": method,
            "calibration_file": str(calibration_path),
            "calibration_sha256": sha256_file(calibration_path),
            "run_id": run_id,
        }
        adapter_summary = save_mixed_adapter(
            model=model,
            tokenizer=tokenizer,
            states=states,
            weights=best_weights,
            adapter_dir=run_dir / "adapter",
            metadata=adapter_metadata,
        )
        summary = {
            "run_id": run_id,
            "target": target,
            "shots_per_class": shots_per_class,
            "calibration_samples": len(calibration_rows),
            "test_file": resolved["target"]["test_file"],
            "test_samples": calibration_metadata["test_count"],
            "compatibility": compatibility,
            "tokenization": token_summary,
            "best": best_payload,
            "adapter": adapter_summary,
            "finished_at_utc": utc_now(),
        }
        write_json(run_dir / "summary.json", summary)
        manifest.update({"status": "completed", "finished_at_utc": utc_now()})
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
