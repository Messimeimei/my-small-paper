#!/usr/bin/env python3
"""Evaluate a base model (optionally + LoRA adapter/checkpoint) with vLLM.

Without --adapter (or with none/None/NONE), loads the base model directly.
With an adapter path, merges LoRA into a temporary full model then loads with
plain vLLM (vLLM 0.8.4 + cachetools>=6 breaks enable_lora in spawned workers).
The merged weights are deleted after evaluation finishes (or on failure).

Prefer YAML under eval_output/configs/<task>/<train_method>/ via --config;
CLI flags override config values. Free greedy generation, score-only RAIL, and two-stage CoT-RAIL
are supported. Supports
data/<task>/{cot,label_only}/test_*.jsonl (labels field) and legacy JSON.

Writes to eval_output/results/<task>/<exp_name>/ (metrics.json, predictions.jsonl,
resolved_config.json) and rebuilds eval_output/evaluation_analysis.md from all
discovered metrics.json files (no comparison_table.md).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from eval_analysis import update_evaluation_analysis
from metrics_utils import (
    REASONING_RE,
    classification_metrics,
    criterion_title,
    extract_score,
    infer_eval_condition,
    infer_supervision_mode,
    infer_task_name,
    short_model_name,
    token_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _import_vllm():
    """Lazy import so --help works before vLLM is installed."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "vLLM is required for evaluation. Install with: "
            "python -m pip install -r requirements-eval.txt"
        ) from exc
    return LLM, SamplingParams


DEFAULT_DATASET = PROJECT_ROOT / "data/rw_gen_coherence/cot/test_cot.jsonl"
# Prefer data disk for temporary merges (system disk is often too small for 4B weights).
_AUTODL_TMP = Path("/root/autodl-tmp")
DEFAULT_MERGE_CACHE = (
    _AUTODL_TMP / "merged" if _AUTODL_TMP.is_dir() else PROJECT_ROOT / "merged"
)
DEFAULT_EVAL_OUTPUT_ROOT = PROJECT_ROOT / "eval_output"
DEFAULT_MERGE_RETENTION_DAYS = 0
RAIL_PROBABILITY_NORMALIZATION = "full_vocab_raw"
RAIL_IMPLEMENTATION = "tract_official_release"
RAIL_EXPECTATION_FORMULA = "sum(score * p_full_vocab(score))"
RAIL_DISCRETE_DECODING = "nearest_legal_score_tie_low"

# YAML keys -> argparse destinations (same names as CLI flags without --).
CONFIG_KEYS = {
    "exp_name",
    "model_name",
    "adapter",
    "train_seed",
    "dataset_file",
    "inference_mode",
    "output_path",
    "max_model_len",
    "max_tokens",
    "temp",
    "top_p",
    "seed",
    "rollout",
    "batch_size",
    "gpu_memory_utilization",
    "merge_cache",
    "merge_retention_days",
    "enable_thinking",
    "train_config",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_name_from_exp(exp_name: str) -> str:
    """Task folder name: part of exp_name before the first '#'."""
    return exp_name.split("#", 1)[0]


def eval_run_dir(output_root: Path, exp_name: str, task_name: str | None = None) -> Path:
    """Layout: <output_root>/<task>/<exp_name>/"""
    task = task_name or task_name_from_exp(exp_name)
    return output_root / task / exp_name


def read_eval_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"Eval config must be a YAML object: {path}")
    required = {"exp_name", "model_name", "dataset_file"}
    missing = required - set(config)
    if missing:
        raise SystemExit(f"Eval config {path} missing fields: {sorted(missing)}")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise SystemExit(
            f"Eval config {path} has unknown fields: {sorted(unknown)}. "
            f"Allowed: {sorted(CONFIG_KEYS)}"
        )
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--refresh-analysis-only", action="store_true")
    pre_args, _ = pre.parse_known_args(argv)

    config_defaults: dict[str, Any] = {}
    config_path: Path | None = None
    analysis_only = bool(pre_args.refresh_analysis_only)
    if pre_args.config:
        config_path = resolve_path(pre_args.config)
        if not config_path.is_file():
            raise SystemExit(f"Eval config not found: {config_path}")
        loaded = read_eval_config(config_path)
        for key, value in loaded.items():
            if key in CONFIG_KEYS:
                config_defaults[key] = value

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-analysis-only",
        action="store_true",
        help="Rebuild eval_output/evaluation_analysis.md from existing metrics.json and exit.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Eval YAML under eval_output/configs/<task>/<train_method>/; "
            "e.g. base/greedy_on_cot.yaml, cot/greedy_on_label_only.yaml. "
            "CLI flags override values from the config."
        ),
    )
    parser.add_argument(
        "--exp_name",
        default=config_defaults.get("exp_name"),
        required=(not analysis_only and "exp_name" not in config_defaults),
    )
    parser.add_argument(
        "--model_name",
        default=config_defaults.get("model_name"),
        required=(not analysis_only and "model_name" not in config_defaults),
        help="Base model path.",
    )
    parser.add_argument(
        "--adapter",
        default=config_defaults.get("adapter"),
        help=(
            "LoRA adapter/checkpoint path, or a training-method output root "
            "whose latest completed adapter is selected by --train_seed. "
            "Omit, or pass none/None/NONE, to evaluate the base model only."
        ),
    )
    parser.add_argument(
        "--train_seed",
        type=int,
        default=(
            int(config_defaults["train_seed"])
            if config_defaults.get("train_seed") is not None
            else None
        ),
        help="Training seed used when --adapter points to a method output root.",
    )
    parser.add_argument(
        "--dataset_file",
        default=config_defaults.get("dataset_file", str(DEFAULT_DATASET)),
    )
    parser.add_argument(
        "--output_path",
        default=config_defaults.get("output_path", str(DEFAULT_EVAL_OUTPUT_ROOT)),
        help=(
            "Eval output root (default: <project>/eval_output). "
            "Each run writes to <output_path>/<task>/<exp_name>/."
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=int(config_defaults.get("max_model_len", 8192)),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(config_defaults.get("max_tokens", 512)),
    )
    parser.add_argument(
        "--inference_mode",
        choices=("greedy", "rail", "cot_rail"),
        default=str(config_defaults.get("inference_mode", "greedy")),
        help=(
            "greedy preserves free generation; rail applies the released TRACT "
            "full-vocabulary scorer after a direct <score> prefix; cot_rail first "
            "generates CoT to <score>, then applies the same scorer."
        ),
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=float(config_defaults.get("temp", 0.0)),
        help="Must be 0: evaluation is deterministic.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(config_defaults.get("top_p", 1.0)),
        help="Must be 1.0: evaluation is deterministic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config_defaults.get("seed", 42)),
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=int(config_defaults.get("rollout", 1)),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config_defaults.get("batch_size", 64)),
        help="Prompt chunk size for progress logging; vLLM schedules internally.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(config_defaults.get("gpu_memory_utilization", 0.9)),
        help="vLLM GPU memory fraction of the visible device.",
    )
    parser.add_argument(
        "--merge_cache",
        default=config_defaults.get("merge_cache", str(DEFAULT_MERGE_CACHE)),
        help=(
            "Scratch directory for temporary merged models "
            f"(default: {DEFAULT_MERGE_CACHE}). Deleted after each eval."
        ),
    )
    parser.add_argument(
        "--merge_retention_days",
        type=float,
        default=float(
            config_defaults.get("merge_retention_days", DEFAULT_MERGE_RETENTION_DAYS)
        ),
        help=(
            "Before eval, delete leftover merged dirs older than this many days "
            f"(default: {DEFAULT_MERGE_RETENTION_DAYS} = remove all leftovers). "
            "The merged model from the current run is always deleted after use."
        ),
    )
    thinking_default = bool(config_defaults.get("enable_thinking", False))
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        default=thinking_default,
        help="Enable native thinking in chat template when supported.",
    )
    thinking_group.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable native thinking even if config enables it.",
    )
    parser.add_argument(
        "--train_config",
        default=config_defaults.get("train_config"),
        help="Optional training YAML to embed in metrics (seed/config provenance).",
    )
    args = parser.parse_args(argv)
    args.config_path = str(config_path) if config_path is not None else None
    return args


def has_adapter_weights(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def training_run_seed(run_directory: Path) -> int | None:
    summary_path = run_directory / "summary.json"
    if summary_path.is_file():
        try:
            value = read_json(summary_path).get("seed")
            return int(value) if value is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    match = re.search(r"(?:^|__)seed(\d+)(?:__|$)", run_directory.name, re.I)
    return int(match.group(1)) if match else None


def normalize_adapter(
    value: str | None, *, train_seed: int | None = None
) -> Path | None:
    """Resolve a direct adapter or the latest completed run under a method root."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    path = resolve_path(text)
    if not path.is_dir():
        raise SystemExit(f"Adapter path does not exist or is not a directory: {path}")
    if has_adapter_weights(path):
        return path

    run_adapter = path / "adapter"
    if has_adapter_weights(run_adapter):
        run_seed = training_run_seed(path)
        if train_seed is not None and run_seed is not None and run_seed != train_seed:
            raise SystemExit(
                f"Adapter run seed is {run_seed}, but --train_seed={train_seed}: {path}"
            )
        return run_adapter.resolve()

    candidates: list[tuple[float, str, Path]] = []
    for run_directory in path.iterdir():
        if not run_directory.is_dir():
            continue
        candidate = run_directory / "adapter"
        summary_path = run_directory / "summary.json"
        if not has_adapter_weights(candidate) or not summary_path.is_file():
            continue
        run_seed = training_run_seed(run_directory)
        if train_seed is not None and run_seed != train_seed:
            continue
        candidates.append(
            (summary_path.stat().st_mtime, run_directory.name, candidate.resolve())
        )
    if not candidates:
        seed_text = f" for training seed {train_seed}" if train_seed is not None else ""
        raise SystemExit(f"No completed adapter found under {path}{seed_text}.")
    selected = max(candidates)[2]
    print(f"selected completed adapter: {selected}", flush=True)
    return selected


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_score_sets(raw: Any, *, context: str) -> list[int]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError(f"{context} has invalid score_sets: {raw!r}")
    return list(raw)


def parse_label(row: dict[str, Any], index: int, allowed_scores: set[int]) -> int:
    raw = row.get("labels", row.get("label"))
    try:
        label = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Row {index} has invalid label: {raw!r}") from error
    if isinstance(raw, bool) or label not in allowed_scores:
        raise ValueError(
            f"Row {index} label {raw!r} is outside score_sets "
            f"{sorted(allowed_scores)}"
        )
    return label


def load_dataset_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load eval rows from JSONL (one object per line) or legacy JSON wrappers."""
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                rows.append(row)
        return rows, None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("test", payload.get("train"))
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = None
    else:
        rows = payload
        metadata = None
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list or a test/train list.")
    return rows, metadata


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows, metadata = load_dataset_rows(path)
    if not rows:
        raise ValueError(f"{path} must contain a non-empty test/train list.")

    declared_score_sets: list[tuple[str, list[int]]] = []
    if isinstance(metadata, dict) and metadata.get("score_sets") is not None:
        declared_score_sets.append(
            (
                "metadata",
                normalize_score_sets(
                    metadata["score_sets"], context=f"{path} metadata"
                ),
            )
        )
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("score_sets") is not None:
            declared_score_sets.append(
                (
                    f"row {index}",
                    normalize_score_sets(
                        row["score_sets"], context=f"{path} row {index}"
                    ),
                )
            )
    if declared_score_sets:
        score_sets = declared_score_sets[0][1]
        for location, values in declared_score_sets[1:]:
            if values != score_sets:
                raise ValueError(
                    f"{path} {location} score_sets {values} does not match "
                    f"{score_sets}"
                )
    else:
        observed_labels = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Row {index} is not an object.")
            raw = row.get("labels", row.get("label"))
            try:
                label = int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {index} has invalid label: {raw!r}") from error
            if isinstance(raw, bool):
                raise ValueError(f"Row {index} has invalid label: {raw!r}")
            observed_labels.append(label)
        score_sets = sorted(set(observed_labels))
    allowed_scores = set(score_sets)

    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), list):
            raise ValueError(f"Row {index} missing a valid prompt list.")
        sample_id = str(row.get("id", "")).strip() or f"row_{index:04d}"
        if sample_id in seen:
            raise ValueError(f"Duplicate id: {sample_id}")
        seen.add(sample_id)
        cleaned.append(
            {
                "id": sample_id,
                "label": parse_label(row, index, allowed_scores),
                "prompt": row["prompt"],
                "task": row.get("task"),
                "aspect": row.get("aspect"),
                "evaluation_mode": row.get("evaluation_mode")
                or row.get("supervision_mode"),
                "prompt_version": row.get("prompt_version"),
                "score_sets": score_sets,
            }
        )
    return cleaned, score_sets


def mean_rollout_metrics(
    rollout_metrics: list[dict[str, Any]], score_sets: list[int]
) -> dict[str, Any]:
    """Average metrics across rollouts instead of voting across predictions."""

    def summarize(values: list[float | int | None]) -> tuple[float | None, float | None]:
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            return None, None
        array = np.asarray(cleaned, dtype=float)
        return float(array.mean()), float(array.std())

    aggregate: dict[str, Any] = {
        "samples": rollout_metrics[0]["samples"],
        "score_sets": score_sets,
        "per_class": {},
    }
    scalar_metrics = (
        "accuracy",
        "macro_f1",
        "format_valid_rate",
        "mae",
        "qwk",
        "rail_mae",
        "rail_mse",
        "rail_rmse",
        "avg_score_probability_mass",
        "score_prefix_valid_rate",
        "reasoning_valid_rate",
    )
    for metric in scalar_metrics:
        if not any(metric in rollout for rollout in rollout_metrics):
            continue
        mean, std = summarize([rollout.get(metric) for rollout in rollout_metrics])
        aggregate[metric] = mean
        aggregate[f"{metric}_std"] = std
    for field in (
        "probability_normalization",
        "candidate_renormalization",
        "rail_implementation",
        "rail_expectation_formula",
        "discrete_decoding",
    ):
        values = {rollout.get(field) for rollout in rollout_metrics}
        values.discard(None)
        if len(values) > 1:
            raise ValueError(f"Rollouts disagree on {field}: {sorted(values)}")
        if values:
            aggregate[field] = values.pop()

    for label in score_sets:
        label_key = str(label)
        per_class = {
            "support": rollout_metrics[0]["per_class"][label_key]["support"]
        }
        for metric in ("precision", "recall", "f1"):
            mean, std = summarize(
                [
                    rollout["per_class"][label_key][metric]
                    for rollout in rollout_metrics
                ]
            )
            per_class[metric] = mean
            per_class[f"{metric}_std"] = std
        aggregate["per_class"][label_key] = per_class
    return aggregate


def chat_template_supports_thinking(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def format_prompts(
    tokenizer,
    prompts: list[list[dict[str, Any]]],
    enable_thinking: bool,
) -> list[str]:
    kwargs: dict[str, Any] = {}
    if chat_template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    return [
        tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        for prompt in prompts
    ]


def format_rail_prompts(
    tokenizer,
    prompts: list[list[dict[str, Any]]],
) -> list[str]:
    kwargs: dict[str, Any] = {}
    if chat_template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": "<score>"}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            **kwargs,
        )
        for prompt in prompts
    ]


def resolve_score_token_ids(tokenizer, score_sets: list[int]) -> list[int]:
    token_ids: list[int] = []
    for score in score_sets:
        encoded = tokenizer.encode(str(score), add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(
                f"RAIL requires score {score!r} to be exactly one tokenizer token; "
                f"got token IDs {encoded}."
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise SystemExit(
            f"RAIL score tokens must be unique: scores={score_sets}, token_ids={token_ids}"
        )
    return token_ids


def official_rail_statistics(
    score_logprobs: dict[int, float],
) -> dict[str, Any]:
    """Apply the released TRACT RAIL scorer to full-vocabulary log-probs.

    Each value is already a log probability under the complete vocabulary.
    TRACT selects the legal score-token probabilities and computes their raw
    weighted sum; it does not renormalize the selected candidates.
    """
    if not score_logprobs:
        raise ValueError("RAIL requires at least one score candidate.")
    if any(
        math.isnan(value) or value == math.inf
        for value in score_logprobs.values()
    ):
        raise ValueError(f"RAIL received invalid log probabilities: {score_logprobs}")

    probabilities = {
        score: math.exp(logprob)
        for score, logprob in score_logprobs.items()
    }
    probability_mass = math.fsum(probabilities.values())
    if probability_mass > 1.0 + 1e-6:
        raise ValueError(
            "RAIL legal score probability mass exceeds one; the supplied values "
            f"are not full-vocabulary log probabilities: {probability_mass}"
        )
    expected_score = math.fsum(
        score * probability for score, probability in probabilities.items()
    )
    return {
        "expected_score": expected_score,
        "score_probabilities": probabilities,
        "score_probability_mass": probability_mass,
    }


def nearest_legal_score(expected_score: float, score_sets: list[int]) -> int:
    """Map a continuous score to the nearest label, breaking ties downward."""
    return min(
        sorted(score_sets),
        key=lambda score: (abs(score - expected_score), score),
    )


def extract_requested_logprobs(
    request_output,
    score_sets: list[int],
    score_token_ids: list[int],
) -> dict[int, float]:
    generated = request_output.outputs[0]
    if not generated.logprobs:
        raise RuntimeError("vLLM did not return token log probabilities for RAIL.")
    first_position = generated.logprobs[0]
    result: dict[int, float] = {}
    for score, token_id in zip(score_sets, score_token_ids, strict=True):
        entry = first_position.get(token_id)
        if entry is None:
            raise RuntimeError(
                f"vLLM omitted requested score token {score!r} (token ID {token_id})."
            )
        value = getattr(entry, "logprob", entry)
        result[score] = float(value)
    return result


def build_rail_sampling_params(score_token_ids: list[int], seed: int):
    _, SamplingParams = _import_vllm()
    return SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        logprobs=len(score_token_ids),
        logprob_token_ids=score_token_ids,
    )


def build_cot_rail_sampling_params(max_tokens: int, seed: int):
    _, SamplingParams = _import_vllm()
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        stop=["<score>"],
        include_stop_str_in_output=True,
    )


def disable_incompatible_torchao() -> str | None:
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


def adapter_weight_file(adapter: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter / name
        if path.is_file():
            return path
    raise SystemExit(f"No adapter weights found under {adapter}")


def merged_model_dir(base: Path, adapter: Path, cache_root: Path) -> Path:
    weight = adapter_weight_file(adapter)
    digest = hashlib.sha1()
    digest.update(str(base).encode())
    digest.update(str(adapter).encode())
    digest.update(str(weight.stat().st_mtime_ns).encode())
    digest.update(str(weight.stat().st_size).encode())
    return cache_root / digest.hexdigest()[:16]


def _merged_entry_mtime(path: Path) -> float:
    marker = path / ".ok"
    if marker.is_file():
        return marker.stat().st_mtime
    return path.stat().st_mtime


def cleanup_merged_cache(cache_root: Path, retention_days: float) -> None:
    """Remove leftover merged-model directories under cache_root.

    retention_days <= 0 means remove every leftover entry (default), since merged
    weights are intended to be temporary and deleted after each eval.
    """
    if not cache_root.is_dir():
        return
    cutoff = None if retention_days <= 0 else time.time() - retention_days * 86400
    removed = 0
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = _merged_entry_mtime(child)
        except OSError:
            continue
        if cutoff is not None and mtime >= cutoff:
            continue
        try:
            shutil.rmtree(child)
            removed += 1
            print(f"removed leftover merged cache: {child}", flush=True)
        except OSError as exc:
            print(f"failed to remove {child}: {exc}", flush=True)
    if removed:
        age_msg = (
            "all leftovers"
            if retention_days <= 0
            else f"older than {retention_days:g} day(s)"
        )
        print(
            f"cleaned {removed} merged cache entr"
            f"{'y' if removed == 1 else 'ies'} ({age_msg})",
            flush=True,
        )


def remove_merged_model(path: Path | None) -> None:
    """Delete a temporary merged-model directory if it still exists."""
    if path is None:
        return
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        print(f"removed temporary merged model: {path}", flush=True)
    except OSError as exc:
        print(f"failed to remove merged model {path}: {exc}", flush=True)
    # Also drop a sibling .tmp directory from a crashed merge.
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        try:
            shutil.rmtree(temporary)
            print(f"removed incomplete merge dir: {temporary}", flush=True)
        except OSError as exc:
            print(f"failed to remove {temporary}: {exc}", flush=True)


def ensure_merged_model(base: Path, adapter: Path, cache_root: Path) -> Path:
    """Merge LoRA on CPU into a temporary directory for plain vLLM loading.

    Caller must delete the returned path after evaluation (see remove_merged_model).
    """
    out = merged_model_dir(base, adapter, cache_root)
    # Never reuse on-disk merges: always rebuild, then delete after eval.
    if out.exists():
        shutil.rmtree(out)

    note = disable_incompatible_torchao()
    if note:
        print(note, flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    print(f"merging LoRA on CPU -> {out} (temporary; deleted after eval)", flush=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    write_json(
        temporary / "merge_meta.json",
        {
            "base_model": str(base),
            "adapter": str(adapter),
            "merged_at_utc": utc_now(),
        },
    )
    del model
    temporary.rename(out)
    (out / ".ok").write_text(utc_now() + "\n", encoding="utf-8")
    print(f"merge done in {time.perf_counter() - started:.1f}s", flush=True)
    return out


def init_vllm(
    model_path: Path,
    *,
    max_model_len: int,
    max_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
) -> tuple[Any, Any]:
    LLM, SamplingParams = _import_vllm()
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    )
    return llm, sampling_params


def load_train_run_metadata(adapter: Path | None) -> dict[str, Any]:
    """Pull exported-checkpoint metadata from a training run directory."""
    if adapter is None:
        return {}
    run_dir = adapter.parent if adapter.name == "adapter" else adapter
    if run_dir.name.startswith("checkpoint-"):
        run_dir = run_dir.parent.parent
    meta: dict[str, Any] = {"train_run_directory": str(run_dir)}
    summary_path = run_dir / "summary.json"
    state_path = run_dir / "trainer_state.json"
    config_path = run_dir / "resolved_config.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        uses_adapter_source = summary.get("adapter_source_checkpoint") is not None
        uses_final_checkpoint = summary.get("final_checkpoint") is not None
        meta.update(
            {
                "checkpoint": summary.get("adapter_source_checkpoint")
                or summary.get("final_checkpoint")
                or summary.get("best_checkpoint"),
                "checkpoint_epoch": summary.get("adapter_source_checkpoint_epoch")
                if uses_adapter_source
                else (
                    summary.get("final_checkpoint_epoch")
                    if uses_final_checkpoint
                    else summary.get("best_checkpoint_epoch")
                ),
                "checkpoint_step": summary.get("adapter_source_checkpoint_step")
                if uses_adapter_source
                else (
                    summary.get("final_checkpoint_step")
                    if uses_final_checkpoint
                    else summary.get("best_checkpoint_step")
                ),
                "generation_accuracy": summary.get("final_generation_accuracy")
                if uses_final_checkpoint
                else summary.get("best_generation_accuracy"),
                "checkpoint_retention": summary.get("checkpoint_retention")
                or "best",
                "adapter_selection": summary.get("adapter_selection")
                or ("final_checkpoint" if uses_final_checkpoint else "best_checkpoint"),
                "train_run_id": summary.get("run_id"),
                "train_seed": summary.get("seed"),
            }
        )
    if state_path.is_file() and meta.get("checkpoint_epoch") is None:
        state = read_json(state_path)
        meta.setdefault("checkpoint_step", state.get("best_global_step"))
        meta.setdefault("generation_accuracy", state.get("best_metric"))
        best_step = state.get("best_global_step")
        for row in reversed(state.get("log_history") or []):
            if best_step is not None and int(row.get("step", -1)) == int(best_step):
                if "epoch" in row:
                    meta["checkpoint_epoch"] = float(row["epoch"])
                break
    if config_path.is_file():
        meta["train_resolved_config"] = read_json(config_path)
    return meta


def load_optional_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else None


def resolved_eval_config(args: argparse.Namespace, *, adapter: Path | None) -> dict[str, Any]:
    """Snapshot of the effective eval settings written next to metrics."""
    return {
        "config": args.config_path,
        "exp_name": args.exp_name,
        "model_name": str(resolve_path(args.model_name)),
        "adapter": str(adapter) if adapter is not None else None,
        "adapter_request": args.adapter,
        "train_seed": args.train_seed,
        "dataset_file": str(resolve_path(args.dataset_file)),
        "output_path": str(resolve_path(args.output_path)),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "effective_max_tokens": 1 if args.inference_mode == "rail" else args.max_tokens,
        "reasoning_max_tokens": (
            args.max_tokens if args.inference_mode == "cot_rail" else None
        ),
        "score_probe_max_tokens": (
            1 if args.inference_mode in {"rail", "cot_rail"} else None
        ),
        "inference_mode": args.inference_mode,
        "probability_normalization": (
            RAIL_PROBABILITY_NORMALIZATION
            if args.inference_mode in {"rail", "cot_rail"}
            else None
        ),
        "candidate_renormalization": (
            False if args.inference_mode in {"rail", "cot_rail"} else None
        ),
        "rail_implementation": (
            RAIL_IMPLEMENTATION
            if args.inference_mode in {"rail", "cot_rail"}
            else None
        ),
        "rail_expectation_formula": (
            RAIL_EXPECTATION_FORMULA
            if args.inference_mode in {"rail", "cot_rail"}
            else None
        ),
        "discrete_decoding": (
            RAIL_DISCRETE_DECODING
            if args.inference_mode in {"rail", "cot_rail"}
            else None
        ),
        "temp": args.temp,
        "top_p": args.top_p,
        "seed": args.seed,
        "rollout": args.rollout,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "merge_cache": str(resolve_path(args.merge_cache)),
        "merge_retention_days": args.merge_retention_days,
        "enable_thinking": args.enable_thinking,
        "train_config": args.train_config,
    }


def gpu_time_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return info
    info["device_name"] = torch.cuda.get_device_name(0)
    info["memory_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
    info["memory_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
    return info


def run_rollout(
    llm,
    sampling_params,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    args: argparse.Namespace,
    rollout_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = llm.get_tokenizer()
    supports_thinking = chat_template_supports_thinking(tokenizer)
    if args.enable_thinking and not supports_thinking:
        raise SystemExit(
            "This model chat template does not support enable_thinking; "
            "remove --enable_thinking."
        )

    predictions: list[dict[str, Any]] = []
    allowed_scores = set(score_sets)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
    else:
        gpu_started = started

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = format_prompts(
            tokenizer, [row["prompt"] for row in batch], args.enable_thinking
        )
        completions = llm.generate(texts, sampling_params, use_tqdm=False)
        outputs = [completion.outputs[0].text for completion in completions]
        for row, output in zip(batch, outputs, strict=True):
            prediction = extract_score(output, allowed_scores)
            predictions.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "output": output,
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )
        done = min(start + args.batch_size, len(rows))
        print(f"[rollout {rollout_index}] {done}/{len(rows)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    metrics["elapsed_sec"] = round(elapsed, 3)
    metrics["gpu_time_sec"] = round(gpu_elapsed, 3)
    metrics["samples_per_sec"] = round(len(rows) / max(elapsed, 1e-9), 3)
    metrics["tokens"] = token_stats(predictions, tokenizer)
    return predictions, metrics


def run_rail_rollout(
    llm,
    sampling_params,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    score_token_ids: list[int],
    args: argparse.Namespace,
    rollout_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply official full-vocabulary RAIL after the fixed <score> prefix."""
    tokenizer = llm.get_tokenizer()
    predictions: list[dict[str, Any]] = []
    generated_token_counts: list[int] = []
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
    else:
        gpu_started = started

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = format_rail_prompts(tokenizer, [row["prompt"] for row in batch])
        completions = llm.generate(texts, sampling_params, use_tqdm=False)
        for row, completion in zip(batch, completions, strict=True):
            generated = completion.outputs[0]
            score_logprobs = extract_requested_logprobs(
                completion, score_sets, score_token_ids
            )
            statistics = official_rail_statistics(score_logprobs)
            expected_score = float(statistics["expected_score"])
            prediction = nearest_legal_score(expected_score, score_sets)
            generated_token_counts.append(len(generated.token_ids))
            predictions.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "output": f"<score>{expected_score:.6f}</score>",
                    "raw_output": generated.text,
                    "generated_token_ids": list(generated.token_ids),
                    "expected_score": expected_score,
                    "score_probability_mass": float(
                        statistics["score_probability_mass"]
                    ),
                    "score_probabilities": {
                        str(score): float(probability)
                        for score, probability in statistics[
                            "score_probabilities"
                        ].items()
                    },
                    "score_logprobs": {
                        str(score): float(logprob)
                        for score, logprob in score_logprobs.items()
                    },
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )
        done = min(start + args.batch_size, len(rows))
        print(f"[rail rollout {rollout_index}] {done}/{len(rows)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    squared_errors = [
        (row["expected_score"] - row["label"]) ** 2 for row in predictions
    ]
    absolute_errors = [
        abs(row["expected_score"] - row["label"]) for row in predictions
    ]
    metrics.update(
        {
            "rail_mae": sum(absolute_errors) / len(absolute_errors),
            "rail_mse": sum(squared_errors) / len(squared_errors),
            "rail_rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
            "avg_score_probability_mass": sum(
                row["score_probability_mass"] for row in predictions
            )
            / len(predictions),
            "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
            "candidate_renormalization": False,
            "rail_implementation": RAIL_IMPLEMENTATION,
            "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
            "discrete_decoding": RAIL_DISCRETE_DECODING,
            "score_prefix": "<score>",
            "elapsed_sec": round(elapsed, 3),
            "gpu_time_sec": round(gpu_elapsed, 3),
            "samples_per_sec": round(len(rows) / max(elapsed, 1e-9), 3),
            "tokens": {
                "avg_output_tokens": sum(generated_token_counts)
                / len(generated_token_counts),
                "avg_reasoning_tokens": 0.0,
                "total_output_tokens": sum(generated_token_counts),
                "total_reasoning_tokens": 0,
                "samples": len(predictions),
            },
        }
    )
    return predictions, metrics


def run_cot_rail_rollout(
    llm,
    cot_sampling_params,
    score_sampling_params,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    score_token_ids: list[int],
    args: argparse.Namespace,
    rollout_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate reasoning to <score>, then apply official full-vocabulary RAIL."""
    tokenizer = llm.get_tokenizer()
    predictions: list[dict[str, Any]] = []
    output_token_counts: list[int] = []
    reasoning_token_counts: list[int] = []
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
    else:
        gpu_started = started

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        base_texts = format_prompts(
            tokenizer,
            [row["prompt"] for row in batch],
            enable_thinking=False,
        )
        cot_completions = llm.generate(
            base_texts,
            cot_sampling_params,
            use_tqdm=False,
        )

        stage_rows: list[dict[str, Any]] = []
        score_prompts: list[str] = []
        score_indexes: list[int] = []
        for index, (row, base_text, completion) in enumerate(
            zip(batch, base_texts, cot_completions, strict=True)
        ):
            generated = completion.outputs[0]
            cot_output = generated.text
            stop_reason = getattr(generated, "stop_reason", None)
            prefix_reached = (
                stop_reason == "<score>" and cot_output.endswith("<score>")
            )
            reasoning_match = REASONING_RE.search(cot_output)
            reasoning_text = (
                reasoning_match.group(1).strip() if reasoning_match is not None else ""
            )
            reasoning_valid = bool(reasoning_text)
            stage_rows.append(
                {
                    "row": row,
                    "cot_output": cot_output,
                    "cot_generated_token_ids": list(generated.token_ids),
                    "cot_finish_reason": getattr(generated, "finish_reason", None),
                    "cot_stop_reason": stop_reason,
                    "score_prefix_reached": prefix_reached,
                    "reasoning_valid": reasoning_valid,
                    "reasoning_tokens": len(
                        tokenizer.encode(reasoning_text, add_special_tokens=False)
                    ),
                }
            )
            if prefix_reached:
                score_indexes.append(index)
                score_prompts.append(base_text + cot_output)

        score_completions = (
            llm.generate(score_prompts, score_sampling_params, use_tqdm=False)
            if score_prompts
            else []
        )
        score_by_index = dict(zip(score_indexes, score_completions, strict=True))

        for index, stage in enumerate(stage_rows):
            row = stage["row"]
            cot_ids = stage["cot_generated_token_ids"]
            score_completion = score_by_index.get(index)
            if score_completion is None:
                output_token_counts.append(len(cot_ids))
                reasoning_token_counts.append(stage["reasoning_tokens"])
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": None,
                        "correct": False,
                        "output": stage["cot_output"],
                        "raw_output": stage["cot_output"],
                        "cot_output": stage["cot_output"],
                        "cot_generated_token_ids": cot_ids,
                        "cot_finish_reason": stage["cot_finish_reason"],
                        "cot_stop_reason": stage["cot_stop_reason"],
                        "score_prefix_reached": False,
                        "reasoning_valid": stage["reasoning_valid"],
                        "score_probe_text": None,
                        "score_probe_token_ids": None,
                        "expected_score": None,
                        "score_probability_mass": None,
                        "score_probabilities": None,
                        "score_logprobs": None,
                        "task": row.get("task"),
                        "aspect": row.get("aspect"),
                    }
                )
                continue

            score_generated = score_completion.outputs[0]
            score_logprobs = extract_requested_logprobs(
                score_completion,
                score_sets,
                score_token_ids,
            )
            statistics = official_rail_statistics(score_logprobs)
            expected_score = float(statistics["expected_score"])
            prediction = nearest_legal_score(expected_score, score_sets)
            probe_ids = list(score_generated.token_ids)
            output_token_counts.append(len(cot_ids) + len(probe_ids))
            reasoning_token_counts.append(stage["reasoning_tokens"])
            predictions.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "output": (
                        stage["cot_output"] + f"{expected_score:.6f}</score>"
                    ),
                    "raw_output": stage["cot_output"] + score_generated.text,
                    "cot_output": stage["cot_output"],
                    "cot_generated_token_ids": cot_ids,
                    "cot_finish_reason": stage["cot_finish_reason"],
                    "cot_stop_reason": stage["cot_stop_reason"],
                    "score_prefix_reached": True,
                    "reasoning_valid": stage["reasoning_valid"],
                    "score_probe_text": score_generated.text,
                    "score_probe_token_ids": probe_ids,
                    "expected_score": expected_score,
                    "score_probability_mass": float(
                        statistics["score_probability_mass"]
                    ),
                    "score_probabilities": {
                        str(score): float(probability)
                        for score, probability in statistics[
                            "score_probabilities"
                        ].items()
                    },
                    "score_logprobs": {
                        str(score): float(logprob)
                        for score, logprob in score_logprobs.items()
                    },
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )

        done = min(start + args.batch_size, len(rows))
        print(f"[cot-rail rollout {rollout_index}] {done}/{len(rows)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    valid = [row for row in predictions if row["expected_score"] is not None]
    squared_errors = [
        (row["expected_score"] - row["label"]) ** 2 for row in valid
    ]
    absolute_errors = [
        abs(row["expected_score"] - row["label"]) for row in valid
    ]
    sample_count = len(predictions)
    valid_count = len(valid)
    metrics.update(
        {
            "rail_mae": (
                sum(absolute_errors) / valid_count if valid_count else None
            ),
            "rail_mse": (
                sum(squared_errors) / valid_count if valid_count else None
            ),
            "rail_rmse": (
                math.sqrt(sum(squared_errors) / valid_count)
                if valid_count
                else None
            ),
            "avg_score_probability_mass": (
                sum(row["score_probability_mass"] for row in valid) / valid_count
                if valid_count
                else None
            ),
            "score_prefix_valid_rate": (
                valid_count / sample_count if sample_count else 0.0
            ),
            "reasoning_valid_rate": (
                sum(bool(row["reasoning_valid"]) for row in predictions) / sample_count
                if sample_count
                else 0.0
            ),
            "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
            "candidate_renormalization": False,
            "rail_implementation": RAIL_IMPLEMENTATION,
            "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
            "discrete_decoding": RAIL_DISCRETE_DECODING,
            "score_prefix": "<score>",
            "cot_stop_strings": ["<score>"],
            "elapsed_sec": round(elapsed, 3),
            "gpu_time_sec": round(gpu_elapsed, 3),
            "samples_per_sec": round(sample_count / max(elapsed, 1e-9), 3),
            "tokens": {
                "avg_output_tokens": (
                    sum(output_token_counts) / sample_count if sample_count else None
                ),
                "avg_reasoning_tokens": (
                    sum(reasoning_token_counts) / sample_count
                    if sample_count
                    else None
                ),
                "total_output_tokens": sum(output_token_counts),
                "total_reasoning_tokens": sum(reasoning_token_counts),
                "samples": sample_count,
            },
        }
    )
    return predictions, metrics


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.refresh_analysis_only:
        output_root = resolve_path(args.output_path)
        analysis_path = update_evaluation_analysis(output_root)
        print(f"updated {analysis_path}")
        return
    if args.rollout < 1:
        raise SystemExit("--rollout must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.temp != 0:
        raise SystemExit("--temp must be 0 because evaluation is deterministic")
    if args.top_p != 1:
        raise SystemExit("--top_p must be 1.0 because evaluation is deterministic")
    if args.inference_mode in {"rail", "cot_rail"} and args.enable_thinking:
        raise SystemExit(
            "RAIL modes require --disable_thinking so their score boundary is stable"
        )
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu_memory_utilization must be in (0, 1]")

    model_name = resolve_path(args.model_name)
    adapter = normalize_adapter(args.adapter, train_seed=args.train_seed)
    dataset_file = resolve_path(args.dataset_file)
    merge_cache = resolve_path(args.merge_cache)
    output_root = resolve_path(args.output_path)
    train_config = (
        load_optional_yaml(resolve_path(args.train_config))
        if args.train_config
        else None
    )

    set_seed(args.seed)
    rows, score_sets = load_rows(dataset_file)
    task_name = infer_task_name(dataset_file, rows)
    # Prefer dataset-derived task for folder layout; fall back to exp_name prefix.
    task_folder = task_name if task_name and task_name != "unknown" else task_name_from_exp(
        args.exp_name
    )
    out_dir = eval_run_dir(output_root, args.exp_name, task_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "resolved_config.json", resolved_eval_config(args, adapter=adapter))

    supervision_mode = infer_supervision_mode(dataset_file, rows)
    aspect = rows[0].get("aspect") or task_name
    criterion = criterion_title(str(aspect))
    run_tag = f"{task_name}|{supervision_mode}|{short_model_name(model_name)}"
    train_meta = load_train_run_metadata(adapter)

    cleanup_merged_cache(merge_cache, args.merge_retention_days)
    # Also sweep the legacy project-local merge dir if it still holds leftovers.
    legacy_merge = PROJECT_ROOT / "merged"
    if legacy_merge.resolve() != merge_cache.resolve() and legacy_merge.is_dir():
        cleanup_merged_cache(legacy_merge, 0)

    wall_started = time.perf_counter()
    gpu_before = gpu_time_snapshot()
    merged_to_cleanup: Path | None = None
    llm = None
    try:
        if adapter is None:
            model_path = model_name
            backend = "vllm-base"
            print(
                f"[{run_tag}] backend={backend} samples={len(rows)} "
                f"base={model_name} adapter=None",
                flush=True,
            )
        else:
            adapter_weight_file(adapter)
            model_path = ensure_merged_model(model_name, adapter, merge_cache)
            merged_to_cleanup = model_path
            backend = "vllm-merged"
            print(
                f"[{run_tag}] backend={backend} samples={len(rows)} base={model_name} "
                f"adapter={adapter} merged={model_path}",
                flush=True,
            )

        llm, sampling_params = init_vllm(
            model_path,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        score_token_ids: list[int] | None = None
        score_sampling_params = None
        cot_sampling_params = None
        if args.inference_mode in {"rail", "cot_rail"}:
            score_token_ids = resolve_score_token_ids(llm.get_tokenizer(), score_sets)
            score_sampling_params = build_rail_sampling_params(
                score_token_ids, args.seed
            )
            if args.inference_mode == "rail":
                sampling_params = score_sampling_params
            else:
                cot_sampling_params = build_cot_rail_sampling_params(
                    args.max_tokens, args.seed
                )
            print(
                f"[{run_tag}] RAIL score tokens: "
                f"{dict(zip(score_sets, score_token_ids, strict=True))}",
                flush=True,
            )

        rollout_metrics = []
        rollout_predictions: list[list[dict[str, Any]]] = []
        for rollout_index in range(1, args.rollout + 1):
            if args.inference_mode == "rail":
                assert score_token_ids is not None
                predictions, metrics = run_rail_rollout(
                    llm,
                    sampling_params,
                    rows,
                    score_sets,
                    score_token_ids,
                    args,
                    rollout_index=rollout_index,
                )
            elif args.inference_mode == "cot_rail":
                assert score_token_ids is not None
                assert cot_sampling_params is not None
                assert score_sampling_params is not None
                predictions, metrics = run_cot_rail_rollout(
                    llm,
                    cot_sampling_params,
                    score_sampling_params,
                    rows,
                    score_sets,
                    score_token_ids,
                    args,
                    rollout_index=rollout_index,
                )
            else:
                predictions, metrics = run_rollout(
                    llm,
                    sampling_params,
                    rows,
                    score_sets,
                    args,
                    rollout_index=rollout_index,
                )
            rollout_predictions.append(predictions)
            rollout_metrics.append(metrics)
            print(
                f"[{run_tag}][rollout {rollout_index}] "
                f"acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"valid={metrics['format_valid_rate']:.4f} "
                f"gpu_s={metrics['gpu_time_sec']:.1f}",
                flush=True,
            )

        prediction_records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            scores = [preds[index]["prediction"] for preds in rollout_predictions]
            outputs = [preds[index]["output"] for preds in rollout_predictions]
            raw_outputs = [
                preds[index].get("raw_output", preds[index]["output"])
                for preds in rollout_predictions
            ]
            correct = [score == row["label"] for score in scores]
            record = {
                "id": row["id"],
                "label": row["label"],
                "rollout_predictions": scores,
                "rollout_correct": correct,
                "mean_correct": sum(correct) / len(correct),
                "outputs": outputs,
                "raw_outputs": raw_outputs,
                "task": row.get("task"),
                "aspect": row.get("aspect"),
            }
            if args.inference_mode in {"rail", "cot_rail"}:
                expected_scores = [
                    preds[index]["expected_score"] for preds in rollout_predictions
                ]
                valid_expected_scores = [
                    score for score in expected_scores if score is not None
                ]
                record.update(
                    {
                        "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
                        "candidate_renormalization": False,
                        "rail_implementation": RAIL_IMPLEMENTATION,
                        "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
                        "discrete_decoding": RAIL_DISCRETE_DECODING,
                        "rollout_expected_scores": expected_scores,
                        "expected_score": (
                            float(np.mean(valid_expected_scores))
                            if valid_expected_scores
                            else None
                        ),
                        "rollout_score_probability_masses": [
                            preds[index]["score_probability_mass"]
                            for preds in rollout_predictions
                        ],
                        "rollout_score_probabilities": [
                            preds[index]["score_probabilities"]
                            for preds in rollout_predictions
                        ],
                        "rollout_score_logprobs": [
                            preds[index]["score_logprobs"]
                            for preds in rollout_predictions
                        ],
                    }
                )
                if args.inference_mode == "cot_rail":
                    record.update(
                        {
                            "rollout_cot_outputs": [
                                preds[index]["cot_output"]
                                for preds in rollout_predictions
                            ],
                            "rollout_cot_generated_token_ids": [
                                preds[index]["cot_generated_token_ids"]
                                for preds in rollout_predictions
                            ],
                            "rollout_score_probe_texts": [
                                preds[index]["score_probe_text"]
                                for preds in rollout_predictions
                            ],
                            "rollout_score_probe_token_ids": [
                                preds[index]["score_probe_token_ids"]
                                for preds in rollout_predictions
                            ],
                            "rollout_score_prefix_reached": [
                                preds[index]["score_prefix_reached"]
                                for preds in rollout_predictions
                            ],
                            "rollout_reasoning_valid": [
                                preds[index]["reasoning_valid"]
                                for preds in rollout_predictions
                            ],
                            "rollout_cot_finish_reasons": [
                                preds[index]["cot_finish_reason"]
                                for preds in rollout_predictions
                            ],
                            "rollout_cot_stop_reasons": [
                                preds[index]["cot_stop_reason"]
                                for preds in rollout_predictions
                            ],
                        }
                    )
            prediction_records.append(record)
        aggregate_metrics = mean_rollout_metrics(rollout_metrics, score_sets)
        avg_output_tokens = None
        avg_reasoning_tokens = None
        token_means = [
            rollout["tokens"]["avg_output_tokens"]
            for rollout in rollout_metrics
            if rollout.get("tokens", {}).get("avg_output_tokens") is not None
        ]
        reason_means = [
            rollout["tokens"]["avg_reasoning_tokens"]
            for rollout in rollout_metrics
            if rollout.get("tokens", {}).get("avg_reasoning_tokens") is not None
        ]
        if token_means:
            avg_output_tokens = float(np.mean(token_means))
        if reason_means:
            avg_reasoning_tokens = float(np.mean(reason_means))
        aggregate_metrics["tokens"] = {
            "avg_output_tokens": avg_output_tokens,
            "avg_reasoning_tokens": avg_reasoning_tokens,
        }
        aggregate_metrics["gpu_time_sec"] = float(
            np.mean([rollout["gpu_time_sec"] for rollout in rollout_metrics])
        )
        aggregate_metrics["elapsed_sec"] = float(
            np.mean([rollout["elapsed_sec"] for rollout in rollout_metrics])
        )

        train_config_path = str(args.train_config) if args.train_config else None
        eval_condition = infer_eval_condition(
            exp_name=args.exp_name,
            supervision_mode=supervision_mode,
            adapter=str(adapter) if adapter is not None else None,
            train_config=train_config_path,
        )
        configured_method = ((train_config or {}).get("supervision") or {}).get(
            "method"
        )
        is_raft_without_cot = (
            configured_method == "raft_without_cot"
            or "raft_without_cot" in args.exp_name.lower()
        )
        is_cot_raft = (
            configured_method == "cot_raft"
            or "cot_raft" in args.exp_name.lower()
        )
        if is_cot_raft:
            eval_condition = (
                "COT-RAFT-R"
                if args.inference_mode == "cot_rail"
                else "COT-RAFT-G"
            )
        elif is_raft_without_cot:
            eval_condition = (
                "RAFT-R" if args.inference_mode == "rail" else "RAFT-G"
            )
        elif args.inference_mode in {"rail", "cot_rail"}:
            eval_condition = f"{eval_condition}-R" if eval_condition else "RAIL"

        full_config = resolved_eval_config(args, adapter=adapter)
        full_config["train_config"] = train_config
        full_config["train_run"] = train_meta

        wall_elapsed = time.perf_counter() - wall_started
        summary = {
            "exp_name": args.exp_name,
            "run_tag": run_tag,
            "task": task_name,
            "criterion": criterion,
            "aspect": aspect,
            "supervision_mode": supervision_mode,
            "evaluation_mode": supervision_mode,
            "eval_condition": eval_condition,
            "output_dir": str(out_dir),
            "backend": backend,
            "model_name": str(model_name),
            "adapter": str(adapter) if adapter is not None else None,
            "merged_model": str(model_path) if adapter is not None else None,
            "dataset_file": str(dataset_file),
            "score_sets": score_sets,
            "seed": args.seed,
            "train_seed_requested": args.train_seed,
            "rollout": args.rollout,
            "inference_mode": args.inference_mode,
            "decoding": {
                "greedy": "greedy",
                "rail": "rail_full_vocab_raw_expected_score",
                "cot_rail": "cot_rail_full_vocab_raw_expected_score",
            }[args.inference_mode],
            "temp": 0.0,
            "top_p": 1.0,
            "max_model_len": args.max_model_len,
            "configured_max_tokens": args.max_tokens,
            "max_tokens": 1 if args.inference_mode == "rail" else args.max_tokens,
            "reasoning_max_tokens": (
                args.max_tokens if args.inference_mode == "cot_rail" else None
            ),
            "score_probe_max_tokens": (
                1 if args.inference_mode in {"rail", "cot_rail"} else None
            ),
            "cot_stop_strings": (
                ["<score>"] if args.inference_mode == "cot_rail" else None
            ),
            "score_token_ids": (
                dict(zip(score_sets, score_token_ids, strict=True))
                if score_token_ids is not None
                else None
            ),
            "probability_normalization": (
                RAIL_PROBABILITY_NORMALIZATION
                if args.inference_mode in {"rail", "cot_rail"}
                else None
            ),
            "candidate_renormalization": (
                False if args.inference_mode in {"rail", "cot_rail"} else None
            ),
            "rail_implementation": (
                RAIL_IMPLEMENTATION
                if args.inference_mode in {"rail", "cot_rail"}
                else None
            ),
            "rail_expectation_formula": (
                RAIL_EXPECTATION_FORMULA
                if args.inference_mode in {"rail", "cot_rail"}
                else None
            ),
            "discrete_decoding": (
                RAIL_DISCRETE_DECODING
                if args.inference_mode in {"rail", "cot_rail"}
                else None
            ),
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_thinking": args.enable_thinking,
            "finished_at_utc": utc_now(),
            "aggregation": "mean_over_rollouts",
            "checkpoint_retention": train_meta.get("checkpoint_retention"),
            "adapter_selection": train_meta.get("adapter_selection"),
            "checkpoint_epoch": train_meta.get("checkpoint_epoch"),
            "checkpoint_step": train_meta.get("checkpoint_step"),
            "generation_accuracy": train_meta.get("generation_accuracy"),
            "test_accuracy": aggregate_metrics.get("accuracy"),
            "test_macro_f1": aggregate_metrics.get("macro_f1"),
            "test_mae": aggregate_metrics.get("mae"),
            "test_qwk": aggregate_metrics.get("qwk"),
            "test_rail_mae": aggregate_metrics.get("rail_mae"),
            "test_rail_mse": aggregate_metrics.get("rail_mse"),
            "test_rail_rmse": aggregate_metrics.get("rail_rmse"),
            "avg_score_probability_mass": aggregate_metrics.get(
                "avg_score_probability_mass"
            ),
            "format_valid_rate": aggregate_metrics.get("format_valid_rate"),
            "score_prefix_valid_rate": aggregate_metrics.get(
                "score_prefix_valid_rate"
            ),
            "reasoning_valid_rate": aggregate_metrics.get("reasoning_valid_rate"),
            "avg_output_tokens": avg_output_tokens,
            "avg_reasoning_tokens": avg_reasoning_tokens,
            "gpu_time_sec": aggregate_metrics.get("gpu_time_sec"),
            "wall_time_sec": round(wall_elapsed, 3),
            "gpu_before": gpu_before,
            "gpu_after": gpu_time_snapshot(),
            "full_config": full_config,
            "aggregate": aggregate_metrics,
            "rollouts": rollout_metrics,
        }
        metrics_path = out_dir / "metrics.json"
        write_json(metrics_path, summary)

        prediction_path = out_dir / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in prediction_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Rebuild the single aggregate analysis report under eval_output/.
        analysis_path = update_evaluation_analysis(output_root)

        print(
            f"[{run_tag}][done] acc={aggregate_metrics['accuracy']:.4f}"
            f"+/-{aggregate_metrics.get('accuracy_std') or 0:.4f} "
            f"macro_f1={aggregate_metrics['macro_f1']:.4f}"
            f"+/-{aggregate_metrics.get('macro_f1_std') or 0:.4f} "
            f"valid={aggregate_metrics['format_valid_rate']:.4f}"
            f"+/-{aggregate_metrics.get('format_valid_rate_std') or 0:.4f}"
        )
        if aggregate_metrics.get("mae") is not None:
            print(
                f"[{run_tag}] mae={aggregate_metrics['mae']:.4f} "
                f"qwk={aggregate_metrics.get('qwk')}",
                flush=True,
            )
        print(
            f"[{run_tag}] tokens: output={avg_output_tokens} "
            f"reasoning={avg_reasoning_tokens} "
            f"gpu_s={aggregate_metrics.get('gpu_time_sec')} "
            f"checkpoint_epoch={train_meta.get('checkpoint_epoch')}",
            flush=True,
        )
        print(f"wrote {out_dir / 'resolved_config.json'}")
        print(f"wrote {metrics_path}")
        print(f"wrote {prediction_path}")
        print(f"updated {analysis_path}")
    finally:
        # Release vLLM before deleting on-disk merged weights.
        if llm is not None:
            try:
                del llm
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        remove_merged_model(merged_to_cleanup)

if __name__ == "__main__":
    main()
