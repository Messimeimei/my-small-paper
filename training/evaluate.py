#!/usr/bin/env python3
"""Evaluate a base model (optionally + LoRA adapter/checkpoint) with vLLM.

Without --adapter (or with none/None/NONE), loads the base model directly.
With an adapter path, merges LoRA into a temporary full model then loads with
plain vLLM (vLLM 0.8.4 + cachetools>=6 breaks enable_lora in spawned workers).
The merged weights are deleted after evaluation finishes (or on failure).

Prefer YAML under evaloutput/configs/<task>/{base,ft}_{cot,score_only}.yaml
via --config; CLI flags override config values. Supports
data/<task>/{cot,score_only}/test_*.jsonl (labels field) and legacy JSON.

Writes to evaloutput/<task>/<exp_name>/ (metrics.json, predictions.jsonl,
resolved_config.json) and updates evaloutput/comparison_table.md with the
four settings: base-score_only, base-cot, score_only(ft), cot(ft).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
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

from metrics_utils import (
    classification_metrics,
    compute_efficiency_metrics,
    criterion_title,
    extract_score,
    infer_comparison_mode,
    infer_supervision_mode,
    infer_task_name,
    merge_comparison_row,
    render_comparison_table,
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
DEFAULT_EVAL_OUTPUT_ROOT = PROJECT_ROOT / "evaloutput"
DEFAULT_MERGE_RETENTION_DAYS = 0

# YAML keys -> argparse destinations (same names as CLI flags without --).
CONFIG_KEYS = {
    "exp_name",
    "model_name",
    "adapter",
    "dataset_file",
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
    pre_args, _ = pre.parse_known_args(argv)

    config_defaults: dict[str, Any] = {}
    config_path: Path | None = None
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
        "--config",
        default=None,
        help=(
            "Eval YAML under evaloutput/configs/<task>/"
            "{base,ft}_{cot,score_only}.yaml. "
            "CLI flags override values from the config."
        ),
    )
    parser.add_argument(
        "--exp_name",
        default=config_defaults.get("exp_name"),
        required="exp_name" not in config_defaults,
    )
    parser.add_argument(
        "--model_name",
        default=config_defaults.get("model_name"),
        required="model_name" not in config_defaults,
        help="Base model path.",
    )
    parser.add_argument(
        "--adapter",
        default=config_defaults.get("adapter"),
        help=(
            "LoRA adapter/ or checkpoints/checkpoint-* path. "
            "Omit, or pass none/None/NONE, to evaluate the base model only."
        ),
    )
    parser.add_argument(
        "--dataset_file",
        default=config_defaults.get("dataset_file", str(DEFAULT_DATASET)),
    )
    parser.add_argument(
        "--output_path",
        default=config_defaults.get("output_path", str(DEFAULT_EVAL_OUTPUT_ROOT)),
        help=(
            "Eval output root (default: <project>/evaloutput). "
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
        "--temp",
        type=float,
        default=float(config_defaults.get("temp", 0.0)),
        help="Must be 0: every rollout uses greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(config_defaults.get("top_p", 1.0)),
        help="Must be 1.0: nucleus sampling is disabled for greedy decoding.",
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


def normalize_adapter(value: str | None) -> Path | None:
    """Return adapter path, or None for base-only eval (omit / none / None / NONE)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return resolve_path(text)


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
    for metric in ("accuracy", "macro_f1", "format_valid_rate", "mae", "qwk"):
        if metric not in rollout_metrics[0] and metric in {"mae", "qwk"}:
            continue
        mean, std = summarize([rollout.get(metric) for rollout in rollout_metrics])
        aggregate[metric] = mean
        aggregate[f"{metric}_std"] = std

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
    """Pull best-checkpoint epoch / config from a training run directory if present."""
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
        meta.update(
            {
                "best_checkpoint": summary.get("best_checkpoint"),
                "best_checkpoint_epoch": summary.get("best_checkpoint_epoch"),
                "best_checkpoint_step": summary.get("best_checkpoint_step"),
                "best_generation_accuracy": summary.get("best_generation_accuracy"),
                "train_run_id": summary.get("run_id"),
                "train_seed": summary.get("seed"),
            }
        )
    if state_path.is_file() and meta.get("best_checkpoint_epoch") is None:
        state = read_json(state_path)
        meta.setdefault("best_checkpoint_step", state.get("best_global_step"))
        meta.setdefault("best_generation_accuracy", state.get("best_metric"))
        best_step = state.get("best_global_step")
        for row in reversed(state.get("log_history") or []):
            if best_step is not None and int(row.get("step", -1)) == int(best_step):
                if "epoch" in row:
                    meta["best_checkpoint_epoch"] = float(row["epoch"])
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
        "dataset_file": str(resolve_path(args.dataset_file)),
        "output_path": str(resolve_path(args.output_path)),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
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


def update_comparison_table(
    output_root: Path,
    *,
    criterion: str,
    mode: str,
    model_key: str,
    accuracy: float,
    macro_f1: float,
    avg_output_tokens: float | None,
    source_metrics: Path,
    gpu_time_sec: float | None = None,
    samples_per_sec: float | None = None,
    tokens_per_sec: float | None = None,
    n_samples: int | None = None,
) -> Path:
    """Merge this run into the 4-way comparison table (base/ft × score_only/cot)."""
    table_json = output_root / "comparison_table.json"
    table_md = output_root / "comparison_table.md"
    payload = {"model_key": model_key, "rows": {}}
    if table_json.is_file():
        loaded = read_json(table_json)
        if isinstance(loaded, dict) and loaded.get("model_key") == model_key:
            payload = loaded
        elif isinstance(loaded, dict):
            # Different model family: keep separate file keyed by model.
            table_json = output_root / f"comparison_table__{model_key}.json"
            table_md = output_root / f"comparison_table__{model_key}.md"
            if table_json.is_file():
                payload = read_json(table_json)
            else:
                payload = {"model_key": model_key, "rows": {}}
    rows = payload.setdefault("rows", {})
    rows[criterion] = merge_comparison_row(
        rows.get(criterion),
        criterion=criterion,
        mode=mode,
        accuracy=accuracy,
        macro_f1=macro_f1,
        avg_output_tokens=avg_output_tokens,
        gpu_time_sec=gpu_time_sec,
        samples_per_sec=samples_per_sec,
        tokens_per_sec=tokens_per_sec,
        n_samples=n_samples,
    )
    rows[criterion]["sources"] = {
        **(rows[criterion].get("sources") or {}),
        mode: str(source_metrics),
    }
    payload["updated_at_utc"] = utc_now()
    ordered = sorted(rows.values(), key=lambda row: row.get("criterion", ""))
    write_json(table_json, payload)
    table_md.write_text(render_comparison_table(ordered), encoding="utf-8")
    return table_md


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


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.rollout < 1:
        raise SystemExit("--rollout must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.temp != 0:
        raise SystemExit("--temp must be 0 because rollout evaluation uses greedy decoding")
    if args.top_p != 1:
        raise SystemExit("--top_p must be 1.0 because rollout evaluation is greedy")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu_memory_utilization must be in (0, 1]")

    model_name = resolve_path(args.model_name)
    adapter = normalize_adapter(args.adapter)
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

        rollout_metrics = []
        rollout_predictions: list[list[dict[str, Any]]] = []
        for rollout_index in range(1, args.rollout + 1):
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
            correct = [score == row["label"] for score in scores]
            prediction_records.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "rollout_predictions": scores,
                    "rollout_correct": correct,
                    "mean_correct": sum(correct) / len(correct),
                    "outputs": outputs,
                    "raw_outputs": outputs,
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )
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
            "output_dir": str(out_dir),
            "backend": backend,
            "model_name": str(model_name),
            "adapter": str(adapter) if adapter is not None else None,
            "merged_model": str(model_path) if adapter is not None else None,
            "dataset_file": str(dataset_file),
            "score_sets": score_sets,
            "seed": args.seed,
            "rollout": args.rollout,
            "decoding": "greedy",
            "temp": 0.0,
            "top_p": 1.0,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_thinking": args.enable_thinking,
            "finished_at_utc": utc_now(),
            "aggregation": "mean_over_rollouts",
            "best_checkpoint_epoch": train_meta.get("best_checkpoint_epoch"),
            "best_checkpoint_step": train_meta.get("best_checkpoint_step"),
            "best_generation_accuracy": train_meta.get("best_generation_accuracy"),
            "test_accuracy": aggregate_metrics.get("accuracy"),
            "test_macro_f1": aggregate_metrics.get("macro_f1"),
            "test_mae": aggregate_metrics.get("mae"),
            "test_qwk": aggregate_metrics.get("qwk"),
            "format_valid_rate": aggregate_metrics.get("format_valid_rate"),
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

        # Per-task comparison table + root aggregate table.
        model_key = short_model_name(model_name)
        mode = infer_comparison_mode(
            supervision_mode=supervision_mode,
            adapter=adapter,
            exp_name=args.exp_name,
        )
        n_samples = int(aggregate_metrics.get("samples") or len(rows))
        gpu_time_sec = aggregate_metrics.get("gpu_time_sec")
        rollout_sps = [
            float(r["samples_per_sec"])
            for r in rollout_metrics
            if r.get("samples_per_sec") is not None
        ]
        samples_per_sec = float(np.mean(rollout_sps)) if rollout_sps else None
        eff = compute_efficiency_metrics(
            n_samples=n_samples,
            avg_output_tokens=avg_output_tokens,
            gpu_time_sec=gpu_time_sec,
            samples_per_sec=samples_per_sec,
        )
        table_kwargs = dict(
            criterion=criterion,
            mode=mode,
            model_key=model_key,
            accuracy=float(aggregate_metrics["accuracy"]),
            macro_f1=float(aggregate_metrics["macro_f1"]),
            avg_output_tokens=avg_output_tokens,
            source_metrics=metrics_path,
            gpu_time_sec=eff["gpu_time_sec"],
            samples_per_sec=eff["samples_per_sec"],
            tokens_per_sec=eff["tokens_per_sec"],
            n_samples=n_samples,
        )
        task_table_path = update_comparison_table(
            output_root / task_folder,
            **table_kwargs,
        )
        root_table_path = update_comparison_table(
            output_root,
            **table_kwargs,
        )

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
            f"best_epoch={train_meta.get('best_checkpoint_epoch')}",
            flush=True,
        )
        print(f"wrote {out_dir / 'resolved_config.json'}")
        print(f"wrote {metrics_path}")
        print(f"wrote {prediction_path}")
        print(f"updated {task_table_path}")
        print(f"updated {root_table_path}")
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
