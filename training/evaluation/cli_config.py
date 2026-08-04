"""Evaluation CLI parsing and effective configuration snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from shared.project_io import resolve_path
from evaluation.defaults import (
    CONFIG_KEYS,
    DEFAULT_DATASET,
    DEFAULT_EVAL_OUTPUT_ROOT,
    DEFAULT_MERGE_CACHE,
    DEFAULT_MERGE_RETENTION_DAYS,
)
from evaluation.methods import available_inference_modes, get_evaluation_method

DESCRIPTION = """Evaluate a base model, optionally with a LoRA adapter, using vLLM.

YAML values provide defaults and explicit CLI flags override them. The command
supports greedy generation, score-only RAIL, and two-stage CoT-RAIL.
"""


def json_object(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected a JSON object: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return payload


def task_name_from_exp(exp_name: str) -> str:
    return exp_name.split("#", 1)[0]


def eval_run_dir(output_root: Path, exp_name: str, task_name: str | None = None) -> Path:
    return output_root / (task_name or task_name_from_exp(exp_name)) / exp_name


def read_eval_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"Eval config must be a YAML object: {path}")
    missing = {"exp_name", "model_name", "dataset_file"} - set(config)
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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_parser.add_argument("--refresh-analysis-only", action="store_true")
    pre_args, _ = pre_parser.parse_known_args(argv)

    defaults: dict[str, Any] = {}
    config_path: Path | None = None
    if pre_args.config:
        config_path = resolve_path(pre_args.config)
        if not config_path.is_file():
            raise SystemExit(f"Eval config not found: {config_path}")
        defaults.update(read_eval_config(config_path))

    analysis_only = bool(pre_args.refresh_analysis_only)
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--refresh-analysis-only",
        action="store_true",
        help="Rebuild evaluation_analysis.md from existing metrics and exit.",
    )
    parser.add_argument("--config", help="Evaluation YAML; CLI flags override it.")
    parser.add_argument(
        "--exp_name",
        default=defaults.get("exp_name"),
        required=not analysis_only and "exp_name" not in defaults,
    )
    parser.add_argument(
        "--model_name",
        default=defaults.get("model_name"),
        required=not analysis_only and "model_name" not in defaults,
        help="Base model path.",
    )
    parser.add_argument(
        "--adapter",
        default=defaults.get("adapter"),
        help=(
            "Adapter/checkpoint path, or method output root selected by --train_seed. "
            "Omit or pass none to evaluate the base model."
        ),
    )
    parser.add_argument(
        "--train_seed",
        type=int,
        default=(
            int(defaults["train_seed"])
            if defaults.get("train_seed") is not None
            else None
        ),
    )
    parser.add_argument(
        "--dataset_file", default=defaults.get("dataset_file", str(DEFAULT_DATASET))
    )
    parser.add_argument(
        "--output_path",
        default=defaults.get("output_path", str(DEFAULT_EVAL_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--max_model_len", type=int, default=int(defaults.get("max_model_len", 8192))
    )
    parser.add_argument(
        "--max_tokens", type=int, default=int(defaults.get("max_tokens", 512))
    )
    parser.add_argument(
        "--inference_mode",
        choices=available_inference_modes(),
        default=str(defaults.get("inference_mode", "greedy")),
    )
    parser.add_argument(
        "--method_options",
        type=json_object,
        default=defaults.get("method_options", {}),
        help="JSON object passed unchanged to the selected inference method.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=float(defaults.get("temp", 0.0)),
        help="Must be 0; evaluation is deterministic.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(defaults.get("top_p", 1.0)),
        help="Must be 1; evaluation is deterministic.",
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument(
        "--rollout", type=int, default=int(defaults.get("rollout", 1))
    )
    parser.add_argument(
        "--batch_size", type=int, default=int(defaults.get("batch_size", 64))
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(defaults.get("gpu_memory_utilization", 0.9)),
    )
    parser.add_argument(
        "--merge_cache", default=defaults.get("merge_cache", str(DEFAULT_MERGE_CACHE))
    )
    parser.add_argument(
        "--merge_retention_days",
        type=float,
        default=float(
            defaults.get("merge_retention_days", DEFAULT_MERGE_RETENTION_DAYS)
        ),
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        default=bool(defaults.get("enable_thinking", False)),
    )
    thinking.add_argument(
        "--disable_thinking", dest="enable_thinking", action="store_false"
    )
    parser.add_argument("--train_config", default=defaults.get("train_config"))
    args = parser.parse_args(argv)
    args.config_path = str(config_path) if config_path is not None else None
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.rollout < 1:
        raise SystemExit("--rollout must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.temp != 0:
        raise SystemExit("--temp must be 0 because evaluation is deterministic")
    if args.top_p != 1:
        raise SystemExit("--top_p must be 1.0 because evaluation is deterministic")
    if not isinstance(args.method_options, dict):
        raise SystemExit("--method_options must be a JSON/YAML object")
    get_evaluation_method(args.inference_mode).validate(args)
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu_memory_utilization must be in (0, 1]")


def load_optional_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else None


def resolved_eval_config(
    args: argparse.Namespace, *, adapter: Path | None
) -> dict[str, Any]:
    method = get_evaluation_method(args.inference_mode)
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
        "inference_mode": method.name,
        "method_options": args.method_options,
        **method.resolved_config_metadata(args),
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
