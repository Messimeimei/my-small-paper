#!/usr/bin/env python3
"""Run Qwen3-8B self-correct CoT/Align training and evaluation serially."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import run_qwen8b_three_method_queue as base

PROJECT_ROOT = base.PROJECT_ROOT
TASKS = base.TASKS
METHODS = ("self_correct_cot", "self_correct_align")
INTERFACES = base.INTERFACES
DEFAULT_SEEDS = base.DEFAULT_SEEDS
TRAIN_CONFIG_ROOT = base.TRAIN_CONFIG_ROOT
EVAL_CONFIG_ROOT = base.EVAL_CONFIG_ROOT
CHECKPOINT_ROOT = base.CHECKPOINT_ROOT
QUEUE_ROOT = PROJECT_ROOT / "outputs" / "qwen8b_self_correct_queue"
RUN_LOG_ROOT = QUEUE_ROOT / "logs"
STATE_PATH = QUEUE_ROOT / "state.json"
EVENTS_PATH = QUEUE_ROOT / "events.jsonl"


def emit(event: str, **fields: Any) -> None:
    payload = {"time_utc": base.utc_now(), "event": event, **fields}
    print(" ".join(f"{key}={value}" for key, value in payload.items()), flush=True)
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def train_config_path(task: str, method: str) -> Path:
    return TRAIN_CONFIG_ROOT / task / f"{method}.yaml"


def checkpoint_method_root(task: str, method: str) -> Path:
    return CHECKPOINT_ROOT / task / method


def eval_config_path(task: str, method: str, interface: str) -> Path:
    filename = f"greedy_on_{interface}.yaml"
    return EVAL_CONFIG_ROOT / task / method / filename


def completed_adapter(task: str, method: str, seed: int) -> Path | None:
    return base.completed_adapter(task, method, seed)


def completion_counts(seeds: tuple[int, ...]) -> tuple[int, int]:
    train_done = 0
    eval_done = 0
    for seed in seeds:
        for task in TASKS:
            for method in METHODS:
                adapter = completed_adapter(task, method, seed)
                if adapter is None:
                    continue
                train_done += 1
                for interface in INTERFACES:
                    if base.evaluation_status(
                        task, method, interface, seed, adapter
                    )[0]:
                        eval_done += 1
    return train_done, eval_done


def write_state(
    seeds: tuple[int, ...],
    *,
    status: str,
    current: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    train_done, eval_done = completion_counts(seeds)
    base.atomic_write_json(
        STATE_PATH,
        {
            "updated_at_utc": base.utc_now(),
            "status": status,
            "seeds": list(seeds),
            "methods": list(METHODS),
            "tasks": list(TASKS),
            "interfaces": list(INTERFACES),
            "completed_training_runs": train_done,
            "total_training_runs": len(seeds) * len(TASKS) * len(METHODS),
            "completed_evaluations": eval_done,
            "total_evaluations": (
                len(seeds) * len(TASKS) * len(METHODS) * len(INTERFACES)
            ),
            "current": current,
            "error": error,
        },
    )


def validate_training_config(task: str, method: str) -> None:
    path = train_config_path(task, method)
    config = base.read_yaml(path)
    expected_exp = f"{task}#qwen3_8b#{method}"
    expected_output = f"checkpoints/training-8B/{task}/{method}"
    expected_root = (
        "data/cot_self_generated_data-8B/"
        if method == "self_correct_cot"
        else "data/align_self_generated_data-8B/"
    )
    if config.get("experiment_name") != expected_exp:
        raise RuntimeError(f"experiment_name mismatch in {path}")
    if config.get("model_name_or_path") != base.MODEL_NAME:
        raise RuntimeError(f"model mismatch in {path}")
    if config.get("output_root") != expected_output:
        raise RuntimeError(f"output_root mismatch in {path}")
    for key in ("dataset_path", "split_path"):
        value = str(config.get(key) or "")
        if not value.startswith(expected_root) or not (PROJECT_ROOT / value).is_file():
            raise RuntimeError(f"Invalid {key} in {path}: {value}")
    label_path = config.get("label_dataset_path")
    if method == "self_correct_align":
        value = str(label_path or "")
        if not value.startswith(expected_root) or not (PROJECT_ROOT / value).is_file():
            raise RuntimeError(f"Invalid label_dataset_path in {path}: {value}")


def validate_evaluation_config(task: str, method: str, interface: str) -> None:
    path = eval_config_path(task, method, interface)
    config = base.read_yaml(path)
    expected_adapter = f"checkpoints/training-8B/{task}/{method}"
    expected_dataset = f"data/{task}/{interface}/test_{interface}.jsonl"
    expected_train_config = str(train_config_path(task, method).relative_to(PROJECT_ROOT))
    if config.get("model_name") != base.MODEL_NAME:
        raise RuntimeError(f"model mismatch in {path}")
    if config.get("adapter") != expected_adapter:
        raise RuntimeError(f"adapter mismatch in {path}")
    if config.get("dataset_file") != expected_dataset:
        raise RuntimeError(f"dataset mismatch in {path}")
    if config.get("train_config") != expected_train_config:
        raise RuntimeError(f"train_config mismatch in {path}")
    if config.get("output_path") != "outputs/evaluations/evaluation-8b":
        raise RuntimeError(f"output_path mismatch in {path}")


def preflight(seeds: tuple[int, ...], *, run_dry_runs: bool) -> None:
    emit("preflight_started", seeds=list(seeds), methods=list(METHODS))
    for task in TASKS:
        for method in METHODS:
            validate_training_config(task, method)
            for interface in INTERFACES:
                validate_evaluation_config(task, method, interface)
            if run_dry_runs:
                command = [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(train_config_path(task, method).relative_to(PROJECT_ROOT)),
                    "--seed",
                    str(seeds[0]),
                    "--dry-run",
                ]
                result = base.subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=os.environ.copy(),
                    stdout=base.subprocess.PIPE,
                    stderr=base.subprocess.STDOUT,
                    text=True,
                )
                if result.returncode != 0:
                    tail = "\n".join(result.stdout.splitlines()[-30:])
                    raise RuntimeError(f"Dry-run failed for {task}/{method}:\n{tail}")
    train_done, eval_done = completion_counts(seeds)
    emit(
        "preflight_completed",
        completed_training=train_done,
        remaining_training=len(seeds) * len(TASKS) * len(METHODS) - train_done,
        completed_evaluations=eval_done,
        note="all runtime commands override adapter, train_seed, and exp_name",
    )


def train(task: str, method: str, seed: int) -> tuple[Path, bool]:
    existing = completed_adapter(task, method, seed)
    if existing is not None:
        emit("training_skipped", seed=seed, task=task, method=method, adapter=existing)
        return existing, False
    current = {"phase": "train", "seed": seed, "task": task, "method": method}
    write_state(DEFAULT_SEEDS, status="running", current=current)
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(train_config_path(task, method).relative_to(PROJECT_ROOT)),
        "--seed",
        str(seed),
        "--fresh",
    ]
    log_path = RUN_LOG_ROOT / f"seed{seed}" / task / f"{method}.train.log"
    emit("training_started", **current, log=log_path)
    return_code = base.run_command(command, log_path)
    adapter = completed_adapter(task, method, seed)
    if adapter is None:
        raise RuntimeError(
            f"Training failed or incomplete: seed={seed} task={task} "
            f"method={method} rc={return_code}"
        )
    emit(
        "training_completed",
        seed=seed,
        task=task,
        method=method,
        return_code=return_code,
        adapter=adapter,
    )
    return adapter, True


def evaluate(
    task: str,
    method: str,
    interface: str,
    seed: int,
    adapter: Path,
) -> None:
    valid, reason = base.evaluation_status(task, method, interface, seed, adapter)
    if valid:
        emit(
            "evaluation_skipped",
            seed=seed,
            task=task,
            method=method,
            interface=interface,
            reason=reason,
        )
        return
    current = {
        "phase": "evaluate",
        "seed": seed,
        "task": task,
        "method": method,
        "interface": interface,
    }
    write_state(DEFAULT_SEEDS, status="running", current=current)
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        str(eval_config_path(task, method, interface).relative_to(PROJECT_ROOT)),
        "--adapter",
        str(checkpoint_method_root(task, method).relative_to(PROJECT_ROOT)),
        "--train_seed",
        str(seed),
        "--exp_name",
        base.eval_exp_name(task, method, interface, seed),
        "--seed",
        str(base.EVAL_DECODE_SEED),
    ]
    log_path = (
        RUN_LOG_ROOT / f"seed{seed}" / task / f"{method}.eval_on_{interface}.log"
    )
    emit("evaluation_started", **current, adapter=adapter, log=log_path)
    return_code = base.run_command(command, log_path)
    valid, reason = base.evaluation_status(task, method, interface, seed, adapter)
    if not valid:
        raise RuntimeError(
            f"Evaluation failed: seed={seed} task={task} method={method} "
            f"interface={interface} rc={return_code}: {reason}"
        )
    emit(
        "evaluation_completed",
        seed=seed,
        task=task,
        method=method,
        interface=interface,
        return_code=return_code,
        result=reason,
    )
    base.write_report(DEFAULT_SEEDS)


def run_queue(seeds: tuple[int, ...]) -> None:
    preflight(seeds, run_dry_runs=True)
    write_state(seeds, status="running")
    try:
        for seed in seeds:
            for task in TASKS:
                for method in METHODS:
                    adapter, _ = train(task, method, seed)
                    for interface in INTERFACES:
                        evaluate(task, method, interface, seed, adapter)
    except Exception as exc:
        write_state(seeds, status="failed", error=str(exc))
        emit("queue_failed", error=str(exc))
        raise
    write_state(seeds, status="completed")
    emit("queue_completed", seeds=list(seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds or any(seed not in DEFAULT_SEEDS for seed in seeds):
        raise SystemExit(f"Seeds must be selected from {DEFAULT_SEEDS}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.preflight_only:
        preflight(seeds, run_dry_runs=True)
        write_state(seeds, status="preflight_passed")
        return
    run_queue(seeds)


if __name__ == "__main__":
    main()
