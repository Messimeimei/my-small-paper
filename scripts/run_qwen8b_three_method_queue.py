#!/usr/bin/env python3
"""Run the Qwen3-8B three-method, three-seed experiment queue serially."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG_ROOT = PROJECT_ROOT / "configs" / "training" / "training-8B"
EVAL_CONFIG_ROOT = PROJECT_ROOT / "configs" / "evaluation" / "evaluation-8b"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "training-8B"
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluations" / "evaluation-8b"
QUEUE_ROOT = PROJECT_ROOT / "outputs" / "qwen8b_three_method_queue"
RUN_LOG_ROOT = QUEUE_ROOT / "logs"
STATE_PATH = QUEUE_ROOT / "state.json"
EVENTS_PATH = QUEUE_ROOT / "events.jsonl"
REPORT_PATH = PROJECT_ROOT / "qwen3_8b_seed42_44_results.md"

TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
METHODS = ("cot", "label_only", "paper_align")
REPORT_METHODS = METHODS + ("self_correct_cot", "self_correct_align")
INTERFACES = ("cot", "label_only")
DEFAULT_SEEDS = (42, 43, 44)
MODEL_NAME = "model/Qwen3-8B"
EVAL_DECODE_SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def emit(event: str, **fields: Any) -> None:
    payload = {"time_utc": utc_now(), "event": event, **fields}
    message = " ".join(f"{key}={value}" for key, value in payload.items())
    print(message, flush=True)
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def train_config_path(task: str, method: str) -> Path:
    return TRAIN_CONFIG_ROOT / task / f"{method}.yaml"


def eval_config_path(task: str, method: str, interface: str) -> Path:
    return EVAL_CONFIG_ROOT / task / method / f"greedy_on_{interface}.yaml"


def checkpoint_method_root(task: str, method: str) -> Path:
    return CHECKPOINT_ROOT / task / method


def eval_exp_name(task: str, method: str, interface: str, seed: int) -> str:
    return (
        f"{task}#qwen3_8b#ft#{method}#greedy"
        f"#on_{interface}#seed_{seed}"
    )


def eval_run_directory(task: str, method: str, interface: str, seed: int) -> Path:
    return EVAL_OUTPUT_ROOT / task / eval_exp_name(task, method, interface, seed)


def method_order(seed: int, task: str) -> tuple[str, ...]:
    return METHODS


def adapter_weight(adapter: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def completed_adapter(task: str, method: str, seed: int) -> Path | None:
    root = checkpoint_method_root(task, method)
    if not root.is_dir():
        return None
    candidates: list[tuple[float, str, Path]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        adapter = run_dir / "adapter"
        if not manifest_path.is_file() or not summary_path.is_file():
            continue
        if adapter_weight(adapter) is None:
            continue
        try:
            manifest = read_json(manifest_path)
            summary = read_json(summary_path)
            run_seed = int(summary.get("seed"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "completed" or run_seed != seed:
            continue
        candidates.append((summary_path.stat().st_mtime, run_dir.name, adapter.resolve()))
    return max(candidates)[2] if candidates else None


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def evaluation_status(
    task: str,
    method: str,
    interface: str,
    seed: int,
    adapter: Path,
) -> tuple[bool, str]:
    out_dir = eval_run_directory(task, method, interface, seed)
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "resolved_config.json"
    predictions_path = out_dir / "predictions.jsonl"
    if not all(path.is_file() for path in (metrics_path, config_path, predictions_path)):
        return False, "missing metrics/resolved_config/predictions"
    try:
        metrics = read_json(metrics_path)
        resolved = read_json(config_path)
        requested_seed = int(metrics.get("train_seed_requested"))
        resolved_seed = int(resolved.get("train_seed"))
        resolved_adapter = Path(str(resolved.get("adapter"))).resolve()
        dataset = Path(str(resolved.get("dataset_file"))).resolve()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return False, f"invalid result metadata: {exc}"
    expected_exp = eval_exp_name(task, method, interface, seed)
    if metrics.get("exp_name") != expected_exp:
        return False, "exp_name mismatch"
    if requested_seed != seed or resolved_seed != seed:
        return False, "train_seed mismatch"
    if resolved_adapter != adapter.resolve():
        return False, "adapter mismatch"
    if not dataset.is_file():
        return False, "resolved dataset missing"
    expected_rows = count_jsonl(dataset)
    actual_rows = count_jsonl(predictions_path)
    if actual_rows != expected_rows:
        return False, f"prediction count {actual_rows} != {expected_rows}"
    if not metrics.get("finished_at_utc"):
        return False, "missing finished_at_utc"
    return True, f"{actual_rows} predictions"


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def report_rows(seeds: tuple[int, ...]) -> list[list[str]]:
    rows: list[list[str]] = []
    for seed in seeds:
        for task in TASKS:
            for method in REPORT_METHODS:
                for interface in INTERFACES:
                    out_dir = eval_run_directory(task, method, interface, seed)
                    metrics_path = out_dir / "metrics.json"
                    if not metrics_path.is_file():
                        continue
                    try:
                        metrics = read_json(metrics_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    if metrics.get("exp_name") != eval_exp_name(
                        task, method, interface, seed
                    ):
                        continue
                    aggregate = metrics.get("aggregate") or {}
                    rows.append(
                        [
                            str(seed),
                            task,
                            method,
                            interface,
                            format_metric(metrics.get("eval_condition")),
                            format_metric(aggregate.get("samples")),
                            format_metric(metrics.get("test_accuracy")),
                            format_metric(metrics.get("test_macro_f1")),
                            format_metric(metrics.get("test_qwk")),
                            format_metric(metrics.get("test_mae")),
                            format_metric(metrics.get("test_rail_mae")),
                            format_metric(metrics.get("test_rail_mse")),
                            format_metric(metrics.get("test_rail_rmse")),
                            format_metric(metrics.get("format_valid_rate")),
                            format_metric(metrics.get("score_prefix_valid_rate")),
                            format_metric(metrics.get("reasoning_valid_rate")),
                            format_metric(metrics.get("avg_score_probability_mass")),
                            format_metric(metrics.get("generation_accuracy")),
                            format_metric(metrics.get("avg_output_tokens")),
                            format_metric(metrics.get("avg_reasoning_tokens")),
                            format_metric(metrics.get("gpu_time_sec")),
                            format_metric(metrics.get("wall_time_sec")),
                        ]
                    )
    return rows


def write_report(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> None:
    headers = [
        "Seed",
        "Task",
        "Train",
        "Interface",
        "Condition",
        "N",
        "Accuracy",
        "Macro-F1",
        "QWK",
        "MAE",
        "RAIL MAE",
        "RAIL MSE",
        "RAIL RMSE",
        "Format valid",
        "Score-prefix valid",
        "Reasoning valid",
        "Score mass",
        "Train-val accuracy",
        "Avg output tok",
        "Avg reasoning tok",
        "GPU sec",
        "Wall sec",
    ]
    rows = report_rows(seeds)
    expected = len(seeds) * len(TASKS) * len(REPORT_METHODS) * len(INTERFACES)
    lines = [
        "# Qwen3-8B Seed 42-44 Results",
        "",
        f"Updated: {utc_now()}",
        "",
        f"Completed evaluations: {len(rows)} / {expected}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    atomic_write_text(REPORT_PATH, "\n".join(lines))


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
                    if evaluation_status(
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
    atomic_write_json(
        STATE_PATH,
        {
            "updated_at_utc": utc_now(),
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
            "report": str(REPORT_PATH),
        },
    )


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = shlex.join(command)
    print(f"+ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] + {printable}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def validate_training_config(task: str, method: str) -> None:
    path = train_config_path(task, method)
    if not path.is_file():
        raise RuntimeError(f"Missing training config: {path}")
    config = read_yaml(path)
    expected_exp = f"{task}#qwen3_8b#{method}"
    expected_output = f"checkpoints/training-8B/{task}/{method}"
    if config.get("experiment_name") != expected_exp:
        raise RuntimeError(f"experiment_name mismatch in {path}")
    if config.get("model_name_or_path") != MODEL_NAME:
        raise RuntimeError(f"model mismatch in {path}")
    if config.get("output_root") != expected_output:
        raise RuntimeError(f"output_root mismatch in {path}")
    for key in ("dataset_path", "split_path"):
        value = config.get(key)
        if not value or not (PROJECT_ROOT / str(value)).is_file():
            raise RuntimeError(f"Missing {key} target in {path}: {value}")
    label_path = config.get("label_dataset_path")
    if label_path and not (PROJECT_ROOT / str(label_path)).is_file():
        raise RuntimeError(f"Missing label_dataset_path target in {path}: {label_path}")


def validate_evaluation_config(task: str, method: str, interface: str) -> None:
    path = eval_config_path(task, method, interface)
    if not path.is_file():
        raise RuntimeError(f"Missing evaluation config: {path}")
    config = read_yaml(path)
    expected_adapter = f"checkpoints/training-8B/{task}/{method}"
    expected_dataset = f"data/{task}/{interface}/test_{interface}.jsonl"
    expected_train_config = rel(train_config_path(task, method))
    if config.get("model_name") != MODEL_NAME:
        raise RuntimeError(f"model mismatch in {path}")
    if config.get("adapter") != expected_adapter:
        raise RuntimeError(f"adapter root mismatch in {path}")
    if config.get("dataset_file") != expected_dataset:
        raise RuntimeError(f"dataset mismatch in {path}")
    if config.get("train_config") != expected_train_config:
        raise RuntimeError(f"train_config mismatch in {path}")
    if config.get("output_path") != "outputs/evaluations/evaluation-8b":
        raise RuntimeError(f"output_path mismatch in {path}")
    if not (PROJECT_ROOT / expected_dataset).is_file():
        raise RuntimeError(f"Missing evaluation dataset: {expected_dataset}")


def preflight(seeds: tuple[int, ...], *, run_dry_runs: bool) -> None:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    emit(
        "preflight_started",
        seeds=list(seeds),
        tasks=len(TASKS),
        methods=list(METHODS),
        interfaces=list(INTERFACES),
    )
    for task in TASKS:
        for method in METHODS:
            validate_training_config(task, method)
            for interface in INTERFACES:
                validate_evaluation_config(task, method, interface)
    model_path = (PROJECT_ROOT / MODEL_NAME).resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Base model directory missing: {model_path}")
    if run_dry_runs:
        for task in TASKS:
            for method in METHODS:
                config = train_config_path(task, method)
                command = [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    rel(config),
                    "--seed",
                    str(seeds[0]),
                    "--dry-run",
                ]
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if result.returncode != 0:
                    tail = "\n".join(result.stdout.splitlines()[-30:])
                    raise RuntimeError(f"Dry-run failed for {config}:\n{tail}")
    train_done, eval_done = completion_counts(seeds)
    remaining_train = len(seeds) * len(TASKS) * len(METHODS) - train_done
    checkpoint_free = shutil.disk_usage(CHECKPOINT_ROOT.resolve()).free
    project_free = shutil.disk_usage(PROJECT_ROOT).free
    compact_mode = os.environ.get("QWEN06B_ADAPTER_ONLY") == "1"
    estimated_per_run = 400 if compact_mode else 800
    reserve_gib = 3 if compact_mode else 5
    estimated_checkpoint_bytes = remaining_train * estimated_per_run * 1024 * 1024
    if checkpoint_free < estimated_checkpoint_bytes + reserve_gib * 1024**3:
        raise RuntimeError(
            "Insufficient checkpoint disk: "
            f"free={checkpoint_free / 1024**3:.1f}GiB "
            f"estimated_need={estimated_checkpoint_bytes / 1024**3:.1f}GiB"
        )
    if project_free < 1024**3:
        raise RuntimeError(
            f"Insufficient project disk: free={project_free / 1024**3:.1f}GiB"
        )
    emit(
        "preflight_completed",
        completed_training=train_done,
        remaining_training=remaining_train,
        completed_evaluations=eval_done,
        checkpoint_free_gib=f"{checkpoint_free / 1024**3:.1f}",
        project_free_gib=f"{project_free / 1024**3:.1f}",
        eval_decode_seed=EVAL_DECODE_SEED,
        note="runtime CLI overrides adapter, train_seed, and exp_name",
    )


def train(task: str, method: str, seed: int) -> Path:
    existing = completed_adapter(task, method, seed)
    if existing is not None:
        emit(
            "training_skipped",
            seed=seed,
            task=task,
            method=method,
            adapter=existing,
        )
        return existing
    current = {"phase": "train", "seed": seed, "task": task, "method": method}
    write_state(DEFAULT_SEEDS, status="running", current=current)
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        rel(train_config_path(task, method)),
        "--seed",
        str(seed),
        "--fresh",
    ]
    log_path = RUN_LOG_ROOT / f"seed{seed}" / task / f"{method}.train.log"
    emit("training_started", **current, log=log_path)
    return_code = run_command(command, log_path)
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
    return adapter


def evaluate(
    task: str,
    method: str,
    interface: str,
    seed: int,
    adapter: Path,
) -> None:
    valid, reason = evaluation_status(task, method, interface, seed, adapter)
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
        rel(eval_config_path(task, method, interface)),
        "--adapter",
        rel(checkpoint_method_root(task, method)),
        "--train_seed",
        str(seed),
        "--exp_name",
        eval_exp_name(task, method, interface, seed),
        "--seed",
        str(EVAL_DECODE_SEED),
    ]
    log_path = (
        RUN_LOG_ROOT
        / f"seed{seed}"
        / task
        / f"{method}.eval_on_{interface}.log"
    )
    emit(
        "evaluation_started",
        **current,
        adapter=adapter,
        previous_status=reason,
        log=log_path,
    )
    return_code = run_command(command, log_path)
    valid, reason = evaluation_status(task, method, interface, seed, adapter)
    if not valid:
        raise RuntimeError(
            f"Evaluation failed or incomplete: seed={seed} task={task} "
            f"method={method} interface={interface} rc={return_code}: {reason}"
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
    write_report(DEFAULT_SEEDS)


def run_queue(seeds: tuple[int, ...]) -> None:
    preflight(seeds, run_dry_runs=True)
    write_report(DEFAULT_SEEDS)
    write_state(seeds, status="running")
    emit("queue_started", seeds=list(seeds))
    try:
        for seed in seeds:
            emit("seed_started", seed=seed)
            for task in TASKS:
                emit("task_started", seed=seed, task=task)
                for method in method_order(seed, task):
                    adapter = train(task, method, seed)
                    for interface in INTERFACES:
                        evaluate(task, method, interface, seed, adapter)
                emit("task_completed", seed=seed, task=task)
            emit("seed_completed", seed=seed)
    except Exception as exc:
        write_report(DEFAULT_SEEDS)
        write_state(seeds, status="failed", error=str(exc))
        emit("queue_failed", error=str(exc))
        raise
    write_report(DEFAULT_SEEDS)
    write_state(seeds, status="completed")
    emit("queue_completed", seeds=list(seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds or any(seed not in DEFAULT_SEEDS for seed in seeds):
        raise SystemExit(f"Seeds must be selected from {DEFAULT_SEEDS}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except ValueError:
        omp_threads = 0
    if omp_threads < 1:
        os.environ["OMP_NUM_THREADS"] = "8"
    if args.report_only:
        write_report(DEFAULT_SEEDS)
        print(REPORT_PATH)
        return
    if args.preflight_only:
        preflight(seeds, run_dry_runs=True)
        write_report(DEFAULT_SEEDS)
        write_state(seeds, status="preflight_passed")
        return
    run_queue(seeds)


if __name__ == "__main__":
    main()
