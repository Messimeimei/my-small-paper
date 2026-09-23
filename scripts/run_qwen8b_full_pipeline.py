#!/usr/bin/env python3
"""Run the complete ordered Qwen3-8B seed42/43/44 pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_qwen8b_self_correct_queue as self_queue
import run_qwen8b_three_method_queue as base_queue

PROJECT_ROOT = base_queue.PROJECT_ROOT
STATE_ROOT = PROJECT_ROOT / "outputs" / "qwen8b_full_pipeline"
STATE_PATH = STATE_ROOT / "state.json"
EVENTS_PATH = STATE_ROOT / "events.jsonl"
SEEDS = (42, 43, 44)
ALL_METHODS = base_queue.METHODS + self_queue.METHODS


def emit(event: str, **fields: Any) -> None:
    payload = {"time_utc": base_queue.utc_now(), "event": event, **fields}
    print(" ".join(f"{key}={value}" for key, value in payload.items()), flush=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def completion_counts() -> tuple[int, int]:
    train_done = 0
    eval_done = 0
    for seed in SEEDS:
        for task in base_queue.TASKS:
            for method in ALL_METHODS:
                adapter = base_queue.completed_adapter(task, method, seed)
                if adapter is None:
                    continue
                train_done += 1
                for interface in base_queue.INTERFACES:
                    if base_queue.evaluation_status(
                        task, method, interface, seed, adapter
                    )[0]:
                        eval_done += 1
    return train_done, eval_done


def write_state(
    *,
    status: str,
    current: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    train_done, eval_done = completion_counts()
    base_queue.atomic_write_json(
        STATE_PATH,
        {
            "updated_at_utc": base_queue.utc_now(),
            "status": status,
            "seeds": list(SEEDS),
            "tasks": list(base_queue.TASKS),
            "methods": list(ALL_METHODS),
            "completed_training_runs": train_done,
            "total_training_runs": len(SEEDS) * len(base_queue.TASKS) * len(ALL_METHODS),
            "completed_evaluations": eval_done,
            "total_evaluations": (
                len(SEEDS)
                * len(base_queue.TASKS)
                * len(ALL_METHODS)
                * len(base_queue.INTERFACES)
            ),
            "current": current,
            "error": error,
            "results_report": str(base_queue.REPORT_PATH),
        },
    )


def ensure_disk_headroom() -> None:
    free = shutil.disk_usage(base_queue.CHECKPOINT_ROOT.resolve()).free
    if free < 1024**3:
        raise RuntimeError(
            f"Checkpoint disk below 1 GiB safety floor: {free / 1024**3:.2f} GiB"
        )


def run_base_seed(seed: int) -> None:
    emit("base_phase_started", seed=seed)
    base_queue.preflight((seed,), run_dry_runs=True)
    for task in base_queue.TASKS:
        for method in base_queue.METHODS:
            ensure_disk_headroom()
            current = {
                "phase": "base",
                "seed": seed,
                "task": task,
                "method": method,
            }
            write_state(status="running", current=current)
            adapter = base_queue.train(task, method, seed)
            for interface in base_queue.INTERFACES:
                base_queue.evaluate(task, method, interface, seed, adapter)
    emit("base_phase_completed", seed=seed)


def run_seed42_generation() -> None:
    current = {"phase": "generate_seed42_self_data", "seed": 42}
    write_state(status="running", current=current)
    emit("generation_phase_started", seed=42)
    result = subprocess.run(
        [sys.executable, "scripts/run_qwen8b_self_data.py"],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Seed42 self-data generation failed: rc={result.returncode}")
    emit("generation_phase_completed", seed=42)


def run_self_seed(seed: int) -> None:
    emit("self_correct_phase_started", seed=seed)
    self_queue.preflight((seed,), run_dry_runs=True)
    for task in self_queue.TASKS:
        for method in self_queue.METHODS:
            ensure_disk_headroom()
            current = {
                "phase": "self_correct",
                "seed": seed,
                "task": task,
                "method": method,
            }
            write_state(status="running", current=current)
            adapter, _ = self_queue.train(task, method, seed)
            for interface in self_queue.INTERFACES:
                self_queue.evaluate(task, method, interface, seed, adapter)
    emit("self_correct_phase_completed", seed=seed)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) < 1:
        os.environ["OMP_NUM_THREADS"] = "8"

    base_queue.write_report(SEEDS)
    write_state(status="running")
    emit("pipeline_started")
    try:
        run_base_seed(42)
        run_seed42_generation()
        run_self_seed(42)
        for seed in (43, 44):
            run_base_seed(seed)
            run_self_seed(seed)
    except Exception as exc:
        base_queue.write_report(SEEDS)
        write_state(status="failed", error=str(exc))
        emit("pipeline_failed", error=str(exc))
        raise
    base_queue.write_report(SEEDS)
    write_state(status="completed")
    emit("pipeline_completed")


if __name__ == "__main__":
    main()
