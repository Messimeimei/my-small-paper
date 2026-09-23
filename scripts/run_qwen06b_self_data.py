#!/usr/bin/env python3
"""Generate Qwen3-0.6B self-CoT datasets for completed seed42 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.adapters import adapter_weight_file, normalize_adapter

GENERATOR = (
    PROJECT_ROOT
    / "data"
    / "align_self_generated_data"
    / "generate_align_self_cot.py"
)
MODEL_NAME = "model/Qwen3-0.6B"
TRAIN_SEED = 42
GENERATION_SEED = 42
TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
VARIANTS = {
    "cot": {
        "method": "cot",
        "output_root": "data/cot_self_generated_data-0.6B",
    },
    "align": {
        "method": "paper_align",
        "output_root": "data/align_self_generated_data-0.6B",
    },
}
STATE_ROOT = PROJECT_ROOT / "outputs" / "qwen06b_self_data_generation"
STATE_PATH = STATE_ROOT / "state.json"
EVENTS_PATH = STATE_ROOT / "events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def emit(event: str, **fields: Any) -> None:
    payload = {"time_utc": utc_now(), "event": event, **fields}
    print(" ".join(f"{key}={value}" for key, value in payload.items()), flush=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_selection(
    values: list[str],
    *,
    all_values: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if values == ["all"]:
        return all_values
    unknown = sorted(set(values) - set(all_values))
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}; expected {list(all_values)}")
    return tuple(dict.fromkeys(values))


def adapter_root(task: str, variant: str) -> Path:
    method = str(VARIANTS[variant]["method"])
    return (
        PROJECT_ROOT
        / "checkpoints"
        / "training-0.6B"
        / task
        / method
    )


def output_root(variant: str) -> Path:
    return PROJECT_ROOT / str(VARIANTS[variant]["output_root"])


def resolve_adapter(task: str, variant: str) -> Path:
    root = adapter_root(task, variant)
    adapter = normalize_adapter(str(root), train_seed=TRAIN_SEED)
    if adapter is None:
        raise RuntimeError(
            f"No seed{TRAIN_SEED} adapter for task={task} variant={variant}"
        )
    weight = adapter_weight_file(adapter)
    if weight.stat().st_size <= 0:
        raise RuntimeError(f"Empty adapter weights: {weight}")
    return adapter.resolve()


def completed_output(task: str, variant: str, adapter: Path) -> tuple[bool, str]:
    root = output_root(variant)
    output_task = root / task
    manifest_path = root / ".runs" / task / "manifest.json"
    summary_path = root / ".runs" / task / "summary.json"
    if not output_task.is_dir() or not manifest_path.is_file() or not summary_path.is_file():
        return False, "missing output task or completion metadata"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid completion metadata: {exc}"
    signature = manifest.get("signature") or {}
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        return False, "generation metadata is not complete"
    if Path(str(signature.get("adapter"))).resolve() != adapter.resolve():
        return False, "adapter provenance mismatch"
    if int(signature.get("train_seed")) != TRAIN_SEED:
        return False, "training seed mismatch"
    if Path(str(signature.get("model_name"))).resolve() != (
        PROJECT_ROOT / MODEL_NAME
    ).resolve():
        return False, "base model mismatch"
    weight_meta = signature.get("adapter_weight") or {}
    weight = adapter_weight_file(adapter)
    if weight_meta.get("sha256") != sha256_file(weight):
        return False, "adapter weight hash mismatch"
    expected_samples = sum(
        1
        for path in sorted((PROJECT_ROOT / "data" / task / "cot").glob("train_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if int(summary.get("samples", -1)) != expected_samples:
        return False, "sample count mismatch"
    if float(summary.get("reasoning_valid_rate", 0.0)) != 1.0:
        return False, "reasoning validity is not 1.0"
    return True, f"{expected_samples} samples"


def build_command(
    task: str,
    variant: str,
    args: argparse.Namespace,
    *,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(GENERATOR.relative_to(PROJECT_ROOT)),
        "--model-name",
        MODEL_NAME,
        "--adapter",
        str(adapter_root(task, variant).relative_to(PROJECT_ROOT)),
        "--dataset-name",
        task,
        "--train-seed",
        str(TRAIN_SEED),
        "--source-root",
        "data",
        "--output-root",
        str(output_root(variant).relative_to(PROJECT_ROOT)),
        "--max_model_len",
        str(args.max_model_len),
        "--max_tokens",
        str(args.max_tokens),
        "--batch_size",
        str(args.batch_size),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(GENERATION_SEED),
        "--merge_cache",
        args.merge_cache,
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.restart:
        command.append("--restart")
    if dry_run:
        command.append("--dry-run")
    return command


def run_command(command: list[str], log_path: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy(), check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] + {shlex.join(command)}\n")
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
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Generator exited with rc={return_code}: {shlex.join(command)}"
        )


def write_state(
    *,
    status: str,
    tasks: tuple[str, ...],
    variants: tuple[str, ...],
    current: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    completed = 0
    for variant in variants:
        for task in tasks:
            adapter = resolve_adapter(task, variant)
            if completed_output(task, variant, adapter)[0]:
                completed += 1
    atomic_json(
        STATE_PATH,
        {
            "updated_at_utc": utc_now(),
            "status": status,
            "tasks": list(tasks),
            "variants": list(variants),
            "train_seed": TRAIN_SEED,
            "generation_seed": GENERATION_SEED,
            "completed": completed,
            "total": len(tasks) * len(variants),
            "current": current,
            "error": error,
        },
    )


def preflight(
    tasks: tuple[str, ...],
    variants: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    if not GENERATOR.is_file():
        raise RuntimeError(f"Generator missing: {GENERATOR}")
    model = (PROJECT_ROOT / MODEL_NAME).resolve()
    if not model.is_dir():
        raise RuntimeError(f"Base model missing: {model}")
    emit(
        "preflight_started",
        tasks=list(tasks),
        variants=list(variants),
        train_seed=TRAIN_SEED,
    )
    for variant in variants:
        root = output_root(variant)
        root.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            adapter = resolve_adapter(task, variant)
            emit(
                "adapter_resolved",
                task=task,
                variant=variant,
                method=VARIANTS[variant]["method"],
                adapter=adapter,
            )
            run_command(build_command(task, variant, args, dry_run=True))
    emit("preflight_completed", plans=len(tasks) * len(variants))


def generate(
    tasks: tuple[str, ...],
    variants: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    preflight(tasks, variants, args)
    write_state(status="running", tasks=tasks, variants=variants)
    try:
        for variant in variants:
            for task in tasks:
                adapter = resolve_adapter(task, variant)
                complete, reason = completed_output(task, variant, adapter)
                if complete and not args.overwrite:
                    emit(
                        "generation_skipped",
                        task=task,
                        variant=variant,
                        reason=reason,
                    )
                    continue
                current = {"task": task, "variant": variant}
                write_state(
                    status="running",
                    tasks=tasks,
                    variants=variants,
                    current=current,
                )
                log_path = STATE_ROOT / "logs" / variant / f"{task}.log"
                emit(
                    "generation_started",
                    task=task,
                    variant=variant,
                    adapter=adapter,
                    log=log_path,
                )
                run_command(
                    build_command(task, variant, args, dry_run=False),
                    log_path,
                )
                complete, reason = completed_output(task, variant, adapter)
                if not complete:
                    raise RuntimeError(
                        f"Published data failed validation: "
                        f"task={task} variant={variant}: {reason}"
                    )
                emit(
                    "generation_completed",
                    task=task,
                    variant=variant,
                    result=reason,
                )
    except Exception as exc:
        write_state(
            status="failed",
            tasks=tasks,
            variants=variants,
            error=str(exc),
        )
        emit("queue_failed", error=str(exc))
        raise
    write_state(status="completed", tasks=tasks, variants=variants)
    emit("queue_completed", outputs=len(tasks) * len(variants))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--max-model-len", "--max_model_len", dest="max_model_len", type=int, default=8192)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=512)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", dest="gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--merge-cache",
        "--merge_cache",
        dest="merge_cache",
        default="/root/autodl-tmp/merged_qwen06b_self_data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = resolve_selection(
        args.tasks,
        all_values=TASKS,
        label="tasks",
    )
    variants = resolve_selection(
        args.variants,
        all_values=tuple(VARIANTS),
        label="variants",
    )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) < 1:
        os.environ["OMP_NUM_THREADS"] = "8"
    if args.dry_run:
        preflight(tasks, variants, args)
        write_state(status="dry_run_passed", tasks=tasks, variants=variants)
        return
    generate(tasks, variants, args)


if __name__ == "__main__":
    main()
