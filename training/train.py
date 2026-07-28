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
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from metrics_utils import (
    SCORE_RE,
    infer_supervision_mode,
    infer_task_name,
    short_model_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
REQUIRED_RESUME_FILES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        "--resume-from-checkpoint",
        dest="resume",
        type=Path,
        default=None,
        help=(
            "Resume an existing run in place. Accepts either a run directory "
            "(latest complete checkpoint is selected) or checkpoint-* directory."
        ),
    )
    mode.add_argument(
        "--fresh",
        action="store_true",
        help="Explicitly start from the base model in a new timestamped run directory.",
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
    required = {"experiment_name", "model_name_or_path", "dataset_path"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing fields: {sorted(missing)}")
    return config


def default_split_path(dataset_path: Path, split_seed: int) -> Path:
    """Place fixed splits next to the dataset: <dir>/splits/<stem>_seedN.json."""
    return dataset_path.parent / "splits" / f"{dataset_path.stem}_seed{split_seed}.json"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_train_label(row: dict[str, Any], line_number: int) -> int:
    raw = row.get("label", row.get("labels"))
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"Invalid label at line {line_number}: {raw!r}")
    return raw


def load_rows(path: Path) -> list[dict[str, Any]]:
    """读训练 JSONL（data/*/cot|score_only/train_*.jsonl 或旧 lora_data）。"""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    declared_score_sets: list[int] | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", "")).strip()
            prompt = row.get("prompt")
            completion = row.get("completion")
            label = parse_train_label(row, line_number)
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Invalid or duplicate id at {path}:{line_number}")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"Invalid prompt at {path}:{line_number}")
            if (
                not isinstance(completion, list)
                or len(completion) != 1
                or completion[0].get("role") != "assistant"
                or not SCORE_RE.search(str(completion[0].get("content", "")))
            ):
                raise ValueError(f"Invalid completion at {path}:{line_number}")
            if row.get("score_sets") is not None:
                score_sets = row["score_sets"]
                if (
                    not isinstance(score_sets, list)
                    or not score_sets
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in score_sets)
                ):
                    raise ValueError(f"Invalid score_sets at {path}:{line_number}")
                if declared_score_sets is None:
                    declared_score_sets = list(score_sets)
                elif list(score_sets) != declared_score_sets:
                    raise ValueError(f"Inconsistent score_sets at {path}:{line_number}")
                if label not in set(declared_score_sets):
                    raise ValueError(
                        f"Label {label} outside score_sets at {path}:{line_number}"
                    )
            seen_ids.add(sample_id)
            normalized = {
                **row,
                "id": sample_id,
                "label": label,
                "prompt": prompt,
                "completion": completion,
            }
            rows.append(normalized)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def score_sets(rows: list[dict[str, Any]]) -> list[int]:
    """Infer the ordered score set represented by a LoRA dataset."""
    return sorted({row["label"] for row in rows})


def validation_counts(
    rows: list[dict[str, Any]], labels: list[int], ratio: float
) -> dict[int, int]:
    """按标签分层，决定验证集每类抽多少条。"""
    counts = {label: sum(row["label"] == label for row in rows) for label in labels}
    target_total = max(len(labels), round(len(rows) * ratio))
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
    labels: list[int],
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
        for label in labels
    }
    for ids in per_label.values():
        rng.shuffle(ids)
    selected_counts = validation_counts(rows, labels, validation_ratio)
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


def label_counts(rows: list[dict[str, Any]], labels: list[int]) -> dict[str, int]:
    return {
        str(label): sum(row["label"] == label for row in rows) for label in labels
    }


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


# 机器时区多为 UTC；run 目录名用北京时间，便于阅读。
_CN_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def create_run_directory(config: dict[str, Any], seed: int) -> tuple[str, Path]:
    """新建独立 run 目录：experiment__seed__北京时间。"""
    timestamp = datetime.now(_CN_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{config['experiment_name']}__seed{seed}__{timestamp}"
    output_root = resolve_path(config.get("output_root", "train_outputs"))
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_id, run_directory


def best_checkpoint_epoch(trainer_state: Any) -> float | None:
    """Resolve the epoch of the accuracy-selected best checkpoint."""
    best_step = getattr(trainer_state, "best_global_step", None)
    history = getattr(trainer_state, "log_history", None) or []
    if best_step is not None:
        for row in reversed(history):
            if int(row.get("step", -1)) == int(best_step) and "epoch" in row:
                return float(row["epoch"])
    if getattr(trainer_state, "epoch", None) is not None:
        return float(trainer_state.epoch)
    return None


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a Trainer checkpoint directory: {path}")
    return int(match.group(1))


def validate_resume_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Require all state needed for a true optimizer/scheduler/RNG resume."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint}")
    expected_step = checkpoint_step(checkpoint)
    missing = sorted(
        filename
        for filename in REQUIRED_RESUME_FILES
        if not (checkpoint / filename).is_file()
        or (checkpoint / filename).stat().st_size == 0
    )
    if not any(
        (checkpoint / filename).is_file() and (checkpoint / filename).stat().st_size > 0
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if missing:
        raise ValueError(
            f"Checkpoint is incomplete and cannot be resumed: {checkpoint}; "
            f"missing={missing}"
        )
    state = read_json(checkpoint / "trainer_state.json")
    if not isinstance(state, dict):
        raise ValueError(f"Invalid trainer_state.json in {checkpoint}")
    actual_step = int(state.get("global_step", -1))
    if actual_step != expected_step:
        raise ValueError(
            f"Checkpoint step mismatch: directory={expected_step}, trainer_state={actual_step}"
        )
    return state


def find_latest_complete_checkpoint(run_directory: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_root = run_directory / "checkpoints"
    if not checkpoint_root.is_dir():
        raise ValueError(f"Run has no checkpoints directory: {run_directory}")
    candidates = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name)
        ),
        key=checkpoint_step,
        reverse=True,
    )
    errors: list[str] = []
    for checkpoint in candidates:
        try:
            return checkpoint.resolve(), validate_resume_checkpoint(checkpoint)
        except ValueError as error:
            errors.append(str(error))
    detail = f"; invalid candidates={errors}" if errors else ""
    raise ValueError(f"No complete checkpoint found in {checkpoint_root}{detail}")


def resolve_resume_target(
    target: str | Path,
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Resolve a run/checkpoint argument to its run and complete checkpoint."""
    resolved = resolve_path(target)
    if CHECKPOINT_RE.fullmatch(resolved.name):
        if resolved.parent.name != "checkpoints":
            raise ValueError(
                f"Checkpoint must be inside a checkpoints/ directory: {resolved}"
            )
        run_directory = resolved.parent.parent
        checkpoint = resolved
        state = validate_resume_checkpoint(checkpoint)
    else:
        run_directory = resolved
        if not run_directory.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_directory}")
        checkpoint, state = find_latest_complete_checkpoint(run_directory)

    for filename in ("resolved_config.json", "data_summary.json", "manifest.json"):
        if not (run_directory / filename).is_file():
            raise ValueError(f"Resume run is missing {filename}: {run_directory}")
    manifest = read_json(run_directory / "manifest.json")
    run_id = str(manifest.get("run_id") or run_directory.name)
    return run_id, run_directory.resolve(), checkpoint.resolve(), state


def comparable_resume_config(config: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Remove paths that may safely differ when continuing an existing optimizer state."""
    comparable = json.loads(json.dumps(config))
    comparable.pop("output_root", None)  # A run may have been relocated after interruption.
    epochs = float(comparable.setdefault("training", {}).get("num_train_epochs", 3))
    return comparable, epochs


def validate_resume_compatibility(
    *,
    run_directory: Path,
    checkpoint_state: dict[str, Any],
    resolved_config: dict[str, Any],
    data_summary: dict[str, Any],
    validation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reject accidental resume with changed model/data/optimizer semantics."""
    previous_config = read_json(run_directory / "resolved_config.json")
    previous_data = read_json(run_directory / "data_summary.json")
    previous_comparable, previous_epochs = comparable_resume_config(previous_config)
    current_comparable, current_epochs = comparable_resume_config(resolved_config)
    if previous_comparable != current_comparable:
        raise ValueError(
            "Current config is incompatible with the interrupted run. Only output_root "
            "may change when resuming; use --fresh for changed training semantics."
        )
    for key in ("dataset_sha256", "all", "train", "validation"):
        if previous_data.get(key) != data_summary.get(key):
            raise ValueError(
                f"Data summary field {key!r} differs from the interrupted run; "
                "start with --fresh."
            )
    if validation_rows is not None:
        saved_validation_path = run_directory / "validation_dataset.json"
        if not saved_validation_path.is_file():
            raise ValueError(
                f"Interrupted run has no saved validation_dataset.json: {run_directory}"
            )
        saved_payload = read_json(saved_validation_path)
        saved_ids = [row["id"] for row in saved_payload.get("test", [])]
        current_ids = [row["id"] for row in validation_rows]
        if saved_ids != current_ids:
            raise ValueError(
                "Fixed split differs from the interrupted run; start with --fresh."
            )
    checkpoint_epoch = float(checkpoint_state.get("epoch") or 0.0)
    if current_epochs <= checkpoint_epoch:
        raise ValueError(
            f"No training remains: checkpoint epoch={checkpoint_epoch}, "
            f"configured num_train_epochs={current_epochs}."
        )
    return {
        "checkpoint_step": int(checkpoint_state["global_step"]),
        "checkpoint_epoch": checkpoint_epoch,
        "previous_num_train_epochs": previous_epochs,
        "configured_num_train_epochs": current_epochs,
    }


def rebase_best_checkpoint_path(
    checkpoint: Path, run_directory: Path
) -> dict[str, str] | None:
    """Repair absolute best-checkpoint paths after a run directory was moved."""
    state_path = checkpoint / "trainer_state.json"
    state = read_json(state_path)
    old_value = state.get("best_model_checkpoint")
    if not old_value:
        return None
    best_step = state.get("best_global_step")
    best_name = f"checkpoint-{int(best_step)}" if best_step is not None else Path(old_value).name
    new_path = (run_directory / "checkpoints" / best_name).resolve()
    if not new_path.is_dir():
        raise ValueError(
            f"Best checkpoint recorded by Trainer is unavailable after relocation: {new_path}"
        )
    if str(old_value) == str(new_path):
        return None
    state["best_model_checkpoint"] = str(new_path)
    write_json(state_path, state)
    return {"old": str(old_value), "new": str(new_path)}


def begin_attempt(
    manifest: dict[str, Any],
    *,
    mode: str,
    command: list[str],
    resume_checkpoint: Path | None,
    resume_state: dict[str, Any] | None,
) -> dict[str, Any]:
    history = manifest.setdefault("attempt_history", [])
    if not history and manifest.get("started_at_utc"):
        history.append(
            {
                "mode": manifest.get("mode", "fresh"),
                "status": manifest.get("status", "unknown"),
                "started_at_utc": manifest.get("started_at_utc"),
                "finished_at_utc": manifest.get("finished_at_utc"),
                "command": manifest.get("command"),
                "error": manifest.get("error"),
                "traceback": manifest.get("traceback"),
            }
        )
    attempt = {
        "mode": mode,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": command,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_from_step": int(resume_state["global_step"]) if resume_state else None,
        "resume_from_epoch": float(resume_state.get("epoch") or 0.0) if resume_state else None,
    }
    history.append(attempt)
    manifest.update(
        {
            "status": "running",
            "last_started_at_utc": attempt["started_at_utc"],
            "last_command": command,
            "resume_from_checkpoint": attempt["resume_from_checkpoint"],
        }
    )
    for key in ("finished_at_utc", "error", "traceback"):
        manifest.pop(key, None)
    return attempt


def finish_attempt(
    manifest: dict[str, Any],
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    finished_at = utc_now()
    attempt = manifest["attempt_history"][-1]
    attempt.update({"status": status, "finished_at_utc": finished_at})
    manifest.update({"status": status, "finished_at_utc": finished_at})
    if error is None:
        manifest.pop("error", None)
        manifest.pop("traceback", None)
        return
    error_text = f"{type(error).__name__}: {error}"
    traceback_text = traceback.format_exc()
    attempt.update({"error": error_text, "traceback": traceback_text})
    manifest.update({"error": error_text, "traceback": traceback_text})


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    config_path = resolve_path(args.config)
    config = read_config(config_path)
    dataset_path = resolve_path(config["dataset_path"])
    split_seed = int(config.get("split_seed", 20260720))
    if config.get("split_path"):
        split_path = resolve_path(config["split_path"])
    else:
        split_path = default_split_path(dataset_path, split_seed)
    model_path = resolve_path(config["model_name_or_path"])

    # --- 数据与固定划分 ---
    dataset_hash = sha256_file(dataset_path)
    rows = load_rows(dataset_path)
    labels = score_sets(rows)
    task_name = infer_task_name(dataset_path, rows)
    supervision_mode = infer_supervision_mode(dataset_path, rows)
    model_short = short_model_name(model_path)
    run_tag = f"{task_name}|{supervision_mode}|{model_short}"
    split = load_or_create_split(
        rows,
        labels,
        split_path,
        dataset_hash,
        split_seed,
        float(config.get("validation_ratio", 0.1)),
    )
    train_rows, validation_rows = split_rows(rows, split)
    data_summary = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "split": str(split_path),
        "task": task_name,
        "supervision_mode": supervision_mode,
        "score_sets": labels,
        "prompt_version": rows[0].get("prompt_version"),
        "teacher_models": rows[0].get("teacher_models"),
        "all": {"samples": len(rows), "labels": label_counts(rows, labels)},
        "train": {
            "samples": len(train_rows),
            "labels": label_counts(train_rows, labels),
        },
        "validation": {
            "samples": len(validation_rows),
            "labels": label_counts(validation_rows, labels),
        },
    }

    resolved_config = {
        **config,
        "model_name_or_path": str(model_path),
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "seed": args.seed,
        "task": task_name,
        "supervision_mode": supervision_mode,
    }
    resume_checkpoint: Path | None = None
    resume_state: dict[str, Any] | None = None
    resume_details: dict[str, Any] | None = None
    if args.resume is not None:
        run_id, run_directory, resume_checkpoint, resume_state = resolve_resume_target(
            args.resume
        )
        resume_details = validate_resume_compatibility(
            run_directory=run_directory,
            checkpoint_state=resume_state,
            resolved_config=resolved_config,
            data_summary=data_summary,
            validation_rows=validation_rows,
        )

    if args.dry_run:
        payload: dict[str, Any] = {
            "mode": "resume" if resume_checkpoint else "fresh",
            "data_summary": data_summary,
        }
        if resume_checkpoint is not None:
            payload["resume"] = {
                "run_directory": str(run_directory),
                "checkpoint": str(resume_checkpoint),
                **(resume_details or {}),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    import torch
    import transformers
    import trl
    from datasets import Dataset, __version__ as datasets_version
    from peft import LoraConfig, TaskType, __version__ as peft_version
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig

    from generative_trainer import (
        CompactTrainLogCallback,
        GenerativeEvalSFTTrainer,
        JsonlLogCallback,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run training on a GPU node.")

    # --- run 目录与复现元数据 ---
    if resume_checkpoint is None:
        run_id, run_directory = create_run_directory(config, args.seed)
    path_relocation = (
        rebase_best_checkpoint_path(resume_checkpoint, run_directory)
        if resume_checkpoint is not None
        else None
    )
    log_path = run_directory / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger(__name__)
    # Keep HuggingFace trainer logs quieter; CompactTrainLogCallback prints the useful lines.
    transformers.logging.set_verbosity_warning()
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

    environment_metadata = {
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
    if resume_checkpoint is None:
        write_json(run_directory / "resolved_config.json", resolved_config)
        write_json(run_directory / "data_summary.json", data_summary)
        manifest = {"run_id": run_id, "mode": "fresh", **environment_metadata}
    else:
        manifest = read_json(run_directory / "manifest.json")
        manifest["last_environment"] = environment_metadata
        manifest["resume_validation"] = resume_details
        if path_relocation is not None:
            manifest["best_checkpoint_path_relocation"] = path_relocation
    attempt = begin_attempt(
        manifest,
        mode="resume" if resume_checkpoint else "fresh",
        command=list(sys.argv),
        resume_checkpoint=resume_checkpoint,
        resume_state=resume_state,
    )
    if resume_checkpoint is None:
        manifest.update(
            {
                "started_at_utc": attempt["started_at_utc"],
                "command": list(sys.argv),
            }
        )
    write_json(run_directory / "manifest.json", manifest)

    try:
        training = config.get("training", {})
        lora = config.get("lora", {})
        generation = config.get("generation", {})
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
            # Save only when generation accuracy improves (not by eval_loss).
            save_strategy="best",
            load_best_model_at_end=True,
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
            score_sets=labels,
            generation_batch_size=int(generation.get("batch_size", 1)),
            generation_max_length=int(training.get("max_length", 8192)),
            generation_max_new_tokens=int(generation.get("max_new_tokens", 512)),
            run_directory=run_directory,
            logger=logger,
            callbacks=[
                JsonlLogCallback(run_directory / "train_history.jsonl", run_tag=run_tag),
                CompactTrainLogCallback(run_tag=run_tag, logger=logger),
            ],
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

        if resume_checkpoint is None:
            logger.info(
                "[%s] Starting fresh run %s seed=%d train=%d val=%d scores=%s",
                run_tag,
                run_id,
                args.seed,
                len(train_rows),
                len(validation_rows),
                labels,
            )
        else:
            logger.info(
                "[%s] Resuming run %s from %s (step=%d epoch=%s)",
                run_tag,
                run_id,
                resume_checkpoint,
                resume_state["global_step"],
                resume_state.get("epoch"),
            )
            if path_relocation is not None:
                logger.info(
                    "Rebased best checkpoint path: %s -> %s",
                    path_relocation["old"],
                    path_relocation["new"],
                )
        logger.info(
            "[%s] Trainable parameters=%d (%.4f%%); save_strategy=best by eval_generation_accuracy",
            run_tag,
            trainable_parameters,
            manifest["parameters"]["trainable_percent"],
        )
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
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

        best_epoch = best_checkpoint_epoch(trainer.state)
        summary = {
            "run_id": run_id,
            "run_tag": run_tag,
            "task": task_name,
            "supervision_mode": supervision_mode,
            "model_name_or_path": str(model_path),
            "seed": args.seed,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_checkpoint_epoch": best_epoch,
            "best_checkpoint_step": trainer.state.best_global_step,
            "best_generation_accuracy": trainer.state.best_metric,
            "metric_for_best_model": "eval_generation_accuracy",
            "train_metrics": train_result.metrics,
            "language_model_validation": language_model_metrics,
            "generation_validation": validation_metrics,
            "adapter_directory": str(adapter_directory),
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "resume_from_step": int(resume_state["global_step"]) if resume_state else None,
            "resolved_config": resolved_config,
        }
        write_json(run_directory / "summary.json", summary)
        finish_attempt(manifest, status="completed")
        write_json(run_directory / "manifest.json", manifest)
        logger.info(
            "[%s] Completed run %s | best_epoch=%s best_acc=%s",
            run_tag,
            run_id,
            best_epoch,
            trainer.state.best_metric,
        )
        logger.info("[%s] Validation metrics: %s", run_tag, validation_metrics)
    except BaseException as error:
        # 失败也写回 manifest，便于事后排查
        finish_attempt(manifest, status="failed", error=error)
        write_json(run_directory / "manifest.json", manifest)
        logger.exception("Run failed")
        raise


if __name__ == "__main__":
    main()
