"""Run directories, resume, manifest, and shared I/O helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
REQUIRED_RESUME_FILES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
}
_CN_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
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
    return dataset_path.parent / "splits" / f"{dataset_path.stem}_seed{split_seed}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
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


def create_run_directory(config: dict[str, Any], seed: int) -> tuple[str, Path]:
    timestamp = datetime.now(_CN_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{config['experiment_name']}__seed{seed}__{timestamp}"
    output_root = resolve_path(config.get("output_root", "train_outputs"))
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_id, run_directory


def best_checkpoint_epoch(trainer_state: Any) -> float | None:
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
    comparable = json.loads(json.dumps(config))
    comparable.pop("output_root", None)
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
