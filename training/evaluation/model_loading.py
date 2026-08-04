"""Adapter discovery, temporary model merging, and vLLM initialization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shared.project_io import read_json, resolve_path, utc_now, write_json


def import_vllm() -> tuple[Any, Any]:
    """Import lazily so configuration and --help work without vLLM installed."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "vLLM is required for evaluation. Install the evaluation dependencies."
        ) from exc
    return LLM, SamplingParams


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


def normalize_adapter(value: str | None, *, train_seed: int | None = None) -> Path | None:
    """Resolve a direct adapter or the latest completed run under a method root."""
    if value is None or not str(value).strip() or str(value).strip().lower() == "none":
        return None
    path = resolve_path(str(value).strip())
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
        suffix = f" for training seed {train_seed}" if train_seed is not None else ""
        raise SystemExit(f"No completed adapter found under {path}{suffix}.")
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


def disable_incompatible_torchao() -> str | None:
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return None
    major, minor, *_ = (int(part) for part in version.split(".")[:2])
    if (major, minor) >= (0, 16):
        return None
    from peft.tuners.lora import torchao as peft_torchao_backend

    peft_torchao_backend.is_torchao_available = lambda: False
    return f"Disabled optional torchao {version}; PEFT requires >=0.16.0."


def adapter_weight_file(adapter: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter / name
        if path.is_file():
            return path
    raise SystemExit(f"No adapter weights found under {adapter}")


def merged_model_dir(base: Path, adapter: Path, cache_root: Path) -> Path:
    weight = adapter_weight_file(adapter)
    digest = hashlib.sha1()
    for value in (base, adapter, weight.stat().st_mtime_ns, weight.stat().st_size):
        digest.update(str(value).encode())
    return cache_root / digest.hexdigest()[:16]


def cleanup_merged_cache(cache_root: Path, retention_days: float) -> None:
    if not cache_root.is_dir():
        return
    cutoff = None if retention_days <= 0 else time.time() - retention_days * 86400
    removed = 0
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        marker = child / ".ok"
        try:
            mtime = (marker if marker.is_file() else child).stat().st_mtime
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
        age = "all leftovers" if retention_days <= 0 else f"older than {retention_days:g} day(s)"
        print(f"cleaned {removed} merged cache entries ({age})", flush=True)


def remove_merged_model(path: Path | None) -> None:
    if path is None:
        return
    targets = (
        (path, "temporary merged model"),
        (
            path.with_name(path.name + ".tmp"),
            "incomplete merge dir",
        ),
    )
    for target, label in targets:
        if not target.exists():
            continue
        try:
            shutil.rmtree(target)
            print(f"removed {label}: {target}", flush=True)
        except OSError as exc:
            print(f"failed to remove {target}: {exc}", flush=True)


def ensure_merged_model(base: Path, adapter: Path, cache_root: Path) -> Path:
    """Merge LoRA on CPU; the caller owns deletion of the returned directory."""
    out = merged_model_dir(base, adapter, cache_root)
    remove_merged_model(out)
    note = disable_incompatible_torchao()
    if note:
        print(note, flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
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
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    write_json(
        temporary / "merge_meta.json",
        {"base_model": str(base), "adapter": str(adapter), "merged_at_utc": utc_now()},
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
    llm_class, sampling_params_class = import_vllm()
    llm = llm_class(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )
    params = sampling_params_class(
        max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=seed
    )
    return llm, params


def load_train_run_metadata(adapter: Path | None) -> dict[str, Any]:
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
        adapter_source = summary.get("adapter_source_checkpoint") is not None
        final = summary.get("final_checkpoint") is not None
        # Exported adapters may originate from a selected or the final checkpoint.
        meta.update(
            {
                "checkpoint": summary.get("adapter_source_checkpoint")
                or summary.get("final_checkpoint")
                or summary.get("best_checkpoint"),
                "checkpoint_epoch": summary.get("adapter_source_checkpoint_epoch")
                if adapter_source
                else summary.get("final_checkpoint_epoch")
                if final
                else summary.get("best_checkpoint_epoch"),
                "checkpoint_step": summary.get("adapter_source_checkpoint_step")
                if adapter_source
                else summary.get("final_checkpoint_step")
                if final
                else summary.get("best_checkpoint_step"),
                "generation_accuracy": summary.get("final_generation_accuracy")
                if final
                else summary.get("best_generation_accuracy"),
                "checkpoint_retention": summary.get("checkpoint_retention") or "best",
                "adapter_selection": summary.get("adapter_selection")
                or ("final_checkpoint" if final else "best_checkpoint"),
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


def gpu_time_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(0)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(0)),
            }
        )
    return info
