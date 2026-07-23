#!/usr/bin/env python3
"""Evaluate a base model (optionally + LoRA adapter/checkpoint) with vLLM.

Without --adapter (or with none/None/NONE), loads the base model directly.
With an adapter path, merges LoRA into a cached full model then loads with
plain vLLM (vLLM 0.8.4 + cachetools>=6 breaks enable_lora in spawned workers).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm import LLM, SamplingParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_RE = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.I)
DEFAULT_DATASET = (
    PROJECT_ROOT / "rw_gen__coherence__exact_user_deduplicated__test__n1046.json"
)
DEFAULT_MERGE_CACHE = PROJECT_ROOT / "merged"
DEFAULT_MERGE_RETENTION_DAYS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--model_name", required=True, help="Base model path.")
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "LoRA adapter/ or checkpoints/checkpoint-* path. "
            "Omit, or pass none/None/NONE, to evaluate the base model only."
        ),
    )
    parser.add_argument("--dataset_file", default=str(DEFAULT_DATASET))
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Must be 0: every rollout uses greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Must be 1.0: nucleus sampling is disabled for greedy decoding.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout", type=int, default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Prompt chunk size for progress logging; vLLM schedules internally.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory fraction of the visible device.",
    )
    parser.add_argument(
        "--merge_cache",
        default=str(DEFAULT_MERGE_CACHE),
        help="Directory for cached merged models (default: <project>/merged).",
    )
    parser.add_argument(
        "--merge_retention_days",
        type=float,
        default=DEFAULT_MERGE_RETENTION_DAYS,
        help=(
            "Delete cached merged models older than this many days "
            f"(default: {DEFAULT_MERGE_RETENTION_DAYS}). "
            "Set <=0 to disable cleanup."
        ),
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable native thinking in chat template when supported.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("test", payload.get("train"))
        metadata = payload.get("metadata")
    else:
        rows = payload
        metadata = None
    if not isinstance(rows, list) or not rows:
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
            }
        )
    return cleaned, score_sets


def extract_score(text: str, allowed_scores: set[int]) -> int | None:
    matches = SCORE_RE.findall(text or "")
    if not matches:
        return None
    score = int(matches[-1])
    return score if score in allowed_scores else None


def classification_metrics(
    predictions: list[dict[str, Any]], score_sets: list[int]
) -> dict[str, Any]:
    total = len(predictions)
    allowed_scores = set(score_sets)
    f1_values = []
    per_class: dict[str, dict[str, float | int]] = {}
    for label in score_sets:
        tp = sum(row["label"] == label and row["prediction"] == label for row in predictions)
        fp = sum(row["label"] != label and row["prediction"] == label for row in predictions)
        fn = sum(row["label"] == label and row["prediction"] != label for row in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(row["label"] == label for row in predictions),
        }
    return {
        "samples": total,
        "score_sets": score_sets,
        "accuracy": sum(bool(row["correct"]) for row in predictions) / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "format_valid_rate": sum(
            row["prediction"] in allowed_scores for row in predictions
        )
        / total,
        "per_class": per_class,
    }


def mean_rollout_metrics(
    rollout_metrics: list[dict[str, Any]], score_sets: list[int]
) -> dict[str, Any]:
    """Average metrics across rollouts instead of voting across predictions."""

    def summarize(values: list[float | int]) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        return float(array.mean()), float(array.std())

    aggregate: dict[str, Any] = {
        "samples": rollout_metrics[0]["samples"],
        "score_sets": score_sets,
        "per_class": {},
    }
    for metric in ("accuracy", "macro_f1", "format_valid_rate"):
        mean, std = summarize([rollout[metric] for rollout in rollout_metrics])
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
    """Remove stale merged-model directories under cache_root."""
    if retention_days <= 0 or not cache_root.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = _merged_entry_mtime(child)
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            shutil.rmtree(child)
            removed += 1
            print(f"removed stale merged cache: {child}", flush=True)
        except OSError as exc:
            print(f"failed to remove {child}: {exc}", flush=True)
    if removed:
        print(
            f"cleaned {removed} merged cache entr"
            f"{'y' if removed == 1 else 'ies'} "
            f"older than {retention_days:g} day(s)",
            flush=True,
        )


def ensure_merged_model(base: Path, adapter: Path, cache_root: Path) -> Path:
    """Merge LoRA on CPU and cache the full weights for plain vLLM loading."""
    out = merged_model_dir(base, adapter, cache_root)
    marker = out / ".ok"
    if marker.is_file() and (out / "config.json").is_file():
        # Refresh mtime so recently reused caches survive retention cleanup.
        now = utc_now()
        marker.write_text(now + "\n", encoding="utf-8")
        print(f"reusing merged model: {out}", flush=True)
        return out

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

    print(f"merging LoRA on CPU -> {out}", flush=True)
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
    if out.exists():
        shutil.rmtree(out)
    temporary.rename(out)
    marker.write_text(utc_now() + "\n", encoding="utf-8")
    print(f"merge done in {time.perf_counter() - started:.1f}s", flush=True)
    return out


def init_vllm(
    model_path: Path,
    *,
    max_model_len: int,
    max_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
) -> tuple[LLM, SamplingParams]:
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


def run_rollout(
    llm: LLM,
    sampling_params: SamplingParams,
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

    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    metrics["elapsed_sec"] = round(elapsed, 3)
    metrics["samples_per_sec"] = round(len(rows) / max(elapsed, 1e-9), 3)
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
    out_dir = resolve_path(args.output_path) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    rows, score_sets = load_rows(dataset_file)
    cleanup_merged_cache(merge_cache, args.merge_retention_days)
    if adapter is None:
        model_path = model_name
        backend = "vllm-base"
        print(
            f"backend={backend} samples={len(rows)} base={model_name} adapter=None",
            flush=True,
        )
    else:
        adapter_weight_file(adapter)
        model_path = ensure_merged_model(model_name, adapter, merge_cache)
        backend = "vllm-merged"
        print(
            f"backend={backend} samples={len(rows)} base={model_name} "
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
            f"[rollout {rollout_index}] "
            f"acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"valid={metrics['format_valid_rate']:.4f}",
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
                "task": row.get("task"),
                "aspect": row.get("aspect"),
            }
        )
    aggregate_metrics = mean_rollout_metrics(rollout_metrics, score_sets)

    summary = {
        "exp_name": args.exp_name,
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
        "aggregate": aggregate_metrics,
        "rollouts": rollout_metrics,
    }
    write_json(out_dir / "metrics.json", summary)

    prediction_path = out_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in prediction_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[done] acc={aggregate_metrics['accuracy']:.4f}"
        f"+/-{aggregate_metrics['accuracy_std']:.4f} "
        f"macro_f1={aggregate_metrics['macro_f1']:.4f}"
        f"+/-{aggregate_metrics['macro_f1_std']:.4f} "
        f"valid={aggregate_metrics['format_valid_rate']:.4f}"
        f"+/-{aggregate_metrics['format_valid_rate_std']:.4f}"
    )
    print(f"wrote {out_dir / 'metrics.json'}")
    print(f"wrote {prediction_path}")


if __name__ == "__main__":
    main()
