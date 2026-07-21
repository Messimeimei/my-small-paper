#!/usr/bin/env python3
"""Evaluate a base model + LoRA adapter/checkpoint with vLLM.

vLLM 0.8.4 + cachetools>=6 breaks enable_lora in spawned workers, so this script
merges the adapter into a cached full model and loads that with plain vLLM.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm import LLM, SamplingParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_RE = re.compile(r"<score>\s*([01])\s*</score>", re.I)
DEFAULT_DATASET = (
    PROJECT_ROOT / "rw_gen__coherence__exact_user_deduplicated__test__n1046.json"
)
DEFAULT_MERGE_CACHE = PROJECT_ROOT / "eval_data" / ".merged_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--model_name", required=True, help="Base model path.")
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA adapter/ or checkpoints/checkpoint-* path.",
    )
    parser.add_argument("--dataset_file", default=str(DEFAULT_DATASET))
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
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
        help="Directory for cached merged models.",
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


def parse_label(row: dict[str, Any], index: int) -> int:
    raw = row.get("labels", row.get("label"))
    if raw in (0, 1, "0", "1"):
        return int(raw)
    raise ValueError(f"Row {index} has invalid label: {raw!r}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("test", payload.get("train"))
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty test/train list.")

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
                "label": parse_label(row, index),
                "prompt": row["prompt"],
                "task": row.get("task"),
                "aspect": row.get("aspect"),
            }
        )
    return cleaned


def extract_score(text: str) -> int | None:
    matches = SCORE_RE.findall(text or "")
    return int(matches[-1]) if matches else None


def classification_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    f1_values = []
    per_class: dict[str, dict[str, float | int]] = {}
    for label in (0, 1):
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
        "accuracy": sum(bool(row["correct"]) for row in predictions) / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "format_valid_rate": sum(row["prediction"] in {0, 1} for row in predictions)
        / total,
        "per_class": per_class,
    }


def majority_vote(scores: list[int | None]) -> int | None:
    valid = [score for score in scores if score in {0, 1}]
    if not valid:
        return None
    counts = Counter(valid)
    return sorted(counts.items(), key=lambda item: (-item[1], -item[0]))[0][0]


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


def ensure_merged_model(base: Path, adapter: Path, cache_root: Path) -> Path:
    """Merge LoRA on CPU and cache the full weights for plain vLLM loading."""
    out = merged_model_dir(base, adapter, cache_root)
    marker = out / ".ok"
    if marker.is_file() and (out / "config.json").is_file():
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
    temp: float,
    top_p: float,
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
        temperature=temp,
        top_p=top_p,
        seed=seed,
    )
    return llm, sampling_params


def run_rollout(
    llm: LLM,
    sampling_params: SamplingParams,
    rows: list[dict[str, Any]],
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
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = format_prompts(
            tokenizer, [row["prompt"] for row in batch], args.enable_thinking
        )
        if args.temp > 0:
            sampling_params.seed = args.seed + rollout_index * 100_000 + start

        completions = llm.generate(texts, sampling_params, use_tqdm=False)
        outputs = [completion.outputs[0].text for completion in completions]
        for row, output in zip(batch, outputs, strict=True):
            prediction = extract_score(output)
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
    metrics = classification_metrics(predictions)
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
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu_memory_utilization must be in (0, 1]")

    model_name = resolve_path(args.model_name)
    adapter = resolve_path(args.adapter)
    dataset_file = resolve_path(args.dataset_file)
    merge_cache = resolve_path(args.merge_cache)
    out_dir = resolve_path(args.output_path) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_weight_file(adapter)
    set_seed(args.seed)
    rows = load_rows(dataset_file)
    merged_path = ensure_merged_model(model_name, adapter, merge_cache)
    llm, sampling_params = init_vllm(
        merged_path,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(
        f"backend=vllm-merged samples={len(rows)} base={model_name} "
        f"adapter={adapter} merged={merged_path}",
        flush=True,
    )

    rollout_metrics = []
    rollout_predictions: list[list[dict[str, Any]]] = []
    for rollout_index in range(1, args.rollout + 1):
        predictions, metrics = run_rollout(
            llm,
            sampling_params,
            rows,
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

    aggregate: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        scores = [preds[index]["prediction"] for preds in rollout_predictions]
        outputs = [preds[index]["output"] for preds in rollout_predictions]
        prediction = majority_vote(scores)
        aggregate.append(
            {
                "id": row["id"],
                "label": row["label"],
                "prediction": prediction,
                "correct": prediction == row["label"],
                "rollout_predictions": scores,
                "outputs": outputs,
                "task": row.get("task"),
                "aspect": row.get("aspect"),
            }
        )
    aggregate_metrics = classification_metrics(aggregate)

    summary = {
        "exp_name": args.exp_name,
        "backend": "vllm-merged",
        "model_name": str(model_name),
        "adapter": str(adapter),
        "merged_model": str(merged_path),
        "dataset_file": str(dataset_file),
        "seed": args.seed,
        "rollout": args.rollout,
        "temp": args.temp,
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_thinking": args.enable_thinking,
        "finished_at_utc": utc_now(),
        "aggregate": aggregate_metrics,
        "rollouts": rollout_metrics,
    }
    write_json(out_dir / "metrics.json", summary)

    prediction_path = out_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in aggregate:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[done] acc={aggregate_metrics['accuracy']:.4f} "
        f"macro_f1={aggregate_metrics['macro_f1']:.4f} "
        f"valid={aggregate_metrics['format_valid_rate']:.4f}"
    )
    print(f"wrote {out_dir / 'metrics.json'}")
    print(f"wrote {prediction_path}")


if __name__ == "__main__":
    main()
