#!/usr/bin/env python3
"""Analyze final CC and LC adapters on the same CoT validation set.

For Base, CC, and LC, this script computes teacher-forced rationale NLL,
rationale loss share, and free-generation QWK or Macro-F1. Score NLL remains
in raw per-model JSON for audit only because a gold rationale can reveal the
score. Each result is saved immediately, so an interrupted run can resume.

Examples:
    python scripts/analyze_cc_lc_nll.py

    python scripts/analyze_cc_lc_nll.py \
        --tasks rev_util_actionability rev_util_helpfulness \
        --seeds 42 43 \
        --model model/Qwen3-4B \
        --output-dir outputs/analysis/cc_lc_nll_subset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.datasets import load_rows, score_sets  # noqa: E402
from models.adapters import (  # noqa: E402
    disable_incompatible_torchao,
    load_train_run_metadata,
    normalize_adapter,
)
from training.validation import generate_validation  # noqa: E402
from utils.io import read_json, resolve_path, utc_now, write_json, write_jsonl  # noqa: E402


DEFAULT_TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
ORDINAL_TASKS = frozenset(DEFAULT_TASKS[:4])
CONDITIONS = ("Base", "CC", "LC")

REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.I | re.S)
SCORE_RE = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.I)


@dataclass(frozen=True)
class EncodedRow:
    input_ids: list[int]
    completion_mask: list[bool]
    rationale_mask: list[bool]
    score_mask: list[bool]


@dataclass(frozen=True)
class TaskData:
    name: str
    dataset_path: Path
    split_path: Path
    dataset_sha256: str
    split_sha256: str
    rows: list[dict[str, Any]]
    encoded_rows: list[EncodedRow]
    labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--model", type=Path, default=Path("model/Qwen3-4B"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cc_lc_nll"),
    )
    parser.add_argument(
        "--dataset-template",
        default="data/{task}/cot/train_cot.jsonl",
        help="CoT JSONL path template.",
    )
    parser.add_argument(
        "--split-template",
        default="data/{task}/cot/splits/train_cot_seed20260720.json",
        help="Fixed training-validation split path template.",
    )
    parser.add_argument(
        "--cc-checkpoint-template",
        default="checkpoints/{task}/cot",
        help="Root containing completed CoT runs.",
    )
    parser.add_argument(
        "--lc-checkpoint-template",
        default="checkpoints/{task}/label_only",
        help="Root containing completed Label-only runs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--nll-batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute per-model files that already exist.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write JSON/CSV/Markdown without figures.",
    )
    args = parser.parse_args()

    if not args.tasks or len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks must be non-empty and contain no duplicates")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must be non-empty and contain no duplicates")
    if args.max_length <= 1 or args.max_new_tokens <= 0:
        parser.error("length limits must be positive")
    if args.nll_batch_size <= 0 or args.generation_batch_size <= 0:
        parser.error("batch sizes must be positive")
    return args


def path_from_template(template: str, task: str) -> Path:
    try:
        return resolve_path(template.format(task=task))
    except KeyError as error:
        raise ValueError(f"Unknown placeholder in {template!r}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_span_mask(
    offsets: list[tuple[int, int]], span_start: int, span_end: int
) -> list[bool]:
    return [
        end > start and start < span_end and end > span_start
        for start, end in offsets
    ]


def encode_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> EncodedRow:
    """Render one chat and locate rationale/score tokens by character offsets."""
    content = str(row["completion"][0]["content"])
    rationale_match = REASONING_RE.search(content)
    score_match = SCORE_RE.search(content)
    if rationale_match is None or score_match is None:
        raise ValueError(f"{row['id']} has no valid rationale and numeric score tags")
    if int(score_match.group(1)) != int(row["label"]):
        raise ValueError(
            f"{row['id']} teacher score {score_match.group(1)} != label {row['label']}"
        )

    prompt_text = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [*row["prompt"], *row["completion"]],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full_text.startswith(prompt_text + content):
        raise ValueError(f"Unexpected assistant boundary for {row['id']}")

    encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if input_ids[: len(prompt_ids)] != list(prompt_ids):
        raise ValueError(f"Unstable prompt token boundary for {row['id']}")
    if len(input_ids) > max_length:
        raise ValueError(
            f"{row['id']} has {len(input_ids)} tokens, above --max-length={max_length}; "
            "increase the limit instead of truncating the score target"
        )

    content_start = len(prompt_text)
    completion = [index >= len(prompt_ids) for index in range(len(input_ids))]
    rationale = token_span_mask(
        offsets,
        content_start + rationale_match.start(1),
        content_start + rationale_match.end(1),
    )
    score = token_span_mask(
        offsets,
        content_start + score_match.start(1),
        content_start + score_match.end(1),
    )
    rationale = [a and b for a, b in zip(completion, rationale, strict=True)]
    score = [a and b for a, b in zip(completion, score, strict=True)]
    if not any(completion) or not any(rationale) or not any(score):
        raise ValueError(f"Empty completion/rationale/score mask for {row['id']}")
    if any(a and b for a, b in zip(rationale, score, strict=True)):
        raise ValueError(f"Overlapping rationale and score masks for {row['id']}")
    return EncodedRow(input_ids, completion, rationale, score)


def load_task(task: str, args: argparse.Namespace, tokenizer: Any) -> TaskData:
    dataset_path = path_from_template(args.dataset_template, task)
    split_path = path_from_template(args.split_template, task)
    if not dataset_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing dataset or split for {task}")

    all_rows = load_rows(dataset_path)
    rows_by_id = {row["id"]: row for row in all_rows}
    validation_ids = read_json(split_path).get("validation_ids")
    if not isinstance(validation_ids, list) or not validation_ids:
        raise ValueError(f"No validation_ids in {split_path}")
    missing = [sample_id for sample_id in validation_ids if sample_id not in rows_by_id]
    if missing:
        raise ValueError(f"Split references missing IDs in {dataset_path}: {missing[:5]}")
    rows = [rows_by_id[sample_id] for sample_id in validation_ids]
    encoded_rows = [encode_row(row, tokenizer, args.max_length) for row in rows]
    labels = score_sets(all_rows)
    print(f"loaded {task}: validation={len(rows)} labels={labels}", flush=True)
    return TaskData(
        task,
        dataset_path,
        split_path,
        sha256_file(dataset_path),
        sha256_file(split_path),
        rows,
        encoded_rows,
        labels,
    )


def batches(rows: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def nll_batch(rows: list[EncodedRow], pad_token_id: int, device: Any) -> dict[str, Any]:
    import torch

    width = max(len(row.input_ids) for row in rows)
    shape = (len(rows), width)
    input_ids = torch.full(shape, pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros(shape, dtype=torch.long, device=device)
    masks = {
        name: torch.zeros(shape, dtype=torch.bool, device=device)
        for name in ("completion", "rationale", "score")
    }
    for index, row in enumerate(rows):
        length = len(row.input_ids)
        input_ids[index, :length] = torch.tensor(row.input_ids, device=device)
        attention[index, :length] = 1
        for name in masks:
            masks[name][index, :length] = torch.tensor(
                getattr(row, f"{name}_mask"), device=device
            )
    return {"input_ids": input_ids, "attention_mask": attention, "masks": masks}


def partitioned_nll(
    model: Any,
    tokenizer: Any,
    task: TaskData,
    batch_size: int,
    device: Any,
) -> dict[str, float | int]:
    """Compute token-micro-average NLL with the causal one-token shift."""
    import torch
    import torch.nn.functional as functional

    loss_sums = {name: 0.0 for name in ("completion", "rationale", "score")}
    token_counts = {name: 0 for name in loss_sums}
    model.eval()
    old_use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.inference_mode():
            total_batches = math.ceil(len(task.encoded_rows) / batch_size)
            for index, row_batch in enumerate(batches(task.encoded_rows, batch_size), 1):
                batch = nll_batch(row_batch, tokenizer.pad_token_id, device)
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits[:, :-1, :]
                targets = batch["input_ids"][:, 1:]
                completion = batch["masks"]["completion"][:, 1:]

                # Only completion positions participate in SFT loss.
                selected_nll = functional.cross_entropy(
                    logits[completion].float(),
                    targets[completion],
                    reduction="none",
                )
                loss_sums["completion"] += float(selected_nll.sum().item())
                token_counts["completion"] += int(completion.sum().item())
                for name in ("rationale", "score"):
                    selected_mask = batch["masks"][name][:, 1:][completion]
                    loss_sums[name] += float(selected_nll[selected_mask].sum().item())
                    token_counts[name] += int(selected_mask.sum().item())

                print(
                    f"  NLL {task.name}: {index}/{total_batches}",
                    end="\r" if index < total_batches else "\n",
                    flush=True,
                )
                del batch, logits, targets, completion, selected_nll
    finally:
        model.config.use_cache = old_use_cache

    if any(count == 0 for count in token_counts.values()):
        raise RuntimeError(f"At least one empty NLL partition for {task.name}")
    completion_loss = loss_sums["completion"]
    return {
        "completion_nll": completion_loss / token_counts["completion"],
        "rationale_nll": loss_sums["rationale"] / token_counts["rationale"],
        "score_nll": loss_sums["score"] / token_counts["score"],
        "rationale_loss_share": loss_sums["rationale"] / completion_loss,
        "score_loss_share": loss_sums["score"] / completion_loss,
        "completion_loss_sum": completion_loss,
        "rationale_loss_sum": loss_sums["rationale"],
        "score_loss_sum": loss_sums["score"],
        "completion_tokens": token_counts["completion"],
        "rationale_tokens": token_counts["rationale"],
        "score_tokens": token_counts["score"],
    }


def result_path(output_dir: Path, task: str, condition: str, seed: int | None) -> Path:
    suffix = "base" if seed is None else f"seed{seed}"
    return output_dir / "runs" / task / f"{condition.lower()}_{suffix}.json"


def evaluate_model(
    model: Any,
    tokenizer: Any,
    task: TaskData,
    condition: str,
    seed: int | None,
    adapter: Path | None,
    train_meta: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: Any,
) -> dict[str, Any]:
    path = result_path(output_dir, task.name, condition, seed)
    if path.is_file() and not args.overwrite:
        print(f"resume: {path}", flush=True)
        return read_json(path)

    print(f"evaluating task={task.name} condition={condition} seed={seed}", flush=True)
    nll = partitioned_nll(model, tokenizer, task, args.nll_batch_size, device)
    generation, predictions = generate_validation(
        model,
        tokenizer,
        task.rows,
        batch_size=args.generation_batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        score_sets=task.labels,
    )
    primary_name = "qwk" if task.name in ORDINAL_TASKS else "macro_f1"
    generation["primary_metric_name"] = primary_name
    generation["primary_metric"] = generation.get(primary_name)

    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "task": task.name,
        "condition": condition,
        "train_seed": seed,
        "model": str(resolve_path(args.model)),
        "adapter": str(adapter) if adapter else None,
        "training_run": train_meta or None,
        "dataset": {
            "path": str(task.dataset_path),
            "sha256": task.dataset_sha256,
            "split_path": str(task.split_path),
            "split_sha256": task.split_sha256,
            "validation_samples": len(task.rows),
            "score_sets": task.labels,
        },
        "nll": nll,
        "generation": generation,
        "notes": {
            "nll": "Teacher-forced token-micro-average negative log likelihood.",
            "loss_share": "Partition NLL sum divided by total completion NLL sum.",
            "qwk": "Computed over format-valid predictions, matching project metrics.",
            "score_nll": (
                "Audit only; not used as scoring evidence because the gold rationale "
                "may explicitly reveal the score."
            ),
        },
    }
    write_json(path, result)
    prediction_file = (
        output_dir / "predictions" / task.name / path.with_suffix(".jsonl").name
    )
    write_jsonl(prediction_file, predictions)
    print(f"wrote {path}", flush=True)
    return result


FIELDS = {
    "rationale_nll": "nll.rationale_nll",
    "completion_nll": "nll.completion_nll",
    "rationale_loss_share": "nll.rationale_loss_share",
    "primary_metric": "generation.primary_metric",
    "accuracy": "generation.accuracy",
    "format_valid_rate": "generation.format_valid_rate",
}


def get_number(payload: dict[str, Any], path: str) -> float | None:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return None if value is None else float(value)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(results: list[dict[str, Any]], tasks: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for task in tasks:
        for condition in CONDITIONS:
            runs = [
                result
                for result in results
                if result["task"] == task and result["condition"] == condition
            ]
            if not runs:
                continue
            row: dict[str, Any] = {
                "task": task,
                "condition": condition,
                "seeds": ",".join(
                    str(run["train_seed"])
                    for run in runs
                    if run["train_seed"] is not None
                ),
                "runs": len(runs),
                "primary_metric_name": "qwk" if task in ORDINAL_TASKS else "macro_f1",
            }
            for field, path in FIELDS.items():
                values = [
                    value
                    for run in runs
                    if (value := get_number(run, path)) is not None
                ]
                row[f"{field}_mean"], row[f"{field}_std"] = mean_std(values)
            summary.append(row)
    return summary


def paired_rows(
    results: list[dict[str, Any]], tasks: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    indexed = {
        (result["task"], result["condition"], result["train_seed"]): result
        for result in results
    }
    rows = []
    for task in tasks:
        for seed in seeds:
            cc, lc = indexed[(task, "CC", seed)], indexed[(task, "LC", seed)]
            row: dict[str, Any] = {
                "task": task,
                "seed": seed,
                "primary_metric_name": "qwk" if task in ORDINAL_TASKS else "macro_f1",
            }
            for field, path in FIELDS.items():
                cc_value, lc_value = get_number(cc, path), get_number(lc, path)
                row[f"cc_{field}"] = cc_value
                row[f"lc_{field}"] = lc_value
                row[f"cc_minus_lc_{field}"] = (
                    cc_value - lc_value
                    if cc_value is not None and lc_value is not None
                    else None
                )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    def show(value: Any) -> str:
        return "--" if value is None else f"{float(value):.4f}"

    lines = [
        "# CC vs LC NLL analysis",
        "",
        "All conditions use the same CoT training-validation examples. NLL is "
        "teacher-forced; QWK/Macro-F1 comes from free greedy generation.",
        "",
        "| Task | Condition | Seeds | Rationale NLL | Completion NLL | Rationale share | Primary metric | Valid rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['task']} | {row['condition']} | {row['seeds'] or 'base'} | "
            f"{show(row['rationale_nll_mean'])} | {show(row['completion_nll_mean'])} | "
            f"{show(row['rationale_loss_share_mean'])} | "
            f"{show(row['primary_metric_mean'])} ({row['primary_metric_name']}) | "
            f"{show(row['format_valid_rate_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rationale modeling is supported when CC has lower rationale NLL than "
            "LC. CC lower than Base additionally shows learning of the teacher "
            "rationale distribution. Scoring ability is judged only by free-generation "
            "QWK (ordinal) or Macro-F1 (classification). Score NLL is audit-only. "
            "NLL does not measure rationale quality or causal gradients.",
            "",
            f"Paired CC-LC rows: {len(paired)}. See paired_differences.csv.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_summary(output_dir: Path, summary: list[dict[str, Any]], tasks: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install matplotlib/numpy or rerun with --no-plots") from error

    by_key = {(row["task"], row["condition"]): row for row in summary}
    colors = {"Base": "#6B7280", "CC": "#C44E52", "LC": "#2A7F62"}
    panels = (
        ("rationale_nll", "Rationale NLL"),
        ("completion_nll", "Completion NLL"),
        ("rationale_loss_share", "Rationale loss share"),
        ("primary_metric", "QWK (ordinal) / Macro-F1 (classification)"),
    )
    x = np.arange(len(tasks))
    width = 0.24
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, panels, strict=True):
        for offset, condition in zip((-width, 0, width), CONDITIONS, strict=True):
            rows = [by_key.get((task, condition), {}) for task in tasks]
            values = [row.get(f"{field}_mean", np.nan) for row in rows]
            errors = [row.get(f"{field}_std", 0.0) for row in rows]
            axis.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=condition,
                color=colors[condition],
            )
        axis.set_title(title)
        axis.set_xticks(
            x,
            [task.replace("rev_util_", "").replace("rw_gen_", "") for task in tasks],
            rotation=30,
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "endpoint_comparison.png", dpi=180)
    figure.savefig(figures / "endpoint_comparison.pdf")
    plt.close(figure)


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "model": str(resolve_path(args.model)),
        "dataset_template": args.dataset_template,
        "split_template": args.split_template,
        "cc_checkpoint_template": args.cc_checkpoint_template,
        "lc_checkpoint_template": args.lc_checkpoint_template,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "nll_batch_size": args.nll_batch_size,
        "generation_batch_size": args.generation_batch_size,
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model does not exist: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on a GPU or choose another --device")

    config = run_config(args)
    config_path = output_dir / "run_config.json"
    if config_path.is_file() and not args.overwrite and read_json(config_path) != config:
        raise ValueError("Output has a different config; use another directory or --overwrite")
    write_json(config_path, config)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for exact token span masks")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tasks = {task: load_task(task, args, tokenizer) for task in args.tasks}

    adapters: dict[tuple[str, str, int], Path] = {}
    for task in args.tasks:
        for seed in args.seeds:
            adapters[(task, "CC", seed)] = normalize_adapter(
                str(path_from_template(args.cc_checkpoint_template, task)),
                train_seed=seed,
            )
            adapters[(task, "LC", seed)] = normalize_adapter(
                str(path_from_template(args.lc_checkpoint_template, task)),
                train_seed=seed,
            )

    note = disable_incompatible_torchao()
    if note:
        print(note, flush=True)
    dtype = getattr(torch, args.dtype)
    print(f"loading base model: {model_path}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    device = next(base_model.parameters()).device
    results: list[dict[str, Any]] = []

    for task in args.tasks:
        results.append(
            evaluate_model(
                base_model,
                tokenizer,
                tasks[task],
                "Base",
                None,
                None,
                {},
                args,
                output_dir,
                device,
            )
        )

    for task in args.tasks:
        for seed in args.seeds:
            for condition in ("CC", "LC"):
                adapter = adapters[(task, condition, seed)]
                existing = result_path(output_dir, task, condition, seed)
                if existing.is_file() and not args.overwrite:
                    print(f"resume: {existing}", flush=True)
                    results.append(read_json(existing))
                    continue
                print(f"loading {condition} adapter: {adapter}", flush=True)
                peft_model = PeftModel.from_pretrained(
                    base_model, str(adapter), is_trainable=False
                )
                results.append(
                    evaluate_model(
                        peft_model,
                        tokenizer,
                        tasks[task],
                        condition,
                        seed,
                        adapter,
                        load_train_run_metadata(adapter),
                        args,
                        output_dir,
                        device,
                    )
                )
                base_model = peft_model.unload()
                del peft_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = summarize(results, args.tasks)
    paired = paired_rows(results, args.tasks, args.seeds)
    write_json(output_dir / "all_results.json", results)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_differences.csv", paired)
    (output_dir / "analysis.md").write_text(
        report_text(summary, paired), encoding="utf-8"
    )
    if not args.no_plots:
        plot_summary(output_dir, summary, args.tasks)
    print(f"analysis complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
