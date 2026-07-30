"""Record human-readable training and inference logic with source fingerprints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "training" / "logic_history"

TRAINING_LOGIC_SOURCES = (
    "training/logic_snapshot.py",
    "training/pipeline.py",
    "training/data_utils.py",
    "training/generative_trainer.py",
    "training/supervision/__init__.py",
    "training/supervision/standard.py",
    "training/supervision/align.py",
    "training/trainers/align_trainer.py",
)

INFERENCE_LOGIC_SOURCES = (
    "training/logic_snapshot.py",
    "training/evaluate.py",
    "training/metrics_utils.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints(relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {
        relative_path: _sha256(PROJECT_ROOT / relative_path)
        for relative_path in relative_paths
    }


def _logic_id(kind: str, variant: str, sources: dict[str, str]) -> str:
    payload = json.dumps(
        {"kind": kind, "variant": variant, "sources": sources},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _standard_objective(supervision_mode: str) -> dict[str, Any]:
    target = (
        "full rationale followed by <score>"
        if supervision_mode == "cot"
        else "<score> only"
    )
    return {
        "name": "standard_completion_sft",
        "views_per_original_sample": 1,
        "prompt_mode": supervision_mode,
        "target": target,
        "loss": (
            "Mean next-token cross-entropy over completion tokens; prompt and padding "
            "tokens are ignored."
        ),
        "micro_batch_backward": (
            "Each micro-batch runs one forward pass, computes its completion loss, "
            "and immediately accumulates gradients with backward()."
        ),
    }


def _align_objective() -> dict[str, Any]:
    return {
        "name": "align_unpaired_split_view",
        "views_per_original_sample": 2,
        "prompt_mode": "The original CoT prompt is reused for both views.",
        "targets": {
            "label_view": "<score> block only",
            "rationale_view": "<reasoning> block only",
        },
        "sampling": (
            "The two views are ordinary independent dataset rows. Trainer shuffling "
            "does not guarantee that views from the same source sample share a batch."
        ),
        "loss": (
            "Within each micro-batch, compute mean CE over label-view tokens and mean "
            "CE over rationale-view tokens separately, then use "
            "label_coeff * label_loss + rationale_coeff * rationale_loss. A missing "
            "view type contributes zero in that micro-batch."
        ),
        "micro_batch_backward": (
            "Each micro-batch computes the weighted loss and immediately accumulates "
            "gradients with backward(); losses are not collected across the full epoch "
            "before backward."
        ),
    }


def build_training_logic_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    training = config.get("training") or {}
    generation = config.get("generation") or {}
    lora = config.get("lora") or {}
    method = str(context["training_method"])
    supervision_mode = str(context["supervision_mode"])
    variant = f"{method}:{supervision_mode}"
    sources = source_fingerprints(TRAINING_LOGIC_SOURCES)
    objective = (
        _align_objective()
        if method == "align"
        else _standard_objective(supervision_mode)
    )
    accumulation = int(training.get("gradient_accumulation_steps", 16))
    micro_batch_size = int(training.get("per_device_train_batch_size", 1))
    supervision = config.get("supervision") or {}

    return {
        "schema_version": 1,
        "kind": "training",
        "logic_id": _logic_id("training", variant, sources),
        "variant": variant,
        "generated_at_utc": utc_now(),
        "logic": {
            "objective": objective,
            "parameter_update": {
                "adapter": "LoRA",
                "target_modules": lora.get("target_modules", "all-linear"),
                "micro_batch_size_per_device": micro_batch_size,
                "gradient_accumulation_steps": accumulation,
                "views_per_optimizer_step_per_device": micro_batch_size * accumulation,
                "order": (
                    "forward -> loss -> backward for each micro-batch; after the "
                    "configured accumulation count: gradient clipping -> optimizer "
                    "step -> scheduler step -> zero gradients"
                ),
            },
            "validation_and_selection": {
                "frequency": "end of every epoch",
                "teacher_forced": "completion loss on the fixed validation split",
                "generation": (
                    "Greedy model.generate on each validation prompt; parse the last "
                    "valid <score> value."
                ),
                "best_checkpoint_metric": "eval_generation_accuracy",
                "generation_batch_size": int(generation.get("batch_size", 1)),
                "max_new_tokens": int(generation.get("max_new_tokens", 512)),
            },
        },
        "effective_run": {
            "run_id": context.get("run_id"),
            "dataset": str(context["data_summary"]["dataset"]),
            "split": str(context["data_summary"]["split"]),
            "train_samples": context["data_summary"]["train"]["samples"],
            "validation_samples": context["data_summary"]["validation"]["samples"],
            "seed": context["resolved_config"]["seed"],
            "model_name_or_path": str(context["model_path"]),
            "learning_rate": float(training.get("learning_rate", 1e-4)),
            "epochs": float(training.get("num_train_epochs", 3)),
            "max_length": int(training.get("max_length", 8192)),
            "precision": "bf16" if training.get("bf16", True) else "fp16",
            "label_coeff": (
                float(supervision.get("label_coeff", 0.5)) if method == "align" else None
            ),
            "rationale_coeff": (
                float(supervision.get("rationale_coeff", 0.5))
                if method == "align"
                else None
            ),
        },
        "source_sha256": sources,
    }


def build_inference_logic_snapshot(
    *,
    args: Any,
    dataset_file: Path,
    sample_count: int,
    score_sets: list[int],
    supervision_mode: str,
    adapter: Path | None,
    train_meta: dict[str, Any],
) -> dict[str, Any]:
    variant = f"vllm:{supervision_mode}"
    sources = source_fingerprints(INFERENCE_LOGIC_SOURCES)
    return {
        "schema_version": 1,
        "kind": "inference",
        "logic_id": _logic_id("inference", variant, sources),
        "variant": variant,
        "generated_at_utc": utc_now(),
        "logic": {
            "model_loading": (
                "Base model directly when adapter is absent; otherwise merge the LoRA "
                "adapter into a temporary full model, then load it with vLLM."
            ),
            "prompt": (
                "Use each dataset row's stored prompt and apply the model chat template; "
                "the dataset mode determines Label-only versus CoT instructions."
            ),
            "decoding": "Greedy decoding with temperature=0 and top_p=1.",
            "prediction": (
                "Parse the last <score> integer in the allowed score set; missing or "
                "out-of-range scores are invalid and count as incorrect."
            ),
            "metrics": (
                "Accuracy, macro-F1, per-class metrics, format-valid rate, confusion "
                "matrix, token statistics, and MAE/QWK for ordinal score sets."
            ),
        },
        "effective_run": {
            "experiment": args.exp_name,
            "dataset": str(dataset_file),
            "samples": sample_count,
            "score_sets": score_sets,
            "adapter": str(adapter) if adapter is not None else None,
            "train_logic_run": train_meta.get("train_run_id"),
            "seed": int(args.seed),
            "rollouts": int(args.rollout),
            "batch_size": int(args.batch_size),
            "max_model_len": int(args.max_model_len),
            "max_tokens": int(args.max_tokens),
            "enable_thinking": bool(args.enable_thinking),
        },
        "source_sha256": sources,
    }


def render_logic_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        f"# {snapshot['kind'].title()} Logic Snapshot",
        "",
        f"- Logic ID: `{snapshot['logic_id']}`",
        f"- Variant: `{snapshot['variant']}`",
        f"- Generated at: `{snapshot['generated_at_utc']}`",
        "",
        "## Logic",
        "",
        "```json",
        json.dumps(snapshot["logic"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Effective Run",
        "",
        "```json",
        json.dumps(snapshot["effective_run"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Source Fingerprints",
        "",
    ]
    lines.extend(
        f"- `{path}`: `{digest}`" for path, digest in snapshot["source_sha256"].items()
    )
    return "\n".join(lines) + "\n"


def write_logic_snapshot(
    output_directory: Path,
    snapshot: dict[str, Any],
    *,
    history_directory: Path = DEFAULT_HISTORY_DIR,
) -> dict[str, str]:
    kind = snapshot["kind"]
    logic_id = snapshot["logic_id"]
    markdown = render_logic_markdown(snapshot)

    json_path = output_directory / f"{kind}_logic.json"
    markdown_path = output_directory / f"{kind}_logic.md"
    _write_json(json_path, snapshot)
    _write_text(markdown_path, markdown)

    version_root = output_directory / "logic_snapshots"
    version_json = version_root / f"{kind}_{logic_id}.json"
    version_markdown = version_root / f"{kind}_{logic_id}.md"
    if not version_json.exists():
        _write_json(version_json, snapshot)
        _write_text(version_markdown, markdown)

    history_payload = {
        key: snapshot[key]
        for key in (
            "schema_version",
            "kind",
            "logic_id",
            "variant",
            "logic",
            "source_sha256",
        )
    }
    history_json = history_directory / f"{kind}_{logic_id}.json"
    history_markdown = history_directory / f"{kind}_{logic_id}.md"
    if not history_json.exists():
        history_snapshot = {
            **history_payload,
            "generated_at_utc": snapshot["generated_at_utc"],
            "effective_run": {"note": "See run-local snapshot for effective settings."},
        }
        _write_json(history_json, history_payload)
        _write_text(history_markdown, render_logic_markdown(history_snapshot))

    return {
        "logic_id": logic_id,
        "snapshot_json": str(json_path),
        "snapshot_markdown": str(markdown_path),
        "history_json": str(history_json),
    }


def write_training_logic_snapshot(
    output_directory: Path, context: dict[str, Any]
) -> dict[str, str]:
    return write_logic_snapshot(output_directory, build_training_logic_snapshot(context))


def write_inference_logic_snapshot(
    output_directory: Path, **kwargs: Any
) -> dict[str, str]:
    return write_logic_snapshot(
        output_directory, build_inference_logic_snapshot(**kwargs)
    )
