"""End-to-end evaluation orchestration."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.io import PROJECT_ROOT, resolve_path, utc_now, write_json
from utils.metrics import (
    criterion_title,
    infer_supervision_mode,
    infer_task_name,
    short_model_name,
)
from evaluation.reporting import update_evaluation_analysis
from evaluation.config import (
    eval_run_dir,
    load_optional_yaml,
    parse_args,
    resolved_eval_config,
    task_name_from_exp,
    validate_args,
)
from evaluation.dataset import load_rows, mean_rollout_metrics
from models import get_model_backend
from models.adapters import (
    adapter_weight_file,
    cleanup_merged_cache,
    ensure_merged_model,
    load_train_run_metadata,
    normalize_adapter,
    remove_merged_model,
)
from evaluation.inference import get_inference_method


def run_evaluation(args: Any) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    validate_args(args)
    method = get_inference_method(args.inference_mode)
    model_backend = get_model_backend(args.backend)
    model_name = resolve_path(args.model_name)
    adapter = normalize_adapter(args.adapter, train_seed=args.train_seed)
    dataset_file = resolve_path(args.dataset_file)
    merge_cache = resolve_path(args.merge_cache)
    output_root = resolve_path(args.output_path)
    train_config = (
        load_optional_yaml(resolve_path(args.train_config))
        if args.train_config
        else None
    )

    model_backend.set_seed(args.seed)
    rows, score_sets = load_rows(dataset_file)
    task_name = infer_task_name(dataset_file, rows)
    # Prefer dataset-derived task for folder layout; fall back to exp_name prefix.
    task_folder = task_name if task_name and task_name != "unknown" else task_name_from_exp(
        args.exp_name
    )
    out_dir = eval_run_dir(output_root, args.exp_name, task_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "resolved_config.json", resolved_eval_config(args, adapter=adapter))

    supervision_mode = infer_supervision_mode(dataset_file, rows)
    aspect = rows[0].get("aspect") or task_name
    criterion = criterion_title(str(aspect))
    run_tag = f"{task_name}|{supervision_mode}|{short_model_name(model_name)}"
    train_meta = load_train_run_metadata(adapter)

    cleanup_merged_cache(merge_cache, args.merge_retention_days)
    # Also sweep the legacy project-local merge dir if it still holds leftovers.
    legacy_merge = PROJECT_ROOT / "merged"
    if legacy_merge.resolve() != merge_cache.resolve() and legacy_merge.is_dir():
        cleanup_merged_cache(legacy_merge, 0)

    wall_started = time.perf_counter()
    gpu_before = model_backend.gpu_snapshot()
    merged_to_cleanup: Path | None = None
    llm = None
    try:
        if adapter is None:
            model_path = model_name
            backend = f"{model_backend.name}-base"
            print(
                f"[{run_tag}] backend={backend} samples={len(rows)} "
                f"base={model_name} adapter=None",
                flush=True,
            )
        else:
            adapter_weight_file(adapter)
            model_path = ensure_merged_model(model_name, adapter, merge_cache)
            merged_to_cleanup = model_path
            backend = f"{model_backend.name}-merged"
            print(
                f"[{run_tag}] backend={backend} samples={len(rows)} base={model_name} "
                f"adapter={adapter} merged={model_path}",
                flush=True,
            )

        llm, sampling_params = model_backend.initialize(
            model_path,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        method_runtime = method.prepare(
            llm=llm,
            default_sampling_params=sampling_params,
            score_sets=score_sets,
            args=args,
            run_tag=run_tag,
        )

        rollout_metrics = []
        rollout_predictions: list[list[dict[str, Any]]] = []
        for rollout_index in range(1, args.rollout + 1):
            predictions, metrics = method.run_rollout(
                llm=llm,
                runtime=method_runtime,
                rows=rows,
                score_sets=score_sets,
                args=args,
                rollout_index=rollout_index,
            )
            rollout_predictions.append(predictions)
            rollout_metrics.append(metrics)
            print(
                f"[{run_tag}][rollout {rollout_index}] "
                f"acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"valid={metrics['format_valid_rate']:.4f} "
                f"gpu_s={metrics['gpu_time_sec']:.1f}",
                flush=True,
            )

        prediction_records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            per_rollout = [
                predictions[index] for predictions in rollout_predictions
            ]
            prediction_records.append(
                method.build_prediction_record(row, per_rollout)
            )
        aggregate_metrics = mean_rollout_metrics(
            rollout_metrics,
            score_sets,
            scalar_metrics=method.aggregate_metrics,
        )
        avg_output_tokens = None
        avg_reasoning_tokens = None
        token_means = [
            rollout["tokens"]["avg_output_tokens"]
            for rollout in rollout_metrics
            if rollout.get("tokens", {}).get("avg_output_tokens") is not None
        ]
        reason_means = [
            rollout["tokens"]["avg_reasoning_tokens"]
            for rollout in rollout_metrics
            if rollout.get("tokens", {}).get("avg_reasoning_tokens") is not None
        ]
        if token_means:
            avg_output_tokens = float(np.mean(token_means))
        if reason_means:
            avg_reasoning_tokens = float(np.mean(reason_means))
        aggregate_metrics["tokens"] = {
            "avg_output_tokens": avg_output_tokens,
            "avg_reasoning_tokens": avg_reasoning_tokens,
        }
        aggregate_metrics["gpu_time_sec"] = float(
            np.mean([rollout["gpu_time_sec"] for rollout in rollout_metrics])
        )
        aggregate_metrics["elapsed_sec"] = float(
            np.mean([rollout["elapsed_sec"] for rollout in rollout_metrics])
        )

        train_config_path = str(args.train_config) if args.train_config else None
        configured_method = ((train_config or {}).get("supervision") or {}).get(
            "method"
        )
        eval_condition = method.resolve_condition(
            exp_name=args.exp_name,
            supervision_mode=supervision_mode,
            adapter=str(adapter) if adapter is not None else None,
            train_config=train_config_path,
            training_method=configured_method,
        )

        full_config = resolved_eval_config(args, adapter=adapter)
        full_config["train_config"] = train_config
        full_config["train_run"] = train_meta

        wall_elapsed = time.perf_counter() - wall_started
        summary = {
            "exp_name": args.exp_name,
            "run_tag": run_tag,
            "task": task_name,
            "criterion": criterion,
            "aspect": aspect,
            "supervision_mode": supervision_mode,
            "evaluation_mode": supervision_mode,
            "eval_condition": eval_condition,
            "output_dir": str(out_dir),
            "backend": backend,
            "model_backend": model_backend.name,
            "model_name": str(model_name),
            "adapter": str(adapter) if adapter is not None else None,
            "merged_model": str(model_path) if adapter is not None else None,
            "dataset_file": str(dataset_file),
            "score_sets": score_sets,
            "seed": args.seed,
            "train_seed_requested": args.train_seed,
            "rollout": args.rollout,
            "inference_mode": method.name,
            **method.summary_metadata(args, method_runtime, score_sets),
            "temp": 0.0,
            "top_p": 1.0,
            "max_model_len": args.max_model_len,
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_thinking": args.enable_thinking,
            "finished_at_utc": utc_now(),
            "aggregation": "mean_over_rollouts",
            "checkpoint_retention": train_meta.get("checkpoint_retention"),
            "adapter_selection": train_meta.get("adapter_selection"),
            "checkpoint_epoch": train_meta.get("checkpoint_epoch"),
            "checkpoint_step": train_meta.get("checkpoint_step"),
            "generation_accuracy": train_meta.get("generation_accuracy"),
            "test_accuracy": aggregate_metrics.get("accuracy"),
            "test_macro_f1": aggregate_metrics.get("macro_f1"),
            "test_mae": aggregate_metrics.get("mae"),
            "test_qwk": aggregate_metrics.get("qwk"),
            "test_rail_mae": aggregate_metrics.get("rail_mae"),
            "test_rail_mse": aggregate_metrics.get("rail_mse"),
            "test_rail_rmse": aggregate_metrics.get("rail_rmse"),
            "avg_score_probability_mass": aggregate_metrics.get(
                "avg_score_probability_mass"
            ),
            "format_valid_rate": aggregate_metrics.get("format_valid_rate"),
            "score_prefix_valid_rate": aggregate_metrics.get(
                "score_prefix_valid_rate"
            ),
            "reasoning_valid_rate": aggregate_metrics.get("reasoning_valid_rate"),
            "avg_output_tokens": avg_output_tokens,
            "avg_reasoning_tokens": avg_reasoning_tokens,
            "gpu_time_sec": aggregate_metrics.get("gpu_time_sec"),
            "wall_time_sec": round(wall_elapsed, 3),
            "gpu_before": gpu_before,
            "gpu_after": model_backend.gpu_snapshot(),
            "full_config": full_config,
            "aggregate": aggregate_metrics,
            "rollouts": rollout_metrics,
        }
        metrics_path = out_dir / "metrics.json"
        write_json(metrics_path, summary)

        prediction_path = out_dir / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in prediction_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Rebuild the single aggregate analysis report under outputs/evaluations/.
        analysis_path = update_evaluation_analysis(output_root)

        print(
            f"[{run_tag}][done] acc={aggregate_metrics['accuracy']:.4f}"
            f"+/-{aggregate_metrics.get('accuracy_std') or 0:.4f} "
            f"macro_f1={aggregate_metrics['macro_f1']:.4f}"
            f"+/-{aggregate_metrics.get('macro_f1_std') or 0:.4f} "
            f"valid={aggregate_metrics['format_valid_rate']:.4f}"
            f"+/-{aggregate_metrics.get('format_valid_rate_std') or 0:.4f}"
        )
        if aggregate_metrics.get("mae") is not None:
            print(
                f"[{run_tag}] mae={aggregate_metrics['mae']:.4f} "
                f"qwk={aggregate_metrics.get('qwk')}",
                flush=True,
            )
        print(
            f"[{run_tag}] tokens: output={avg_output_tokens} "
            f"reasoning={avg_reasoning_tokens} "
            f"gpu_s={aggregate_metrics.get('gpu_time_sec')} "
            f"checkpoint_epoch={train_meta.get('checkpoint_epoch')}",
            flush=True,
        )
        print(f"wrote {out_dir / 'resolved_config.json'}")
        print(f"wrote {metrics_path}")
        print(f"wrote {prediction_path}")
        print(f"updated {analysis_path}")
    finally:
        # Release vLLM before deleting on-disk merged weights.
        if llm is not None:
            try:
                del llm
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        remove_merged_model(merged_to_cleanup)


def main() -> None:
    args = parse_args()
    if args.refresh_analysis_only:
        output_root = resolve_path(args.output_path)
        analysis_path = update_evaluation_analysis(output_root)
        print(f"updated {analysis_path}")
        return
    run_evaluation(args)
