"""Training pipeline orchestration."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import sys
from pathlib import Path
from typing import Any

from data_utils import (
    label_counts,
    load_or_create_split,
    load_rows,
    score_sets,
    split_rows,
)
from logic_snapshot import write_training_logic_snapshot
from metrics_utils import infer_supervision_mode, infer_task_name, short_model_name
from run_utils import (
    begin_attempt,
    best_checkpoint_epoch,
    create_run_directory,
    default_split_path,
    disable_incompatible_torchao,
    finish_attempt,
    git_metadata,
    read_config,
    rebase_best_checkpoint_path,
    resolve_path,
    resolve_resume_target,
    sha256_file,
    validate_resume_compatibility,
    write_json,
    write_jsonl,
)
from supervision import get_supervision_strategy, resolve_training_method
from supervision.base import TrainingBuildContext


def prepare_run_context(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_path(args.config)
    config = read_config(config_path)
    dataset_path = resolve_path(config["dataset_path"])
    split_seed = int(config.get("split_seed", 20260720))
    split_path = (
        resolve_path(config["split_path"])
        if config.get("split_path")
        else default_split_path(dataset_path, split_seed)
    )
    model_path = resolve_path(config["model_name_or_path"])

    dataset_hash = sha256_file(dataset_path)
    rows = load_rows(dataset_path)
    labels = score_sets(rows)
    task_name = infer_task_name(dataset_path, rows)
    dataset_supervision_mode = infer_supervision_mode(dataset_path, rows)
    training_method = resolve_training_method(config, dataset_supervision_mode)
    supervision_mode = (
        "align" if training_method == "align" else dataset_supervision_mode
    )
    model_short = short_model_name(model_path)
    run_tag = f"{task_name}|{supervision_mode}|{model_short}"

    split = load_or_create_split(
        rows,
        labels,
        split_path,
        dataset_hash,
        split_seed,
        float(config.get("validation_ratio", 0.1)),
        write_json=write_json,
    )
    train_rows, validation_rows = split_rows(rows, split)
    data_summary = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "split": str(split_path),
        "task": task_name,
        "supervision_mode": supervision_mode,
        "training_method": training_method,
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
    if training_method == "align":
        data_summary["align_train_views"] = len(train_rows) * 2

    resolved_config = {
        **config,
        "model_name_or_path": str(model_path),
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "seed": args.seed,
        "task": task_name,
        "supervision_mode": supervision_mode,
        "training_method": training_method,
    }

    resume_checkpoint = None
    resume_state = None
    resume_details = None
    run_id = None
    run_directory = None
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

    return {
        "args": args,
        "config": config,
        "model_path": model_path,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "labels": labels,
        "task_name": task_name,
        "supervision_mode": supervision_mode,
        "training_method": training_method,
        "run_tag": run_tag,
        "data_summary": data_summary,
        "resolved_config": resolved_config,
        "resume_checkpoint": resume_checkpoint,
        "resume_state": resume_state,
        "resume_details": resume_details,
        "run_id": run_id,
        "run_directory": run_directory,
    }


def run_training(context: dict[str, Any]) -> None:
    args = context["args"]
    config = context["config"]
    resume_checkpoint = context["resume_checkpoint"]
    resume_state = context["resume_state"]
    run_directory = context["run_directory"]
    run_id = context["run_id"]

    import torch
    import transformers
    import trl
    from datasets import __version__ as datasets_version
    from peft import LoraConfig, TaskType, __version__ as peft_version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run training on a GPU node.")

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
    logger = logging.getLogger("training.pipeline")
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
    environment_metadata["logic"] = write_training_logic_snapshot(
        run_directory, {**context, "run_id": run_id}
    )
    if resume_checkpoint is None:
        write_json(run_directory / "resolved_config.json", context["resolved_config"])
        write_json(run_directory / "data_summary.json", context["data_summary"])
        manifest = {"run_id": run_id, "mode": "fresh", **environment_metadata}
    else:
        from run_utils import read_json

        manifest = read_json(run_directory / "manifest.json")
        manifest["last_environment"] = environment_metadata
        manifest["resume_validation"] = context["resume_details"]
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
        if training.get("bf16", True) and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured bf16=true, but this GPU does not support BF16.")

        model_path = context["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if training.get("bf16", True) else torch.float16,
            attn_implementation=training.get("attn_implementation", "sdpa"),
        )
        model.config.use_cache = False
        torchao_note = disable_incompatible_torchao()
        if torchao_note:
            logger.warning(torchao_note)
            manifest["torchao_compatibility"] = torchao_note
            write_json(run_directory / "manifest.json", manifest)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora.get("target_modules", "all-linear"),
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias=str(lora.get("bias", "none")),
        )

        strategy = get_supervision_strategy(context["training_method"])
        build_context = TrainingBuildContext(
            config=config,
            seed=args.seed,
            run_id=run_id,
            run_directory=run_directory,
            run_tag=context["run_tag"],
            model_path=model_path,
            train_rows=context["train_rows"],
            validation_rows=context["validation_rows"],
            labels=context["labels"],
            logger=logger,
            manifest=manifest,
        )
        trainer = strategy.build_trainer(
            model=model,
            tokenizer=tokenizer,
            peft_config=peft_config,
            context=build_context,
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
                "[%s] Starting fresh run %s seed=%d train=%d val=%d scores=%s method=%s",
                context["run_tag"],
                run_id,
                args.seed,
                len(context["train_rows"]),
                len(context["validation_rows"]),
                context["labels"],
                context["training_method"],
            )
        else:
            logger.info(
                "[%s] Resuming run %s from %s (step=%d epoch=%s)",
                context["run_tag"],
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
            context["run_tag"],
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
            "run_tag": context["run_tag"],
            "task": context["task_name"],
            "supervision_mode": context["supervision_mode"],
            "training_method": context["training_method"],
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
            "resolved_config": context["resolved_config"],
        }
        write_json(run_directory / "summary.json", summary)
        finish_attempt(manifest, status="completed")
        write_json(run_directory / "manifest.json", manifest)
        logger.info(
            "[%s] Completed run %s | best_epoch=%s best_acc=%s",
            context["run_tag"],
            run_id,
            best_epoch,
            trainer.state.best_metric,
        )
        logger.info("[%s] Validation metrics: %s", context["run_tag"], validation_metrics)
    except BaseException as error:
        finish_attempt(manifest, status="failed", error=error)
        write_json(run_directory / "manifest.json", manifest)
        logger.exception("Run failed")
        raise
