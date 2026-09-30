#!/usr/bin/env python3
"""Run and summarize multi-seed evaluations for a SciRM-family model."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_NAME = os.environ.get("SCIRM_EVAL_MODEL_DIR", "SciRM-7B")
MODEL_PATH = PROJECT_ROOT / "model" / MODEL_DIR_NAME
MODEL_REPO = os.environ.get("SCIRM_EVAL_MODEL_REPO", "UKPLab/SciRM-7B")
MODEL_REVISION = os.environ.get(
    "SCIRM_EVAL_MODEL_REVISION", "d0475b1725de05287d7e0b4a72a741ea6e452b6a"
)
MODEL_TAG = os.environ.get("SCIRM_EVAL_MODEL_TAG", "scirm_7b")
MODEL_DISPLAY_NAME = os.environ.get("SCIRM_EVAL_MODEL_DISPLAY", MODEL_DIR_NAME)
SEED42_TAG = os.environ.get("SCIRM_EVAL_SEED42_TAG", "base")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluations"
STATE_ROOT = PROJECT_ROOT / "outputs" / os.environ.get(
    "SCIRM_EVAL_STATE_NAME", "scirm_multiseed"
)
STATE_PATH = STATE_ROOT / "state.json"
EVENTS_PATH = STATE_ROOT / "events.jsonl"
SUMMARY_JSON = STATE_ROOT / "summary.json"
SUMMARY_MD = STATE_ROOT / "summary.md"

TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
ORDINAL_TASKS = frozenset(TASKS[:4])
INTERFACES = ("label_only", "cot")
RUN_SEEDS = tuple(
    int(value) for value in os.environ.get("SCIRM_EVAL_RUN_SEEDS", "43,44").split(",")
)
ALL_SEEDS = tuple(
    int(value) for value in os.environ.get("SCIRM_EVAL_ALL_SEEDS", "42,43,44").split(",")
)
METRIC_KEYS = (
    "test_accuracy",
    "test_macro_f1",
    "test_mae",
    "test_qwk",
    "format_valid_rate",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def emit(event: str, **fields: Any) -> None:
    payload = {"time_utc": utc_now(), "event": event, **fields}
    print(" ".join(f"{key}={value}" for key, value in payload.items()), flush=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def exp_name(task: str, interface: str, seed: int) -> str:
    seed_tag = SEED42_TAG if seed == 42 else str(seed)
    return f"{task}#{MODEL_TAG}#base#greedy#on_{interface}#seed_{seed_tag}"


def run_dir(task: str, interface: str, seed: int) -> Path:
    return OUTPUT_ROOT / task / exp_name(task, interface, seed)


def config_path(task: str, interface: str) -> Path:
    return (
        PROJECT_ROOT
        / "configs"
        / "evaluation"
        / task
        / "base"
        / f"greedy_on_{interface}.yaml"
    )


def nonempty_jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def load_complete_metrics(
    task: str, interface: str, seed: int
) -> dict[str, Any] | None:
    directory = run_dir(task, interface, seed)
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    resolved_path = directory / "resolved_config.json"
    if not all(path.is_file() for path in (metrics_path, predictions_path, resolved_path)):
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        dataset_path = Path(metrics["dataset_file"])
        if int(metrics["seed"]) != seed or metrics["task"] != task:
            return None
        if metrics["evaluation_mode"] != interface:
            return None
        if int(resolved["seed"]) != seed or not dataset_path.is_file():
            return None
        if Path(resolved["model_name"]).resolve() != MODEL_PATH.resolve():
            return None
        if nonempty_jsonl_rows(predictions_path) != nonempty_jsonl_rows(dataset_path):
            return None
        return metrics
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def completion_count(seeds: tuple[int, ...] = RUN_SEEDS) -> int:
    return sum(
        load_complete_metrics(task, interface, seed) is not None
        for seed in seeds
        for task in TASKS
        for interface in INTERFACES
    )


def write_state(
    *, status: str, current: dict[str, Any] | None = None, error: str | None = None
) -> None:
    atomic_write_json(
        STATE_PATH,
        {
            "updated_at_utc": utc_now(),
            "status": status,
            "model": str(MODEL_PATH),
            "run_seeds": list(RUN_SEEDS),
            "summary_seeds": list(ALL_SEEDS),
            "tasks": list(TASKS),
            "interfaces": list(INTERFACES),
            "completed_evaluations": completion_count(),
            "total_evaluations": len(RUN_SEEDS) * len(TASKS) * len(INTERFACES),
            "current": current,
            "error": error,
            "summary": str(SUMMARY_MD),
        },
    )


def preflight() -> None:
    required_model_files = (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    )
    missing = [name for name in required_model_files if not (MODEL_PATH / name).is_file()]
    if missing:
        raise RuntimeError(f"{MODEL_DISPLAY_NAME} download is incomplete; missing: {missing}")
    for shard in MODEL_PATH.glob("model-*.safetensors"):
        if shard.stat().st_size < 100_000_000:
            raise RuntimeError(f"Model shard is unexpectedly small: {shard}")
    missing_configs = [
        str(config_path(task, interface))
        for task in TASKS
        for interface in INTERFACES
        if not config_path(task, interface).is_file()
    ]
    if missing_configs:
        raise RuntimeError(f"Missing evaluation configs: {missing_configs}")


def run_evaluation(task: str, interface: str, seed: int, *, force: bool) -> None:
    existing = load_complete_metrics(task, interface, seed)
    if existing is not None and not force:
        emit("evaluation_skipped", task=task, interface=interface, seed=seed)
        return

    current = {"seed": seed, "task": task, "interface": interface}
    write_state(status="running", current=current)
    emit("evaluation_started", **current)
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        str(config_path(task, interface).relative_to(PROJECT_ROOT)),
        "--model_name",
        f"model/{MODEL_DIR_NAME}",
        "--seed",
        str(seed),
        "--exp_name",
        exp_name(task, interface, seed),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    completed = load_complete_metrics(task, interface, seed)
    if completed is None:
        raise RuntimeError(
            f"Evaluation failed or produced incomplete output: task={task} "
            f"interface={interface} seed={seed} rc={result.returncode}"
        )
    emit(
        "evaluation_completed",
        **current,
        returncode=result.returncode,
        accuracy=completed.get("test_accuracy"),
        macro_f1=completed.get("test_macro_f1"),
        qwk=completed.get("test_qwk"),
    )


def metric_stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    aggregates: dict[str, Any] = {}
    for task in TASKS:
        runs[task] = {}
        aggregates[task] = {}
        for interface in INTERFACES:
            seed_metrics: dict[str, Any] = {}
            loaded: list[dict[str, Any]] = []
            for seed in ALL_SEEDS:
                metrics = load_complete_metrics(task, interface, seed)
                if metrics is None:
                    raise RuntimeError(
                        f"Missing complete result for task={task}, "
                        f"interface={interface}, seed={seed}"
                    )
                selected = {key: metrics.get(key) for key in METRIC_KEYS}
                selected["finished_at_utc"] = metrics.get("finished_at_utc")
                selected["output_dir"] = metrics.get("output_dir")
                seed_metrics[str(seed)] = selected
                loaded.append(metrics)
            runs[task][interface] = seed_metrics
            aggregates[task][interface] = {}
            for key in METRIC_KEYS:
                values = [float(metrics[key]) for metrics in loaded if metrics.get(key) is not None]
                aggregates[task][interface][key] = metric_stats(values) if values else None

    primary_by_seed: dict[str, dict[str, float]] = {
        interface: {} for interface in INTERFACES
    }
    for interface in INTERFACES:
        for seed in ALL_SEEDS:
            values = []
            for task in TASKS:
                key = "test_qwk" if task in ORDINAL_TASKS else "test_macro_f1"
                values.append(float(runs[task][interface][str(seed)][key]))
            primary_by_seed[interface][str(seed)] = statistics.fmean(values)

    cross_task = {
        interface: metric_stats(list(primary_by_seed[interface].values()))
        | {"by_seed": primary_by_seed[interface]}
        for interface in INTERFACES
    }
    payload = {
        "generated_at_utc": utc_now(),
        "model": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH.resolve()),
        "seeds": list(ALL_SEEDS),
        "note": (
            "All evaluations use deterministic greedy decoding (temperature=0); "
            "seed variance is expected to be zero or negligible."
        ),
        "runs": runs,
        "aggregates": aggregates,
        "cross_task_primary": cross_task,
    }
    atomic_write_json(SUMMARY_JSON, payload)
    write_markdown_summary(payload)
    return payload


def display_task(task: str) -> str:
    return task.replace("rev_util_", "").replace("rw_gen_", "").replace("_", " ").title()


def write_markdown_summary(payload: dict[str, Any]) -> None:
    lines = [
        f"# {MODEL_DISPLAY_NAME} multi-seed evaluation",
        "",
        f"Generated at `{payload['generated_at_utc']}`.",
        f"Model: `{payload['model']}` at revision `{payload['model_revision']}`.",
        "",
        "Primary metric: QWK for the four review-utility tasks and Macro-F1 for "
        "the three writing-quality tasks. Higher is better.",
        "",
        "| Task | Interface | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        key = "test_qwk" if task in ORDINAL_TASKS else "test_macro_f1"
        for interface in INTERFACES:
            values = [
                float(payload["runs"][task][interface][str(seed)][key])
                for seed in ALL_SEEDS
            ]
            stats = payload["aggregates"][task][interface][key]
            lines.append(
                f"| {display_task(task)} | {interface} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + f" | {stats['mean']:.4f} +/- {stats['sample_std']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Cross-task primary average",
            "",
            "| Interface | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for interface in INTERFACES:
        stats = payload["cross_task_primary"][interface]
        values = [stats["by_seed"][str(seed)] for seed in ALL_SEEDS]
        lines.append(
            f"| {interface} | "
            + " | ".join(f"{value:.4f}" for value in values)
            + f" | {stats['mean']:.4f} +/- {stats['sample_std']:.4f} |"
        )
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun seed 43/44 evaluations even when complete outputs exist.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only validate and aggregate existing seed 42/43/44 outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if int(os.environ.get("OMP_NUM_THREADS", "0") or 0) < 1:
        os.environ["OMP_NUM_THREADS"] = "8"

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        preflight()
        if not args.summarize_only:
            write_state(status="running")
            emit("queue_started", seeds=list(RUN_SEEDS))
            for seed in RUN_SEEDS:
                for task in TASKS:
                    for interface in INTERFACES:
                        run_evaluation(task, interface, seed, force=args.force)
        summary = summarize()
        write_state(status="completed")
        emit(
            "queue_completed",
            completed=completion_count(),
            label_only_mean=summary["cross_task_primary"]["label_only"]["mean"],
            cot_mean=summary["cross_task_primary"]["cot"]["mean"],
        )
    except Exception as exc:
        write_state(status="failed", error=str(exc))
        emit("queue_failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
