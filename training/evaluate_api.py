#!/usr/bin/env python3
"""Evaluate version-pinned API models without touching local-model results."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from api_evaluation.pipeline import (
    execute_run,
    expand_runs,
    load_env,
    load_matrix_config,
    total_samples,
)
from api_evaluation.report import update_api_reports
from shared.project_io import resolve_path


def comparison_supplemental_roots(
    config: dict[str, object],
) -> tuple[tuple[Path, str], ...]:
    entries = config.get("comparison_supplemental_results", [])
    if not isinstance(entries, list):
        raise ValueError("comparison_supplemental_results must be a list")
    roots = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError("each supplemental result requires a path")
        roots.append(
            (
                resolve_path(str(entry["path"])),
                str(entry.get("label_suffix") or ""),
            )
        )
    return tuple(roots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable OpenAI-compatible API baselines."
    )
    parser.add_argument(
        "--config",
        default="eval_output/api_configs/fixed_version_baselines.yaml",
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument(
        "--mode", action="append", choices=("label_only", "cot"), dest="modes"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually issue paid API requests; without this flag only validate and preview.",
    )
    parser.add_argument("--refresh-report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    config_path = resolve_path(args.config)
    config = load_matrix_config(config_path)
    output_root = resolve_path(config["output_path"])
    local_root = resolve_path(config.get("local_results_path", "eval_output/results"))
    supplemental_roots = comparison_supplemental_roots(config)
    if args.refresh_report_only:
        paths = update_api_reports(
            output_root,
            local_root,
            supplemental_api_roots=supplemental_roots,
        )
        print("updated " + " and ".join(map(str, paths)))
        return

    runs = expand_runs(
        config,
        model_filters=set(args.models or []) or None,
        task_filters=set(args.tasks or []) or None,
        mode_filters=set(args.modes or []) or None,
    )
    if not runs:
        raise SystemExit("no API evaluation runs matched the filters")
    request_count = total_samples(runs, limit=args.limit)
    print(f"config: {config_path}")
    print(f"runs: {len(runs)}")
    print(f"API requests: {request_count}")
    print(f"output root: {output_root}")
    for run in runs:
        print(f"  {run.model} | {run.task} | {run.mode} | {run.dataset_file}")
    if not args.execute:
        print("preview only; add --execute to issue API requests")
        return

    env_file = resolve_path(config.get("env_file", ".env"))
    load_env(env_file, override=True)
    api_key_env = str(config["api_key_env"])
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing API key in environment variable {api_key_env}")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=str(config["api_base_url"]).rstrip("/"),
        timeout=float(config.get("timeout_seconds", 180)),
        max_retries=int(config.get("max_retries", 5)),
    )
    for run in runs:
        output_dir = execute_run(
            client=client,
            run=run,
            config=config,
            output_root=output_root,
            limit=args.limit,
        )
        print(f"wrote {output_dir}")
        update_api_reports(
            output_root,
            local_root,
            supplemental_api_roots=supplemental_roots,
        )
    paths = update_api_reports(
        output_root,
        local_root,
        supplemental_api_roots=supplemental_roots,
    )
    print("updated " + " and ".join(map(str, paths)))


if __name__ == "__main__":
    main()
