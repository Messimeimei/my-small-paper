#!/usr/bin/env python3
"""Run SSA v2 training and matched independent-score evaluation serially."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)
ORDINAL_TASKS = ALL_TASKS[:4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="'all', 'ordinal', or explicit task names.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phase",
        choices=("all", "train", "evaluate"),
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate training and evaluation configs without starting GPU work.",
    )
    return parser.parse_args()


def resolve_tasks(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(ALL_TASKS)
    if values == ["ordinal"]:
        return list(ORDINAL_TASKS)
    unknown = sorted(set(values) - set(ALL_TASKS))
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}; expected {list(ALL_TASKS)}")
    return list(dict.fromkeys(values))


def run(command: list[str], *, quiet: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def train(tasks: list[str], seed: int, *, dry_run: bool) -> None:
    for task in tasks:
        command = [
            sys.executable,
            "scripts/train.py",
            "--config",
            f"configs/training/{task}/ssa_v2.yaml",
            "--seed",
            str(seed),
        ]
        command.append("--dry-run" if dry_run else "--fresh")
        run(command)


def evaluate(tasks: list[str], seed: int, *, dry_run: bool) -> None:
    for task in tasks:
        for mode in ("cot", "label_only"):
            config = (
                f"configs/evaluation/{task}/ssa_v2/"
                f"greedy_on_{mode}.yaml"
            )
            if dry_run:
                run(
                    [
                        sys.executable,
                        "scripts/evaluate.py",
                        "--config",
                        config,
                        "--help",
                    ],
                    quiet=True,
                )
                continue
            run(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--config",
                    config,
                    "--train_seed",
                    str(seed),
                    "--exp_name",
                    (
                        f"{task}#qwen3_4b#ft#ssa_v2#greedy"
                        f"#on_{mode}#seed_{seed}"
                    ),
                ]
            )


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except ValueError:
        omp_threads = 0
    if omp_threads < 1:
        os.environ["OMP_NUM_THREADS"] = "8"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    print(
        f"SSA v2 tasks={tasks} seed={args.seed} phase={args.phase} "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    if args.phase in {"all", "train"}:
        train(tasks, args.seed, dry_run=args.dry_run)
    if args.phase in {"all", "evaluate"}:
        evaluate(tasks, args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
