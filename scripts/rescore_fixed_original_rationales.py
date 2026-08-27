#!/usr/bin/env python3
"""Run the fixed-original-rationale control with the original Qwen adapters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rescore_corrected_rationales as base


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/analysis/interface_switch_harmful_samples/corrected_harmful_samples.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("fixed_original_rescored_harmful_samples.jsonl")
DEFAULT_WORK_DIR = DEFAULT_INPUT.parent / "qwen_rescore_fixed_original"
PROMPT_VERSION = "fixed_original_rationale_v1"


def original_rationale(item: dict[str, Any]) -> str:
    rationale = str((item.get("cot_result") or {}).get("original_rationale", "")).strip()
    if not rationale:
        raise ValueError(f"{item.get('record_id')}: original rationale is empty")
    lowered = rationale.lower()
    if any(tag in lowered for tag in ("<reasoning", "</reasoning", "<score")):
        raise ValueError(f"{item.get('record_id')}: original rationale contains control tags")
    return rationale


def assistant_prefix(item: dict[str, Any]) -> str:
    template = str(item["qwen3_4b_scorer"].get("assistant_prefix_template", ""))
    if template != base.ASSISTANT_PREFIX_TEMPLATE:
        raise ValueError(
            f"{item.get('record_id')}: unexpected assistant prefix template {template!r}"
        )
    return template.format(corrected_rationale=original_rationale(item))


# Reuse the verified worker implementation with the control-arm intervention.
base.PROMPT_VERSION = PROMPT_VERSION
base.corrected_rationale = original_rationale
base.assistant_prefix = assistant_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-group", help=argparse.SUPPRESS)
    return parser.parse_args()


def prepare(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = base.read_jsonl(input_path)
    if not rows:
        raise ValueError(f"{input_path} is empty")
    plan = base.build_plan(rows)
    verification = base.validate_prompt_messages(rows)
    verification.update(
        {
            "rationale_source": "cot_result.original_rationale",
            "only_added_message": "assistant original rationale + <score> prefix",
        }
    )
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "arm": "fixed_original_rationale_control",
        "input": {
            "path": base.portable_path(input_path),
            "sha256": base.sha256(input_path),
            "records": len(rows),
        },
        "output": base.portable_path(output_path),
        "groups": len(plan["groups"]),
        "execution": "serial subprocess per adapter group",
        "sampling": "exact original resolved greedy configuration",
        "gold_in_model_prompt": False,
        "prompt_verification": verification,
    }
    manifest_path, plan_path = base.plan_paths(work_dir)
    if manifest_path.is_file() and base.read_json(manifest_path) != manifest:
        raise ValueError(
            f"configuration differs from existing {manifest_path}; use a new --work-dir"
        )
    if plan_path.is_file() and base.read_json(plan_path) != plan:
        raise ValueError(
            f"plan differs from existing {plan_path}; use a new --work-dir"
        )
    base.write_json(manifest_path, manifest)
    base.write_json(plan_path, plan)
    return rows, plan


def run_worker_process(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    group: dict[str, Any],
) -> None:
    group_dir = work_dir / "groups" / group["group_id"]
    group_dir.mkdir(parents=True, exist_ok=True)
    log_path = group_dir / "run.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--work-dir",
        str(work_dir),
        "--worker-group",
        group["group_id"],
    ]
    print(
        f"[{group['group_id']}] start samples={group['samples']} "
        f"adapter={group['signature']['adapter']}",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"worker {group['group_id']} failed with {process.returncode}:\n"
            + "\n".join(tail)
        )
    if not base.group_is_complete(work_dir, group):
        raise RuntimeError(f"worker {group['group_id']} exited without complete output")
    print(f"[{group['group_id']}] completed", flush=True)


def collect_results(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    output_path: Path,
    work_dir: Path,
) -> None:
    results: dict[str, dict[str, Any]] = {}
    for group in plan["groups"]:
        path = base.group_result_path(work_dir, group["group_id"])
        if not path.is_file():
            raise RuntimeError(f"missing group output: {path}")
        for result in base.read_jsonl(path):
            record_id = str(result["record_id"])
            if record_id in results:
                raise RuntimeError(f"duplicate worker result: {record_id}")
            results[record_id] = result
    expected_ids = {row["record_id"] for row in rows}
    if set(results) != expected_ids:
        raise RuntimeError("fixed-original worker result IDs do not match input")

    output_rows: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for item in rows:
        result = results[item["record_id"]]
        value = copy.deepcopy(item)
        workflow = value["third_layer_workflow"]
        prediction = result["prediction"]
        natural_prediction = int(item["cot_result"]["prediction"])
        gold_score = int(item["gold_score"])
        reproduced = prediction == natural_prediction if prediction is not None else None
        fixed_result = {
            "status": "completed",
            "model_name": result["model_name"],
            "adapter": result["adapter"],
            "resolved_config_path": result["resolved_config_path"],
            "prompt_version": result["prompt_version"],
            "input_prompt_field": result["input_prompt_field"],
            "input_rationale_field": "cot_result.original_rationale",
            "assistant_prefix": result["assistant_prefix"],
            "prediction": prediction,
            "completion": result["completion"],
            "raw_output": result["raw_output"],
            "format_valid": result["format_valid"],
            "output_token_ids": result["output_token_ids"],
            "finish_reason": result["finish_reason"],
            "stop_reason": result["stop_reason"],
            "original_rendered_prompt_sha256": result[
                "original_rendered_prompt_sha256"
            ],
            "model_input_sha256": result["model_input_sha256"],
            "sampling": result["sampling"],
            "natural_original_prediction": natural_prediction,
            "reproduces_natural_original_prediction": reproduced,
        }
        workflow["fixed_original_rescore"] = fixed_result
        workflow["causal_comparison"].update(
            {
                "fixed_original_rationale_prediction": prediction,
                "fixed_original_reproduces_natural_prediction": reproduced,
            }
        )
        output_rows.append(value)
        compact.append(
            {
                "record_id": item["record_id"],
                "task": item["task"],
                "training_method": item["training_method"],
                "seed": item["seed"],
                "natural_original_prediction": natural_prediction,
                "fixed_original_rationale_prediction": prediction,
                "gold_score": gold_score,
                "format_valid": result["format_valid"],
                "reproduces_natural_original_prediction": reproduced,
                "fixed_original_is_gold": (
                    prediction == gold_score if prediction is not None else None
                ),
            }
        )

    base.write_jsonl(output_path, output_rows)
    base.write_jsonl(work_dir / "predictions.jsonl", compact)
    valid = [row for row in compact if row["format_valid"]]
    summary = {
        "prompt_version": PROMPT_VERSION,
        "arm": "fixed_original_rationale_control",
        "samples": len(compact),
        "format_valid": len(valid),
        "format_invalid": len(compact) - len(valid),
        "reproduced_natural_original": sum(
            row["reproduces_natural_original_prediction"] is True for row in valid
        ),
        "did_not_reproduce_natural_original": sum(
            row["reproduces_natural_original_prediction"] is False for row in valid
        ),
        "fixed_original_is_gold": sum(
            row["fixed_original_is_gold"] is True for row in valid
        ),
        "by_task": dict(Counter(row["task"] for row in compact)),
        "output": base.portable_path(output_path),
    }
    base.write_json(work_dir / "summary.json", summary)
    print(
        f"finished: samples={summary['samples']} valid={summary['format_valid']} "
        f"reproduced={summary['reproduced_natural_original']} "
        f"output={output_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    input_path = base.resolve_path(args.input)
    output_path = base.resolve_path(args.output)
    work_dir = base.resolve_path(args.work_dir)
    if input_path == output_path:
        raise SystemExit("--output must differ from --input")
    rows, plan = prepare(input_path, output_path, work_dir)
    if args.prepare_only:
        print(
            f"validated {len(rows)} fixed-original records across "
            f"{len(plan['groups'])} adapter groups",
            flush=True,
        )
        return
    if args.worker_group:
        base.run_worker(input_path, work_dir, args.worker_group)
        return

    for index, group in enumerate(plan["groups"], 1):
        if base.group_is_complete(work_dir, group):
            print(
                f"[{index}/{len(plan['groups'])}] reuse completed {group['group_id']}",
                flush=True,
            )
            continue
        print(f"[{index}/{len(plan['groups'])}] dispatch {group['group_id']}", flush=True)
        run_worker_process(input_path, output_path, work_dir, group)
    collect_results(rows, plan, output_path, work_dir)


if __name__ == "__main__":
    main()
