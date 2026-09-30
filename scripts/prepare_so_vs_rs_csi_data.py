#!/usr/bin/env python3
"""Combine the final seven-task SO-vs-RS audit sample files for CSI judging."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "outputs/analysis/rationale_comparation_all_tasks_v2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/analysis/so_vs_rs_csi_quality_bundle"
OUTPUT = OUTPUT_ROOT / "so_vs_rs_matched_350.jsonl"
TASKS = (
    "rev_util_actionability",
    "rev_util_grounding_specificity",
    "rev_util_helpfulness",
    "rev_util_verifiability",
    "rw_gen_coherence",
    "rw_gen_positioning_check",
    "rw_gen_positioning_type",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    inherited_config: dict[str, Any] | None = None
    stratum_map = {
        "both_correct": "both_correct",
        "lc_only_correct": "so_only_correct",
        "cc_only_correct": "rs_only_correct",
        "both_wrong": "both_wrong",
    }

    for task in TASKS:
        task_root = REFERENCE_ROOT / task
        samples_path = task_root / "sampled_items.jsonl"
        manifest_path = task_root / "manifest.json"
        samples = read_jsonl(samples_path)
        manifest = read_json(manifest_path)
        if len(samples) != 50:
            raise ValueError(f"{task}: expected 50 samples, found {len(samples)}")
        config = {
            "prompt_version": manifest["prompt_version"],
            "prediction_seed": manifest["prediction_seed"],
            "sample_seed": manifest["sample_seed"],
            "num": manifest["num"],
            "sampling": manifest["sampling"],
            "judge_max_tokens": manifest["judge_max_tokens"],
        }
        if inherited_config is None:
            inherited_config = config
        elif config != inherited_config:
            raise ValueError(f"{task}: reference evaluation config differs")

        for sample in samples:
            assignment = {
                side: "so" if condition == "lc" else "rs"
                for side, condition in sample["blind_assignment"].items()
            }
            values = {"so": sample["lc"], "rs": sample["cc"]}
            rows.append(
                {
                    "item_id": f"{task}__{sample['item_id']}",
                    "reference_item_id": sample["item_id"],
                    "source_id": sample["source_id"],
                    "task": task,
                    "query": sample["query"],
                    "criteria": sample["criteria"],
                    "evaluated_text": sample["evaluated_text"],
                    "gold_label": sample["gold_label"],
                    "outcome_stratum": stratum_map[sample["outcome_stratum"]],
                    "blind_assignment": assignment,
                    "A": {
                        "reasoning": values[assignment["A"]]["reasoning"],
                        "prediction": values[assignment["A"]]["prediction"],
                    },
                    "B": {
                        "reasoning": values[assignment["B"]]["reasoning"],
                        "prediction": values[assignment["B"]]["prediction"],
                    },
                    "so": sample["lc"],
                    "rs": sample["cc"],
                }
            )
        sources[task] = {
            "samples": str(samples_path.relative_to(PROJECT_ROOT)),
            "samples_sha256": sha256(samples_path),
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256(manifest_path),
        }

    if inherited_config is None or len(rows) != 350:
        raise ValueError(f"expected 350 samples, found {len(rows)}")
    write_jsonl(OUTPUT, rows)
    write_json(
        OUTPUT_ROOT / "data_manifest.json",
        {
            "schema_version": 1,
            "description": (
                "Exact final SO-vs-RS seed-42 sample IDs, contents, and A/B "
                "assignments, combined across seven tasks for CSI judging."
            ),
            "output": str(OUTPUT.relative_to(PROJECT_ROOT)),
            "output_sha256": sha256(OUTPUT),
            "samples": len(rows),
            "by_task": dict(Counter(row["task"] for row in rows)),
            "by_outcome_stratum": dict(
                Counter(row["outcome_stratum"] for row in rows)
            ),
            "inherited_prompt_and_sampling_config": inherited_config,
            "csi_judge_config": {
                "primary_judges": ["Qwen3.8-Max", "DeepSeek-V4-Pro"],
                "tiebreaker": "GLM-5.3",
                "base_url": "http://113.46.219.251:8080/v1",
                "temperature": 0,
                "judge_max_tokens": 8192,
                "model_request_options": {
                    "Qwen3.8-Max": {
                        "enable_thinking": True,
                        "thinking_budget": 8192,
                    },
                    "DeepSeek-V4-Pro": {
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": "high",
                    },
                    "GLM-5.3": {
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": "high",
                    },
                },
            },
            "sources": sources,
        },
    )
    print(f"prepared {len(rows)} exact SO-vs-RS samples: {OUTPUT}")


if __name__ == "__main__":
    main()
