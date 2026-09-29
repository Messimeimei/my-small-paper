#!/usr/bin/env python3
"""Run the final SO-vs-RS rationale audit with the three CSI judge models."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
import judge_protocol as base  # noqa: E402


DEFAULT_INPUT = PACKAGE_ROOT / "so_vs_rs_matched_350.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results"
DEFAULT_BASE_URL = "http://113.46.219.251:8080/v1"
DEFAULT_JUDGES = ("Qwen3.8-Max", "DeepSeek-V4-Pro")
DEFAULT_TIEBREAKER = "GLM-5.3"
DEFAULT_API_KEY_ENV = "CSI_API_KEY"
CONDITIONS = ("so", "rs")
LABELS = {"so": "SO", "rs": "RS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--judge-models", nargs=2, default=list(DEFAULT_JUDGES))
    parser.add_argument("--tiebreaker-model", default=DEFAULT_TIEBREAKER)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--model-options-file",
        type=Path,
        default=PACKAGE_ROOT / "model_request_options.json",
        help="Per-model thinking/reasoning request options.",
    )
    parser.add_argument("--disable-model-options", action="store_true")
    parser.add_argument(
        "--extra-request-json",
        default="{}",
        help=(
            "Additional top-level request fields as a JSON object, for example "
            "enable_thinking/thinking_budget after provider support is verified."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def deblind(
    item: dict[str, Any], round1: dict[str, Any], round2: dict[str, Any]
) -> dict[str, Any]:
    side_for = {
        condition: side for side, condition in item["blind_assignment"].items()
    }
    return {
        "so_dimensions": round1[side_for["so"]],
        "rs_dimensions": round1[side_for["rs"]],
        "so_support": round2[f"{side_for['so']}_support"],
        "rs_support": round2[f"{side_for['rs']}_support"],
    }


def run_judge(
    client: base.JudgeClient,
    item: dict[str, Any],
    model: str,
    *,
    quiet: bool,
) -> dict[str, Any]:
    if not quiet:
        print(f"    {model}: round 1 quality", flush=True)
    first = client.call(
        model,
        base.ROUND1_SYSTEM_PROMPT,
        base.round1_prompt(item),
        base.validate_round1,
        base.ROUND1_OUTPUT_SCHEMA,
        item["item_id"],
        "round1",
    )
    if not quiet:
        print(f"    {model}: round 2 consistency", flush=True)
    second = client.call(
        model,
        base.ROUND2_SYSTEM_PROMPT,
        base.round2_prompt(item),
        base.validate_round2,
        base.ROUND2_OUTPUT_SCHEMA,
        item["item_id"],
        "round2",
    )
    return {
        "model": model,
        "round1": first["parsed"],
        "round2": second["parsed"],
        "deblinded": deblind(item, first["parsed"], second["parsed"]),
        "raw_responses": first["raw_responses"] + second["raw_responses"],
    }


def dimension_mean(scores: dict[str, float | int]) -> float:
    return statistics.fmean(float(scores[key]) for key in base.DIMENSIONS)


def aggregate(judges: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for dimension in base.DIMENSIONS:
            dimensions[condition][dimension] = statistics.mean(
                judge["deblinded"][f"{condition}_dimensions"][dimension]
                for judge in judges
            )
    averages = {
        condition: dimension_mean(dimensions[condition])
        for condition in CONDITIONS
    }
    preference = (
        "so"
        if averages["so"] > averages["rs"]
        else "rs"
        if averages["rs"] > averages["so"]
        else "tie"
    )
    order = {"not_supported": 0, "partially_supported": 1, "supported": 2}
    reverse = {value: key for key, value in order.items()}
    support = {}
    for condition in CONDITIONS:
        values = [
            order[judge["deblinded"][f"{condition}_support"]]
            for judge in judges
        ]
        support[condition] = reverse[int(statistics.median(values))]
    return {
        "dimensions": dimensions,
        "dimension_averages": averages,
        "overall_preference": preference,
        "overall_preference_method": "compare_four_dimension_means",
        "support": support,
    }


def summary_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "completed_items": 0,
            "overall_preference": {},
            "dimension_means": {condition: {} for condition in CONDITIONS},
            "dimension_average_means": {condition: None for condition in CONDITIONS},
            "score_support": {condition: {} for condition in CONDITIONS},
        }
    dimensions = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for dimension in base.DIMENSIONS:
            dimensions[condition][dimension] = statistics.mean(
                row["aggregate"]["dimensions"][condition][dimension]
                for row in rows
            )
    return {
        "completed_items": len(rows),
        "overall_preference": dict(
            Counter(row["aggregate"]["overall_preference"] for row in rows)
        ),
        "dimension_means": dimensions,
        "dimension_average_means": {
            condition: dimension_mean(dimensions[condition])
            for condition in CONDITIONS
        },
        "score_support": {
            condition: dict(
                Counter(row["aggregate"]["support"][condition] for row in rows)
            )
            for condition in CONDITIONS
        },
    }


def analyze(items: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [results[item["item_id"]] for item in items if item["item_id"] in results]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_task[row["task"]].append(row)
    overall = summary_block(ordered)
    overall.update(
        {
            "total_selected": len(items),
            "failed_items": len(items) - len(ordered),
            "tiebreaker_items": sum(row["tiebreaker_used"] for row in ordered),
            "sample_strata": dict(Counter(row["outcome_stratum"] for row in ordered)),
        }
    )
    return {
        "overall": overall,
        "by_task": {
            task: summary_block(rows) for task, rows in sorted(by_task.items())
        },
    }


def render(analysis: dict[str, Any]) -> str:
    value = analysis["overall"]
    lines = [
        "# SO vs RS rationale quality with CSI judges",
        "",
        f"- selected: {value['total_selected']}",
        f"- completed: {value['completed_items']}",
        f"- failed: {value['failed_items']}",
        f"- tiebreaker used: {value['tiebreaker_items']}",
        "",
        "## Overall preference",
        "",
        "| Result | Count |",
        "| --- | ---: |",
        f"| SO better | {value['overall_preference'].get('so', 0)} |",
        f"| RS better | {value['overall_preference'].get('rs', 0)} |",
        f"| Tie | {value['overall_preference'].get('tie', 0)} |",
        "",
        "## Four-dimension means",
        "",
        "| Dimension | SO | RS | RS - SO |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dimension in base.DIMENSIONS:
        so_value = value["dimension_means"]["so"][dimension]
        rs_value = value["dimension_means"]["rs"][dimension]
        lines.append(
            f"| {base.DIMENSION_LABELS[dimension]} | {so_value:.3f} | "
            f"{rs_value:.3f} | {rs_value-so_value:+.3f} |"
        )
    lines += [
        "",
        "## Rationale-score consistency",
        "",
        "| Condition | Supported | Partially supported | Not supported |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        counts = value["score_support"][condition]
        lines.append(
            f"| {LABELS[condition]} | {counts.get('supported', 0)} | "
            f"{counts.get('partially_supported', 0)} | "
            f"{counts.get('not_supported', 0)} |"
        )
    lines += [
        "",
        "Gold labels, condition names, correctness, and training methods are hidden",
        "from both judge rounds. Gold is retained only for post-hoc strata.",
        "",
    ]
    return "\n".join(lines)


def validate(args: argparse.Namespace) -> None:
    if args.timeout <= 0 or args.max_retries <= 0 or args.judge_max_tokens <= 0:
        raise ValueError("timeout, retries, and judge-max-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    models = [*args.judge_models, args.tiebreaker_model]
    if any(not str(model).strip() for model in models) or len(set(models)) != 3:
        raise ValueError("configure three distinct judge models")


def request_options(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = json.loads(args.extra_request_json)
    if not isinstance(value, dict):
        raise ValueError("--extra-request-json must be a JSON object")
    protected = {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "response_format",
    }
    overlap = protected.intersection(value)
    if overlap:
        raise ValueError(
            "--extra-request-json cannot override: " + ", ".join(sorted(overlap))
        )
    model_options: dict[str, dict[str, Any]] = {}
    if not args.disable_model_options:
        raw = json.loads(args.model_options_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or any(
            not isinstance(options, dict) for options in raw.values()
        ):
            raise ValueError("--model-options-file must contain an object of objects")
        for model, options in raw.items():
            model_overlap = protected.intersection(options)
            if model_overlap:
                raise ValueError(
                    f"model options for {model} cannot override: "
                    + ", ".join(sorted(model_overlap))
                )
            model_options[str(model)] = dict(options)
    return value, model_options


def main() -> None:
    args = parse_args()
    validate(args)
    extra_options, model_options = request_options(args)
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    all_items = base.read_jsonl(input_path)
    items = all_items[: args.limit] if args.limit is not None else all_items
    if not items or len({item["item_id"] for item in items}) != len(items):
        raise ValueError("input is empty or contains duplicate item IDs")

    manifest = {
        "schema_version": 1,
        "prompt_version": base.PROMPT_VERSION,
        "input": input_path.name,
        "input_sha256": base.sha256(input_path),
        "samples": len(items),
        "limit": args.limit,
        "conditions": ["so", "rs"],
        "prediction_seed": 42,
        "sample_seed": 20260824,
        "sampling": "representative",
        "samples_per_task": 50,
        "judge_models": list(args.judge_models),
        "tiebreaker_model": args.tiebreaker_model,
        "judge_max_tokens": args.judge_max_tokens,
        "extra_request_fields": extra_options,
        "model_request_options": model_options,
        "base_url": args.base_url,
        "temperature": 0,
        "tiebreaker_rule": (
            "primary failure, different A/B dimension-derived preference, "
            "different support label for either side, or >=2-point gap on any dimension"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("output directory contains a different configuration")
    else:
        base.write_json(manifest_path, manifest)
    base.write_jsonl(output_dir / "sampled_items.jsonl", items)
    base.write_jsonl(
        output_dir / "judge_prompts.jsonl",
        [
            {
                "item_id": item["item_id"],
                "round1": base.round1_prompt(item),
                "round2": base.round2_prompt(item),
            }
            for item in items
        ],
    )
    if args.prepare_only or not args.run_api:
        print(f"prepared {len(items)} SO-vs-RS samples in {output_dir}")
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    client = base.JudgeClient(
        args.base_url,
        api_key,
        output_dir,
        args.timeout,
        args.max_retries,
        args.judge_max_tokens,
        extra_options,
        model_options,
    )
    results_path = output_dir / "judge_results.jsonl"
    failures_path = output_dir / "judge_failures.jsonl"
    results = {
        row["item_id"]: row for row in base.read_jsonl(results_path)
    } if results_path.is_file() else {}
    failures = {
        row["item_id"]: row for row in base.read_jsonl(failures_path)
    } if failures_path.is_file() else {}

    for index, item in enumerate(items, 1):
        if item["item_id"] in results:
            print(f"[{index}/{len(items)}] reuse {item['item_id']}")
            continue
        print(f"[{index}/{len(items)}] {item['task']} {item['source_id']}", flush=True)
        judges: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for model in args.judge_models:
            try:
                judges.append(run_judge(client, item, model, quiet=args.quiet))
            except Exception as exc:
                errors.append({"model": model, "error": str(exc)})
                print(f"    {model} failed: {exc}", flush=True)
        use_tiebreaker = len(judges) < 2 or base.judges_disagree(judges[0], judges[1])
        reason = None
        if use_tiebreaker:
            reason = "primary failure" if len(judges) < 2 else "primary disagreement"
            try:
                judges.append(run_judge(client, item, args.tiebreaker_model, quiet=args.quiet))
            except Exception as exc:
                errors.append({"model": args.tiebreaker_model, "error": str(exc)})
                print(f"    {args.tiebreaker_model} failed: {exc}", flush=True)
        if len(judges) < 2:
            failures[item["item_id"]] = {
                "item_id": item["item_id"],
                "source_id": item["source_id"],
                "judge_errors": errors,
            }
            base.write_jsonl(failures_path, [failures[key] for key in sorted(failures)])
            continue
        result = {
            "item_id": item["item_id"],
            "reference_item_id": item["reference_item_id"],
            "source_id": item["source_id"],
            "task": item["task"],
            "outcome_stratum": item["outcome_stratum"],
            "gold_label": item["gold_label"],
            "so_prediction": item["so"]["prediction"],
            "rs_prediction": item["rs"]["prediction"],
            "so_reasoning": item["so"]["reasoning"],
            "rs_reasoning": item["rs"]["reasoning"],
            "blind_assignment": item["blind_assignment"],
            "judges": judges,
            "tiebreaker_used": use_tiebreaker,
            "tiebreaker_reason": reason,
            "judge_errors": errors,
            "aggregate": aggregate(judges),
        }
        results[item["item_id"]] = result
        failures.pop(item["item_id"], None)
        base.write_jsonl(
            results_path,
            [results[current["item_id"]] for current in items if current["item_id"] in results],
        )
        base.write_jsonl(failures_path, [failures[key] for key in sorted(failures)])

    analysis = analyze(items, results)
    base.write_json(output_dir / "analysis.json", analysis)
    (output_dir / "analysis.md").write_text(render(analysis), encoding="utf-8")
    print(f"analysis written to {output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
