#!/usr/bin/env python3
"""Judge RS versus RePaIR using the exact final SO-versus-RS protocol."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import judge_cot_vs_label_training_rationales as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/analysis/rs_vs_repair_rationale_quality_bundle"
    / "rs_vs_repair_matched_350.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/analysis/rs_vs_repair_rationale_quality_bundle/results"
)
CONDITIONS = ("rs", "repair")
CONDITION_LABELS = {"rs": "RS", "repair": "RePaIR"}
STRATUM_LABELS = {
    "both_correct": "Both correct",
    "rs_only_correct": "RS only correct",
    "repair_only_correct": "RePaIR only correct",
    "both_wrong": "Both wrong",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=base.DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=base.DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--judge-models", nargs=2, default=list(base.DEFAULT_JUDGE_MODELS)
    )
    parser.add_argument("--tiebreaker-model", default=base.DEFAULT_TIEBREAKER_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def deblind(
    item: dict[str, Any], round1: dict[str, Any], round2: dict[str, Any]
) -> dict[str, Any]:
    side_for = {
        condition: side for side, condition in item["blind_assignment"].items()
    }
    return {
        "rs_dimensions": round1[side_for["rs"]],
        "repair_dimensions": round1[side_for["repair"]],
        "rs_support": round2[f"{side_for['rs']}_support"],
        "repair_support": round2[f"{side_for['repair']}_support"],
    }


def run_judge(
    client: base.JudgeClient,
    item: dict[str, Any],
    model: str,
    *,
    quiet: bool,
) -> dict[str, Any]:
    if not quiet:
        print(f"    {model}: round 1 rationale quality", flush=True)
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
        print(f"    {model}: round 2 rationale-score consistency", flush=True)
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


def mean_dimension_score(scores: dict[str, float | int]) -> float:
    return statistics.fmean(float(scores[key]) for key in base.DIMENSIONS)


def condition_preference(dimensions: dict[str, dict[str, float]]) -> str:
    rs_mean = mean_dimension_score(dimensions["rs"])
    repair_mean = mean_dimension_score(dimensions["repair"])
    if rs_mean > repair_mean:
        return "rs"
    if repair_mean > rs_mean:
        return "repair"
    return "tie"


def aggregate(judges: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for dimension in base.DIMENSIONS:
            dimensions[condition][dimension] = statistics.mean(
                judge["deblinded"][f"{condition}_dimensions"][dimension]
                for judge in judges
            )
    support_order = {
        "not_supported": 0,
        "partially_supported": 1,
        "supported": 2,
    }
    reverse_support = {value: key for key, value in support_order.items()}
    support: dict[str, str] = {}
    for condition in CONDITIONS:
        values = [
            support_order[judge["deblinded"][f"{condition}_support"]]
            for judge in judges
        ]
        support[condition] = reverse_support[int(statistics.median(values))]
    dimension_averages = {
        condition: mean_dimension_score(dimensions[condition])
        for condition in CONDITIONS
    }
    return {
        "dimensions": dimensions,
        "dimension_averages": dimension_averages,
        "overall_preference": condition_preference(dimensions),
        "overall_preference_method": "compare_four_dimension_means",
        "support": support,
    }


def analysis_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            condition: mean_dimension_score(dimensions[condition])
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
    overall = analysis_block(ordered)
    overall.update(
        {
            "total_selected": len(items),
            "failed_items": len(items) - len(ordered),
            "tiebreaker_items": sum(row["tiebreaker_used"] for row in ordered),
            "sample_strata": dict(
                Counter(row["outcome_stratum"] for row in ordered)
            ),
        }
    )
    return {
        "overall": overall,
        "by_task": {
            task: analysis_block(rows) for task, rows in sorted(by_task.items())
        },
    }


def render(analysis: dict[str, Any]) -> str:
    overall = analysis["overall"]
    lines = [
        "# RS vs RePaIR Rationale Quality",
        "",
        f"- selected: {overall['total_selected']}",
        f"- completed: {overall['completed_items']}",
        f"- failed: {overall['failed_items']}",
        f"- tiebreaker used: {overall['tiebreaker_items']}",
        "",
        "## Overall preference",
        "",
        "| Result | Count |",
        "| --- | ---: |",
        f"| RS better | {overall['overall_preference'].get('rs', 0)} |",
        f"| RePaIR better | {overall['overall_preference'].get('repair', 0)} |",
        f"| Tie | {overall['overall_preference'].get('tie', 0)} |",
        "",
        "## Four-dimension means",
        "",
        "| Dimension | RS | RePaIR | RePaIR - RS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dimension in base.DIMENSIONS:
        rs_value = overall["dimension_means"]["rs"][dimension]
        repair_value = overall["dimension_means"]["repair"][dimension]
        lines.append(
            f"| {base.DIMENSION_LABELS[dimension]} | {rs_value:.3f} | "
            f"{repair_value:.3f} | {repair_value-rs_value:+.3f} |"
        )
    lines += [
        "",
        "## Rationale-score consistency",
        "",
        "| Condition | Supported | Partially supported | Not supported |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        counts = overall["score_support"][condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {counts.get('supported', 0)} | "
            f"{counts.get('partially_supported', 0)} | "
            f"{counts.get('not_supported', 0)} |"
        )
    lines += [
        "",
        "The two rounds use the unchanged final SO-vs-RS prompts. Gold labels,",
        "condition names, correctness, and training methods are hidden from judges.",
        "",
    ]
    return "\n".join(lines)


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0 or args.max_retries <= 0 or args.judge_max_tokens <= 0:
        raise ValueError("timeout, retries, and judge-max-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    models = [*args.judge_models, args.tiebreaker_model]
    if any(not str(model).strip() for model in models) or len(set(models)) != 3:
        raise ValueError("configure three distinct judge models")


def main() -> None:
    base.load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    validate_args(args)
    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)
    all_items = base.read_jsonl(input_path)
    items = all_items[: args.limit] if args.limit is not None else all_items
    if not items:
        raise ValueError("input contains no samples")
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("input contains duplicate item IDs")

    manifest = {
        "schema_version": 1,
        "prompt_version": base.PROMPT_VERSION,
        "prompt_source": "scripts/judge_cot_vs_label_training_rationales.py",
        "input": base.portable_path(input_path),
        "input_sha256": base.sha256(input_path),
        "samples": len(items),
        "limit": args.limit,
        "conditions": ["rs", "repair"],
        "prediction_seed": 42,
        "reference_sampling": {
            "source": "outputs/analysis/rationale_comparation_all_tasks_v2",
            "sample_seed": 20260824,
            "sampling": "representative",
            "samples_per_task": 50,
        },
        "judge_models": list(args.judge_models),
        "tiebreaker_model": args.tiebreaker_model,
        "judge_max_tokens": args.judge_max_tokens,
        "base_url": args.base_url,
        "tiebreaker_rule": (
            "primary failure, different A/B dimension-derived preference, "
            "different support label for either side, or >=2-point gap on any "
            "dimension"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
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
        print(f"prepared {len(items)} matched RS-vs-RePaIR pairs in {output_dir}")
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
    )
    results_path = output_dir / "judge_results.jsonl"
    failures_path = output_dir / "judge_failures.jsonl"
    existing_results = {
        row["item_id"]: row for row in base.read_jsonl(results_path)
    } if results_path.is_file() else {}
    failures = {
        row["item_id"]: row for row in base.read_jsonl(failures_path)
    } if failures_path.is_file() else {}

    print(f"sending {len(items)} matched blind pairs to {args.base_url}")
    for index, item in enumerate(items, 1):
        if item["item_id"] in existing_results:
            print(f"[{index}/{len(items)}] reuse {item['item_id']}")
            continue
        print(
            f"[{index}/{len(items)}] {item['task']} {item['source_id']} "
            f"({STRATUM_LABELS[item['outcome_stratum']]})",
            flush=True,
        )
        judges: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for model in args.judge_models:
            try:
                judges.append(run_judge(client, item, model, quiet=args.quiet))
            except Exception as exc:
                errors.append({"model": model, "error": str(exc)})
                print(f"    {model} failed: {exc}", flush=True)

        use_tiebreaker = len(judges) < 2 or base.judges_disagree(
            judges[0], judges[1]
        )
        reason = None
        if use_tiebreaker:
            reason = "primary failure" if len(judges) < 2 else "primary disagreement"
            try:
                judges.append(
                    run_judge(
                        client,
                        item,
                        args.tiebreaker_model,
                        quiet=args.quiet,
                    )
                )
            except Exception as exc:
                errors.append({"model": args.tiebreaker_model, "error": str(exc)})
                print(f"    {args.tiebreaker_model} failed: {exc}", flush=True)
        if len(judges) < 2:
            failures[item["item_id"]] = {
                "item_id": item["item_id"],
                "source_id": item["source_id"],
                "judge_errors": errors,
            }
            base.write_jsonl(
                failures_path, [failures[key] for key in sorted(failures)]
            )
            continue

        result = {
            "item_id": item["item_id"],
            "reference_item_id": item["reference_item_id"],
            "source_id": item["source_id"],
            "task": item["task"],
            "outcome_stratum": item["outcome_stratum"],
            "gold_label": item["gold_label"],
            "rs_prediction": item["rs"]["prediction"],
            "repair_prediction": item["repair"]["prediction"],
            "rs_reasoning": item["rs"]["reasoning"],
            "repair_reasoning": item["repair"]["reasoning"],
            "blind_assignment": item["blind_assignment"],
            "judges": judges,
            "tiebreaker_used": use_tiebreaker,
            "tiebreaker_reason": reason,
            "judge_errors": errors,
            "aggregate": aggregate(judges),
        }
        existing_results[item["item_id"]] = result
        failures.pop(item["item_id"], None)
        base.write_jsonl(
            results_path,
            [
                existing_results[current["item_id"]]
                for current in items
                if current["item_id"] in existing_results
            ],
        )
        base.write_jsonl(
            failures_path, [failures[key] for key in sorted(failures)]
        )

    analysis = analyze(items, existing_results)
    base.write_json(output_dir / "analysis.json", analysis)
    (output_dir / "analysis.md").write_text(render(analysis), encoding="utf-8")
    print(f"analysis written to {output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
