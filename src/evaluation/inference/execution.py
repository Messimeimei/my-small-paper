"""Greedy, score-only RAIL, and two-stage CoT-RAIL inference loops."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import torch

from utils.metrics import REASONING_RE, classification_metrics, extract_score, token_stats
from evaluation.inference.rail_scoring import (
    RAIL_DISCRETE_DECODING,
    RAIL_EXPECTATION_FORMULA,
    RAIL_IMPLEMENTATION,
    RAIL_PROBABILITY_NORMALIZATION,
)
from evaluation.inference.rail_scoring import (
    chat_template_supports_thinking,
    extract_requested_logprobs,
    format_prompts,
    format_rail_prompts,
    nearest_legal_score,
    official_rail_statistics,
)

def run_rollout(
    llm,
    sampling_params,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    args: argparse.Namespace,
    rollout_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = llm.get_tokenizer()
    supports_thinking = chat_template_supports_thinking(tokenizer)
    if args.enable_thinking and not supports_thinking:
        raise SystemExit(
            "This model chat template does not support enable_thinking; "
            "remove --enable_thinking."
        )

    predictions: list[dict[str, Any]] = []
    allowed_scores = set(score_sets)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
    else:
        gpu_started = started

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = format_prompts(
            tokenizer, [row["prompt"] for row in batch], args.enable_thinking
        )
        completions = llm.generate(texts, sampling_params, use_tqdm=False)
        outputs = [completion.outputs[0].text for completion in completions]
        for row, output in zip(batch, outputs, strict=True):
            prediction = extract_score(output, allowed_scores)
            predictions.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "output": output,
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )
        done = min(start + args.batch_size, len(rows))
        print(f"[rollout {rollout_index}] {done}/{len(rows)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    metrics["elapsed_sec"] = round(elapsed, 3)
    metrics["gpu_time_sec"] = round(gpu_elapsed, 3)
    metrics["samples_per_sec"] = round(len(rows) / max(elapsed, 1e-9), 3)
    metrics["tokens"] = token_stats(predictions, tokenizer)
    return predictions, metrics


def run_rail_rollout(
    llm,
    sampling_params,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    score_token_ids: list[int],
    args: argparse.Namespace,
    rollout_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply official full-vocabulary RAIL after the fixed <score> prefix."""
    tokenizer = llm.get_tokenizer()
    predictions: list[dict[str, Any]] = []
    generated_token_counts: list[int] = []
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
    else:
        gpu_started = started

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = format_rail_prompts(tokenizer, [row["prompt"] for row in batch])
        completions = llm.generate(texts, sampling_params, use_tqdm=False)
        for row, completion in zip(batch, completions, strict=True):
            generated = completion.outputs[0]
            score_logprobs = extract_requested_logprobs(
                completion, score_sets, score_token_ids
            )
            statistics = official_rail_statistics(score_logprobs)
            expected_score = float(statistics["expected_score"])
            prediction = nearest_legal_score(expected_score, score_sets)
            generated_token_counts.append(len(generated.token_ids))
            predictions.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "output": f"<score>{expected_score:.6f}</score>",
                    "raw_output": generated.text,
                    "generated_token_ids": list(generated.token_ids),
                    "expected_score": expected_score,
                    "score_probability_mass": float(
                        statistics["score_probability_mass"]
                    ),
                    "score_probabilities": {
                        str(score): float(probability)
                        for score, probability in statistics[
                            "score_probabilities"
                        ].items()
                    },
                    "score_logprobs": {
                        str(score): float(logprob)
                        for score, logprob in score_logprobs.items()
                    },
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )
        done = min(start + args.batch_size, len(rows))
        print(f"[rail rollout {rollout_index}] {done}/{len(rows)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    squared_errors = [
        (row["expected_score"] - row["label"]) ** 2 for row in predictions
    ]
    absolute_errors = [
        abs(row["expected_score"] - row["label"]) for row in predictions
    ]
    metrics.update(
        {
            "rail_mae": sum(absolute_errors) / len(absolute_errors),
            "rail_mse": sum(squared_errors) / len(squared_errors),
            "rail_rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
            "avg_score_probability_mass": sum(
                row["score_probability_mass"] for row in predictions
            )
            / len(predictions),
            "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
            "candidate_renormalization": False,
            "rail_implementation": RAIL_IMPLEMENTATION,
            "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
            "discrete_decoding": RAIL_DISCRETE_DECODING,
            "score_prefix": "<score>",
            "elapsed_sec": round(elapsed, 3),
            "gpu_time_sec": round(gpu_elapsed, 3),
            "samples_per_sec": round(len(rows) / max(elapsed, 1e-9), 3),
            "tokens": {
                "avg_output_tokens": sum(generated_token_counts)
                / len(generated_token_counts),
                "avg_reasoning_tokens": 0.0,
                "total_output_tokens": sum(generated_token_counts),
                "total_reasoning_tokens": 0,
                "samples": len(predictions),
            },
        }
    )
    return predictions, metrics


def run_cot_rail_rollout(
    llm,
    cot_sampling_params,
    score_sampling_params,
    rows: list[dict[str, Any]],
    score_sets: list[int],
    score_token_ids: list[int],
    args: argparse.Namespace,
    rollout_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate reasoning to <score>, then apply official full-vocabulary RAIL."""
    tokenizer = llm.get_tokenizer()
    predictions: list[dict[str, Any]] = []
    output_token_counts: list[int] = []
    reasoning_token_counts: list[int] = []
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
    else:
        gpu_started = started

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        base_texts = format_prompts(
            tokenizer,
            [row["prompt"] for row in batch],
            enable_thinking=False,
        )
        cot_completions = llm.generate(
            base_texts,
            cot_sampling_params,
            use_tqdm=False,
        )

        stage_rows: list[dict[str, Any]] = []
        score_prompts: list[str] = []
        score_indexes: list[int] = []
        for index, (row, base_text, completion) in enumerate(
            zip(batch, base_texts, cot_completions, strict=True)
        ):
            generated = completion.outputs[0]
            cot_output = generated.text
            stop_reason = getattr(generated, "stop_reason", None)
            prefix_reached = (
                stop_reason == "<score>" and cot_output.endswith("<score>")
            )
            reasoning_match = REASONING_RE.search(cot_output)
            reasoning_text = (
                reasoning_match.group(1).strip() if reasoning_match is not None else ""
            )
            reasoning_valid = bool(reasoning_text)
            stage_rows.append(
                {
                    "row": row,
                    "cot_output": cot_output,
                    "cot_generated_token_ids": list(generated.token_ids),
                    "cot_finish_reason": getattr(generated, "finish_reason", None),
                    "cot_stop_reason": stop_reason,
                    "score_prefix_reached": prefix_reached,
                    "reasoning_valid": reasoning_valid,
                    "reasoning_tokens": len(
                        tokenizer.encode(reasoning_text, add_special_tokens=False)
                    ),
                }
            )
            if prefix_reached:
                score_indexes.append(index)
                score_prompts.append(base_text + cot_output)

        score_completions = (
            llm.generate(score_prompts, score_sampling_params, use_tqdm=False)
            if score_prompts
            else []
        )
        score_by_index = dict(zip(score_indexes, score_completions, strict=True))

        for index, stage in enumerate(stage_rows):
            row = stage["row"]
            cot_ids = stage["cot_generated_token_ids"]
            score_completion = score_by_index.get(index)
            if score_completion is None:
                output_token_counts.append(len(cot_ids))
                reasoning_token_counts.append(stage["reasoning_tokens"])
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": None,
                        "correct": False,
                        "output": stage["cot_output"],
                        "raw_output": stage["cot_output"],
                        "cot_output": stage["cot_output"],
                        "cot_generated_token_ids": cot_ids,
                        "cot_finish_reason": stage["cot_finish_reason"],
                        "cot_stop_reason": stage["cot_stop_reason"],
                        "score_prefix_reached": False,
                        "reasoning_valid": stage["reasoning_valid"],
                        "score_probe_text": None,
                        "score_probe_token_ids": None,
                        "expected_score": None,
                        "score_probability_mass": None,
                        "score_probabilities": None,
                        "score_logprobs": None,
                        "task": row.get("task"),
                        "aspect": row.get("aspect"),
                    }
                )
                continue

            score_generated = score_completion.outputs[0]
            score_logprobs = extract_requested_logprobs(
                score_completion,
                score_sets,
                score_token_ids,
            )
            statistics = official_rail_statistics(score_logprobs)
            expected_score = float(statistics["expected_score"])
            prediction = nearest_legal_score(expected_score, score_sets)
            probe_ids = list(score_generated.token_ids)
            output_token_counts.append(len(cot_ids) + len(probe_ids))
            reasoning_token_counts.append(stage["reasoning_tokens"])
            predictions.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                    "output": (
                        stage["cot_output"] + f"{expected_score:.6f}</score>"
                    ),
                    "raw_output": stage["cot_output"] + score_generated.text,
                    "cot_output": stage["cot_output"],
                    "cot_generated_token_ids": cot_ids,
                    "cot_finish_reason": stage["cot_finish_reason"],
                    "cot_stop_reason": stage["cot_stop_reason"],
                    "score_prefix_reached": True,
                    "reasoning_valid": stage["reasoning_valid"],
                    "score_probe_text": score_generated.text,
                    "score_probe_token_ids": probe_ids,
                    "expected_score": expected_score,
                    "score_probability_mass": float(
                        statistics["score_probability_mass"]
                    ),
                    "score_probabilities": {
                        str(score): float(probability)
                        for score, probability in statistics[
                            "score_probabilities"
                        ].items()
                    },
                    "score_logprobs": {
                        str(score): float(logprob)
                        for score, logprob in score_logprobs.items()
                    },
                    "task": row.get("task"),
                    "aspect": row.get("aspect"),
                }
            )

        done = min(start + args.batch_size, len(rows))
        print(f"[cot-rail rollout {rollout_index}] {done}/{len(rows)}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(predictions, score_sets)
    valid = [row for row in predictions if row["expected_score"] is not None]
    squared_errors = [
        (row["expected_score"] - row["label"]) ** 2 for row in valid
    ]
    absolute_errors = [
        abs(row["expected_score"] - row["label"]) for row in valid
    ]
    sample_count = len(predictions)
    valid_count = len(valid)
    metrics.update(
        {
            "rail_mae": (
                sum(absolute_errors) / valid_count if valid_count else None
            ),
            "rail_mse": (
                sum(squared_errors) / valid_count if valid_count else None
            ),
            "rail_rmse": (
                math.sqrt(sum(squared_errors) / valid_count)
                if valid_count
                else None
            ),
            "avg_score_probability_mass": (
                sum(row["score_probability_mass"] for row in valid) / valid_count
                if valid_count
                else None
            ),
            "score_prefix_valid_rate": (
                valid_count / sample_count if sample_count else 0.0
            ),
            "reasoning_valid_rate": (
                sum(bool(row["reasoning_valid"]) for row in predictions) / sample_count
                if sample_count
                else 0.0
            ),
            "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
            "candidate_renormalization": False,
            "rail_implementation": RAIL_IMPLEMENTATION,
            "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
            "discrete_decoding": RAIL_DISCRETE_DECODING,
            "score_prefix": "<score>",
            "cot_stop_strings": ["<score>"],
            "elapsed_sec": round(elapsed, 3),
            "gpu_time_sec": round(gpu_elapsed, 3),
            "samples_per_sec": round(sample_count / max(elapsed, 1e-9), 3),
            "tokens": {
                "avg_output_tokens": (
                    sum(output_token_counts) / sample_count if sample_count else None
                ),
                "avg_reasoning_tokens": (
                    sum(reasoning_token_counts) / sample_count
                    if sample_count
                    else None
                ),
                "total_output_tokens": sum(output_token_counts),
                "total_reasoning_tokens": sum(reasoning_token_counts),
                "samples": sample_count,
            },
        }
    )
    return predictions, metrics
