"""Score-only RAIL and two-stage CoT-RAIL evaluation methods."""

from __future__ import annotations

from typing import Any

from evaluation.inference.rail_scoring import (
    RAIL_DISCRETE_DECODING,
    RAIL_EXPECTATION_FORMULA,
    RAIL_IMPLEMENTATION,
    RAIL_PROBABILITY_NORMALIZATION,
)
from evaluation.inference.base import InferenceMethod, InferenceRuntime


def rail_metadata() -> dict[str, Any]:
    return {
        "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
        "candidate_renormalization": False,
        "rail_implementation": RAIL_IMPLEMENTATION,
        "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
        "discrete_decoding": RAIL_DISCRETE_DECODING,
    }


def prepare_score_probe(
    llm: Any,
    score_sets: list[int],
    seed: int,
    run_tag: str,
) -> tuple[list[int], Any]:
    from evaluation.inference.rail_scoring import build_rail_sampling_params, resolve_score_token_ids

    score_token_ids = resolve_score_token_ids(llm.get_tokenizer(), score_sets)
    sampling_params = build_rail_sampling_params(score_token_ids, seed)
    print(
        f"[{run_tag}] RAIL score tokens: "
        f"{dict(zip(score_sets, score_token_ids, strict=True))}",
        flush=True,
    )
    return score_token_ids, sampling_params


class RailMethod(InferenceMethod):
    name = "rail"
    decoding = "rail_full_vocab_raw_expected_score"
    supports_thinking = False
    aggregate_metrics = InferenceMethod.aggregate_metrics + (
        "rail_mae",
        "rail_mse",
        "rail_rmse",
        "avg_score_probability_mass",
    )

    def prepare(
        self,
        *,
        llm: Any,
        default_sampling_params: Any,
        score_sets: list[int],
        args: Any,
        run_tag: str,
    ) -> InferenceRuntime:
        score_token_ids, score_sampling_params = prepare_score_probe(
            llm, score_sets, args.seed, run_tag
        )
        return InferenceRuntime(
            sampling_params=score_sampling_params,
            score_token_ids=score_token_ids,
            score_sampling_params=score_sampling_params,
        )

    def run_rollout(
        self,
        *,
        llm: Any,
        runtime: InferenceRuntime,
        rows: list[dict[str, Any]],
        score_sets: list[int],
        args: Any,
        rollout_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from evaluation.inference.execution import run_rail_rollout

        assert runtime.score_token_ids is not None
        return run_rail_rollout(
            llm,
            runtime.sampling_params,
            rows,
            score_sets,
            runtime.score_token_ids,
            args,
            rollout_index=rollout_index,
        )

    def build_prediction_record(
        self,
        row: dict[str, Any],
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record = super().build_prediction_record(row, predictions)
        expected_scores = [
            prediction["expected_score"] for prediction in predictions
        ]
        valid_expected_scores = [
            score for score in expected_scores if score is not None
        ]
        record.update(
            {
                **rail_metadata(),
                "rollout_expected_scores": expected_scores,
                "expected_score": (
                    sum(valid_expected_scores) / len(valid_expected_scores)
                    if valid_expected_scores
                    else None
                ),
                "rollout_score_probability_masses": [
                    prediction["score_probability_mass"]
                    for prediction in predictions
                ],
                "rollout_score_probabilities": [
                    prediction["score_probabilities"] for prediction in predictions
                ],
                "rollout_score_logprobs": [
                    prediction["score_logprobs"] for prediction in predictions
                ],
            }
        )
        return record

    def resolved_config_metadata(self, args: Any) -> dict[str, Any]:
        return {
            "effective_max_tokens": 1,
            "reasoning_max_tokens": None,
            "score_probe_max_tokens": 1,
            **rail_metadata(),
        }

    def summary_metadata(
        self,
        args: Any,
        runtime: InferenceRuntime,
        score_sets: list[int],
    ) -> dict[str, Any]:
        assert runtime.score_token_ids is not None
        return {
            "decoding": self.decoding,
            "configured_max_tokens": args.max_tokens,
            "max_tokens": 1,
            "reasoning_max_tokens": None,
            "score_probe_max_tokens": 1,
            "cot_stop_strings": None,
            "score_token_ids": dict(
                zip(score_sets, runtime.score_token_ids, strict=True)
            ),
            **rail_metadata(),
        }


class CotRailMethod(RailMethod):
    name = "cot_rail"
    decoding = "cot_rail_full_vocab_raw_expected_score"
    aggregate_metrics = RailMethod.aggregate_metrics + (
        "score_prefix_valid_rate",
        "reasoning_valid_rate",
    )

    def prepare(
        self,
        *,
        llm: Any,
        default_sampling_params: Any,
        score_sets: list[int],
        args: Any,
        run_tag: str,
    ) -> InferenceRuntime:
        from evaluation.inference.rail_scoring import build_cot_rail_sampling_params

        score_token_ids, score_sampling_params = prepare_score_probe(
            llm, score_sets, args.seed, run_tag
        )
        return InferenceRuntime(
            sampling_params=default_sampling_params,
            score_token_ids=score_token_ids,
            cot_sampling_params=build_cot_rail_sampling_params(
                args.max_tokens, args.seed
            ),
            score_sampling_params=score_sampling_params,
        )

    def run_rollout(
        self,
        *,
        llm: Any,
        runtime: InferenceRuntime,
        rows: list[dict[str, Any]],
        score_sets: list[int],
        args: Any,
        rollout_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from evaluation.inference.execution import run_cot_rail_rollout

        assert runtime.score_token_ids is not None
        assert runtime.cot_sampling_params is not None
        assert runtime.score_sampling_params is not None
        return run_cot_rail_rollout(
            llm,
            runtime.cot_sampling_params,
            runtime.score_sampling_params,
            rows,
            score_sets,
            runtime.score_token_ids,
            args,
            rollout_index=rollout_index,
        )

    def build_prediction_record(
        self,
        row: dict[str, Any],
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record = super().build_prediction_record(row, predictions)
        record.update(
            {
                "rollout_cot_outputs": [
                    prediction["cot_output"] for prediction in predictions
                ],
                "rollout_cot_generated_token_ids": [
                    prediction["cot_generated_token_ids"]
                    for prediction in predictions
                ],
                "rollout_score_probe_texts": [
                    prediction["score_probe_text"] for prediction in predictions
                ],
                "rollout_score_probe_token_ids": [
                    prediction["score_probe_token_ids"]
                    for prediction in predictions
                ],
                "rollout_score_prefix_reached": [
                    prediction["score_prefix_reached"] for prediction in predictions
                ],
                "rollout_reasoning_valid": [
                    prediction["reasoning_valid"] for prediction in predictions
                ],
                "rollout_cot_finish_reasons": [
                    prediction["cot_finish_reason"] for prediction in predictions
                ],
                "rollout_cot_stop_reasons": [
                    prediction["cot_stop_reason"] for prediction in predictions
                ],
            }
        )
        return record

    def resolved_config_metadata(self, args: Any) -> dict[str, Any]:
        return {
            "effective_max_tokens": args.max_tokens,
            "reasoning_max_tokens": args.max_tokens,
            "score_probe_max_tokens": 1,
            **rail_metadata(),
        }

    def summary_metadata(
        self,
        args: Any,
        runtime: InferenceRuntime,
        score_sets: list[int],
    ) -> dict[str, Any]:
        metadata = super().summary_metadata(args, runtime, score_sets)
        metadata.update(
            {
                "decoding": self.decoding,
                "max_tokens": args.max_tokens,
                "reasoning_max_tokens": args.max_tokens,
                "cot_stop_strings": ["<score>"],
            }
        )
        return metadata
