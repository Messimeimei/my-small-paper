"""Base types shared by evaluation inference methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from evaluation.condition_labels import resolve_eval_condition


@dataclass
class MethodRuntime:
    sampling_params: Any
    score_token_ids: list[int] | None = None
    cot_sampling_params: Any = None
    score_sampling_params: Any = None


class EvaluationMethod(ABC):
    name: str
    decoding: str
    supports_thinking = True
    aggregate_metrics = (
        "accuracy",
        "macro_f1",
        "format_valid_rate",
        "mae",
        "qwk",
    )

    def validate(self, args: Any) -> None:
        if args.enable_thinking and not self.supports_thinking:
            raise SystemExit(
                f"{self.name} requires --disable_thinking so its score boundary is stable"
            )

    def prepare(
        self,
        *,
        llm: Any,
        default_sampling_params: Any,
        score_sets: list[int],
        args: Any,
        run_tag: str,
    ) -> MethodRuntime:
        return MethodRuntime(sampling_params=default_sampling_params)

    @abstractmethod
    def run_rollout(
        self,
        *,
        llm: Any,
        runtime: MethodRuntime,
        rows: list[dict[str, Any]],
        score_sets: list[int],
        args: Any,
        rollout_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...

    def build_prediction_record(
        self,
        row: dict[str, Any],
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scores = [prediction["prediction"] for prediction in predictions]
        outputs = [prediction["output"] for prediction in predictions]
        return {
            "id": row["id"],
            "label": row["label"],
            "rollout_predictions": scores,
            "rollout_correct": [score == row["label"] for score in scores],
            "mean_correct": sum(score == row["label"] for score in scores) / len(scores),
            "outputs": outputs,
            "raw_outputs": [
                prediction.get("raw_output", prediction["output"])
                for prediction in predictions
            ],
            "task": row.get("task"),
            "aspect": row.get("aspect"),
        }

    def resolved_config_metadata(self, args: Any) -> dict[str, Any]:
        return {
            "effective_max_tokens": args.max_tokens,
            "reasoning_max_tokens": None,
            "score_probe_max_tokens": None,
            "probability_normalization": None,
            "candidate_renormalization": None,
            "rail_implementation": None,
            "rail_expectation_formula": None,
            "discrete_decoding": None,
        }

    def summary_metadata(
        self,
        args: Any,
        runtime: MethodRuntime,
        score_sets: list[int],
    ) -> dict[str, Any]:
        return {
            "decoding": self.decoding,
            "configured_max_tokens": args.max_tokens,
            "max_tokens": args.max_tokens,
            "reasoning_max_tokens": None,
            "score_probe_max_tokens": None,
            "cot_stop_strings": None,
            "score_token_ids": None,
            "probability_normalization": None,
            "candidate_renormalization": None,
            "rail_implementation": None,
            "rail_expectation_formula": None,
            "discrete_decoding": None,
        }

    def resolve_condition(
        self,
        *,
        exp_name: str,
        supervision_mode: str,
        adapter: str | None,
        train_config: str | None,
        training_method: str | None,
    ) -> str | None:
        return resolve_eval_condition(
            exp_name=exp_name,
            supervision_mode=supervision_mode,
            adapter=adapter,
            train_config=train_config,
            training_method=training_method,
            inference_mode=self.name,
        )
