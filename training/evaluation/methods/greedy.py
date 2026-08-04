"""Greedy generation evaluation method."""

from __future__ import annotations

from typing import Any

from evaluation.methods.base import EvaluationMethod, MethodRuntime


class GreedyMethod(EvaluationMethod):
    name = "greedy"
    decoding = "greedy"

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
        from evaluation.inference_loops import run_rollout

        return run_rollout(
            llm,
            runtime.sampling_params,
            rows,
            score_sets,
            args,
            rollout_index=rollout_index,
        )
