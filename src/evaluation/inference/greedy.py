"""Greedy generation evaluation method."""

from __future__ import annotations

from typing import Any

from evaluation.inference.base import InferenceMethod, InferenceRuntime


class GreedyMethod(InferenceMethod):
    name = "greedy"
    decoding = "greedy"

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
        from evaluation.inference.execution import run_rollout

        return run_rollout(
            llm,
            runtime.sampling_params,
            rows,
            score_sets,
            args,
            rollout_index=rollout_index,
        )
