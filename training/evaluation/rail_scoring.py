"""Prompt construction and the TRACT-compatible RAIL score probe."""

from __future__ import annotations

import math
from typing import Any

from evaluation.defaults import (
    RAIL_DISCRETE_DECODING,
    RAIL_EXPECTATION_FORMULA,
    RAIL_IMPLEMENTATION,
    RAIL_PROBABILITY_NORMALIZATION,
)

RAIL_METADATA = {
    "probability_normalization": RAIL_PROBABILITY_NORMALIZATION,
    "candidate_renormalization": False,
    "rail_implementation": RAIL_IMPLEMENTATION,
    "rail_expectation_formula": RAIL_EXPECTATION_FORMULA,
    "discrete_decoding": RAIL_DISCRETE_DECODING,
}


def chat_template_supports_thinking(tokenizer: Any) -> bool:
    return "enable_thinking" in (getattr(tokenizer, "chat_template", None) or "")


def format_prompts(
    tokenizer: Any,
    prompts: list[list[dict[str, Any]]],
    enable_thinking: bool,
) -> list[str]:
    kwargs: dict[str, Any] = {}
    if chat_template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    return [
        tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, **kwargs
        )
        for prompt in prompts
    ]


def format_rail_prompts(
    tokenizer: Any, prompts: list[list[dict[str, Any]]]
) -> list[str]:
    kwargs: dict[str, Any] = {}
    if chat_template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(
            [*prompt, {"role": "assistant", "content": "<score>"}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            **kwargs,
        )
        for prompt in prompts
    ]


def resolve_score_token_ids(tokenizer: Any, score_sets: list[int]) -> list[int]:
    token_ids: list[int] = []
    for score in score_sets:
        encoded = tokenizer.encode(str(score), add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(
                f"RAIL requires score {score!r} to be exactly one tokenizer token; "
                f"got token IDs {encoded}."
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise SystemExit(
            f"RAIL score tokens must be unique: scores={score_sets}, token_ids={token_ids}"
        )
    return token_ids


def official_rail_statistics(score_logprobs: dict[int, float]) -> dict[str, Any]:
    """Compute TRACT's raw full-vocabulary expected score without renormalizing."""
    if not score_logprobs:
        raise ValueError("RAIL requires at least one score candidate.")
    if any(math.isnan(value) or value == math.inf for value in score_logprobs.values()):
        raise ValueError(f"RAIL received invalid log probabilities: {score_logprobs}")
    probabilities = {
        score: math.exp(logprob) for score, logprob in score_logprobs.items()
    }
    probability_mass = math.fsum(probabilities.values())
    if probability_mass > 1.0 + 1e-6:
        raise ValueError(
            "RAIL legal score probability mass exceeds one; the supplied values "
            f"are not full-vocabulary log probabilities: {probability_mass}"
        )
    return {
        "expected_score": math.fsum(
            score * probability for score, probability in probabilities.items()
        ),
        "score_probabilities": probabilities,
        "score_probability_mass": probability_mass,
    }


def nearest_legal_score(expected_score: float, score_sets: list[int]) -> int:
    return min(
        sorted(score_sets), key=lambda score: (abs(score - expected_score), score)
    )


def extract_requested_logprobs(
    request_output: Any, score_sets: list[int], score_token_ids: list[int]
) -> dict[int, float]:
    generated = request_output.outputs[0]
    if not generated.logprobs:
        raise RuntimeError("vLLM did not return token log probabilities for RAIL.")
    first_position = generated.logprobs[0]
    result: dict[int, float] = {}
    for score, token_id in zip(score_sets, score_token_ids, strict=True):
        entry = first_position.get(token_id)
        if entry is None:
            raise RuntimeError(
                f"vLLM omitted requested score token {score!r} (token ID {token_id})."
            )
        result[score] = float(getattr(entry, "logprob", entry))
    return result


def build_rail_sampling_params(score_token_ids: list[int], seed: int) -> Any:
    from evaluation.model_loading import import_vllm

    _, sampling_params = import_vllm()
    return sampling_params(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        logprobs=len(score_token_ids),
        logprob_token_ids=score_token_ids,
    )


def build_cot_rail_sampling_params(max_tokens: int, seed: int) -> Any:
    from evaluation.model_loading import import_vllm

    _, sampling_params = import_vllm()
    return sampling_params(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        stop=["<score>"],
        include_stop_str_in_output=True,
    )
