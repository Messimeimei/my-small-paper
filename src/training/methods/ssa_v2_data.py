"""Tokenization and branch metadata for rationale-independent SSA v2 scoring."""

from __future__ import annotations

from typing import Any

from utils.metrics import REASONING_RE, SCORE_RE


def _prompt_text(tokenizer, row: dict[str, Any]) -> str:
    template_kwargs: dict[str, Any] = {}
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def _overlapping_positions(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < end and token_end > start
    ]


def tokenize_ssa_v2_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
    include_branch_masks: bool = True,
) -> dict[str, Any]:
    """Build one CoT sequence with an attention-isolated score branch."""
    row_id = str(row["id"])
    completion = str(row["completion"][0]["content"])
    reasoning_matches = list(REASONING_RE.finditer(completion))
    score_matches = list(SCORE_RE.finditer(completion))
    if len(reasoning_matches) != 1 or len(score_matches) != 1:
        raise ValueError(
            f"SSA v2 row {row_id!r} requires exactly one reasoning block and "
            f"one score block; found {len(reasoning_matches)} and "
            f"{len(score_matches)}."
        )

    reasoning_match = reasoning_matches[0]
    score_match = score_matches[0]
    if not reasoning_match.group(1).strip():
        raise ValueError(f"SSA v2 row {row_id!r} has an empty reasoning block.")
    if (
        reasoning_match.start() > score_match.start()
        or completion[: reasoning_match.start()].strip()
        or completion[reasoning_match.end() : score_match.start()].strip()
        or completion[score_match.end() :].strip()
    ):
        raise ValueError(
            f"SSA v2 row {row_id!r} must contain only reasoning followed by score."
        )
    completion_score = int(score_match.group(1))
    if completion_score != int(row["label"]):
        raise ValueError(
            f"SSA v2 row {row_id!r} completion score {completion_score} does "
            f"not match label {row['label']}."
        )

    prompt_ids = list(
        tokenizer(_prompt_text(tokenizer, row), add_special_tokens=False)[
            "input_ids"
        ]
    )
    completion_encoding = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    completion_ids = list(completion_encoding["input_ids"])
    raw_offsets = completion_encoding.get("offset_mapping")
    if raw_offsets is None:
        raise ValueError("SSA v2 requires a fast tokenizer with offset mappings.")
    offsets = [(int(start), int(end)) for start, end in raw_offsets]

    score_branch_positions = _overlapping_positions(
        offsets,
        score_match.start(),
        score_match.end(),
    )
    if not score_branch_positions:
        raise ValueError(f"SSA v2 row {row_id!r} has no tokenized score block.")
    score_branch_start_in_completion = min(score_branch_positions)
    branch_start_offset = offsets[score_branch_start_in_completion][0]
    if branch_start_offset != score_match.start():
        raise ValueError(
            f"SSA v2 row {row_id!r} has a token crossing the <score> boundary."
        )

    score_value_start, score_value_end = score_match.span(1)
    score_value_positions = _overlapping_positions(
        offsets,
        score_value_start,
        score_value_end,
    )
    if not score_value_positions:
        raise ValueError(f"SSA v2 row {row_id!r} has no numeric score token.")

    rationale_mask: list[int] = []
    score_mask: list[int] = []
    score_value_mask: list[int] = []
    for index, (start, end) in enumerate(offsets):
        if end <= score_value_start:
            rationale_mask.append(1)
            score_mask.append(0)
        elif start >= score_value_start:
            rationale_mask.append(0)
            score_mask.append(1)
        else:
            raise ValueError(
                f"SSA v2 row {row_id!r} has a token crossing the score-value "
                "boundary."
            )
        score_value_mask.append(int(index in score_value_positions))

    if tokenizer.eos_token_id is not None:
        completion_ids.append(int(tokenizer.eos_token_id))
        rationale_mask.append(0)
        score_mask.append(1)
        score_value_mask.append(0)

    original_prompt_length = len(prompt_ids)
    score_attention_start = (
        original_prompt_length + score_branch_start_in_completion
    )
    input_ids = prompt_ids + completion_ids
    labels = [-100] * original_prompt_length + completion_ids
    rationale_loss_mask = [0] * original_prompt_length + rationale_mask
    score_loss_mask = [0] * original_prompt_length + score_mask
    score_value_loss_mask = [0] * original_prompt_length + score_value_mask

    overflow = max(0, len(input_ids) - max_length)
    if overflow:
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        rationale_loss_mask = rationale_loss_mask[overflow:]
        score_loss_mask = score_loss_mask[overflow:]
        score_value_loss_mask = score_value_loss_mask[overflow:]
        score_attention_start -= overflow
    prompt_length = max(0, original_prompt_length - overflow)

    if prompt_length <= 0:
        raise ValueError(
            f"SSA v2 row {row_id!r} lost its complete prompt at "
            f"max_length={max_length}."
        )
    if score_attention_start <= prompt_length:
        raise ValueError(
            f"SSA v2 row {row_id!r} lost its rationale branch at "
            f"max_length={max_length}."
        )
    if (
        sum(rationale_loss_mask) == 0
        or sum(score_loss_mask) == 0
        or sum(score_value_loss_mask) == 0
    ):
        raise ValueError(
            f"SSA v2 row {row_id!r} lost a supervised region at "
            f"max_length={max_length}."
        )
    if any(
        rationale + score != int(label != -100)
        for rationale, score, label in zip(
            rationale_loss_mask, score_loss_mask, labels, strict=True
        )
    ):
        raise RuntimeError(f"SSA v2 row {row_id!r} has incomplete loss masks.")

    first_score_target = score_loss_mask.index(1)
    if first_score_target - 1 < score_attention_start:
        raise RuntimeError(
            f"SSA v2 row {row_id!r} numeric score predictor is not inside the "
            "isolated score branch."
        )

    position_ids = list(range(len(input_ids)))
    for index in range(score_attention_start, len(position_ids)):
        position_ids[index] = prompt_length + index - score_attention_start

    record: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    if include_branch_masks:
        record.update(
            position_ids=position_ids,
            rationale_loss_mask=rationale_loss_mask,
            score_loss_mask=score_loss_mask,
            score_value_loss_mask=score_value_loss_mask,
            prompt_length=prompt_length,
            score_attention_start=score_attention_start,
        )
    return record
