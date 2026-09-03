"""Validation and token-region construction for Single Sample Align."""

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


def tokenize_ssa_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
    include_region_masks: bool = True,
) -> dict[str, list[int]]:
    """Tokenize one CoT row and split its completion at the <score> tag."""
    row_id = str(row["id"])
    completion = str(row["completion"][0]["content"])
    reasoning_matches = list(REASONING_RE.finditer(completion))
    score_matches = list(SCORE_RE.finditer(completion))
    if len(reasoning_matches) != 1 or len(score_matches) != 1:
        raise ValueError(
            f"SSA row {row_id!r} requires exactly one reasoning block and one "
            f"score block; found {len(reasoning_matches)} and {len(score_matches)}."
        )

    reasoning_match = reasoning_matches[0]
    score_match = score_matches[0]
    if not reasoning_match.group(1).strip():
        raise ValueError(f"SSA row {row_id!r} has an empty reasoning block.")
    if (
        reasoning_match.start() > score_match.start()
        or completion[: reasoning_match.start()].strip()
        or completion[reasoning_match.end() : score_match.start()].strip()
        or completion[score_match.end() :].strip()
    ):
        raise ValueError(
            f"SSA row {row_id!r} must contain only reasoning followed by score."
        )
    completion_score = int(score_match.group(1))
    if completion_score != int(row["label"]):
        raise ValueError(
            f"SSA row {row_id!r} completion score {completion_score} does not "
            f"match label {row['label']}."
        )

    prompt_ids = tokenizer(
        _prompt_text(tokenizer, row), add_special_tokens=False
    )["input_ids"]
    completion_encoding = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    completion_ids = list(completion_encoding["input_ids"])
    offsets = completion_encoding.get("offset_mapping")
    if offsets is None:
        raise ValueError("SSA requires a fast tokenizer with offset mappings.")

    score_boundary = score_match.start()
    rationale_mask: list[int] = []
    score_mask: list[int] = []
    for start, end in offsets:
        if end <= score_boundary:
            rationale_mask.append(1)
            score_mask.append(0)
        elif start >= score_boundary:
            rationale_mask.append(0)
            score_mask.append(1)
        else:
            raise ValueError(
                f"SSA row {row_id!r} has a token crossing the <score> boundary."
            )

    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
        rationale_mask.append(0)
        score_mask.append(1)

    prompt_ids = list(prompt_ids)
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    rationale_loss_mask = [0] * len(prompt_ids) + rationale_mask
    score_loss_mask = [0] * len(prompt_ids) + score_mask
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        rationale_loss_mask = rationale_loss_mask[overflow:]
        score_loss_mask = score_loss_mask[overflow:]

    if sum(rationale_loss_mask) == 0 or sum(score_loss_mask) == 0:
        raise ValueError(
            f"SSA row {row_id!r} lost a supervised region at max_length={max_length}."
        )
    if any(
        rationale + score != int(label != -100)
        for rationale, score, label in zip(
            rationale_loss_mask, score_loss_mask, labels, strict=True
        )
    ):
        raise RuntimeError(f"SSA row {row_id!r} has incomplete region masks.")

    record = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    if include_region_masks:
        record.update(
            rationale_loss_mask=rationale_loss_mask,
            score_loss_mask=score_loss_mask,
        )
    return record
