"""Lightweight validation for paired Direct/Reason supervision data."""

from __future__ import annotations

from typing import Any

from utils.metrics import REASONING_RE, SCORE_RE


def completion_content(row: dict[str, Any]) -> str:
    return str(row["completion"][0]["content"])


def validate_and_pair_rows(
    cot_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair CoT and label-only rows and enforce the two-view contract."""
    labels_by_id = {row["id"]: row for row in label_rows}
    cot_ids = {row["id"] for row in cot_rows}
    if len(cot_ids) != len(cot_rows):
        raise ValueError("paper_align CoT dataset IDs must be unique.")
    if len(labels_by_id) != len(label_rows) or cot_ids != set(labels_by_id):
        missing = sorted(cot_ids - set(labels_by_id))[:5]
        extra = sorted(set(labels_by_id) - cot_ids)[:5]
        raise ValueError(
            "paper_align datasets must have identical unique IDs; "
            f"missing_label_ids={missing}, extra_label_ids={extra}"
        )

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cot_row in cot_rows:
        label_row = labels_by_id[cot_row["id"]]
        if cot_row["label"] != label_row["label"]:
            raise ValueError(f"Label mismatch for paper_align id={cot_row['id']}")
        if cot_row["prompt"][1:] != label_row["prompt"][1:]:
            raise ValueError(f"User/context prompt mismatch for id={cot_row['id']}")
        if cot_row["prompt"][0] == label_row["prompt"][0]:
            raise ValueError(
                f"Direct and Reason system prompts must differ for id={cot_row['id']}"
            )

        cot_completion = completion_content(cot_row)
        label_completion = completion_content(label_row)
        reasoning_match = REASONING_RE.search(cot_completion)
        cot_score_match = SCORE_RE.search(cot_completion)
        label_score_match = SCORE_RE.search(label_completion)
        if (
            reasoning_match is None
            or cot_score_match is None
            or reasoning_match.start() > cot_score_match.start()
        ):
            raise ValueError(
                f"Reason view must contain reasoning before score for id={cot_row['id']}"
            )
        if REASONING_RE.search(label_completion) is not None or label_score_match is None:
            raise ValueError(
                f"Direct view must contain score only for id={cot_row['id']}"
            )
        expected_score = cot_row["label"]
        if (
            int(cot_score_match.group(1)) != expected_score
            or int(label_score_match.group(1)) != expected_score
        ):
            raise ValueError(
                f"Completion score does not match label for id={cot_row['id']}"
            )
        pairs.append((label_row, cot_row))
    return pairs
