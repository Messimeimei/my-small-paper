#!/usr/bin/env python3
"""Rebuild the five RevUtil cleaned datasets without changing sample content."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SEED = 20260721
DEFAULT_SAMPLE_SIZE = 4800

ASPECT_SPECS = {
    "actionability": {
        "origin": "rev_util__actionability__n10432.json",
        "synthetic_aspect": "actionability",
    },
    "grounding_specificity": {
        "origin": "rev_util__grounding_specificity__n10431.json",
        "synthetic_aspect": "grounding_specificity",
    },
    "helpfulness": {
        "origin": "rev_util__helpfulness__n10430.json",
        "synthetic_aspect": "helpfulness",
    },
    "verifiability": {
        "origin": "rev_util__verifiability__n8323.json",
        "synthetic_aspect": "verifiability",
    },
    "verifiability_extraction": {
        "origin": "rev_util__verifiability_extraction__n10430.json",
        "synthetic_aspect": "verifiability",
    },
}

# These characters are typical remnants of UTF-8 text decoded as Latin-1.
MOJIBAKE_LEAD_CHARS = frozenset("âÎÏÃÂðÐÑØÙ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate candidate pools and print statistics without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def extract_answer(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"Malformed prompt in row: {row.get('id', '<origin>')}")
    user_content = prompt[1].get("content", "")
    if user_content.count("[ANSWER]:") != 1:
        raise ValueError("Expected exactly one [ANSWER]: marker")
    return user_content.rsplit("[ANSWER]:", 1)[1].strip()


def normalize_for_matching(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def contains_mojibake(text: str) -> bool:
    if "\uFFFD" in text:
        return True
    if any(0x80 <= ord(char) <= 0x9F for char in text):
        return True
    return any(char in MOJIBAKE_LEAD_CHARS for char in text)


def allocate_proportional_quotas(
    counts: Counter[int], sample_size: int
) -> dict[int, int]:
    total = sum(counts.values())
    if total < sample_size:
        raise ValueError(f"Candidate pool has {total} rows, fewer than {sample_size}")

    exact = {label: count * sample_size / total for label, count in counts.items()}
    quotas = {label: int(value) for label, value in exact.items()}
    remainder = sample_size - sum(quotas.values())
    order = sorted(
        exact,
        key=lambda label: (-(exact[label] - quotas[label]), label),
    )
    for label in order[:remainder]:
        quotas[label] += 1

    for label, quota in quotas.items():
        if quota > counts[label]:
            raise ValueError(
                f"Label {label} needs {quota} rows but only {counts[label]} are eligible"
            )
    return dict(sorted(quotas.items()))


def stable_order_key(seed: int, aspect: str, purpose: str, answer: str) -> bytes:
    value = f"{seed}:{aspect}:{purpose}:{normalize_for_matching(answer)}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def parse_synthetic_label(value: Any, target_aspect: str) -> int | None:
    text = str(value).strip()
    if target_aspect == "verifiability_extraction":
        return 0 if text.upper() == "X" else 1
    if text.upper() == "X":
        return None
    return int(float(text))


def load_synthetic_labels(
    raw_root: Path, source_aspect: str, target_aspect: str
) -> dict[str, int | None]:
    path = raw_root / source_aspect / "train-00000-of-00001.parquet"
    score_column = f"chatgpt_{source_aspect}_score"
    frame = pd.read_parquet(path, columns=["review_point", score_column])
    labels: dict[str, int | None] = {}
    for review_point, raw_label in zip(frame["review_point"], frame[score_column]):
        key = normalize_for_matching(str(review_point).strip())
        label = parse_synthetic_label(raw_label, target_aspect)
        if key in labels:
            raise ValueError(f"Synthetic source contains duplicate review points: {path}")
        labels[key] = label
    return labels


def build_preview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_by_label: dict[int, dict[str, Any]] = {}
    for row in rows:
        first_by_label.setdefault(row["labels"], row)
    return [first_by_label[label] for label in sorted(first_by_label)]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    origin_root = project_root / "train_data" / "origin_data"
    output_root = project_root / "train_data" / "cleaned_data"
    preview_root = output_root / "preview"
    raw_root = (
        project_root
        / "test_data"
        / "Reward Modeling for Scientific Writing Evaluation"
        / "unprocessed_data"
        / "RevUtil_synthetic"
    )
    if not args.check_only:
        preview_root.mkdir(parents=True, exist_ok=True)

    official_test = load_json(project_root / "test_data" / "prompted_rev_util_data.json")
    blocked_answers = {
        normalize_for_matching(extract_answer(row)) for row in official_test["test"]
    }

    summary: dict[str, Any] = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "blocked_test_answers": len(blocked_answers),
        "aspects": {},
    }

    for aspect, spec in ASPECT_SPECS.items():
        origin_path = origin_root / spec["origin"]
        origin_rows = load_json(origin_path)["train"]
        synthetic_labels = load_synthetic_labels(
            raw_root, spec["synthetic_aspect"], aspect
        )

        eligible: list[dict[str, Any]] = []
        excluded = Counter()
        for row in origin_rows:
            answer = extract_answer(row)
            answer_key = normalize_for_matching(answer)
            if answer_key not in synthetic_labels:
                excluded["non_synthetic"] += 1
            elif synthetic_labels[answer_key] != row["labels"]:
                raise ValueError(
                    f"Origin/synthetic label mismatch for {aspect}: {row['labels']} != "
                    f"{synthetic_labels[answer_key]}"
                )
            elif answer_key in blocked_answers:
                excluded["test_content_overlap"] += 1
            elif contains_mojibake(answer):
                excluded["mojibake"] += 1
            else:
                eligible.append(row)

        label_counts = Counter(row["labels"] for row in eligible)
        quotas = allocate_proportional_quotas(label_counts, args.sample_size)
        rows_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            rows_by_label[row["labels"]].append(row)

        selected: list[dict[str, Any]] = []
        for label, quota in quotas.items():
            ordered = sorted(
                rows_by_label[label],
                key=lambda row: stable_order_key(
                    args.seed, aspect, f"label-{label}", extract_answer(row)
                ),
            )
            selected.extend(ordered[:quota])
        selected.sort(
            key=lambda row: stable_order_key(
                args.seed, aspect, "output", extract_answer(row)
            )
        )

        cleaned_rows: list[dict[str, Any]] = []
        for index, source_row in enumerate(selected, start=1):
            if "id" in source_row:
                raise ValueError(f"Origin row unexpectedly contains id: {origin_path}")
            cleaned_row = {"id": f"train_{index:04d}"}
            cleaned_row.update(source_row)
            cleaned_rows.append(cleaned_row)

        output_name = f"rev_util_{aspect}_{args.sample_size}.json"
        preview_name = f"rev_util_{aspect}_{args.sample_size}_preview.json"
        if not args.check_only:
            write_json(output_root / output_name, {"train": cleaned_rows})
            write_json(preview_root / preview_name, {"train": build_preview(cleaned_rows)})

        summary["aspects"][aspect] = {
            "origin_rows": len(origin_rows),
            "excluded": dict(sorted(excluded.items())),
            "eligible_rows": len(eligible),
            "eligible_labels": dict(sorted(label_counts.items())),
            "selected_labels": dict(
                sorted(Counter(row["labels"] for row in cleaned_rows).items())
            ),
            "output": output_name,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
