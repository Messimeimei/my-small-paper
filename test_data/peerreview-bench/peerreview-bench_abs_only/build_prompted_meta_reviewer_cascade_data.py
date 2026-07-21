#!/usr/bin/env python3
"""
Build zero-shot prompted JSON for PeerReview-Bench primary setting:
per-axis cascade labels (correctness -> significance -> evidence).

Aligned with arXiv:2605.20668 Appendix F primary setting + F.6 principles /
Part A decision procedure. Output schema matches inference.py load_data().

Usage:
  python build_prompted_meta_reviewer_cascade_data.py
  python build_prompted_meta_reviewer_cascade_data.py --limit 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = ROOT / "meta_reviewer" / "eval-00000-of-00001.parquet"
DEFAULT_OUT = ROOT / "prompted_meta_reviewer_cascade_abs_only.json"

MARGINAL_OR_ABOVE = {"Significant", "Marginally Significant"}

SYSTEM_PROMPT = (
    "You are an evaluator of expert-domain scientific writing. You will get a "
    "query-answer pair along with criteria explaining the specific evaluation "
    "aspect and the scoring rubric. You should evaluate whether the answer "
    "satisfy the query based on the given criteria. First output your reasoning "
    "enclosed between <reasoning> and </reasoning>. Then output the three cascade "
    "axis labels using the tags <correctness>, <significance>, and <evidence>. "
    "Use exact label strings from the criteria. For cascade-skipped axes, write "
    "null inside the tag."
)

USER_PREFIX = """[QUERY]: You are a meta-reviewer for scientific papers. You will be given a paper's content AND one reconstructed review item written by a reviewer (human or AI). For this review item, you must judge the item along three axes under the cascade protocol: correctness, then significance, then evidence sufficiency. This is your own meta-review judgment (primary setting). You are NOT writing a new review. You are judging an existing review item by verifying its claims against the paper. You are NOT asked to predict inter-expert disagreement classes.


[CRITERIA]: ### Principles (ordered by importance)
1. Correctness -- judge the main point, not peripheral details. If the reviewer's core concern is valid even though one specific supporting claim is inaccurate, the item is still Correct. Only mark Not Correct when the main point itself is wrong.
2. Significance -- the bar is "would this improve the paper?" Any criticism that would genuinely help the paper if addressed is Significant -- it does NOT need to threaten the paper's validity. Missing statistics, undefined figure annotations, unreported methodological details, internal inconsistencies between text and figures, and missing ablations are typically Significant. Only pure stylistic preferences and trivial typos are Marginally Significant.
3. Evidence -- verifiability, not exhaustiveness. If a meta-reviewer can verify the reviewer's claim from what the reviewer wrote plus the paper, the evidence is Sufficient. When the reviewer's criticism IS that something is absent, identifying the specific absence IS the evidence. Reserve Requires More for cases where the meta-reviewer cannot even locate what part of the paper the reviewer is talking about.

### Decision procedure (cascade; follow for EACH item)
Step 1 (Understand). Read the item. What is the main point?
Step 2 (Correctness). Verify the MAIN POINT against the paper.
- Core concern valid even if a peripheral detail is off? -> "Correct". Continue to Step 3.
- Main point itself factually wrong? -> "Not Correct". Set significance and evidence to null. Stop.
Step 3 (Significance). Would addressing this criticism genuinely improve the paper?
- Would fixing it make the paper substantively better? -> "Significant"
- Minor presentation or stylistic issue that does not affect substance? -> "Marginally Significant"
- Trivial enough that the review would be better without it? -> "Not Significant". Set evidence to null. Stop.
Step 4 (Evidence). Can a meta-reviewer verify the claim from the review item + the paper without substantial extra work?
- Yes -> "Sufficient"
- No -> "Requires More"

### Allowed label strings (exact)
- correctness: "Correct" | "Not Correct"
- significance: "Significant" | "Marginally Significant" | "Not Significant" | null
- evidence: "Sufficient" | "Requires More" | null

### Tips
1. The paper text may contain OCR errors. Do not penalize reviewers for pointing out things that are actually OCR artifacts; infer content from context.
2. Do not try to read or invent every supplementary/code file -- focus on what is provided in [PAPER] and what the review item explicitly references.
3. Apply the same significance bar consistently.
4. Your judgment must be independent of who wrote the review. Do not infer reviewer identity (human/AI) from writing style.
5. Judge the item solely from the paper content provided below.

### Output requirement
In <reasoning>, briefly state the main point and the cascade decisions (2-5 sentences).
Then output exactly these tags (one line each is fine):
<correctness>...</correctness>
<significance>...</significance>
<evidence>...</evidence>


"""


def _norm_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s if s and s.lower() != "nan" else None


def both_agree_correctness(row) -> bool:
    a, b = _norm_str(row["correctness_primary"]), _norm_str(row["correctness_secondary"])
    return a is not None and b is not None and a == b


def both_agree_significance(row) -> bool:
    if _norm_str(row["correctness_primary"]) != "Correct":
        return False
    if _norm_str(row["correctness_secondary"]) != "Correct":
        return False
    a, b = _norm_str(row["significance_primary"]), _norm_str(row["significance_secondary"])
    return a is not None and b is not None and a == b


def both_agree_evidence(row) -> bool:
    if not both_agree_significance(row):
        return False
    if _norm_str(row["significance_primary"]) not in MARGINAL_OR_ABOVE:
        return False
    a, b = _norm_str(row["evidence_primary"]), _norm_str(row["evidence_secondary"])
    return a is not None and b is not None and a == b


def is_fully_positive_primary(row) -> bool:
    return (
        _norm_str(row["correctness_primary"]) == "Correct"
        and _norm_str(row["significance_primary"]) == "Significant"
        and _norm_str(row["evidence_primary"]) == "Sufficient"
    )


def build_user_prompt(paper_title: str, paper_content: str, review_item: str) -> str:
    return (
        USER_PREFIX
        + f"[PAPER TITLE]: {paper_title}\n\n"
        + f"[PAPER]:\n{paper_content}\n\n"
        + f"[ANSWER]: {review_item}"
    )


def row_to_example(row: pd.Series) -> dict:
    # labels/score_sets: fully-positive binary vs primary (side metric + reward fallback)
    fp = int(is_fully_positive_primary(row))
    return {
        "task": "meta_reviewer_cascade_eval",
        "aspect": "three_axis",
        "labels": fp,
        "score_sets": [0, 1],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    str(row["paper_title"] or ""),
                    str(row["paper_content"] or ""),
                    str(row["review_item"] or ""),
                ),
            },
        ],
        "paper_id": int(row["paper_id"]),
        "paper_title": row["paper_title"],
        "reviewer_id": row["reviewer_id"],
        "reviewer_type": row["reviewer_type"],
        "review_item_number": int(row["review_item_number"]),
        "label_id": int(row["label_id"]) if pd.notna(row["label_id"]) else None,
        "label": row["label"] if pd.notna(row.get("label")) else None,
        "correctness_primary": _norm_str(row["correctness_primary"]),
        "correctness_secondary": _norm_str(row["correctness_secondary"]),
        "significance_primary": _norm_str(row["significance_primary"]),
        "significance_secondary": _norm_str(row["significance_secondary"]),
        "evidence_primary": _norm_str(row["evidence_primary"]),
        "evidence_secondary": _norm_str(row["evidence_secondary"]),
        # Paper Table 47 denominators: both annotators agree through that axis.
        "eval_correctness": bool(both_agree_correctness(row)),
        "eval_significance": bool(both_agree_significance(row)),
        "eval_evidence": bool(both_agree_evidence(row)),
        "cascade_mode": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.limit > 0:
        df = df.head(args.limit)

    test = [row_to_example(row) for _, row in df.iterrows()]
    payload = {"train": [], "test": test}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)

    n_c = sum(1 for x in test if x["eval_correctness"])
    n_s = sum(1 for x in test if x["eval_significance"])
    n_e = sum(1 for x in test if x["eval_evidence"])
    print(f"wrote {args.out}")
    print(f"test={len(test)} train=0")
    print(f"Table47-style eval subsets: correctness={n_c} significance={n_s} evidence={n_e}")
    if test:
        lens = [len(ex["prompt"][1]["content"]) for ex in test]
        print(
            f"user prompt chars: min={min(lens)} mean={sum(lens)/len(lens):.0f} max={max(lens)}"
        )


if __name__ == "__main__":
    main()
