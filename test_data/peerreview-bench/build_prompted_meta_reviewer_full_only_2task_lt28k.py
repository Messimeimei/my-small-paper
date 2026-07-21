#!/usr/bin/env python3
"""
Build full-text prompted JSON for PeerReview-Bench Task 2 (two-task meta-reviewer):
the prompt asks the model to both
  1) judge the item on correctness/significance/evidence, and
  2) predict the collapsed 10-class expert-joint label.

Unlike the abs-only version, this script uses the full paper text from the
full `meta_reviewer` parquet and filters prompts by token length.

Output schema matches inference.py load_data():
  {"train": [], "test": [ {task, aspect, labels, score_sets, prompt, ...}, ... ]}

Default output:
  /data01/public/yangxin/small-paper/eval_data/prompted_meta_reviewer_full_only_2task_lt28k.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT.parent
DEFAULT_PARQUET = ROOT / "meta_reviewer" / "eval-00000-of-00001.parquet"
DEFAULT_OUT = EVAL_ROOT / "prompted_meta_reviewer_full_only_2task_lt28k.json"
DEFAULT_TOKENIZER = EVAL_ROOT.parent / "model" / "Qwen2.5-3B-Instruct"
DEFAULT_THRESHOLD = 28000

SCORE_SETS = list(range(1, 11))

MARGINAL_OR_ABOVE = {"Significant", "Marginally Significant"}

SYSTEM_PROMPT = (
    "You are an evaluator of expert-domain scientific writing. You will get a "
    "query-answer pair along with criteria explaining the specific evaluation "
    "aspect and the scoring rubric. You should evaluate whether the answer "
    "satisfy the query based on the given criteria. First output your reasoning "
    "enclosed between <reasoning> and </reasoning>. Then output the three axis "
    "labels using the tags <correctness>, <significance>, and <evidence>. Use "
    "exact label strings from the criteria. For cascade-skipped axes, write "
    "null inside the tag. Finally, output your 10-class prediction enclosed "
    "between <score> and </score>. Inside <score> provide only the numeric "
    "class id (1-10) and nothing else."
)

USER_PREFIX = """[QUERY]: You are a meta-reviewer for scientific papers. You will be given a paper's content AND one reconstructed review item written by a reviewer (human or AI). For this review item, you must: (1) Judge the item along three axes: correctness, significance, and evidence sufficiency (your own meta-review judgment). (2) Predict how two independent expert meta-reviewers would jointly judge the item, expressed as one of 10 collapsed class labels that encode both the cascade outcome and inter-expert agreement. You are NOT writing a new review. You are judging an existing review item by verifying its claims against the paper.


[CRITERIA]: ### Principles (ordered by importance)
1. Correctness -- judge the main point, not peripheral details. If the reviewer's core concern is valid even though one specific supporting claim is inaccurate, the item is still Correct. Only mark Not Correct when the main point itself is wrong.
2. Significance -- the bar is "would this improve the paper?" Any criticism that would genuinely help the paper if addressed is Significant -- it does NOT need to threaten the paper's validity. Missing statistics, undefined figure annotations, unreported methodological details, internal inconsistencies between text and figures, and missing ablations are typically Significant. Only pure stylistic preferences and trivial typos are Marginally Significant.
3. Evidence -- verifiability, not exhaustiveness. If a meta-reviewer can verify the reviewer's claim from what the reviewer wrote plus the paper, the evidence is Sufficient. When the reviewer's criticism IS that something is absent, identifying the specific absence IS the evidence. Reserve Requires More for cases where the meta-reviewer cannot even locate what part of the paper the reviewer is talking about.

### Decision procedure (follow for EACH item)
Part A -- Your initial judgment (axis labels)
Step 1 (Understand). Read the item. What is the main point?
Step 2 (Correctness). Verify the MAIN POINT against the paper.
- Core concern valid even if a peripheral detail is off? -> "Correct". Continue to Step 3.
- Main point itself factually wrong? -> "Not Correct". Set significance and evidence to null.
Step 3 (Significance). Would addressing this criticism genuinely improve the paper?
- Would fixing it make the paper substantively better? -> "Significant"
- Minor presentation or stylistic issue that does not affect substance? -> "Marginally Significant"
- Trivial enough that the review would be better without it? -> "Not Significant". Set evidence to null.
Step 4 (Evidence). Can a meta-reviewer verify the claim from the review item + the paper without substantial extra work?
- Yes -> "Sufficient"
- No -> "Requires More"

Part B -- Devil's advocate + expert disagreement prediction
Expert disagreement is common in meta-review -- it is not a rare edge case. For each axis, you must argue the other side before deciding whether experts would agree.
Step 5 (Correctness -- devil's advocate).
- If "Not Correct" in Step 2: argue the reviewer's defense. Could a charitable expert still find the core concern valid? If plausible -> predict "disagree_on_correctness" (and you may revise Step 2 to "Correct" if the defense convinces you).
- If "Correct" in Step 2: argue the prosecution. Could a skeptical expert find the claim factually wrong? If plausible -> predict "disagree_on_correctness".
- If neither side is plausible -> experts agree on correctness.
Step 6 (Significance -- devil's advocate). Only if Step 5 = agree on Correct.
- If "Significant": argue for downgrading. Could an expert see this as merely Marginally Significant?
- If "Marginally Significant": argue for upgrading. Could an expert see this as genuinely improving the paper?
- If counter-argument plausible -> predict "correct_disagree_on_significance".
IMPORTANT: Correctness disagreement means the factual truth of the claim is ambiguous. Significance disagreement means the fact is clear but its importance is debatable. Do not mislabel correctness uncertainty as significance uncertainty.
Step 7 (Evidence boundary). Only if Step 6 = agree on >= Marginal. Is evidence clearly sufficient or clearly lacking, or borderline?
- Firm -> experts agree. Pick the matching agreement class.
- Borderline -> "correct_significant_disagree_on_evidence" or "correct_marginal_disagree_on_evidence".

### Consistency constraint (critical)
Your prediction_of_expert_judgments MUST be consistent with your FINAL axis labels (after any Part B revisions):
- If final correctness = "Correct", the prediction must be a "correct_*" class OR "disagree_on_correctness". Never "incorrect".
- If final correctness = "Not Correct", the prediction must be "incorrect" OR "disagree_on_correctness". Never a "correct_*" class.
- Similar logic applies downstream: significance and evidence labels constrain which agreement/disagreement classes are valid.

### The 10 valid prediction_of_expert_judgments labels (map to numeric score)
1: correct_significant_sufficient
2: correct_significant_insufficient
3: correct_significant_disagree_on_evidence
4: correct_marginal_sufficient
5: correct_marginal_insufficient
6: correct_marginal_disagree_on_evidence
7: correct_not_significant
8: correct_disagree_on_significance
9: incorrect
10: disagree_on_correctness

Use exact axis label strings in your reasoning and tags: "Correct" / "Not Correct"; "Significant" / "Marginally Significant" / "Not Significant"; "Sufficient" / "Requires More". Use null for cascade-skipped fields. Evidence axis "Requires More" maps to the "*_insufficient" class names above.

### Tips
1. The paper text may contain OCR errors. Do not penalize reviewers for pointing out things that are actually OCR artifacts; infer content from context.
2. Do not try to read or invent every supplementary/code file -- focus on what is provided in [PAPER] and what the review item explicitly references.
3. Apply the same significance bar consistently. Do not be lenient on one item style and strict on another.
4. Your judgment must be independent of who wrote the review. Do not infer reviewer identity (human/AI) from writing style.
5. Judge the item solely from the paper content provided below. Do not rely on external published versions, referee reports, or outside webpages about this paper.

### Output requirement for this evaluation format
In <reasoning>, include: (1) axis judgment + why, (2) devil's advocate argument for correctness (and significance if applicable) and whether plausible, (3) the resulting expert-agreement prediction and chosen class. Aim for 4-6 sentences.
Then output exactly these tags:
<correctness>...</correctness>
<significance>...</significance>
<evidence>...</evidence>
Then output:
<score>...</score>
Inside <score> provide only the numeric class id (1-10) and nothing else.

Scoring rubric is as follows.
1: correct_significant_sufficient (class 1)
2: correct_significant_insufficient (class 2)
3: correct_significant_disagree_on_evidence (class 3)
4: correct_marginal_sufficient (class 4)
5: correct_marginal_insufficient (class 5)
6: correct_marginal_disagree_on_evidence (class 6)
7: correct_not_significant (class 7)
8: correct_disagree_on_significance (class 8)
9: incorrect (class 9)
10: disagree_on_correctness (class 10)


"""


def percentile(vals: list[int], p: float) -> float:
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def build_user_prompt(user_prefix: str, paper_title: str, paper_content: str, review_item: str) -> str:
    return (
        user_prefix
        + f"[PAPER TITLE]: {paper_title}\n\n"
        + f"[PAPER]:\n{paper_content}\n\n"
        + f"[ANSWER]: {review_item}"
    )


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


def row_to_example(row: pd.Series, system_prompt: str, user_prefix: str, tokenizer, threshold: int):
    prompt = [
        {"role": "system", "content": system_prompt},
         {
             "role": "user",
             "content": build_user_prompt(
                user_prefix,
                 str(row["paper_title"] or ""),
                 str(row["paper_content"] or ""),
                 str(row["review_item"] or ""),
             ),
         },
    ]
    tok_len = len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
    if tok_len >= threshold:
        return None, tok_len

    label_id = int(row["label_id"])
    return {
        "task": "meta_reviewer_eval",
        "aspect": "cascade_10way",
        "labels": label_id,
        "score_sets": SCORE_SETS,
        "prompt": prompt,
        "paper_id": int(row["paper_id"]),
        "paper_title": row["paper_title"],
        "reviewer_id": row["reviewer_id"],
        "reviewer_type": row["reviewer_type"],
        "review_item_number": int(row["review_item_number"]),
        "label": row["label"],
        "label_id": label_id,
        "correctness_primary": row.get("correctness_primary"),
        "correctness_secondary": row.get("correctness_secondary"),
        "significance_primary": row.get("significance_primary"),
        "significance_secondary": row.get("significance_secondary"),
        "evidence_primary": row.get("evidence_primary"),
        "evidence_secondary": row.get("evidence_secondary"),
        "eval_correctness": bool(both_agree_correctness(row)),
        "eval_significance": bool(both_agree_significance(row)),
        "eval_evidence": bool(both_agree_evidence(row)),
        "prompt_token_len": tok_len,
    }, tok_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tokenizer_model", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    system_prompt, user_prefix = SYSTEM_PROMPT, USER_PREFIX
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_model), trust_remote_code=True)

    df = pd.read_parquet(args.parquet)
    if args.limit > 0:
        df = df.head(args.limit)

    kept = []
    dropped = 0
    kept_lens: list[int] = []
    label_dist: Counter[int] = Counter()
    total_paper_items = Counter(int(x) for x in df["paper_id"].tolist())
    kept_paper_items = Counter()

    for _, row in df.iterrows():
        example, tok_len = row_to_example(row, system_prompt, user_prefix, tokenizer, args.threshold)
        if example is None:
            dropped += 1
            continue
        kept.append(example)
        kept_lens.append(tok_len)
        label_dist[int(example["label_id"])] += 1
        kept_paper_items[int(example["paper_id"])] += 1

    payload = {"train": [], "test": kept}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)

    excluded_all = sorted(pid for pid in total_paper_items if kept_paper_items[pid] == 0)
    mixed = sorted(
        pid for pid in total_paper_items if 0 < kept_paper_items[pid] < total_paper_items[pid]
    )

    paper_title_map = (
        df[["paper_id", "paper_title"]].drop_duplicates().set_index("paper_id")["paper_title"].to_dict()
    )

    stats = {
        "output_path": str(args.out),
        "parquet": str(args.parquet),
        "tokenizer_model": str(args.tokenizer_model),
        "threshold": args.threshold,
        "n_total_rows": int(len(df)),
        "n_kept": int(len(kept)),
        "n_dropped": int(dropped),
        "n_unique_papers_total": int(df["paper_id"].nunique()),
        "n_unique_papers_kept": int(sum(1 for pid in total_paper_items if kept_paper_items[pid] > 0)),
        "n_papers_all_items_kept": int(
            sum(1 for pid in total_paper_items if kept_paper_items[pid] == total_paper_items[pid])
        ),
        "n_papers_mixed": int(len(mixed)),
        "n_papers_all_items_dropped": int(len(excluded_all)),
        "kept_label_id_dist": dict(sorted(label_dist.items())),
        "kept_token_distribution": {
            "min": min(kept_lens),
            "p25": round(percentile(kept_lens, 0.25), 2),
            "median": round(percentile(kept_lens, 0.5), 2),
            "mean": round(sum(kept_lens) / len(kept_lens), 2),
            "p75": round(percentile(kept_lens, 0.75), 2),
            "p90": round(percentile(kept_lens, 0.9), 2),
            "p95": round(percentile(kept_lens, 0.95), 2),
            "max": max(kept_lens),
        },
        "excluded_papers_all_items_dropped": [
            {
                "paper_id": int(pid),
                "paper_title": paper_title_map[pid],
                "n_items_total": int(total_paper_items[pid]),
            }
            for pid in excluded_all
        ],
        "mixed_papers": [
            {
                "paper_id": int(pid),
                "paper_title": paper_title_map[pid],
                "n_items_total": int(total_paper_items[pid]),
                "n_kept": int(kept_paper_items[pid]),
                "n_dropped": int(total_paper_items[pid] - kept_paper_items[pid]),
            }
            for pid in mixed
        ],
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
