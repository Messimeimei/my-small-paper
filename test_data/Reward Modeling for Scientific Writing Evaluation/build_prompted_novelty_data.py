#!/usr/bin/env python3
"""
Build prompted_novelty_data.json for SciRM Novelty Alignment evaluation.

Pipeline (3 stages, printed step-by-step):
  1. prepare  — copy human-labeled pairs into data/
  2. label      — call multiple LLM APIs, majority-vote on summary.txt verdict
  3. build      — compare human vs LLM verdicts and write final JSON

Data pairing (per SciRM Section 4.2 + Afzal data_for_release):
  Assessment 1 (human): human_novelty_assessments/review_{id}.txt
  Assessment 2 (LLM):  ours/summary.txt
  Human label:         annotation.json output[i].class
  LLM label:           majority vote across configured models on summary.txt

Model config:
  label_models.json    — default list of models
  OPENBITFUN_MODELS    — comma-separated override in .env
  --models             — CLI override

Usage:
  export OPENBITFUN_API_KEY="sk-..."
  python build_prompted_novelty_data.py                    # run all stages
  python build_prompted_novelty_data.py --stage prepare    # only stage 1
  python build_prompted_novelty_data.py --stage label      # only stage 2 (multi-model)
  python build_prompted_novelty_data.py --stage build      # only stage 3
  python build_prompted_novelty_data.py --models glm-5,glm-5.1  # override models
  python build_prompted_novelty_data.py --limit 3          # debug: first 3 samples
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


VALID_VERDICTS = {"novel", "not_novel"}
DEFAULT_MODELS_CONFIG = Path(__file__).parent / "label_models.json"


# ---------------------------------------------------------------------------
# SciRM evaluation prompt templates (Appendix C)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an evaluator of expert-domain scientific writing. You will get a "
    "query-answer pair along with criteria explaining the specific evaluation "
    "aspect and the scoring rubric. You should evaluate whether the answer "
    "satisfy the query based on the given criteria. In addition, examples "
    "demonstrating how the evaluation should be performed will be provided. "
    "First output your reasoning enclosed between <reasoning> and </reasoning>. "
    "Then, output your score enclosed between <score> and </score>. Inside "
    "<score> provide only the numeric score and nothing else."
)

NOVELTY_QUERY = (
    "[QUERY]: Your task is to write two assessments regarding novelty of a "
    "scientific paper. Each one should lead to implicitly/explicitly a verdict "
    'of either "novel" or "not novel". The assessments should be aligned in '
    "terms if their novelty decision.\n\n\n"
)

NOVELTY_CRITERIA = (
    "[CRITERIA]: Assessments with verdict \"novel\" states that the paper "
    "introduces new, original, or significant ideas, methods, metrics, or "
    "frameworks; emphasizes a meaningful, notable, or important contribution; "
    "describes the work as advancing the field in a substantive or distinct "
    "way. Even if it is not groundbreaking fundamental new contribution, the "
    "paper can be still novel based on its significant contributions. On the "
    "other hand, assessments implying \"not novel\" state that the "
    "contribution is totally incremental, limited, weak, already known; "
    "emphasize that prior work already covers most of the ideas or findings; "
    "say the paper lacks significant, substantial, or original contributions; "
    "describe the work as mainly empirical, confirmatory, or replicating "
    "existing knowledge without new insights. You need to check whether these "
    "two assessments' final conclusion is the same or not. It is possible that "
    "two assessments can have similar observations on some points but they can "
    "arrive different conclusions. Your evaluation should be according to final "
    "conclusions. Evaluation two assessments should be binary. Scoring rubric "
    "is as follows.\n"
    "0: Two assessments do not come to the same conclusion in terms of novelty "
    "verdict.\n"
    "1: Two assessments come to the same conclusion in terms of novelty "
    "verdict.\n\n\n"
)

NOVELTY_EXAMPLES = """[EXAMPLES]:

<START OF EXAMPLE 1>

ANSWER: Assessment 1: This paper introduces a novel framework for analyzing the self-improvement capabilities of large language models, centering on the proposal of the generation-verification gap (GV-gap) as a new metric to quantify the limits of self-improvement. The GV-gap is clearly defined and illustrated with real-world examples, offering a fresh perspective for measuring and understanding where self-improvement may be fundamentally constrained. This approach represents a meaningful contribution to the research direction, as it provides the community with a new tool for future studies on model self-improvement. While the concept of self-improvement in language models has been explored in prior work, the introduction of the GV-gap offers a distinct and unified way to formalize and assess these capabilities. The novelty of the paper lies in this new quantification method, which can help clarify and advance discussions in the field. However, the practical application of the GV-gap is somewhat limited by the noisiness of real-world utility functions, as acknowledged by the authors, which may affect the robustness of the measurements. Despite this limitation, the proposed framework and metric are likely to be valuable for future research, marking the work as a notable and useful contribution to ongoing efforts in understanding and improving language model self-improvement. Assessment 2: This submission offers a comprehensive and systematic empirical/theoretical study of LLM self-improvement, with its main novelty being the formalization and central use of the "generation-verification gap" (GV-Gap) as a unifying metric. While the concept of a gap between generation and verification is present in prior work, the explicit metric and its application across models and tasks, as well as the discovery of a scaling law for GV-Gap, are new contributions. The paper also provides a detailed, cross-model analysis of verification mechanisms, including ensemble verification, which is a substantive but incremental extension of existing meta-judging and reward aggregation methods. However, the submission tends to overstate the lack of systematic analysis and diversity in prior work, and does not fully engage with risks such as bias and diversity collapse highlighted in the literature. Overall, the work represents a significant incremental advance in empirical rigor and formalization, rather than a fundamentally new paradigm for LLM self-improvement.

EVALUATION: <reasoning>Assessment 1 clearly concludes that the work is novel, emphasizing that the GV-gap constitutes a "meaningful contribution" and "new metric," marking the paper as a "notable and useful contribution." Assessment 2, while calling the work "a significant incremental advance," still attributes new contributions (formalizing GV-Gap, discovering a scaling law). It does not declare the work non-novel or merely confirmatory; it frames the contribution as incremental "but still novel". Thus both assessments align on a "novel" verdict, the score should be 1.</reasoning><score>1</score>

<END OF EXAMPLE 1>


<START OF EXAMPLE 2>

ANSWER: Assessment 1: This paper presents a comprehensive experimental analysis of self-improvement in Large Language Models (LLMs), focusing on the concept of the generation-verification gap and its relationship to model pre-training computational effort. While the study offers a modular framework and conducts extensive experiments to examine scaling phenomena and conditions for self-improvement, its novelty is limited. Most of the conclusions, such as the monotonic scaling of the verification gap with pre-training FLOPs and the identification of saturation limits, are already established in the literature. As a result, the paper does not provide fundamentally new insights or advances beyond what is already known. The contribution is primarily empirical, and while the analysis is thorough, it does not introduce novel theoretical perspectives or experimental findings that significantly advance the field. Consequently, the work's impact on the community is relatively weak from a novelty standpoint. Assessment 2: This submission offers a comprehensive and systematic empirical/theoretical study of LLM self-improvement, with its main novelty being the formalization and central use of the "generation-verification gap" (GV-Gap) as a unifying metric. While the concept of a gap between generation and verification is present in prior work, the explicit metric and its application across models and tasks, as well as the discovery of a scaling law for GV-Gap, are new contributions. The paper also provides a detailed, cross-model analysis of verification mechanisms, including ensemble verification, which is a substantive but incremental extension of existing meta-judging and reward aggregation methods. However, the submission tends to overstate the lack of systematic analysis and diversity in prior work, and does not fully engage with risks such as bias and diversity collapse highlighted in the literature. Overall, the work represents a significant incremental advance in empirical rigor and formalization, rather than a fundamentally new paradigm for LLM self-improvement.

EVALUATION: <reasoning>Assessment 1 concludes the paper is not novel, emphasizing that it offers only empirical, already-known findings, and no fundamentally new insights. Assessment 2 acknowledges the work as a **significant incremental advance** with *new contributions* (formalizing the metric, discovering a scaling law). Although incremental, it is still treated as novel. Since assessments do not align in terms of novelty of the paper, the evaluation score should 0.</reasoning><score>0</score>

<END OF EXAMPLE 2>


"""

VERDICT_INFERENCE_SYSTEM = (
    "You classify scientific novelty assessments. Read the assessment text and "
    "determine its FINAL implied verdict about whether the paper is novel."
)

VERDICT_INFERENCE_USER = """Read the following novelty assessment of a scientific paper and determine its FINAL verdict.

Verdict definitions:
- "novel": The assessment concludes the paper introduces new, original, or significant ideas/methods; makes a meaningful or notable contribution; or advances the field in a substantive way. A "significant incremental advance" with genuine new contributions still counts as novel.
- "not_novel": The assessment concludes the contribution is incremental/limited/weak/already known; prior work covers most ideas; the paper lacks substantial original contributions; or the work is mainly confirmatory without new insights.

Focus on the overall final conclusion, not isolated phrases.

Assessment:
\"\"\"
{assessment_text}
\"\"\"

Respond with ONLY valid JSON (no markdown fences):
{{"verdict": "novel" or "not_novel", "reasoning": "one sentence explaining the final conclusion"}}"""


@dataclass
class SampleRecord:
    sample_id: str
    forum_id: str
    review_id: int
    paper_title: str
    human_class: str
    human_assessment_path: str
    llm_summary_path: str


def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def progress_iter(items, desc: str):
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def class_to_verdict(class_label: str | None) -> str | None:
    if class_label == "novel":
        return "novel"
    if class_label == "not_novel":
        return "not_novel"
    return None


def build_answer(assessment_1: str, assessment_2: str) -> str:
    return f"Assessment 1: {assessment_1.strip()} Assessment 2: {assessment_2.strip()}"


def build_user_prompt(assessment_1: str, assessment_2: str) -> str:
    answer = build_answer(assessment_1, assessment_2)
    return NOVELTY_QUERY + NOVELTY_CRITERIA + NOVELTY_EXAMPLES + f"[ANSWER]: {answer}"


def build_chat_prompt(assessment_1: str, assessment_2: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(assessment_1, assessment_2)},
    ]


def deduplicate_records(
    records: list[SampleRecord], skipped: list[str]
) -> list[SampleRecord]:
    """Keep one record per (forum_id, review_id); log duplicates to skipped."""
    seen: dict[tuple[str, int], SampleRecord] = {}
    unique: list[SampleRecord] = []

    for rec in records:
        key = (rec.forum_id, rec.review_id)
        if key in seen:
            prev = seen[key]
            if prev.human_class != rec.human_class:
                skipped.append(
                    f"{rec.forum_id}/review_{rec.review_id}: duplicate with conflicting "
                    f"class ({prev.human_class} vs {rec.human_class}), kept first"
                )
            else:
                skipped.append(
                    f"{rec.forum_id}/review_{rec.review_id}: duplicate annotation entry, kept first"
                )
            continue
        seen[key] = rec
        unique.append(rec)

    return unique


def filter_single_review_papers(
    records: list[SampleRecord], skipped: list[str]
) -> list[SampleRecord]:
    """Keep only papers with exactly one valid human annotation."""
    counts = Counter(rec.forum_id for rec in records)
    multi_review_forums = {fid for fid, n in counts.items() if n > 1}

    filtered: list[SampleRecord] = []
    for rec in records:
        if rec.forum_id in multi_review_forums:
            skipped.append(
                f"{rec.forum_id}/review_{rec.review_id}: excluded (paper has "
                f"{counts[rec.forum_id]} reviews, single-review only)"
            )
        else:
            filtered.append(rec)

    if multi_review_forums:
        log(
            "prepare",
            f"  excluded {len(multi_review_forums)} multi-review papers: "
            f"{sorted(multi_review_forums)}",
        )

    return filtered


def deduplicate_manifest_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Safety dedup when reading manifest (by sample_id, then forum_id+review_id)."""
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for sample in samples:
        sid = sample["sample_id"]
        key = (sample["forum_id"], int(sample["review_id"]))
        if sid in by_id or key in by_key:
            continue
        by_id[sid] = sample
        by_key[key] = sample

    return list(by_id.values())


def discover_pairs(source_root: Path) -> tuple[list[SampleRecord], list[str]]:
    records: list[SampleRecord] = []
    skipped: list[str] = []

    for forum_dir in sorted(source_root.iterdir()):
        if not forum_dir.is_dir():
            continue
        forum_id = forum_dir.name
        ann_path = forum_dir / "annotation.json"
        if not ann_path.exists():
            continue

        summary_path = forum_dir / "ours" / "summary.txt"
        human_dir = forum_dir / "human_novelty_assessments"
        if not summary_path.exists():
            skipped.append(f"{forum_id}: missing ours/summary.txt")
            continue
        if not human_dir.is_dir():
            skipped.append(f"{forum_id}: missing human_novelty_assessments/")
            continue

        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)

        paper_title = ann.get("input", {}).get("title", "")

        for review in ann.get("output", []):
            review_id = review.get("review_id")
            human_class = review.get("class")
            if review_id is None:
                skipped.append(f"{forum_id}: review missing review_id")
                continue
            if class_to_verdict(human_class) is None:
                skipped.append(f"{forum_id}/review_{review_id}: invalid human class={human_class!r}")
                continue

            human_path = human_dir / f"review_{review_id}.txt"
            if not human_path.exists():
                skipped.append(f"{forum_id}: missing {human_path.name}")
                continue

            sample_id = f"{forum_id}__review_{review_id}"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    forum_id=forum_id,
                    review_id=int(review_id),
                    paper_title=paper_title,
                    human_class=human_class,
                    human_assessment_path=str(human_path.relative_to(source_root)),
                    llm_summary_path=str(summary_path.relative_to(source_root)),
                )
            )

    records = deduplicate_records(records, skipped)
    records = filter_single_review_papers(records, skipped)
    return records, skipped


def stage_prepare(source_root: Path, output_dir: Path, limit: int | None) -> list[SampleRecord]:
    log("prepare", f"Scanning source data: {source_root}")
    records, skipped = discover_pairs(source_root)

    if limit is not None:
        records = records[:limit]

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for rec in progress_iter(records, "Copying samples"):
        sample_dir = samples_dir / rec.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        human_src = source_root / rec.human_assessment_path
        llm_src = source_root / rec.llm_summary_path
        human_dst = sample_dir / "human_assessment.txt"
        llm_dst = sample_dir / "llm_summary.txt"

        shutil.copy2(human_src, human_dst)
        shutil.copy2(llm_src, llm_dst)

        meta = {
            **asdict(rec),
            "sample_dir": str(sample_dir.relative_to(output_dir)),
            "human_label": rec.human_class,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        manifest.append(meta)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(manifest), "samples": manifest}, f, indent=2, ensure_ascii=False)

    # Remove sample dirs that are no longer in the manifest
    valid_ids = {m["sample_id"] for m in manifest}
    samples_dir = output_dir / "samples"
    if samples_dir.exists():
        removed = 0
        for child in samples_dir.iterdir():
            if child.is_dir() and child.name not in valid_ids:
                shutil.rmtree(child)
                removed += 1
        if removed:
            log("prepare", f"  removed {removed} stale sample directories")

    skip_path = output_dir / "skipped.json"
    with open(skip_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    log("prepare", f"Collected {len(manifest)} human-labeled pairs -> {output_dir}")
    log("prepare", f"  manifest: {manifest_path}")
    log("prepare", f"  skipped : {len(skipped)} items (see {skip_path})")

    human_dist = {}
    for m in manifest:
        human_dist[m["human_label"]] = human_dist.get(m["human_label"], 0) + 1
    log("prepare", f"  human label distribution: {human_dist}")

    return records


def parse_verdict_response(text: str) -> tuple[str | None, str]:
    text = text.strip()
    # Try direct JSON parse
    try:
        obj = json.loads(text)
        verdict = obj.get("verdict", "").strip().lower()
        reasoning = obj.get("reasoning", "")
        if verdict in {"novel", "not_novel"}:
            return verdict, reasoning
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = obj.get("verdict", "").strip().lower()
            reasoning = obj.get("reasoning", "")
            if verdict in {"novel", "not_novel"}:
                return verdict, reasoning
        except json.JSONDecodeError:
            pass

    # Fallback keyword detection
    lower = text.lower()
    if "not_novel" in lower or "not novel" in lower:
        return "not_novel", text[:200]
    if re.search(r"\bnovel\b", lower):
        return "novel", text[:200]
    return None, text[:200]


def infer_llm_verdict(client: OpenAI, model: str, summary_text: str, retries: int = 3) -> dict[str, Any]:
    prompt = VERDICT_INFERENCE_USER.format(assessment_text=summary_text.strip())
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VERDICT_INFERENCE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            verdict, reasoning = parse_verdict_response(content)
            if verdict is None:
                raise ValueError(f"Could not parse verdict from: {content[:300]}")
            return {
                "verdict": verdict,
                "reasoning": reasoning,
                "raw_response": content,
                "model": model,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"LLM inference failed after {retries} retries: {last_error}")


def load_models_config(args) -> tuple[list[str], str]:
    """Load model list and base_url from config file, env, or CLI."""
    if load_dotenv is not None:
        load_dotenv()

    config_path = Path(args.models_config) if args.models_config else DEFAULT_MODELS_CONFIG
    models: list[str] = []
    base_url = args.base_url or os.environ.get("OPENBITFUN_BASE_URL", "https://api.openbitfun.com/v1")

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif os.environ.get("OPENBITFUN_MODELS"):
        models = [m.strip() for m in os.environ["OPENBITFUN_MODELS"].split(",") if m.strip()]
    elif config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        models = list(cfg.get("models", []))
        base_url = args.base_url or cfg.get("base_url") or base_url
    elif args.model:
        models = [args.model]
    elif os.environ.get("OPENBITFUN_MODEL"):
        models = [os.environ["OPENBITFUN_MODEL"]]

    if not models:
        models = ["deepseek-v4-pro"]

    return models, base_url


def normalize_label_entry(entry: dict[str, Any], models: list[str] | None = None) -> dict[str, Any]:
    """Migrate legacy single-model cache entries to multi-model format."""
    if "model_votes" in entry:
        return entry

    if "verdict" in entry and entry.get("verdict") in VALID_VERDICTS:
        model_name = entry.get("model", "unknown")
        return {
            "sample_id": entry.get("sample_id"),
            "human_label": entry.get("human_label"),
            "model_votes": {
                model_name: {
                    "verdict": entry["verdict"],
                    "reasoning": entry.get("reasoning", ""),
                    "raw_response": entry.get("raw_response", ""),
                    "model": model_name,
                    "timestamp": entry.get("timestamp"),
                }
            },
            "verdict": entry["verdict"],
            "vote_counts": {entry["verdict"]: 1},
            "status": "legacy_single_model",
            "models_configured": models or [model_name],
            "timestamp": entry.get("timestamp"),
        }

    return entry


def compute_majority_verdict(model_votes: dict[str, Any], models: list[str]) -> dict[str, Any]:
    """Compute final verdict from per-model votes (strict majority, e.g. 3/5)."""
    counts: Counter[str] = Counter()
    missing_models: list[str] = []

    for model in models:
        vote = model_votes.get(model, {})
        verdict = vote.get("verdict")
        if verdict in VALID_VERDICTS:
            counts[verdict] += 1
        else:
            missing_models.append(model)

    completed = len(models) - len(missing_models)
    threshold = len(models) // 2 + 1
    vote_counts = dict(counts)

    if missing_models:
        return {
            "verdict": None,
            "vote_counts": vote_counts,
            "status": "incomplete",
            "missing_models": missing_models,
            "completed_votes": completed,
            "required_for_majority": threshold,
        }

    if counts["novel"] >= threshold:
        return {
            "verdict": "novel",
            "vote_counts": vote_counts,
            "status": "majority",
            "completed_votes": completed,
            "required_for_majority": threshold,
        }
    if counts["not_novel"] >= threshold:
        return {
            "verdict": "not_novel",
            "vote_counts": vote_counts,
            "status": "majority",
            "completed_votes": completed,
            "required_for_majority": threshold,
        }

    return {
        "verdict": None,
        "vote_counts": vote_counts,
        "status": "tie",
        "completed_votes": completed,
        "required_for_majority": threshold,
    }


def migrate_labels_cache(cache: dict[str, Any], models: list[str]) -> dict[str, Any]:
    migrated = {}
    for sid, entry in cache.items():
        migrated[sid] = normalize_label_entry(entry, models)
    return migrated


def format_vote_summary(model_votes: dict[str, Any], models: list[str]) -> str:
    parts = []
    for model in models:
        vote = model_votes.get(model, {}).get("verdict", "?")
        parts.append(f"{model}={vote}")
    return ", ".join(parts)


def build_comparison_rows(
    full_manifest: list[dict[str, Any]],
    labels_cache: dict[str, Any],
    models: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in full_manifest:
        sid = sample["sample_id"]
        entry = labels_cache.get(sid)
        if not entry or entry.get("verdict") not in VALID_VERDICTS:
            continue

        human_label = sample["human_label"]
        llm_label = entry["verdict"]
        row: dict[str, Any] = {
            "sample_id": sid,
            "forum_id": sample["forum_id"],
            "review_id": sample["review_id"],
            "paper_title": sample["paper_title"],
            "human_label": human_label,
            "llm_label_majority": llm_label,
            "vote_novel": entry.get("vote_counts", {}).get("novel", 0),
            "vote_not_novel": entry.get("vote_counts", {}).get("not_novel", 0),
            "aligned": int(human_label == llm_label),
            "vote_status": entry.get("status"),
        }
        for model in models:
            row[f"model_{model}"] = entry.get("model_votes", {}).get(model, {}).get("verdict", "")
        rows.append(row)
    return rows


def stage_label(
    output_dir: Path,
    api_key: str,
    base_url: str,
    models: list[str],
    limit: int | None,
    sleep_s: float,
) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai package required. Run: pip install openai")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run --stage prepare first. Missing {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = deduplicate_manifest_samples(json.load(f)["samples"])

    if limit is not None:
        manifest = manifest[:limit]

    labels_path = output_dir / "llm_labels.json"
    per_model_dir = output_dir / "model_labels"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            labels_cache: dict[str, Any] = migrate_labels_cache(json.load(f), models)
    else:
        labels_cache = {}

    client = OpenAI(api_key=api_key, base_url=base_url)
    log("label", f"API base_url={base_url}")
    log("label", f"Models ({len(models)}): {', '.join(models)}")
    log("label", f"Majority rule: need {len(models) // 2 + 1}/{len(models)} agreeing votes")
    log("label", f"Samples to label: {len(manifest)}")

    for i, sample in enumerate(progress_iter(manifest, "Multi-model labeling"), start=1):
        sid = sample["sample_id"]
        human_label = sample["human_label"]

        entry = labels_cache.get(sid, {})
        entry = normalize_label_entry(entry, models)
        entry.setdefault("sample_id", sid)
        entry.setdefault("human_label", human_label)
        entry.setdefault("model_votes", {})
        entry["models_configured"] = models

        llm_summary = (output_dir / sample["sample_dir"] / "llm_summary.txt").read_text(encoding="utf-8")
        log("label", f"  [{i}/{len(manifest)}] {sid}")

        for model in models:
            existing = entry["model_votes"].get(model, {})
            if existing.get("verdict") in VALID_VERDICTS:
                log("label", f"    {model}: cached -> {existing['verdict']}")
                continue

            log("label", f"    {model}: inferring...")
            result = infer_llm_verdict(client, model, llm_summary)
            entry["model_votes"][model] = result
            log("label", f"    {model}: -> {result['verdict']} | {result['reasoning'][:100]}")

            # Per-model cache for resume/debug
            model_cache_path = per_model_dir / f"{model.replace('/', '_')}.json"
            model_cache: dict[str, Any] = {}
            if model_cache_path.exists():
                with open(model_cache_path, encoding="utf-8") as f:
                    model_cache = json.load(f)
            model_cache[sid] = result
            with open(model_cache_path, "w", encoding="utf-8") as f:
                json.dump(model_cache, f, indent=2, ensure_ascii=False)

            labels_cache[sid] = entry
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels_cache, f, indent=2, ensure_ascii=False)

            if sleep_s > 0:
                time.sleep(sleep_s)

        majority = compute_majority_verdict(entry["model_votes"], models)
        entry.update(
            {
                "verdict": majority["verdict"],
                "vote_counts": majority["vote_counts"],
                "status": majority["status"],
                "missing_models": majority.get("missing_models", []),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        labels_cache[sid] = entry
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_cache, f, indent=2, ensure_ascii=False)

        if entry["verdict"] in VALID_VERDICTS:
            aligned = "aligned" if entry["verdict"] == human_label else "NOT aligned"
            log(
                "label",
                f"    MAJORITY -> {entry['verdict']} ({majority['vote_counts']}) "
                f"vs human={human_label} [{aligned}]",
            )
        else:
            log(
                "label",
                f"    MAJORITY pending ({majority['status']}): "
                f"{format_vote_summary(entry['model_votes'], models)}",
            )

    with open(manifest_path, encoding="utf-8") as f:
        full_manifest = deduplicate_manifest_samples(json.load(f)["samples"])
    comparison_rows = build_comparison_rows(full_manifest, labels_cache, models)

    csv_path = output_dir / "comparison.csv"
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

    complete = sum(
        1 for s in full_manifest if labels_cache.get(s["sample_id"], {}).get("verdict") in VALID_VERDICTS
    )
    aligned_count = sum(r["aligned"] for r in comparison_rows)
    log("label", f"Comparison saved -> {csv_path}")
    log("label", f"  complete majority labels: {complete}/{len(full_manifest)}")
    log("label", f"  aligned={aligned_count}, not_aligned={len(comparison_rows) - aligned_count}")
    log("label", f"  per-model cache -> {per_model_dir}/")
    log("label", f"  merged labels   -> {labels_path}")

    return labels_cache


def stage_build(output_dir: Path, limit: int | None) -> Path:
    manifest_path = output_dir / "manifest.json"
    labels_path = output_dir / "llm_labels.json"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = deduplicate_manifest_samples(json.load(f)["samples"])
    with open(labels_path, encoding="utf-8") as f:
        labels_cache = migrate_labels_cache(json.load(f), [])

    if limit is not None:
        manifest = manifest[:limit]

    test_instances = []
    missing_labels = []

    for sample in manifest:
        sid = sample["sample_id"]
        entry = labels_cache.get(sid, {})
        if entry.get("verdict") not in VALID_VERDICTS:
            missing_labels.append(sid)
            continue

        sample_dir = output_dir / sample["sample_dir"]
        human_text = (sample_dir / "human_assessment.txt").read_text(encoding="utf-8")
        llm_text = (sample_dir / "llm_summary.txt").read_text(encoding="utf-8")

        human_label = sample["human_label"]
        llm_label = entry["verdict"]
        alignment_label = int(human_label == llm_label)

        instance = {
            "task": "novelty_eval",
            "aspect": "coherence",
            "labels": alignment_label,
            "score_sets": [0, 1],
            "prompt": build_chat_prompt(human_text, llm_text),
            # metadata (ignored by inference.py but useful for debugging)
            "forum_id": sample["forum_id"],
            "review_id": sample["review_id"],
            "paper_title": sample["paper_title"],
            "human_verdict": human_label,
            "llm_verdict": llm_label,
            "llm_verdict_method": "majority_vote",
            "llm_vote_counts": entry.get("vote_counts", {}),
            "llm_model_votes": {
                model: vote.get("verdict")
                for model, vote in entry.get("model_votes", {}).items()
            },
            "human_assessment_source": sample["human_assessment_path"],
            "llm_assessment_source": sample["llm_summary_path"],
        }
        test_instances.append(instance)

    if missing_labels:
        log("build", f"WARNING: {len(missing_labels)} samples missing LLM labels, skipped")

    dataset = {"train": [], "test": test_instances}
    out_path = output_dir / "prompted_novelty_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Also copy to project root for inference.py convenience
    root_copy = output_dir.parent / "prompted_novelty_data.json"
    shutil.copy2(out_path, root_copy)

    aligned = sum(x["labels"] == 1 for x in test_instances)
    not_aligned = sum(x["labels"] == 0 for x in test_instances)

    log("build", f"Built {len(test_instances)} test instances")
    log("build", f"  aligned (label=1): {aligned}")
    log("build", f"  not aligned (label=0): {not_aligned}")
    log("build", f"  output: {out_path}")
    log("build", f"  copy : {root_copy}")

    # Print a few examples
    log("build", "Sample instances:")
    for ex in test_instances[:3]:
        log(
            "build",
            f"  {ex['forum_id']}/review_{ex['review_id']}: "
            f"human={ex['human_verdict']}, llm={ex['llm_verdict']}, labels={ex['labels']}",
        )

    return out_path


def resolve_api_config(args) -> tuple[str, list[str], str]:
    if load_dotenv is not None:
        load_dotenv()

    api_key = args.api_key or os.environ.get("OPENBITFUN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    models, base_url = load_models_config(args)

    if not api_key and args.stage in {"label", "all"}:
        raise ValueError(
            "API key required. Set OPENBITFUN_API_KEY env var, use --api-key, or create a .env file."
        )
    return api_key or "", models, base_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompted_novelty_data.json with LLM labeling")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).parent / "data_for_release",
        help="Original Afzal data_for_release directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for prepared data and final JSON",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "label", "build", "all"],
        default="all",
        help="Which pipeline stage to run",
    )
    parser.add_argument("--api-key", default=None, help="OpenBitFun API key (prefer env var)")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument(
        "--models-config",
        default=None,
        help=f"JSON config for models (default: {DEFAULT_MODELS_CONFIG.name})",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model list, overrides config file and env",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single model shortcut (use --models for multiple)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (debug)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between API calls")
    args = parser.parse_args()

    api_key, models, base_url = resolve_api_config(args)

    print("=" * 72)
    print("SciRM Novelty Alignment Dataset Builder")
    print("=" * 72)
    log("main", f"source_root : {args.source_root}")
    log("main", f"output_dir  : {args.output_dir}")
    log("main", f"stage       : {args.stage}")
    if args.stage in {"label", "all"}:
        log("main", f"models      : {', '.join(models)}")
        log("main", f"base_url    : {base_url}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"prepare", "all"}:
        print("\n" + "-" * 72)
        print("STAGE 1/3: PREPARE — collect human-labeled data")
        print("-" * 72)
        stage_prepare(args.source_root, args.output_dir, args.limit)

    if args.stage in {"label", "all"}:
        print("\n" + "-" * 72)
        print("STAGE 2/3: LABEL — multi-model inference + majority vote")
        print("-" * 72)
        stage_label(args.output_dir, api_key, base_url, models, args.limit, args.sleep)

    if args.stage in {"build", "all"}:
        print("\n" + "-" * 72)
        print("STAGE 3/3: BUILD — compare labels and write prompted_novelty_data.json")
        print("-" * 72)
        stage_build(args.output_dir, args.limit)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Next: python inference.py --dataset_file {args.output_dir / 'prompted_novelty_data.json'} ...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved in data/llm_labels.json (if labeling started).")
        sys.exit(130)
