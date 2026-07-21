import re
import os
import time
import random
import argparse
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value


def model_display_name(model_name: str) -> str:
    """Basename of --model_name for logs / progress bars."""
    return os.path.basename(str(model_name).rstrip("/")) or str(model_name)


def completion_token_counts(completions, tokenizer=None, prompts=None):
    """
    Exact token counts from vLLM RequestOutput when available;
    otherwise fall back to the model tokenizer.
    """
    out_toks = []
    in_toks = []
    for i, c in enumerate(completions):
        out_ids = c.outputs[0].token_ids if c.outputs else None
        if out_ids is not None:
            out_toks.append(len(out_ids))
        elif tokenizer is not None:
            text = c.outputs[0].text if c.outputs else ""
            out_toks.append(len(tokenizer.encode(str(text), add_special_tokens=False)))
        else:
            out_toks.append(0)

        prompt_ids = getattr(c, "prompt_token_ids", None)
        if prompt_ids is not None:
            in_toks.append(len(prompt_ids))
        elif tokenizer is not None and prompts is not None:
            msg = prompts[i]
            if isinstance(msg, str):
                in_toks.append(len(tokenizer.encode(msg, add_special_tokens=False)))
            else:
                in_toks.append(
                    len(
                        tokenizer.apply_chat_template(
                            msg, tokenize=True, add_generation_prompt=True
                        )
                    )
                )
        else:
            in_toks.append(0)
    return out_toks, in_toks


def summarize_rollout_timing(
    n_samples: int,
    elapsed_sec: float,
    outputs,
    output_token_lens=None,
    prompt_token_lens=None,
    tokenizer=None,
) -> dict:
    elapsed_sec = float(max(elapsed_sec, 1e-9))
    out_chars = int(sum(len(str(t)) for t in outputs if t is not None))
    if output_token_lens is not None:
        out_toks = int(sum(int(x) for x in output_token_lens))
    elif tokenizer is not None:
        out_toks = int(
            sum(
                len(tokenizer.encode(str(t), add_special_tokens=False))
                for t in outputs
                if t is not None
            )
        )
    else:
        out_toks = 0
    if prompt_token_lens is not None:
        in_toks = int(sum(int(x) for x in prompt_token_lens))
    else:
        in_toks = 0
    return {
        "n_samples": int(n_samples),
        "elapsed_sec": round(elapsed_sec, 3),
        "samples_per_sec": round(n_samples / elapsed_sec, 3),
        "sec_per_sample": round(elapsed_sec / max(n_samples, 1), 4),
        "output_chars": out_chars,
        "output_tokens": out_toks,
        "prompt_tokens": in_toks,
        "output_toks_per_sec": round(out_toks / elapsed_sec, 2),
        "prompt_toks_per_sec": round(in_toks / elapsed_sec, 2),
        "total_toks_per_sec": round((in_toks + out_toks) / elapsed_sec, 2),
    }

def set_global_seed(seed: int):
    """Fix Python / NumPy / Torch RNGs for as-reproducible-as-possible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

SCORE_PATTERN = re.compile(r"<score>([^<]+)</score>")
CORRECTNESS_TAG = re.compile(r"<correctness>\s*([^<]*?)\s*</correctness>", re.IGNORECASE | re.DOTALL)
SIGNIFICANCE_TAG = re.compile(r"<significance>\s*([^<]*?)\s*</significance>", re.IGNORECASE | re.DOTALL)
EVIDENCE_TAG = re.compile(r"<evidence>\s*([^<]*?)\s*</evidence>", re.IGNORECASE | re.DOTALL)

CORRECTNESS_LABELS = ("Correct", "Not Correct")
SIGNIFICANCE_LABELS = ("Significant", "Marginally Significant", "Not Significant")
EVIDENCE_LABELS = ("Sufficient", "Requires More")
MARGINAL_OR_ABOVE = {"Significant", "Marginally Significant"}
CASCADE_TASK = "meta_reviewer_cascade_eval"


def chat_template_supports_thinking(tokenizer):
    """Return True if the model chat template exposes enable_thinking."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def validate_thinking_config(model_name, enable_thinking):
    """Validate thinking flag against model capabilities before loading vLLM."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    supports_thinking = chat_template_supports_thinking(tokenizer)
    if enable_thinking and not supports_thinking:
        raise ValueError(
            f"Model '{model_name}' does not support thinking mode. "
            "Its chat template does not define 'enable_thinking'. "
            "Remove --enable_thinking or use a thinking-capable model (e.g., Qwen3)."
        )
    return supports_thinking


def format_chat_prompts(tokenizer, prompts, enable_thinking):
    """Apply chat template, optionally controlling native thinking mode."""
    kwargs = {}
    if chat_template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    return [
        tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        for prompt in prompts
    ]


def load_data(dataset_path):

    with open(dataset_path) as fr:
        data = json.load(fr)

    dataset = DatasetDict({"train": Dataset.from_list(data["train"]),
                           "test": Dataset.from_list(data["test"])})
    return dataset


def map_features_with_predictions(dataset, task=None):
    """
    Keep the original dataset schema and pin prediction/output column dtypes so
    malformed early batches do not get inferred as Arrow `null` columns.
    """
    features = Features(dict(dataset.features))
    features["output"] = Value("string")
    features["reward"] = Value("float64")
    features["output_token_len"] = Value("int64")
    features["prompt_token_len"] = Value("int64")

    if task == "meta_reviewer_eval":
        features["pred_score"] = Value("float64")
        features["pred_correctness"] = Value("string")
        features["pred_significance"] = Value("string")
        features["pred_evidence"] = Value("string")
    elif task == CASCADE_TASK:
        features["pred_correctness"] = Value("string")
        features["pred_significance"] = Value("string")
        features["pred_evidence"] = Value("string")
        features["pred_label_id"] = Value("int64")
    else:
        features["pred_score"] = Value("float64")

    return features


def parse_score(text):
    """Extract numeric score from <score>...</score>, or None if missing/invalid."""
    match = re.search(SCORE_PATTERN, text or "")
    if not match:
        return None
    try:
        return float(match.group(1).strip())
    except (ValueError, TypeError):
        return None


def _canon_label(raw, allowed):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "null":
        return None
    for lab in allowed:
        if s.lower() == lab.lower():
            return lab
    return None


def parse_cascade_axes(text):
    """Parse <correctness>/<significance>/<evidence> tags (PeerReview primary setting)."""
    text = text or ""

    def _one(pattern, allowed):
        m = pattern.search(text)
        if not m:
            return None
        return _canon_label(m.group(1), allowed)

    return {
        "correctness": _one(CORRECTNESS_TAG, CORRECTNESS_LABELS),
        "significance": _one(SIGNIFICANCE_TAG, SIGNIFICANCE_LABELS),
        "evidence": _one(EVIDENCE_TAG, EVIDENCE_LABELS),
    }


# Table 46 classes that assume both annotators AGREE on the cascade outcome.
# Primary cascade outputs cannot emit disagreement classes 3 / 6 / 8 / 10.
AGREED_TEN_CLASS_IDS = (1, 2, 4, 5, 7, 9)
TEN_CLASS_SCORE_SET = list(range(1, 11))


def is_peerreview_ten_class_task(task=None, score_sets=None):
    """
    Detect PeerReview Bench Task-2 secondary setting:
    one numeric label in {1..10} encoding the collapsed expert-joint class.
    """
    if task == "meta_reviewer_eval":
        return True
    if not score_sets:
        return False
    target = set(float(x) for x in TEN_CLASS_SCORE_SET)
    try:
        return all(set(float(x) for x in ss) == target for ss in score_sets)
    except Exception:
        return False


def axes_to_agreed_label_id(correctness, significance=None, evidence=None):
    """
    Map primary-setting cascade axes -> Table 46 label_id under the assumption
    that both experts would agree with these labels.

    Returns one of {1,2,4,5,7,9}, or None if axes are incomplete/invalid.
    Cannot represent disagreement classes 3/6/8/10 (needs secondary setting).
    """
    if isinstance(correctness, dict):
        significance = correctness.get("significance")
        evidence = correctness.get("evidence")
        correctness = correctness.get("correctness")
    if correctness is None:
        return None
    if correctness == "Not Correct":
        return 9
    if correctness != "Correct":
        return None
    if significance is None:
        return None
    if significance == "Not Significant":
        return 7
    if significance == "Significant":
        if evidence == "Sufficient":
            return 1
        if evidence == "Requires More":
            return 2
        return None
    if significance == "Marginally Significant":
        if evidence == "Sufficient":
            return 4
        if evidence == "Requires More":
            return 5
        return None
    return None


def get_cascade_reward(axes, gold_c, gold_s, gold_e):
    """
    Reward for cascade outputs:
      -0.5 missing/invalid correctness tag
       0.25 correctness ok but required downstream tags missing
       0.5  format ok on required axes but not all match gold
       1.5  all gold-required axes match
    """
    if axes.get("correctness") is None:
        return -0.5
    required_ok = True
    matches = [axes["correctness"] == gold_c]
    if gold_c == "Correct" and gold_s is not None:
        if axes.get("significance") is None:
            required_ok = False
        else:
            matches.append(axes["significance"] == gold_s)
            if gold_s in MARGINAL_OR_ABOVE and gold_e is not None:
                if axes.get("evidence") is None:
                    required_ok = False
                else:
                    matches.append(axes["evidence"] == gold_e)
    if not required_ok:
        return 0.25
    return float(all(matches)) + 0.5


def _as_bool_list(values):
    out = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
        elif v is None:
            out.append(False)
        else:
            out.append(bool(v))
    return out


def is_cascade_example_batch(batch):
    tasks = batch.get("task") or []
    if any(t == CASCADE_TASK for t in tasks):
        return True
    # When task labels are present, avoid treating secondary-setting combined
    # prompts as cascade-only just because they also carry eval_* columns.
    if tasks:
        return False
    if "cascade_mode" in batch and any(bool(x) for x in batch["cascade_mode"]):
        return True
    return "eval_correctness" in batch and "correctness_primary" in batch


def is_combined_peerreview_batch(batch):
    tasks = batch.get("task") or []
    if not any(t == "meta_reviewer_eval" for t in tasks):
        return False
    needed = {"correctness_primary", "significance_primary", "evidence_primary", "eval_correctness", "labels", "score_sets"}
    return all(k in batch for k in needed)


def cascade_axis_accuracies(
    pred_correctness,
    pred_significance,
    pred_evidence,
    gold_correctness,
    gold_significance,
    gold_evidence,
    eval_correctness,
    eval_significance,
    eval_evidence,
):
    """
    PeerReview Bench Appendix F Table 47 style percent agreement.
    Acc on subsets where both annotators agree through that axis (eval_* flags).
    Compared against primary annotator gold labels.
    """
    eval_correctness = _as_bool_list(eval_correctness)
    eval_significance = _as_bool_list(eval_significance)
    eval_evidence = _as_bool_list(eval_evidence)

    def _acc(preds, golds, masks):
        num = den = 0
        for p, g, m in zip(preds, golds, masks):
            if not m:
                continue
            den += 1
            if p is not None and p == g:
                num += 1
        return (float(num / den) if den else 0.0), int(num), int(den)

    corr_acc, corr_hit, corr_n = _acc(pred_correctness, gold_correctness, eval_correctness)
    sig_acc, sig_hit, sig_n = _acc(pred_significance, gold_significance, eval_significance)
    evi_acc, evi_hit, evi_n = _acc(pred_evidence, gold_evidence, eval_evidence)
    return {
        "correctness_accuracy": corr_acc,
        "significance_accuracy": sig_acc,
        "evidence_accuracy": evi_acc,
        "correctness_n": corr_n,
        "significance_n": sig_n,
        "evidence_n": evi_n,
        "correctness_hits": corr_hit,
        "significance_hits": sig_hit,
        "evidence_hits": evi_hit,
    }


def get_reward_from_score(pred, label, score_set):
    if pred is None:
        return -0.5
    try:
        score_set = set(float(x) for x in score_set)
        if pred in score_set:
            return float(pred == float(label)) + 0.5
        return 0.25
    except (ValueError, TypeError):
        return 0.0


def compute_score(completions, labels, score_sets):
    """
    Reward function that scores LLM outputs which contain a score in <score></score> format.
    The reward is based on:
      - Correct use of the format
      - Presence of a valid numeric value
      - Equality of value and expected score
    """
    preds = [parse_score(completion.outputs[0].text) for completion in completions]
    rewards = np.array([
        get_reward_from_score(pred, label, score_set)
        for pred, label, score_set in zip(preds, labels, score_sets)
    ])

    print(f"\nEXAMPLE FROM BATCH\n\nCompletion: {completions[0].outputs[0].text}\n\nLabel:{labels[0]}\n\nReward: {rewards[0]}\n\n")

    return rewards


def _f1_for_label(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _pearsonr(x, y):
    """Pearson r; 0.0 if undefined (too few points or zero variance)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _micro_f1(y_true, y_pred):
    """Micro-F1 over all classes (equals accuracy for single-label multiclass)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    # Single-label multiclass: micro-F1 == accuracy.
    if y_true.ndim == 1:
        return float(np.mean(y_true == y_pred))
    return 0.0


def classification_metrics(labels, preds, score_sets):
    """
    Classification metrics used by SciRM (Appendix D) and PeerReview Bench Task 2.

    PeerReview Bench Task 2 (AI meta-reviewer, secondary / 10-class setting;
    arXiv:2605.20668 Appendix F Table 49 + 数据集说明 Acc/F1):
      - accuracy: 10-way exact match on label_id in {1..10}
      - f1 / macro_f1: macro-averaged F1 over observed classes
      - weighted_f1: support-weighted F1

    Invalid / out-of-set predictions count as incorrect.
    Binary aspects ({0,1}) also report positive-class F1 (label=1).
    Pearson and MSE are computed on valid in-set predictions only
    (RevUtil-style filtering).
    """
    y_true = np.asarray([float(x) for x in labels], dtype=float)
    y_pred = []
    valid_true = []
    valid_pred = []
    n_valid = 0
    for pred, label, score_set in zip(preds, y_true, score_sets):
        score_set = set(float(x) for x in score_set)
        if pred is not None and pred in score_set:
            y_pred.append(pred)
            valid_true.append(label)
            valid_pred.append(pred)
            n_valid += 1
        else:
            # Force a wrong class so invalid outputs reduce accuracy/F1.
            wrong = next((s for s in sorted(score_set) if s != label), None)
            y_pred.append(wrong if wrong is not None else label + 1.0)
    y_pred = np.asarray(y_pred, dtype=float)
    valid_true = np.asarray(valid_true, dtype=float)
    valid_pred = np.asarray(valid_pred, dtype=float)

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = [_f1_for_label(y_true, y_pred, c) for c in classes]
    support = np.array([np.sum(y_true == c) for c in classes], dtype=float)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=support)) if support.sum() else 0.0
    micro_f1 = _micro_f1(y_true, y_pred)
    mse = float(np.mean((valid_pred - valid_true) ** 2)) if len(valid_true) else 0.0
    pearson = _pearsonr(valid_pred, valid_true) if len(valid_true) else 0.0

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "mse": mse,
        "pearson": pearson,
        "n_valid_score": int(n_valid),
        "n_total": int(len(y_true)),
        "valid_rate": float(n_valid / len(y_true)) if len(y_true) else 0.0,
    }
    # Binary novelty-style labels: also report positive-class F1 (label=1).
    if set(classes).issubset({0.0, 1.0}):
        metrics["f1"] = _f1_for_label(y_true, y_pred, 1.0)
    else:
        # PeerReview Task 2 / multiclass: F1 := macro-F1 (数据集说明 Acc/F1).
        metrics["f1"] = macro_f1
    # Explicit PeerReview Bench Task-2 alias (10-class secondary setting).
    metrics["ten_class_accuracy"] = accuracy
    return metrics


def empty_result_bucket():
    return {
        "peerreview_ten_class": False,
        "rollout_reward_dist": [],
        "rollout_sums": [],
        "rollout_means": [],
        "rollout_stds": [],
        "rollout_accuracy": [],
        "rollout_f1": [],
        "rollout_macro_f1": [],
        "rollout_weighted_f1": [],
        "rollout_micro_f1": [],
        "rollout_ten_class_accuracy": [],
        "rollout_mse": [],
        "rollout_pearson": [],
        "rollout_valid_rate": [],
        "rollout_corr_acc": [],
        "rollout_sig_acc": [],
        "rollout_evi_acc": [],
        "rollout_corr_n": [],
        "rollout_sig_n": [],
        "rollout_evi_n": [],
        "rollout_derived_ten_class_accuracy": [],
        "rollout_derived_ten_class_f1": [],
        "rollout_derived_ten_class_macro_f1": [],
        "rollout_derived_ten_class_reachable_accuracy": [],
        "rollout_derived_ten_class_reachable_n": [],
    }


def derived_ten_class_from_axes(gold_label_ids, pred_correctness, pred_significance, pred_evidence):
    """
    Proxy Table-49 metric for primary cascade runs:
    map predicted axes -> agreed label_id {1,2,4,5,7,9}, compare to gold label_id.
    Disagreement gold classes {3,6,8,10} are unreachable from primary-only outputs.
    """
    pred_ids = [
        axes_to_agreed_label_id(c, s, e)
        for c, s, e in zip(pred_correctness, pred_significance, pred_evidence)
    ]
    gold = [float(x) for x in gold_label_ids]
    score_sets = [TEN_CLASS_SCORE_SET for _ in gold]
    metrics = classification_metrics(gold, pred_ids, score_sets)

    reachable = [
        (g, p)
        for g, p in zip(gold, pred_ids)
        if int(g) in AGREED_TEN_CLASS_IDS
    ]
    if reachable:
        g_r = [g for g, _ in reachable]
        p_r = [p for _, p in reachable]
        reach_m = classification_metrics(
            g_r, p_r, [TEN_CLASS_SCORE_SET for _ in g_r]
        )
        metrics["reachable_accuracy"] = reach_m["accuracy"]
        metrics["reachable_f1"] = reach_m["f1"]
        metrics["reachable_n"] = len(g_r)
    else:
        metrics["reachable_accuracy"] = 0.0
        metrics["reachable_f1"] = 0.0
        metrics["reachable_n"] = 0
    return metrics


def update_result_bucket(
    bucket,
    rewards,
    labels=None,
    preds=None,
    score_sets=None,
    cascade_metrics=None,
    derived_ten_class_metrics=None,
    skip_classification=False,
    peerreview_ten_class=False,
):
    rewards = np.asarray(rewards, dtype=float)
    metrics = {}
    if peerreview_ten_class:
        bucket["peerreview_ten_class"] = True
    bucket["rollout_reward_dist"].append({reward: int(np.sum(rewards == reward)) for reward in [-0.5, 0.0, 0.25, 0.5, 1.5]})
    bucket["rollout_sums"].append(float(rewards.sum()))
    bucket["rollout_means"].append(float(rewards.mean()) if len(rewards) else 0.0)
    bucket["rollout_stds"].append(float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0)
    if not skip_classification:
        metrics = classification_metrics(labels, preds, score_sets)
        bucket["rollout_accuracy"].append(metrics["accuracy"])
        bucket["rollout_f1"].append(metrics["f1"])
        bucket["rollout_macro_f1"].append(metrics["macro_f1"])
        bucket["rollout_weighted_f1"].append(metrics["weighted_f1"])
        bucket["rollout_micro_f1"].append(metrics["micro_f1"])
        bucket["rollout_ten_class_accuracy"].append(metrics["ten_class_accuracy"])
        bucket["rollout_mse"].append(metrics["mse"])
        bucket["rollout_pearson"].append(metrics["pearson"])
        bucket["rollout_valid_rate"].append(metrics["valid_rate"])
    if cascade_metrics is not None:
        bucket["rollout_corr_acc"].append(cascade_metrics["correctness_accuracy"])
        bucket["rollout_sig_acc"].append(cascade_metrics["significance_accuracy"])
        bucket["rollout_evi_acc"].append(cascade_metrics["evidence_accuracy"])
        bucket["rollout_corr_n"].append(cascade_metrics["correctness_n"])
        bucket["rollout_sig_n"].append(cascade_metrics["significance_n"])
        bucket["rollout_evi_n"].append(cascade_metrics["evidence_n"])
        metrics["cascade"] = cascade_metrics
    if derived_ten_class_metrics is not None:
        bucket["rollout_derived_ten_class_accuracy"].append(
            derived_ten_class_metrics["accuracy"]
        )
        bucket["rollout_derived_ten_class_f1"].append(derived_ten_class_metrics["f1"])
        bucket["rollout_derived_ten_class_macro_f1"].append(
            derived_ten_class_metrics["macro_f1"]
        )
        bucket["rollout_derived_ten_class_reachable_accuracy"].append(
            derived_ten_class_metrics["reachable_accuracy"]
        )
        bucket["rollout_derived_ten_class_reachable_n"].append(
            derived_ten_class_metrics["reachable_n"]
        )
        metrics["derived_ten_class"] = derived_ten_class_metrics
    return metrics


def _mean_std(values):
    values = list(values)
    mean = float(np.mean(values)) if values else 0.0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def finalize_result_bucket(bucket, prefix):
    if not bucket["rollout_reward_dist"]:
        return
    keys = bucket["rollout_reward_dist"][0].keys()
    bucket[f"overall_{prefix}_reward_dist"] = {
        k: int(sum(d[k] for d in bucket["rollout_reward_dist"])) for k in keys
    }
    bucket[f"overall_{prefix}_reward_mean"] = float(np.mean(bucket["rollout_means"]))
    bucket[f"{prefix}_reward_mean_stds"] = float(np.mean(bucket["rollout_stds"]))

    is_cascade = bool(bucket["rollout_corr_acc"])
    bucket["paper_metrics"] = {}

    # Classification metrics (SciRM / Task2 10-class). Skipped for cascade runs.
    if bucket["rollout_accuracy"]:
        bucket["accuracy_mean"], bucket["accuracy_std"] = _mean_std(bucket["rollout_accuracy"])
        bucket["f1_mean"], bucket["f1_std"] = _mean_std(bucket["rollout_f1"])
        bucket["macro_f1_mean"], bucket["macro_f1_std"] = _mean_std(bucket["rollout_macro_f1"])
        bucket["weighted_f1_mean"], bucket["weighted_f1_std"] = _mean_std(bucket["rollout_weighted_f1"])
        bucket["micro_f1_mean"], bucket["micro_f1_std"] = _mean_std(bucket["rollout_micro_f1"])
        bucket["ten_class_accuracy_mean"], bucket["ten_class_accuracy_std"] = _mean_std(
            bucket["rollout_ten_class_accuracy"]
        )
        bucket["mse_mean"], bucket["mse_std"] = _mean_std(bucket["rollout_mse"])
        bucket["pearson_mean"], bucket["pearson_std"] = _mean_std(bucket["rollout_pearson"])
        bucket["valid_rate_mean"] = float(np.mean(bucket["rollout_valid_rate"]))
        bucket["paper_metrics"] = {
            "accuracy": f"{bucket['accuracy_mean']:.4f} ± {bucket['accuracy_std']:.4f}",
            "f1": f"{bucket['f1_mean']:.4f} ± {bucket['f1_std']:.4f}",
            "macro_f1": f"{bucket['macro_f1_mean']:.4f} ± {bucket['macro_f1_std']:.4f}",
            "weighted_f1": f"{bucket['weighted_f1_mean']:.4f} ± {bucket['weighted_f1_std']:.4f}",
            "micro_f1": f"{bucket['micro_f1_mean']:.4f} ± {bucket['micro_f1_std']:.4f}",
            "ten_class_accuracy": (
                f"{bucket['ten_class_accuracy_mean']:.4f} ± {bucket['ten_class_accuracy_std']:.4f}"
            ),
            "mse": f"{bucket['mse_mean']:.4f} ± {bucket['mse_std']:.4f}",
            "pearson": f"{bucket['pearson_mean']:.4f} ± {bucket['pearson_std']:.4f}",
        }
        if bucket.get("peerreview_ten_class"):
            bucket["peerreview_ten_class_metrics"] = {
                "accuracy": f"{bucket['accuracy_mean']:.4f} ± {bucket['accuracy_std']:.4f}",
                "f1": f"{bucket['f1_mean']:.4f} ± {bucket['f1_std']:.4f}",
                "macro_f1": f"{bucket['macro_f1_mean']:.4f} ± {bucket['macro_f1_std']:.4f}",
                "weighted_f1": f"{bucket['weighted_f1_mean']:.4f} ± {bucket['weighted_f1_std']:.4f}",
                "micro_f1": f"{bucket['micro_f1_mean']:.4f} ± {bucket['micro_f1_std']:.4f}",
                "ten_class_accuracy": (
                    f"{bucket['ten_class_accuracy_mean']:.4f} ± {bucket['ten_class_accuracy_std']:.4f}"
                ),
                "valid_rate": f"{bucket['valid_rate_mean']:.4f}",
                "note": (
                    "Paper-aligned secondary-setting summary for PeerReview Bench Task 2. "
                    "This prompt format explicitly evaluates the 10-class prediction via <score>. "
                    "Primary per-axis metrics require structured axis labels in the model output."
                ),
            }

    if is_cascade:
        corr_m, corr_s = _mean_std(bucket["rollout_corr_acc"])
        sig_m, sig_s = _mean_std(bucket["rollout_sig_acc"])
        evi_m, evi_s = _mean_std(bucket["rollout_evi_acc"])
        bucket["correctness_accuracy_mean"] = corr_m
        bucket["significance_accuracy_mean"] = sig_m
        bucket["evidence_accuracy_mean"] = evi_m
        bucket["peerreview_cascade_metrics"] = {
            "correctness_accuracy": f"{corr_m:.4f} ± {corr_s:.4f}",
            "significance_accuracy": f"{sig_m:.4f} ± {sig_s:.4f}",
            "evidence_accuracy": f"{evi_m:.4f} ± {evi_s:.4f}",
            "correctness_n": int(bucket["rollout_corr_n"][-1]) if bucket["rollout_corr_n"] else 0,
            "significance_n": int(bucket["rollout_sig_n"][-1]) if bucket["rollout_sig_n"] else 0,
            "evidence_n": int(bucket["rollout_evi_n"][-1]) if bucket["rollout_evi_n"] else 0,
        }
        bucket["paper_metrics"]["correctness_accuracy"] = bucket["peerreview_cascade_metrics"]["correctness_accuracy"]
        bucket["paper_metrics"]["significance_accuracy"] = bucket["peerreview_cascade_metrics"]["significance_accuracy"]
        bucket["paper_metrics"]["evidence_accuracy"] = bucket["peerreview_cascade_metrics"]["evidence_accuracy"]

    if bucket["rollout_derived_ten_class_accuracy"]:
        d_acc_m, d_acc_s = _mean_std(bucket["rollout_derived_ten_class_accuracy"])
        d_f1_m, d_f1_s = _mean_std(bucket["rollout_derived_ten_class_f1"])
        d_mac_m, d_mac_s = _mean_std(bucket["rollout_derived_ten_class_macro_f1"])
        d_reach_m, d_reach_s = _mean_std(bucket["rollout_derived_ten_class_reachable_accuracy"])
        reach_n = int(bucket["rollout_derived_ten_class_reachable_n"][-1])
        bucket["peerreview_derived_ten_class_metrics"] = {
            "accuracy": f"{d_acc_m:.4f} ± {d_acc_s:.4f}",
            "f1": f"{d_f1_m:.4f} ± {d_f1_s:.4f}",
            "macro_f1": f"{d_mac_m:.4f} ± {d_mac_s:.4f}",
            "reachable_accuracy": f"{d_reach_m:.4f} ± {d_reach_s:.4f}",
            "reachable_n": reach_n,
            "note": (
                "Mapped predicted C/S/E -> agreed label_id {1,2,4,5,7,9} vs gold label_id. "
                "Disagreement classes {3,6,8,10} are unreachable from primary cascade outputs."
            ),
        }


def init_model(model_name, max_model_len, max_tokens, temp, top_p, enable_thinking=False, gpu_util=0.9, seed=42):
    import torch
    from vllm import LLM

    supports_thinking = validate_thinking_config(model_name, enable_thinking)

    model = LLM(
        model=model_name,
        dtype=torch.bfloat16,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_util,
        seed=seed,
    )

    sampling_params = model.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    sampling_params.temperature = temp
    sampling_params.top_p = top_p
    sampling_params.seed = seed

    return model, sampling_params, supports_thinking


def inference(batch, model, sampling_params, enable_thinking=False, supports_thinking=False):
    tokenizer = model.get_tokenizer()
    if supports_thinking:
        prompts = format_chat_prompts(tokenizer, batch['prompt'], enable_thinking)
        completions = model.generate(prompts, sampling_params)
        out_toks, in_toks = completion_token_counts(completions, tokenizer=tokenizer, prompts=prompts)
    else:
        completions = model.chat(batch['prompt'], sampling_params)
        out_toks, in_toks = completion_token_counts(
            completions, tokenizer=tokenizer, prompts=batch['prompt']
        )
    outputs = [completion.outputs[0].text for completion in completions]

    if is_combined_peerreview_batch(batch):
        axes_list = [parse_cascade_axes(text) for text in outputs]
        pred_c = [a["correctness"] for a in axes_list]
        pred_s = [a["significance"] for a in axes_list]
        pred_e = [a["evidence"] for a in axes_list]
        preds = [parse_score(text) for text in outputs]
        rewards = np.array([
            get_reward_from_score(pred, label, score_set)
            for pred, label, score_set in zip(preds, batch['labels'], batch['score_sets'])
        ])
        if outputs:
            print(
                f"\nEXAMPLE FROM BATCH\n\nCompletion: {outputs[0]}\n\n"
                f"Gold primary: C={batch['correctness_primary'][0]} "
                f"S={batch['significance_primary'][0]} E={batch['evidence_primary'][0]}\n"
                f"Gold 10-class label: {batch['labels'][0]}\n"
                f"Pred: C={pred_c[0]} S={pred_s[0]} E={pred_e[0]} score={preds[0]}\n"
                f"Reward: {rewards[0]}\n\n"
            )
        return {
            "output": outputs,
            "reward": rewards,
            "pred_score": preds,
            "pred_correctness": pred_c,
            "pred_significance": pred_s,
            "pred_evidence": pred_e,
            "output_token_len": out_toks,
            "prompt_token_len": in_toks,
        }

    if is_cascade_example_batch(batch):
        axes_list = [parse_cascade_axes(text) for text in outputs]
        pred_c = [a["correctness"] for a in axes_list]
        pred_s = [a["significance"] for a in axes_list]
        pred_e = [a["evidence"] for a in axes_list]
        pred_label_ids = [axes_to_agreed_label_id(a) for a in axes_list]
        rewards = np.array([
            get_cascade_reward(axes, gc, gs, ge)
            for axes, gc, gs, ge in zip(
                axes_list,
                batch["correctness_primary"],
                batch["significance_primary"],
                batch["evidence_primary"],
            )
        ])
        if outputs:
            print(
                f"\nEXAMPLE FROM BATCH\n\nCompletion: {outputs[0]}\n\n"
                f"Gold primary: C={batch['correctness_primary'][0]} "
                f"S={batch['significance_primary'][0]} E={batch['evidence_primary'][0]}\n"
                f"Pred: C={pred_c[0]} S={pred_s[0]} E={pred_e[0]}\n"
                f"Reward: {rewards[0]}\n\n"
            )
        return {
            "output": outputs,
            "reward": rewards,
            "pred_correctness": pred_c,
            "pred_significance": pred_s,
            "pred_evidence": pred_e,
            "pred_label_id": pred_label_ids,
            "output_token_len": out_toks,
            "prompt_token_len": in_toks,
        }

    preds = [parse_score(text) for text in outputs]
    rewards = np.array([
        get_reward_from_score(pred, label, score_set)
        for pred, label, score_set in zip(preds, batch['labels'], batch['score_sets'])
    ])
    if outputs:
        print(f"\nEXAMPLE FROM BATCH\n\nCompletion: {outputs[0]}\n\nLabel:{batch['labels'][0]}\n\nReward: {rewards[0]}\n\n")
    return {
        'output': outputs,
        'reward': rewards,
        'pred_score': preds,
        'output_token_len': out_toks,
        'prompt_token_len': in_toks,
    }


def _cascade_metrics_from_df(df, preds=None, out_col=None):
    """Build cascade Acc dict from a dataframe (+ optional precomputed preds)."""
    if out_col is not None:
        axes_list = [parse_cascade_axes(t) for t in df[out_col].tolist()]
        pred_c = [a["correctness"] for a in axes_list]
        pred_s = [a["significance"] for a in axes_list]
        pred_e = [a["evidence"] for a in axes_list]
    else:
        pred_c = df["pred_correctness"].tolist()
        pred_s = df["pred_significance"].tolist()
        pred_e = df["pred_evidence"].tolist()
    return cascade_axis_accuracies(
        pred_c,
        pred_s,
        pred_e,
        df["correctness_primary"].tolist(),
        df["significance_primary"].tolist(),
        df["evidence_primary"].tolist(),
        df["eval_correctness"].tolist(),
        df["eval_significance"].tolist(),
        df["eval_evidence"].tolist(),
    )


def aggregate_from_dataframe(df, rollout):
    """Recompute reward + paper metrics from an existing outputs parquet."""
    aspects = sorted(set(df["aspect"].tolist()))
    results = {aspect: empty_result_bucket() for aspect in aspects}
    results["whole_task"] = empty_result_bucket()
    is_cascade = (
        CASCADE_TASK in set(df["task"].tolist())
        or ("eval_correctness" in df.columns and "correctness_primary" in df.columns)
    )
    if "task" in df.columns and any(t == "meta_reviewer_eval" for t in df["task"].tolist()):
        is_cascade = False
    is_combined = (
        not is_cascade
        and "correctness_primary" in df.columns
        and "eval_correctness" in df.columns
        and "labels" in df.columns
        and "score_sets" in df.columns
        and any(t == "meta_reviewer_eval" for t in df["task"].tolist())
    )
    is_peerreview_secondary = (
        not is_cascade
        and is_peerreview_ten_class_task(
            task=df["task"].iloc[0] if len(df) and "task" in df.columns else None,
            score_sets=df["score_sets"].tolist() if "score_sets" in df.columns else None,
        )
    )
    is_peerreview_secondary = (
        not is_cascade
        and is_peerreview_ten_class_task(
            task=df["task"].iloc[0] if len(df) else None,
            score_sets=df["score_sets"].tolist() if "score_sets" in df.columns else None,
        )
    )

    for turn in range(1, rollout + 1):
        out_col = f"output_{turn}"
        reward_col = f"reward_{turn}"
        if out_col not in df.columns:
            raise ValueError(f"Missing column {out_col} in parquet")

        if is_cascade:
            axes_list = [parse_cascade_axes(text) for text in df[out_col].tolist()]
            if reward_col in df.columns:
                rewards = df[reward_col].tolist()
            else:
                rewards = [
                    get_cascade_reward(axes, gc, gs, ge)
                    for axes, gc, gs, ge in zip(
                        axes_list,
                        df["correctness_primary"].tolist(),
                        df["significance_primary"].tolist(),
                        df["evidence_primary"].tolist(),
                    )
                ]
            cascade_m = _cascade_metrics_from_df(df, out_col=out_col)
            derived_m = None
            if "label_id" in df.columns:
                derived_m = derived_ten_class_from_axes(
                    df["label_id"].tolist(),
                    [a["correctness"] for a in axes_list],
                    [a["significance"] for a in axes_list],
                    [a["evidence"] for a in axes_list],
                )
            update_result_bucket(
                results["whole_task"],
                rewards,
                cascade_metrics=cascade_m,
                derived_ten_class_metrics=derived_m,
                skip_classification=True,
            )
            for aspect in aspects:
                mask = df["aspect"] == aspect
                sub = df.loc[mask]
                sub_rewards = np.asarray(rewards)[mask.values]
                sub_cascade = _cascade_metrics_from_df(sub, out_col=out_col)
                sub_derived = None
                if "label_id" in sub.columns:
                    sub_axes = [parse_cascade_axes(t) for t in sub[out_col].tolist()]
                    sub_derived = derived_ten_class_from_axes(
                        sub["label_id"].tolist(),
                        [a["correctness"] for a in sub_axes],
                        [a["significance"] for a in sub_axes],
                        [a["evidence"] for a in sub_axes],
                    )
                update_result_bucket(
                    results[aspect],
                    sub_rewards,
                    cascade_metrics=sub_cascade,
                    derived_ten_class_metrics=sub_derived,
                    skip_classification=True,
                )
        elif is_combined:
            axes_list = [parse_cascade_axes(text) for text in df[out_col].tolist()]
            preds = [parse_score(text) for text in df[out_col].tolist()]
            labels = df["labels"].tolist()
            score_sets = df["score_sets"].tolist()
            if reward_col in df.columns:
                rewards = df[reward_col].tolist()
            else:
                rewards = [
                    get_reward_from_score(pred, label, score_set)
                    for pred, label, score_set in zip(preds, labels, score_sets)
                ]
            cascade_m = _cascade_metrics_from_df(df, out_col=out_col)
            update_result_bucket(
                results["whole_task"],
                rewards,
                labels,
                preds,
                score_sets,
                cascade_metrics=cascade_m,
                peerreview_ten_class=is_peerreview_secondary,
            )
            for aspect in aspects:
                mask = df["aspect"] == aspect
                sub = df.loc[mask]
                sub_rewards = np.asarray(rewards)[mask.values]
                sub_preds = [p for p, m in zip(preds, mask.tolist()) if m]
                sub_cascade = _cascade_metrics_from_df(sub, out_col=out_col)
                update_result_bucket(
                    results[aspect],
                    sub_rewards,
                    sub["labels"].tolist(),
                    sub_preds,
                    sub["score_sets"].tolist(),
                    cascade_metrics=sub_cascade,
                    peerreview_ten_class=is_peerreview_secondary,
                )
        else:
            preds = [parse_score(text) for text in df[out_col].tolist()]
            labels = df["labels"].tolist()
            score_sets = df["score_sets"].tolist()
            is_peerreview_secondary = is_peerreview_ten_class_task(
                task=df["task"].iloc[0] if len(df) and "task" in df.columns else None,
                score_sets=score_sets,
            )
            if reward_col in df.columns:
                rewards = df[reward_col].tolist()
            else:
                rewards = [
                    get_reward_from_score(pred, label, score_set)
                    for pred, label, score_set in zip(preds, labels, score_sets)
                ]

            update_result_bucket(
                results["whole_task"],
                rewards,
                labels,
                preds,
                score_sets,
                peerreview_ten_class=is_peerreview_secondary,
            )
            for aspect in aspects:
                mask = df["aspect"] == aspect
                update_result_bucket(
                    results[aspect],
                    np.asarray(rewards)[mask.values],
                    df.loc[mask, "labels"].tolist(),
                    [p for p, m in zip(preds, mask.tolist()) if m],
                    df.loc[mask, "score_sets"].tolist(),
                    peerreview_ten_class=is_peerreview_secondary,
                )

    finalize_result_bucket(results["whole_task"], "task")
    for aspect in aspects:
        finalize_result_bucket(results[aspect], "aspect")
    return results


def _print_task_summary(task, wt, prefix=""):
    """Pretty-print metrics; cascade prints Table47 (+ optional derived 10-class), not fully-positive."""
    pfx = f"{prefix}[{task}] " if prefix == "" else prefix
    if "peerreview_cascade_metrics" in wt:
        print(f"{pfx}reward_mean: {wt['overall_task_reward_mean']:.4f}")
        casc = wt["peerreview_cascade_metrics"]
        print(
            f"{pfx}PeerReview Cascade (Table47-style): "
            f"Corr={casc['correctness_accuracy']} | "
            f"Sig={casc['significance_accuracy']} | "
            f"Evid={casc['evidence_accuracy']} "
            f"(n={casc['correctness_n']}/{casc['significance_n']}/{casc['evidence_n']})"
        )
        derived = wt.get("peerreview_derived_ten_class_metrics")
        if derived:
            print(
                f"{pfx}Derived 10-class (axes→label_id vs gold label_id): "
                f"Acc={derived['accuracy']} | F1={derived['f1']} | "
                f"reachable_acc={derived['reachable_accuracy']} "
                f"(n={derived['reachable_n']}; unreachable gold classes 3/6/8/10)"
            )
        sec = wt.get("peerreview_ten_class_metrics")
        if sec:
            print(
                f"{pfx}PeerReview 10-class (Table49-style): "
                f"Acc={sec['accuracy']} | "
                f"F1={sec['f1']} | "
                f"Macro-F1={sec['macro_f1']} | "
                f"valid_rate={sec['valid_rate']}"
            )
    elif "peerreview_ten_class_metrics" in wt:
        sec = wt["peerreview_ten_class_metrics"]
        print(f"{pfx}reward_mean: {wt['overall_task_reward_mean']:.4f}")
        print(
            f"{pfx}PeerReview 10-class (Table49-style): "
            f"Acc={sec['accuracy']} | "
            f"F1={sec['f1']} | "
            f"Macro-F1={sec['macro_f1']} | "
            f"valid_rate={sec['valid_rate']}"
        )
    else:
        print(
            f"{pfx}accuracy: {wt['paper_metrics']['accuracy']} | "
            f"f1: {wt['paper_metrics']['f1']} | "
            f"mse: {wt['paper_metrics']['mse']} | "
            f"pearson: {wt['paper_metrics']['pearson']} | "
            f"reward_mean: {wt['overall_task_reward_mean']:.4f}"
        )


def main(args):
    out_dir = os.path.join(args.output_path, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.recompute_from:
        df = pd.read_parquet(args.recompute_from)
        tasks = sorted(set(df["task"].tolist()))
        for task in tasks:
            task_df = df[df["task"] == task].reset_index(drop=True)
            rollout = args.rollout
            # Infer rollout count from columns if not overridden
            inferred = sorted(
                int(c.split("_")[1]) for c in task_df.columns if c.startswith("output_")
            )
            if inferred:
                rollout = max(inferred) if args.rollout <= 0 else min(args.rollout, max(inferred))
            results = aggregate_from_dataframe(task_df, rollout)

            task_dict = {k: v for k, v in vars(args).items() if k != "recompute_from"}
            task_dict["task"] = task
            task_dict["rollout"] = rollout
            task_dict["recomputed_from"] = args.recompute_from
            task_dict["results"] = results

            out_json = os.path.join(out_dir, f"{task}_results.json")
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w") as fw:
                json.dump(task_dict, fw, indent=4)
            print(f"[recompute] wrote {out_json}")
            _print_task_summary(task, results["whole_task"], prefix="  ")
        return

    set_global_seed(args.seed)
    eval_dataset = load_data(args.dataset_file)['test']
    model, sampling_params, supports_thinking = init_model(
        args.model_name,
        args.max_model_len,
        args.max_tokens,
        args.temp,
        args.top_p,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )
    inference_kwargs = {
        'model': model,
        'sampling_params': sampling_params,
        'enable_thinking': args.enable_thinking,
        'supports_thinking': supports_thinking,
    }

    for task in set(eval_dataset['task']):

        task_dataset = eval_dataset.filter(lambda ex: ex['task'] == task)
        model_tag = model_display_name(args.model_name)
        task_label = f"{task}|{model_tag}"
        n_samples = len(task_dataset)
        print(
            f"[{task_label}] start | n={n_samples} | "
            f"batch_size={args.batch_size} | rollout={args.rollout} | "
            f"temp={args.temp} | seed={args.seed} | model={args.model_name}"
        )

        results = {aspect: empty_result_bucket() for aspect in set(task_dataset['aspect'])}
        results['whole_task'] = empty_result_bucket()

        outputs = []
        timing_rollouts = []
        task_t0 = time.perf_counter()

        for turn in range(args.rollout):
            t0 = time.perf_counter()
            ds = task_dataset.map(
                inference,
                fn_kwargs=inference_kwargs,
                features=map_features_with_predictions(task_dataset, task=task),
                batched=True,
                batch_size=args.batch_size,
                desc=f"[{task_label}] Rollout {turn + 1}/{args.rollout}",
            ).to_pandas()
            elapsed = time.perf_counter() - t0
            rollout_timing = summarize_rollout_timing(
                len(ds),
                elapsed,
                ds["output"].tolist(),
                output_token_lens=ds["output_token_len"].tolist() if "output_token_len" in ds.columns else None,
                prompt_token_lens=ds["prompt_token_len"].tolist() if "prompt_token_len" in ds.columns else None,
            )
            rollout_timing["rollout"] = turn + 1
            timing_rollouts.append(rollout_timing)
            print(
                f"[{task_label}] Rollout {turn + 1}/{args.rollout} done | "
                f"{rollout_timing['elapsed_sec']:.2f}s | "
                f"{rollout_timing['samples_per_sec']:.2f} samples/s | "
                f"{rollout_timing['sec_per_sample']:.3f} s/sample | "
                f"{rollout_timing['output_toks_per_sec']:.0f} out_tok/s | "
                f"{rollout_timing['prompt_toks_per_sec']:.0f} in_tok/s"
            )

            keep_cols = ["output", "reward", "output_token_len", "prompt_token_len"]
            rename = {
                "output": f"output_{turn+1}",
                "reward": f"reward_{turn+1}",
                "output_token_len": f"output_token_len_{turn+1}",
                "prompt_token_len": f"prompt_token_len_{turn+1}",
            }
            for c in ("pred_correctness", "pred_significance", "pred_evidence", "pred_label_id"):
                if c in ds.columns:
                    keep_cols.append(c)
                    rename[c] = f"{c}_{turn+1}"
            keep_cols = [c for c in keep_cols if c in ds.columns]
            outputs.append(ds[keep_cols].rename(columns=rename))

            cascade_m = None
            derived_m = None
            is_combined = (
                task == "meta_reviewer_eval"
                and "eval_correctness" in ds.columns
                and "correctness_primary" in ds.columns
                and "pred_correctness" in ds.columns
            )
            if task == CASCADE_TASK:
                cascade_m = cascade_axis_accuracies(
                    ds["pred_correctness"].tolist(),
                    ds["pred_significance"].tolist(),
                    ds["pred_evidence"].tolist(),
                    ds["correctness_primary"].tolist(),
                    ds["significance_primary"].tolist(),
                    ds["evidence_primary"].tolist(),
                    ds["eval_correctness"].tolist(),
                    ds["eval_significance"].tolist(),
                    ds["eval_evidence"].tolist(),
                )
                if "label_id" in ds.columns:
                    derived_m = derived_ten_class_from_axes(
                        ds["label_id"].tolist(),
                        ds["pred_correctness"].tolist(),
                        ds["pred_significance"].tolist(),
                        ds["pred_evidence"].tolist(),
                    )
                update_result_bucket(
                    results['whole_task'],
                    ds['reward'].tolist(),
                    cascade_metrics=cascade_m,
                    derived_ten_class_metrics=derived_m,
                    skip_classification=True,
                )
                for aspect in [k for k in results.keys() if k != 'whole_task']:
                    aspect_subset = ds[ds['aspect'] == aspect]
                    aspect_cascade = cascade_axis_accuracies(
                        aspect_subset["pred_correctness"].tolist(),
                        aspect_subset["pred_significance"].tolist(),
                        aspect_subset["pred_evidence"].tolist(),
                        aspect_subset["correctness_primary"].tolist(),
                        aspect_subset["significance_primary"].tolist(),
                        aspect_subset["evidence_primary"].tolist(),
                        aspect_subset["eval_correctness"].tolist(),
                        aspect_subset["eval_significance"].tolist(),
                        aspect_subset["eval_evidence"].tolist(),
                    )
                    aspect_derived = None
                    if "label_id" in aspect_subset.columns:
                        aspect_derived = derived_ten_class_from_axes(
                            aspect_subset["label_id"].tolist(),
                            aspect_subset["pred_correctness"].tolist(),
                            aspect_subset["pred_significance"].tolist(),
                            aspect_subset["pred_evidence"].tolist(),
                        )
                    update_result_bucket(
                        results[aspect],
                        aspect_subset['reward'].tolist(),
                        cascade_metrics=aspect_cascade,
                        derived_ten_class_metrics=aspect_derived,
                        skip_classification=True,
                    )
            elif is_combined:
                is_peerreview_secondary = is_peerreview_ten_class_task(
                    task=task,
                    score_sets=ds['score_sets'].tolist(),
                )
                cascade_m = cascade_axis_accuracies(
                    ds["pred_correctness"].tolist(),
                    ds["pred_significance"].tolist(),
                    ds["pred_evidence"].tolist(),
                    ds["correctness_primary"].tolist(),
                    ds["significance_primary"].tolist(),
                    ds["evidence_primary"].tolist(),
                    ds["eval_correctness"].tolist(),
                    ds["eval_significance"].tolist(),
                    ds["eval_evidence"].tolist(),
                )
                update_result_bucket(
                    results['whole_task'],
                    ds['reward'].tolist(),
                    ds['labels'].tolist(),
                    ds['pred_score'].tolist(),
                    ds['score_sets'].tolist(),
                    cascade_metrics=cascade_m,
                    peerreview_ten_class=is_peerreview_secondary,
                )
                for aspect in [k for k in results.keys() if k != 'whole_task']:
                    aspect_subset = ds[ds['aspect'] == aspect]
                    aspect_cascade = cascade_axis_accuracies(
                        aspect_subset["pred_correctness"].tolist(),
                        aspect_subset["pred_significance"].tolist(),
                        aspect_subset["pred_evidence"].tolist(),
                        aspect_subset["correctness_primary"].tolist(),
                        aspect_subset["significance_primary"].tolist(),
                        aspect_subset["evidence_primary"].tolist(),
                        aspect_subset["eval_correctness"].tolist(),
                        aspect_subset["eval_significance"].tolist(),
                        aspect_subset["eval_evidence"].tolist(),
                    )
                    update_result_bucket(
                        results[aspect],
                        aspect_subset['reward'].tolist(),
                        aspect_subset['labels'].tolist(),
                        aspect_subset['pred_score'].tolist(),
                        aspect_subset['score_sets'].tolist(),
                        cascade_metrics=aspect_cascade,
                        peerreview_ten_class=is_peerreview_secondary,
                    )
            else:
                is_peerreview_secondary = is_peerreview_ten_class_task(
                    task=task,
                    score_sets=ds['score_sets'].tolist(),
                )
                update_result_bucket(
                    results['whole_task'],
                    ds['reward'].tolist(),
                    ds['labels'].tolist(),
                    ds['pred_score'].tolist(),
                    ds['score_sets'].tolist(),
                    peerreview_ten_class=is_peerreview_secondary,
                )
                for aspect in [k for k in results.keys() if k != 'whole_task']:
                    aspect_subset = ds[ds['aspect'] == aspect]
                    update_result_bucket(
                        results[aspect],
                        aspect_subset['reward'].tolist(),
                        aspect_subset['labels'].tolist(),
                        aspect_subset['pred_score'].tolist(),
                        aspect_subset['score_sets'].tolist(),
                        peerreview_ten_class=is_peerreview_secondary,
                    )

        finalize_result_bucket(results['whole_task'], 'task')
        for aspect in [k for k in results.keys() if k != 'whole_task']:
            finalize_result_bucket(results[aspect], 'aspect')

        total_infer_sec = time.perf_counter() - task_t0
        total_samples = sum(r["n_samples"] for r in timing_rollouts)
        total_out_toks = sum(r["output_tokens"] for r in timing_rollouts)
        total_in_toks = sum(r["prompt_tokens"] for r in timing_rollouts)
        efficiency = {
            "model_name": args.model_name,
            "model_tag": model_tag,
            "task": task,
            "n_samples": n_samples,
            "batch_size": args.batch_size,
            "rollout": args.rollout,
            "token_source": "vllm_request_output_or_model_tokenizer",
            "rollouts": timing_rollouts,
            "total_infer_sec": round(total_infer_sec, 3),
            "total_sample_forwards": total_samples,
            "mean_samples_per_sec": round(total_samples / max(total_infer_sec, 1e-9), 3),
            "mean_sec_per_sample": round(total_infer_sec / max(total_samples, 1), 4),
            "total_output_tokens": total_out_toks,
            "total_prompt_tokens": total_in_toks,
            "mean_output_toks_per_sec": round(total_out_toks / max(total_infer_sec, 1e-9), 2),
            "mean_prompt_toks_per_sec": round(total_in_toks / max(total_infer_sec, 1e-9), 2),
            "mean_total_toks_per_sec": round((total_in_toks + total_out_toks) / max(total_infer_sec, 1e-9), 2),
        }

        drop_cols = ['output', 'reward', 'output_token_len', 'prompt_token_len']
        for c in ("pred_correctness", "pred_significance", "pred_evidence", "pred_label_id", "pred_score"):
            if c in ds.columns:
                drop_cols.append(c)
        drop_cols = [c for c in drop_cols if c in ds.columns]
        final_outputs = pd.concat(
            [ds.drop(columns=drop_cols).reset_index(drop=True)] + outputs,
            axis=1,
        )

        task_dict = vars(args)
        task_dict['task'] = task
        task_dict['model_tag'] = model_tag
        task_dict['efficiency'] = efficiency
        task_dict['results'] = results

        # Re-create right before write: parent dirs may have been moved/deleted mid-run.
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{task}_results.json")
        outputs_path = os.path.join(out_dir, f"{task}_outputs.parquet")
        with open(results_path, 'w') as fw:
            json.dump(task_dict, fw, indent=4)

        final_outputs.to_parquet(outputs_path, index=False)
        print(f"[{task_label}] wrote {results_path}")
        print(f"[{task_label}] wrote {outputs_path}")
        print(
            f"[{task_label}] efficiency | total={efficiency['total_infer_sec']:.2f}s | "
            f"{efficiency['mean_samples_per_sec']:.2f} samples/s | "
            f"{efficiency['mean_sec_per_sample']:.3f} s/sample | "
            f"{efficiency['mean_output_toks_per_sec']:.0f} out_tok/s | "
            f"{efficiency['mean_prompt_toks_per_sec']:.0f} in_tok/s | "
            f"{efficiency['mean_total_toks_per_sec']:.0f} total_tok/s"
        )
        _print_task_summary(task_label, results['whole_task'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--dataset_file', default='', type=str)
    parser.add_argument('--max_model_len', default=32768, type=int)
    parser.add_argument('--max_tokens', default=2048, type=int)
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--rollout', default=5, type=int,
                        help='Number of rollouts. For --recompute_from, <=0 means use all output_* columns.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='Global + vLLM sampling seed for reproducible generation (default: 42).',
    )
    parser.add_argument(
        '--enable_thinking',
        action='store_true',
        help='Enable model-native thinking mode (e.g., Qwen3 ). '
             'Raises an error if the model chat template does not support it. '
             'Default: disabled for faster inference.',
    )
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument(
        '--recompute_from',
        default='',
        type=str,
        help='Path to an existing *_outputs.parquet; recompute metrics without running the model.',
    )

    arguments = parser.parse_args()
    if not arguments.recompute_from:
        if not arguments.model_name or not arguments.dataset_file:
            parser.error('--model_name and --dataset_file are required unless --recompute_from is set')
    main(arguments)
