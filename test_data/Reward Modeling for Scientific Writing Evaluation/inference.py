import re
import os
import argparse
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

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


def pred_fully_positive(axes):
    return int(
        axes.get("correctness") == "Correct"
        and axes.get("significance") == "Significant"
        and axes.get("evidence") == "Sufficient"
    )


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
    if "cascade_mode" in batch and any(bool(x) for x in batch["cascade_mode"]):
        return True
    return "eval_correctness" in batch and "correctness_primary" in batch


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
    }


def update_result_bucket(bucket, rewards, labels, preds, score_sets, cascade_metrics=None):
    rewards = np.asarray(rewards, dtype=float)
    metrics = classification_metrics(labels, preds, score_sets)
    bucket["rollout_reward_dist"].append({reward: int(np.sum(rewards == reward)) for reward in [-0.5, 0.0, 0.25, 0.5, 1.5]})
    bucket["rollout_sums"].append(float(rewards.sum()))
    bucket["rollout_means"].append(float(rewards.mean()) if len(rewards) else 0.0)
    bucket["rollout_stds"].append(float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0)
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

    # Paper-style: mean ± std over rollouts
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
    # PeerReview Bench Task 2 (meta-reviewer 10-class): Acc + F1.
    bucket["peerreview_task2_metrics"] = {
        "accuracy": bucket["paper_metrics"]["accuracy"],
        "f1": bucket["paper_metrics"]["f1"],
        "ten_class_accuracy": bucket["paper_metrics"]["ten_class_accuracy"],
        "macro_f1": bucket["paper_metrics"]["macro_f1"],
    }
    if bucket["rollout_corr_acc"]:
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


def init_model(model_name, max_model_len, max_tokens, temp, top_p, enable_thinking=False, gpu_util=0.9):
    import torch
    from vllm import LLM

    supports_thinking = validate_thinking_config(model_name, enable_thinking)

    model = LLM(model=model_name, dtype=torch.bfloat16, max_model_len=max_model_len, trust_remote_code=True, gpu_memory_utilization=gpu_util)

    sampling_params = model.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    sampling_params.temperature = temp
    sampling_params.top_p = top_p

    return model, sampling_params, supports_thinking


def inference(batch, model, sampling_params, enable_thinking=False, supports_thinking=False):
    if supports_thinking:
        tokenizer = model.get_tokenizer()
        prompts = format_chat_prompts(tokenizer, batch['prompt'], enable_thinking)
        completions = model.generate(prompts, sampling_params)
    else:
        completions = model.chat(batch['prompt'], sampling_params)
    outputs = [completion.outputs[0].text for completion in completions]

    if is_cascade_example_batch(batch):
        axes_list = [parse_cascade_axes(text) for text in outputs]
        pred_c = [a["correctness"] for a in axes_list]
        pred_s = [a["significance"] for a in axes_list]
        pred_e = [a["evidence"] for a in axes_list]
        preds = [float(pred_fully_positive(a)) if a["correctness"] is not None else None for a in axes_list]
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
        # HF datasets dislikes None in string columns sometimes; keep None for metrics.
        return {
            "output": outputs,
            "reward": rewards,
            "pred_score": preds,
            "pred_correctness": pred_c,
            "pred_significance": pred_s,
            "pred_evidence": pred_e,
        }

    preds = [parse_score(text) for text in outputs]
    rewards = np.array([
        get_reward_from_score(pred, label, score_set)
        for pred, label, score_set in zip(preds, batch['labels'], batch['score_sets'])
    ])
    if outputs:
        print(f"\nEXAMPLE FROM BATCH\n\nCompletion: {outputs[0]}\n\nLabel:{batch['labels'][0]}\n\nReward: {rewards[0]}\n\n")
    return {'output': outputs, 'reward': rewards, 'pred_score': preds}


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

    for turn in range(1, rollout + 1):
        out_col = f"output_{turn}"
        reward_col = f"reward_{turn}"
        if out_col not in df.columns:
            raise ValueError(f"Missing column {out_col} in parquet")

        if is_cascade:
            axes_list = [parse_cascade_axes(text) for text in df[out_col].tolist()]
            preds = [
                float(pred_fully_positive(a)) if a["correctness"] is not None else None
                for a in axes_list
            ]
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
            labels = df["labels"].tolist()
            score_sets = df["score_sets"].tolist()
            cascade_m = _cascade_metrics_from_df(df, out_col=out_col)
            update_result_bucket(
                results["whole_task"], rewards, labels, preds, score_sets, cascade_metrics=cascade_m
            )
            for aspect in aspects:
                mask = df["aspect"] == aspect
                sub = df.loc[mask]
                sub_preds = [p for p, m in zip(preds, mask.tolist()) if m]
                sub_rewards = np.asarray(rewards)[mask.values]
                sub_cascade = _cascade_metrics_from_df(sub, out_col=out_col)
                update_result_bucket(
                    results[aspect],
                    sub_rewards,
                    sub["labels"].tolist(),
                    sub_preds,
                    sub["score_sets"].tolist(),
                    cascade_metrics=sub_cascade,
                )
        else:
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

            update_result_bucket(results["whole_task"], rewards, labels, preds, score_sets)
            for aspect in aspects:
                mask = df["aspect"] == aspect
                update_result_bucket(
                    results[aspect],
                    np.asarray(rewards)[mask.values],
                    df.loc[mask, "labels"].tolist(),
                    [p for p, m in zip(preds, mask.tolist()) if m],
                    df.loc[mask, "score_sets"].tolist(),
                )

    finalize_result_bucket(results["whole_task"], "task")
    for aspect in aspects:
        finalize_result_bucket(results[aspect], "aspect")
    return results


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
            wt = results["whole_task"]
            t2 = wt.get("peerreview_task2_metrics", {})
            print(
                f"  accuracy: {wt['paper_metrics']['accuracy']} | "
                f"f1: {wt['paper_metrics']['f1']} | "
                f"macro_f1: {wt['paper_metrics']['macro_f1']} | "
                f"mse: {wt['paper_metrics']['mse']} | "
                f"pearson: {wt['paper_metrics']['pearson']} | "
                f"reward_mean: {wt['overall_task_reward_mean']:.4f}"
            )
            if t2:
                print(
                    f"  [PeerReview Task2] Acc: {t2['accuracy']} | "
                    f"F1: {t2['f1']} | ten_class_acc: {t2['ten_class_accuracy']}"
                )
            casc = wt.get("peerreview_cascade_metrics")
            if casc:
                print(
                    f"  [PeerReview Cascade] Corr: {casc['correctness_accuracy']} | "
                    f"Sig: {casc['significance_accuracy']} | "
                    f"Evid: {casc['evidence_accuracy']} "
                    f"(n={casc['correctness_n']}/{casc['significance_n']}/{casc['evidence_n']})"
                )
        return

    eval_dataset = load_data(args.dataset_file)['test']
    model, sampling_params, supports_thinking = init_model(
        args.model_name,
        args.max_model_len,
        args.max_tokens,
        args.temp,
        args.top_p,
        enable_thinking=args.enable_thinking,
    )
    inference_kwargs = {
        'model': model,
        'sampling_params': sampling_params,
        'enable_thinking': args.enable_thinking,
        'supports_thinking': supports_thinking,
    }

    for task in set(eval_dataset['task']):

        task_dataset = eval_dataset.filter(lambda ex: ex['task'] == task)

        results = {aspect: empty_result_bucket() for aspect in set(task_dataset['aspect'])}
        results['whole_task'] = empty_result_bucket()

        outputs = []

        for turn in range(args.rollout):
            ds = task_dataset.map(
                inference,
                fn_kwargs=inference_kwargs,
                batched=True,
                batch_size=args.batch_size,
                desc=f"[{task}] Rollout {turn + 1}/{args.rollout}",
            ).to_pandas()

            keep_cols = ["output", "reward", "pred_score"]
            rename = {
                "output": f"output_{turn+1}",
                "reward": f"reward_{turn+1}",
                "pred_score": f"pred_score_{turn+1}",
            }
            for c in ("pred_correctness", "pred_significance", "pred_evidence"):
                if c in ds.columns:
                    keep_cols.append(c)
                    rename[c] = f"{c}_{turn+1}"
            outputs.append(ds[keep_cols].rename(columns=rename))

            cascade_m = None
            if task == CASCADE_TASK or "eval_correctness" in ds.columns:
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
            )

            for aspect in [k for k in results.keys() if k != 'whole_task']:
                aspect_subset = ds[ds['aspect'] == aspect]
                aspect_cascade = None
                if cascade_m is not None:
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
                )

        finalize_result_bucket(results['whole_task'], 'task')
        for aspect in [k for k in results.keys() if k != 'whole_task']:
            finalize_result_bucket(results[aspect], 'aspect')

        drop_cols = ['output', 'reward', 'pred_score']
        for c in ("pred_correctness", "pred_significance", "pred_evidence"):
            if c in ds.columns:
                drop_cols.append(c)
        final_outputs = pd.concat(
            [ds.drop(columns=drop_cols).reset_index(drop=True)] + outputs,
            axis=1,
        )

        task_dict = vars(args)
        task_dict['task'] = task
        task_dict['results'] = results

        # Re-create right before write: parent dirs may have been moved/deleted mid-run.
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{task}_results.json")
        outputs_path = os.path.join(out_dir, f"{task}_outputs.parquet")
        with open(results_path, 'w') as fw:
            json.dump(task_dict, fw, indent=4)

        final_outputs.to_parquet(outputs_path, index=False)
        print(f"[{task}] wrote {results_path}")
        print(f"[{task}] wrote {outputs_path}")
        wt = results['whole_task']
        print(
            f"[{task}] accuracy: {wt['paper_metrics']['accuracy']} | "
            f"f1: {wt['paper_metrics']['f1']} | "
            f"mse: {wt['paper_metrics']['mse']} | "
            f"pearson: {wt['paper_metrics']['pearson']} | "
            f"reward_mean: {wt['overall_task_reward_mean']:.4f}"
        )
        if task == "meta_reviewer_eval" or "peerreview_task2_metrics" in wt:
            t2 = wt["peerreview_task2_metrics"]
            print(
                f"[{task}] PeerReview Task2 (10-class): "
                f"Acc={t2['accuracy']} | F1={t2['f1']}"
            )
        if "peerreview_cascade_metrics" in wt:
            casc = wt["peerreview_cascade_metrics"]
            print(
                f"[{task}] PeerReview Cascade (Table47-style): "
                f"Corr={casc['correctness_accuracy']} | "
                f"Sig={casc['significance_accuracy']} | "
                f"Evid={casc['evidence_accuracy']} "
                f"(n={casc['correctness_n']}/{casc['significance_n']}/{casc['evidence_n']})"
            )


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