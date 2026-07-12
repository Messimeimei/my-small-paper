import re
import os
import argparse
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

SCORE_PATTERN = re.compile(r"<score>([^<]+)</score>")


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


def classification_metrics(labels, preds, score_sets):
    """
    Paper-style metrics (Appendix D Tables 8-13): accuracy and F1.
    Invalid / out-of-set predictions count as incorrect.
    Reports binary F1 on the positive class when labels are {0,1},
    plus macro / weighted F1 for multi-class aspects.
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
    mse = float(np.mean((valid_pred - valid_true) ** 2)) if len(valid_true) else 0.0
    pearson = _pearsonr(valid_pred, valid_true) if len(valid_true) else 0.0

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
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
        metrics["f1"] = macro_f1
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
        "rollout_mse": [],
        "rollout_pearson": [],
        "rollout_valid_rate": [],
    }


def update_result_bucket(bucket, rewards, labels, preds, score_sets):
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
    bucket["rollout_mse"].append(metrics["mse"])
    bucket["rollout_pearson"].append(metrics["pearson"])
    bucket["rollout_valid_rate"].append(metrics["valid_rate"])
    return metrics


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
    bucket["accuracy_mean"] = float(np.mean(bucket["rollout_accuracy"]))
    bucket["accuracy_std"] = float(np.std(bucket["rollout_accuracy"], ddof=1)) if len(bucket["rollout_accuracy"]) > 1 else 0.0
    bucket["f1_mean"] = float(np.mean(bucket["rollout_f1"]))
    bucket["f1_std"] = float(np.std(bucket["rollout_f1"], ddof=1)) if len(bucket["rollout_f1"]) > 1 else 0.0
    bucket["macro_f1_mean"] = float(np.mean(bucket["rollout_macro_f1"]))
    bucket["macro_f1_std"] = float(np.std(bucket["rollout_macro_f1"], ddof=1)) if len(bucket["rollout_macro_f1"]) > 1 else 0.0
    bucket["weighted_f1_mean"] = float(np.mean(bucket["rollout_weighted_f1"]))
    bucket["weighted_f1_std"] = float(np.std(bucket["rollout_weighted_f1"], ddof=1)) if len(bucket["rollout_weighted_f1"]) > 1 else 0.0
    bucket["mse_mean"] = float(np.mean(bucket["rollout_mse"]))
    bucket["mse_std"] = float(np.std(bucket["rollout_mse"], ddof=1)) if len(bucket["rollout_mse"]) > 1 else 0.0
    bucket["pearson_mean"] = float(np.mean(bucket["rollout_pearson"]))
    bucket["pearson_std"] = float(np.std(bucket["rollout_pearson"], ddof=1)) if len(bucket["rollout_pearson"]) > 1 else 0.0
    bucket["valid_rate_mean"] = float(np.mean(bucket["rollout_valid_rate"]))
    bucket["paper_metrics"] = {
        "accuracy": f"{bucket['accuracy_mean']:.4f} ± {bucket['accuracy_std']:.4f}",
        "f1": f"{bucket['f1_mean']:.4f} ± {bucket['f1_std']:.4f}",
        "macro_f1": f"{bucket['macro_f1_mean']:.4f} ± {bucket['macro_f1_std']:.4f}",
        "weighted_f1": f"{bucket['weighted_f1_mean']:.4f} ± {bucket['weighted_f1_std']:.4f}",
        "mse": f"{bucket['mse_mean']:.4f} ± {bucket['mse_std']:.4f}",
        "pearson": f"{bucket['pearson_mean']:.4f} ± {bucket['pearson_std']:.4f}",
    }


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
    preds = [parse_score(text) for text in outputs]
    rewards = np.array([
        get_reward_from_score(pred, label, score_set)
        for pred, label, score_set in zip(preds, batch['labels'], batch['score_sets'])
    ])
    if outputs:
        print(f"\nEXAMPLE FROM BATCH\n\nCompletion: {outputs[0]}\n\nLabel:{batch['labels'][0]}\n\nReward: {rewards[0]}\n\n")
    return {'output': outputs, 'reward': rewards, 'pred_score': preds}


def aggregate_from_dataframe(df, rollout):
    """Recompute reward + paper metrics from an existing outputs parquet."""
    aspects = sorted(set(df["aspect"].tolist()))
    results = {aspect: empty_result_bucket() for aspect in aspects}
    results["whole_task"] = empty_result_bucket()

    for turn in range(1, rollout + 1):
        out_col = f"output_{turn}"
        reward_col = f"reward_{turn}"
        if out_col not in df.columns:
            raise ValueError(f"Missing column {out_col} in parquet")

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
            with open(out_json, "w") as fw:
                json.dump(task_dict, fw, indent=4)
            print(f"[recompute] wrote {out_json}")
            wt = results["whole_task"]
            print(
                f"  accuracy: {wt['paper_metrics']['accuracy']} | "
                f"f1: {wt['paper_metrics']['f1']} | "
                f"macro_f1: {wt['paper_metrics']['macro_f1']} | "
                f"mse: {wt['paper_metrics']['mse']} | "
                f"pearson: {wt['paper_metrics']['pearson']} | "
                f"reward_mean: {wt['overall_task_reward_mean']:.4f}"
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

            outputs.append(
                ds[['output', 'reward', 'pred_score']].rename(
                    columns={
                        'output': f'output_{turn+1}',
                        'reward': f'reward_{turn+1}',
                        'pred_score': f'pred_score_{turn+1}',
                    }
                )
            )

            update_result_bucket(
                results['whole_task'],
                ds['reward'].tolist(),
                ds['labels'].tolist(),
                ds['pred_score'].tolist(),
                ds['score_sets'].tolist(),
            )

            for aspect in [k for k in results.keys() if k != 'whole_task']:
                aspect_subset = ds[ds['aspect'] == aspect]
                update_result_bucket(
                    results[aspect],
                    aspect_subset['reward'].tolist(),
                    aspect_subset['labels'].tolist(),
                    aspect_subset['pred_score'].tolist(),
                    aspect_subset['score_sets'].tolist(),
                )

        finalize_result_bucket(results['whole_task'], 'task')
        for aspect in [k for k in results.keys() if k != 'whole_task']:
            finalize_result_bucket(results[aspect], 'aspect')

        final_outputs = pd.concat(
            [ds.drop(columns=['output', 'reward', 'pred_score']).reset_index(drop=True)] + outputs,
            axis=1,
        )

        task_dict = vars(args)
        task_dict['task'] = task
        task_dict['results'] = results

        with open(os.path.join(out_dir, f"{task}_results.json"), 'w') as fw:
            json.dump(task_dict, fw, indent=4)

        final_outputs.to_parquet(os.path.join(out_dir, f"{task}_outputs.parquet"), index=False)
        wt = results['whole_task']
        print(
            f"[{task}] accuracy: {wt['paper_metrics']['accuracy']} | "
            f"f1: {wt['paper_metrics']['f1']} | "
            f"mse: {wt['paper_metrics']['mse']} | "
            f"pearson: {wt['paper_metrics']['pearson']} | "
            f"reward_mean: {wt['overall_task_reward_mean']:.4f}"
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