# Inference Logic Snapshot

- Logic ID: `fe20c57a31788368`
- Variant: `vllm:cot`
- Generated at: `2026-07-31T05:26:50.232347+00:00`

## Logic

```json
{
  "model_loading": "Base model directly when adapter is absent; otherwise merge the LoRA adapter into a temporary full model, then load it with vLLM.",
  "prompt": "Use each dataset row's stored prompt and apply the model chat template; the dataset mode determines Label-only versus CoT instructions.",
  "decoding": "Greedy decoding with temperature=0 and top_p=1.",
  "prediction": "Parse the last <score> integer in the allowed score set; missing or out-of-range scores are invalid and count as incorrect.",
  "metrics": "Accuracy, macro-F1, per-class metrics, format-valid rate, confusion matrix, token statistics, and MAE/QWK for ordinal score sets."
}
```

## Effective Run

```json
{
  "experiment": "rev_util_grounding_specificity#qwen3_4b#ft#paper_align#on_cot",
  "dataset": "/root/my-small-paper/data/rev_util_grounding_specificity/cot/test_cot.jsonl",
  "samples": 1000,
  "score_sets": [
    1,
    2,
    3,
    4,
    5
  ],
  "adapter": "/root/autodl-tmp/train_outputs/rev_util_grounding_specificity/paper_align/rev_util_grounding_specificity#qwen3_4b#paper_align__seed42__2026-07-31_12-12-51/adapter",
  "train_logic_run": "rev_util_grounding_specificity#qwen3_4b#paper_align__seed42__2026-07-31_12-12-51",
  "seed": 42,
  "rollouts": 1,
  "batch_size": 64,
  "max_model_len": 8192,
  "max_tokens": 512,
  "enable_thinking": false
}
```

## Source Fingerprints

- `training/logic_snapshot.py`: `0acfb8624872c4b2b7ac20014b44f48280969a4f2ec16182f414b01c1cdec2ca`
- `training/evaluate.py`: `68c47819c52a855377f220fb037e413f357ecd486a68394144906ff20d405967`
- `training/metrics_utils.py`: `1b7c7de947d97df09e32b9c8242db76080377f55a624396905bccf27db279857`
