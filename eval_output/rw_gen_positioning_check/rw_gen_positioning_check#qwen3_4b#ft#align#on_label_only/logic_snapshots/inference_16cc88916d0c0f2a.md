# Inference Logic Snapshot

- Logic ID: `16cc88916d0c0f2a`
- Variant: `vllm:label_only`
- Generated at: `2026-07-30T13:59:57.491582+00:00`

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
  "experiment": "rw_gen_positioning_check#qwen3_4b#ft#align#on_label_only",
  "dataset": "/root/my-small-paper/data/rw_gen_positioning_check/label_only/test_label_only.jsonl",
  "samples": 603,
  "score_sets": [
    0,
    1
  ],
  "adapter": "/root/autodl-tmp/train_outputs/rw_gen_positioning_check/align/rw_gen_positioning_check#qwen3_4b#align__seed42__2026-07-30_20-17-59/adapter",
  "train_logic_run": "rw_gen_positioning_check#qwen3_4b#align__seed42__2026-07-30_20-17-59",
  "seed": 42,
  "rollouts": 1,
  "batch_size": 64,
  "max_model_len": 8192,
  "max_tokens": 32,
  "enable_thinking": false
}
```

## Source Fingerprints

- `training/logic_snapshot.py`: `49d1780903c5ecbfbd5b27d3712599995ed727ab8585e2e3949144e15db49318`
- `training/evaluate.py`: `68c47819c52a855377f220fb037e413f357ecd486a68394144906ff20d405967`
- `training/metrics_utils.py`: `bcf5d952e05aa9e04ce7bda5deae2b0931bfe07a2fab031aa764fe4b1bb34ddb`
