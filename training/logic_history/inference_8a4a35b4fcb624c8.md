# Inference Logic Snapshot

- Logic ID: `8a4a35b4fcb624c8`
- Variant: `vllm:label_only`
- Generated at: `2026-07-31T05:15:38.519948+00:00`

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
  "note": "See run-local snapshot for effective settings."
}
```

## Source Fingerprints

- `training/logic_snapshot.py`: `0acfb8624872c4b2b7ac20014b44f48280969a4f2ec16182f414b01c1cdec2ca`
- `training/evaluate.py`: `68c47819c52a855377f220fb037e413f357ecd486a68394144906ff20d405967`
- `training/metrics_utils.py`: `f57a537806a194c2ec10bb03d701dec78c4e91d122d1189e0d0bfae93ea441ea`
