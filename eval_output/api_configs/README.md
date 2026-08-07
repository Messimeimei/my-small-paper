# Fixed-version API baselines

API results are isolated under `eval_output/api_results/`; the existing local-model
results under `eval_output/results/` are read-only inputs to the combined report.
The evaluator loads `GPTPLUS5_KEY` from the project-root `.env` and sends
requests to the exact OpenAI-compatible endpoint `https://az.gptplus5.com/v1`.
Values in that file override stale variables inherited by the current process.

Validate the complete matrix without making requests:

```bash
python training/evaluate_api.py
```

Run one smoke-test sample. Limited runs use a separate `#limit_1` directory and are
excluded from the full comparison report:

```bash
python training/evaluate_api.py \
  --model claude_opus_4_5_20251101 \
  --task rev_util_actionability \
  --mode cot \
  --limit 1 \
  --execute
```

Run or resume the full matrix only after confirming the request budget:

```bash
python training/evaluate_api.py --execute
```

Run or resume the dated OpenBitFun Doubao baseline:

```bash
python training/evaluate_api.py \
  --config eval_output/api_configs/openbitfun_doubao_260215.yaml \
  --execute
```

Run or resume the callable dated OpenAI snapshots exposed by gptplus5 on the
first task (`rev_util_actionability`) in both prompt modes. The catalog also
advertises `gpt-4o-2024-08-06`, but both prompt-mode probes returned `503 No
available channel` on 2026-08-05, so it is documented in the config and smoke
log rather than included in the full-run matrix.

```bash
python training/evaluate_api.py \
  --config eval_output/api_configs/gptplus5_openai_dated_actionability.yaml \
  --execute
```

These runs use model-specific `gpt_*_YYYY_MM_DD` result directories and are
therefore distinct from the `doubao_seed_2_0_pro_260215` results.

The dated request ID `doubao-seed-2-0-pro-260215` is preserved in all result
metadata. OpenBitFun reports `doubao-seed-2.0-pro` in responses, so only that
normalized response name is explicitly accepted for this model.

Every successful response is flushed immediately to `api_responses.jsonl`. Repeating
the same command skips completed sample IDs. A changed dataset, prompt, model, or
decoding configuration gets a different request hash and cannot be mixed into the
same run directory.

Rebuild reports without making API requests:

```bash
python training/evaluate_api.py --refresh-report-only
```

