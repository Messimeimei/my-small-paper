# RS vs RePaIR rationale-quality evaluation

This bundle evaluates 4B RS and 4B RePaIR rationales on the exact 350 samples
used by the final SO-vs-RS audit:

- seed 42 predictions;
- 50 representative samples per task across seven tasks;
- sample seed 20260824;
- the original A/B side assignment;
- the unchanged two-round blind prompts;
- round 1: four-dimensional rationale quality (1--3);
- round 2: rationale-predicted-score consistency;
- gold labels, condition names, correctness, and training methods are hidden
  from both judge prompts.

RS stays on the original CC side. RePaIR replaces SO on the original LC side,
so RS is the repeated anchor between the old SO-vs-RS and new RS-vs-RePaIR
comparisons.

## Files

- `rs_vs_repair_matched_350.jsonl`: portable evaluation input.
- `data_manifest.json`: hashes, sources, sample counts, and inherited config.
- `scripts/judge_rs_vs_repair_rationales.py`: evaluation entry point.
- `scripts/judge_cot_vs_label_training_rationales.py`: unchanged prompt and API
  implementation imported by the entry point.

No API key is included.

## Strictly comparable run

This reproduces the original judge identities and settings. Use the same
OpenBitFun credential used for the final SO-vs-RS audit.

```bash
export OPENBITFUN_API_KEY='YOUR_KEY'

python scripts/judge_rs_vs_repair_rationales.py \
  --output-dir outputs/analysis/rs_vs_repair_rationale_quality_bundle/results_openbitfun \
  --run-api
```

Defaults:

- primary judges: `glm-5.3-flash`, `doubao-seed-2.0-lite`;
- tiebreaker: `MiniMax-M3`;
- base URL: `https://api.openbitfun.com/v1`;
- temperature: 0;
- maximum judge completion: 4096 tokens;
- tiebreaker rule: unchanged from the SO-vs-RS audit.

Run a two-sample smoke test in a separate directory first:

```bash
python scripts/judge_rs_vs_repair_rationales.py \
  --output-dir outputs/analysis/rs_vs_repair_rationale_quality_bundle/smoke_openbitfun \
  --limit 2 \
  --run-api
```

## CSI alternative

This keeps the same samples, A/B sides, prompts, aggregation, and tiebreaker
rule, but changes judge identities. Its absolute scores must not be mixed with
the existing OpenBitFun SO-vs-RS scores. It is valid as a self-contained
RS-vs-RePaIR comparison. For a three-method CSI comparison, SO-vs-RS must also
be rerun with these CSI judges.

```bash
export CSI_API_KEY='YOUR_KEY'

python scripts/judge_rs_vs_repair_rationales.py \
  --output-dir outputs/analysis/rs_vs_repair_rationale_quality_bundle/results_csi \
  --base-url http://113.46.219.251:8080/v1 \
  --api-key-env CSI_API_KEY \
  --judge-models Qwen3.8-Max DeepSeek-V4-Pro \
  --tiebreaker-model GLM-5.3 \
  --run-api
```

Use a separate output directory for every model/configuration change. Results
are written after each sample, so rerunning the identical command resumes.
