# SO vs RS rationale evaluation with CSI judges

This directory is self-contained. It uses the exact final SO-vs-RS evaluation
data and protocol:

- Qwen3-4B seed 42 outputs;
- 50 representative samples per task, seven tasks, 350 pairs total;
- sample seed 20260824;
- unchanged A/B assignments;
- round 1: four-dimensional rationale quality (1--3);
- round 2: rationale-predicted-score consistency;
- no gold label, condition name, correctness, or training method in judge prompts;
- unchanged disagreement and tiebreaker rules;
- temperature 0 and at most 8192 completion tokens.

CSI judges:

- primary: `Qwen3.8-Max` and `DeepSeek-V4-Pro`;
- disagreement judge: `GLM-5.3`;
- base URL: `http://113.46.219.251:8080/v1`.

No API key is included.

## Files

- `so_vs_rs_matched_350.jsonl`: all evaluation inputs.
- `data_manifest.json`: source hashes and exact inherited configuration.
- `judge_so_vs_rs_csi.py`: command-line entry point.
- `judge_protocol.py`: final SO-vs-RS prompts and parsers, with a transport-only
  extension for optional reasoning request fields.
- `model_request_options.json`: verified per-model thinking configuration.
- `probe_csi_models.py`: model metadata, version, and thinking-control probe.
- `model_request_options.json`: per-model verified thinking fields.
- `MODEL_VERSIONS.md`: verified model-version limits and reasoning guidance.

## Run

Run from a company-network machine whose public IP is allowed by CSI:

```bash
cd so_vs_rs_csi_quality_bundle
export CSI_API_KEY='YOUR_KEY'
```

First run two samples in a separate directory:

```bash
python judge_so_vs_rs_csi.py \
  --output-dir smoke \
  --limit 2 \
  --run-api
```

The default sends these model-specific settings:

- `Qwen3.8-Max`: `enable_thinking=true`, `thinking_budget=8192`;
- `DeepSeek-V4-Pro`: `thinking.type=enabled`, `reasoning_effort=high`;
- `GLM-5.3`: `thinking.type=enabled`, `reasoning_effort=high`.

All three use `max_tokens=8192`. DeepSeek documents that temperature is ignored
in thinking mode; it remains present at 0 to preserve the original audit
request shape for the other judges.

On the company network, probe the aliases and options before the smoke test:

```bash
python probe_csi_models.py --reasoning-matrix
```

The effective per-model options and `judge_max_tokens=8192` are saved in every
run manifest. Use `--disable-model-options` only for a provider-default control.
Use `--extra-request-json` only for CSI-confirmed global fields. See
`MODEL_VERSIONS.md` for version and parameter evidence.

Then run all 350 pairs:

```bash
python judge_so_vs_rs_csi.py \
  --output-dir results \
  --run-api
```

The script writes each completed sample immediately. Rerunning the identical
command resumes from `judge_results.jsonl`.

Final files include:

- `results/manifest.json`
- `results/judge_results.jsonl`
- `results/judge_failures.jsonl`
- `results/analysis.json`
- `results/analysis.md`
- `results/raw_responses/`

Keep `results/` together when transferring it back to AutoDL.
