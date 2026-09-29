# Publication Figures

Store only manuscript-ready figure outputs here, preferably vector PDF.
Every figure must have a reproducible generator under `analysis/scripts/` or
a standalone TikZ source in this directory, and must be mapped to its inputs in
`reproducibility/result_to_figure_map.md`.

## RS/SO training templates

- TikZ source: `rs_so_training_templates.tex`
- Manuscript vector: `rs_so_training_templates.pdf`

Rebuild from the repository root without Python:

```bash
make -C paper figure-rs-so
```

Suggested caption: *Prompt templates for rationale-supervised (RS) and
score-only (SO) fine-tuning. Shared instructions and the common user prompt
are shown once; colored labels identify only the mode-specific system-prompt
clauses. User-prompt content is abridged for readability.*

## Matched evaluation prompts

- TikZ source: `scirm_evaluation_example.tex`
- Manuscript vector: `scirm_evaluation_example.pdf`
- Prompt source: `../../../data/rev_util_actionability/cot/test_cot.jsonl`
  and its label-only counterpart (`actionability_test_0669`, gold label 5)
- DS output: `../../../outputs/evaluations/rev_util_actionability/rev_util_actionability#qwen3_4b#base#greedy#on_label_only#seed_base/predictions.jsonl`
- RF output: `../../../outputs/evaluations/rev_util_actionability/rev_util_actionability#qwen3_4b#base#greedy#on_cot#seed_base/predictions.jsonl`
- Layout reference: Figure 1 of *Reward Modeling for Scientific Writing
  Evaluation* (`../../literature/Reward Modeling for Scientific Writing Evaluation.pdf`)

Rebuild from the repository root without Python:

```bash
make -C paper figure-scirm
```
