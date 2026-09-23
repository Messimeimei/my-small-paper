# Qwen3-0.6B Align self-generated data

This directory is reserved for regenerated CoT training data from the completed
Qwen3-0.6B seed42 paper-align adapters.

The generator replaces only reasoning in cot/train_*.jsonl, restores the final
score from the gold label, and copies test data, label-only data, and split files
unchanged. Generation caches and provenance manifests are written under .runs/.

Prepared tasks:

- rev_util_actionability
- rev_util_grounding_specificity
- rev_util_helpfulness
- rev_util_verifiability

One-click tmux generation:

    bash scripts/launch_qwen06b_self_data_tmux.sh

Dry-run:

    python scripts/run_qwen06b_self_data.py --variants align --dry-run
