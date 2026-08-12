# Paper Analysis

Use the isolated environment at `../.venv-paper`; do not install these packages
into the training environment.

Activate it from `paper_workspace/` with:

```bash
source analysis/activate.sh
```

Canonical aggregate input:

```text
../eval_output/results/evaluation_analysis_records.json
```

This JSON is an object with a `records` array, not a top-level array. Its cached
absolute `metrics_path` values currently use an obsolete `/root/my-small-paper`
prefix. Scripts must resolve current paths with `pathlib` from the repository
project root, task, and experiment name.

Planned scripts:

```text
scripts/build_task_table.py
scripts/build_main_results.py
scripts/build_paired_transitions.py
scripts/build_error_decomposition.py
scripts/build_raft_diagnostics.py
```

Every script should write machine-readable data to `derived/` before writing a
LaTeX table or PDF figure. Never derive a manuscript claim directly from a plot.
