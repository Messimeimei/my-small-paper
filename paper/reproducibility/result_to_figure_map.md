# Result-to-Figure Map

| Manuscript artifact | Generator | Raw inputs | Derived data | Verification status |
|---|---|---|---|---|
| Table 1: Tasks | `analysis/scripts/build_task_table.py` | `../data/README.md`, task data | `analysis/derived/task_table.csv` | pending |
| Table 2: Main results | `analysis/scripts/build_main_results.py` | metrics and aggregate records | `analysis/derived/main_results.csv` | pending |
| Figure 2: Paired transitions | `analysis/scripts/build_paired_transitions.py` | predictions JSONL | `analysis/derived/paired_transitions.csv` | pending |
| Figure 3: Error decomposition | `analysis/scripts/build_error_decomposition.py` | predictions JSONL | `analysis/derived/error_decomposition.csv` | pending |
| Figure 4: RAFT diagnostics | `analysis/scripts/build_raft_diagnostics.py` | RAFT predictions JSONL | `analysis/derived/raft_diagnostics.csv` | pending |
