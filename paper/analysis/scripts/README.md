# Analysis Scripts

Each script must:

1. accept paths relative to `paper_workspace/`;
2. validate sample IDs and duplicate IDs;
3. fail on missing conditions rather than silently dropping them;
4. write the derived CSV/JSON used by the manuscript;
5. record its input files and output files in
   `reproducibility/result_to_figure_map.md`.
