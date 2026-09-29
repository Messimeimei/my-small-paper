# SciRM-7B multi-seed evaluation

Generated at `2026-09-29T05:32:41.329879+00:00`.
Model: `UKPLab/SciRM-7B` at revision `d0475b1725de05287d7e0b4a72a741ea6e452b6a`.

Primary metric: QWK for the four review-utility tasks and Macro-F1 for the three writing-quality tasks. Higher is better.

| Task | Interface | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Actionability | label_only | 0.6933 | 0.6955 | 0.6937 | 0.6942 +/- 0.0012 |
| Actionability | cot | 0.7496 | 0.7505 | 0.7460 | 0.7487 +/- 0.0024 |
| Grounding Specificity | label_only | 0.2858 | 0.2885 | 0.2844 | 0.2862 +/- 0.0021 |
| Grounding Specificity | cot | 0.4692 | 0.4907 | 0.4709 | 0.4770 +/- 0.0119 |
| Helpfulness | label_only | 0.5729 | 0.5765 | 0.5804 | 0.5766 +/- 0.0038 |
| Helpfulness | cot | 0.5784 | 0.5827 | 0.5803 | 0.5805 +/- 0.0022 |
| Verifiability | label_only | 0.5709 | 0.5654 | 0.5644 | 0.5669 +/- 0.0035 |
| Verifiability | cot | 0.6600 | 0.6775 | 0.6649 | 0.6675 +/- 0.0090 |
| Coherence | label_only | 0.5764 | 0.5751 | 0.5751 | 0.5756 +/- 0.0008 |
| Coherence | cot | 0.6598 | 0.6491 | 0.6612 | 0.6567 +/- 0.0067 |
| Positioning Check | label_only | 0.8690 | 0.8657 | 0.8673 | 0.8673 +/- 0.0017 |
| Positioning Check | cot | 0.8614 | 0.8581 | 0.8624 | 0.8606 +/- 0.0022 |
| Positioning Type | label_only | 0.5938 | 0.5887 | 0.5732 | 0.5853 +/- 0.0107 |
| Positioning Type | cot | 0.6829 | 0.6910 | 0.6716 | 0.6818 +/- 0.0098 |

## Cross-task primary average

| Interface | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| label_only | 0.5946 | 0.5936 | 0.5912 | 0.5932 +/- 0.0017 |
| cot | 0.6659 | 0.6714 | 0.6653 | 0.6675 +/- 0.0033 |
