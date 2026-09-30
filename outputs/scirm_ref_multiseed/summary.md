# SciRM-Ref-7B multi-seed evaluation

Generated at `2026-09-29T07:20:21.531842+00:00`.
Model: `UKPLab/SciRM-Ref-7B` at revision `ba3656c87660440b8b4377055ccbe11f5fa60f5a`.

Primary metric: QWK for the four review-utility tasks and Macro-F1 for the three writing-quality tasks. Higher is better.

| Task | Interface | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Actionability | label_only | 0.6438 | 0.6430 | 0.6423 | 0.6430 +/- 0.0008 |
| Actionability | cot | 0.6852 | 0.6859 | 0.6850 | 0.6854 +/- 0.0005 |
| Grounding Specificity | label_only | -0.0017 | -0.0045 | -0.0045 | -0.0035 +/- 0.0016 |
| Grounding Specificity | cot | 0.2436 | 0.2583 | 0.2420 | 0.2480 +/- 0.0090 |
| Helpfulness | label_only | 0.5764 | 0.5786 | 0.5786 | 0.5778 +/- 0.0013 |
| Helpfulness | cot | 0.5886 | 0.5958 | 0.5985 | 0.5943 +/- 0.0051 |
| Verifiability | label_only | 0.4233 | 0.4209 | 0.4207 | 0.4216 +/- 0.0014 |
| Verifiability | cot | 0.6137 | 0.6104 | 0.5917 | 0.6053 +/- 0.0119 |
| Coherence | label_only | 0.6100 | 0.6087 | 0.6112 | 0.6100 +/- 0.0012 |
| Coherence | cot | 0.6378 | 0.6379 | 0.6365 | 0.6374 +/- 0.0008 |
| Positioning Check | label_only | 0.9425 | 0.9409 | 0.9409 | 0.9414 +/- 0.0009 |
| Positioning Check | cot | 0.9466 | 0.9483 | 0.9483 | 0.9477 +/- 0.0010 |
| Positioning Type | label_only | 0.5590 | 0.5695 | 0.5669 | 0.5651 +/- 0.0055 |
| Positioning Type | cot | 0.6680 | 0.6664 | 0.6471 | 0.6605 +/- 0.0117 |

## Cross-task primary average

| Interface | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| label_only | 0.5362 | 0.5367 | 0.5366 | 0.5365 +/- 0.0003 |
| cot | 0.6262 | 0.6290 | 0.6213 | 0.6255 +/- 0.0039 |
