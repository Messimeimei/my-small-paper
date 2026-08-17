# Qwen3-4B and SciRM-7B evaluation results

> Generated at 2026-08-14T17:13:27.849161+00:00; 141 deduplicated task/configuration/seed records are included.
> This file is rebuilt by `training/evaluate.py` after evaluation.

## 1. Reporting protocol

Included configurations: B-L, B-C, SciRM-L, SciRM-C, LL, LC, CL, CC, PAL, PAC, SCAL, SCAC, LL-R.

Tables follow the main-result layout of TRACT: training and inference configurations are listed on the left, tasks are expanded across columns, and the strict macro average is reported last. **Bold** marks the best result and <u>underline</u> marks the second-best distinct result in each column. Ties share a rank. MAE is ranked in ascending order; all other metrics are ranked in descending order.

`CoT` indicates whether the evaluation prompt requests an explicit rationale; it does not indicate hidden/internal model reasoning. Multi-seed cells report `mean +/- sample standard deviation`. Variances remain available in `evaluation_analysis_records.json`. An average is shown only when a configuration covers every task to which that metric applies; `—` means not applicable or unavailable.

Task sample counts: Actionability=1000; Grounding Specificity=1000; Helpfulness=1000; Verifiability=788; Coherence=1046; Positioning Check=603; Positioning Type=204.

## 2. Main results by primary metric

The primary metric is QWK for the four ordinal review-utility tasks and Macro-F1 for the three binary writing-quality tasks.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | base | 7/7 | 0.413 | 0.282 | 0.289 | 0.474 | 0.339 | 0.423 | 0.351 | 0.367 |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | base | 7/7 | 0.492 | 0.518 | 0.388 | 0.577 | 0.629 | 0.911 | 0.589 | 0.586 |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 0.693 | 0.286 | 0.573 | 0.571 | 0.576 | 0.869 | 0.594 | 0.595 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 0.750 | 0.469 | 0.578 | 0.660 | 0.660 | 0.861 | 0.683 | 0.666 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43 | 6/7 | 0.770 +/- 0.006 | 0.646 +/- 0.107 | — | 0.412 +/- 0.583 | 0.508 +/- 0.097 | 0.650 +/- 0.202 | 0.615 +/- 0.017 | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43 | 7/7 | 0.735 +/- 0.004 | 0.726 +/- 0.001 | 0.701 +/- 0.008 | 0.672 +/- 0.019 | 0.778 +/- 0.005 | <u>0.997 +/- 0.004</u> | 0.995 +/- 0.001 | 0.801 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43 | 7/7 | 0.764 +/- 0.005 | 0.686 +/- 0.049 | 0.669 +/- 0.002 | 0.659 +/- 0.046 | 0.726 +/- 0.001 | 0.920 +/- 0.087 | 0.843 +/- 0.041 | 0.752 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43 | 7/7 | 0.742 +/- 0.003 | 0.691 +/- 0.010 | 0.693 +/- 0.002 | 0.701 +/- 0.036 | 0.769 +/- 0.004 | 0.993 +/- 0.000 | <u>0.997 +/- 0.004</u> | 0.798 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43 | 7/7 | <u>0.781 +/- 0.004</u> | <u>0.748 +/- 0.011</u> | <u>0.727 +/- 0.019</u> | **0.770 +/- 0.002** | **0.806 +/- 0.025** | **0.998 +/- 0.000** | **1.000 +/- 0.000** | **0.833** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43 | 7/7 | 0.732 +/- 0.009 | 0.701 +/- 0.015 | 0.682 +/- 0.047 | 0.662 +/- 0.013 | 0.795 +/- 0.005 | 0.997 +/- 0.000 | **1.000 +/- 0.000** | 0.796 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 7/7 | **0.786 +/- 0.002** | **0.749 +/- 0.002** | **0.727 +/- 0.004** | <u>0.764 +/- 0.003</u> | <u>0.797 +/- 0.003</u> | 0.997 +/- 0.000 | **1.000 +/- 0.000** | <u>0.831</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43 | 7/7 | 0.764 +/- 0.003 | 0.724 +/- 0.004 | 0.726 +/- 0.004 | 0.729 +/- 0.011 | 0.779 +/- 0.001 | 0.996 +/- 0.001 | **1.000 +/- 0.000** | 0.817 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/7 | — | 0.739 | — | — | — | — | — | — |

*Table 1. Primary-metric results. Average requires complete coverage of all seven tasks.*

## 3. QWK

QWK applies to four ordinal tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | base | 4/4 | 0.413 | 0.282 | 0.289 | 0.474 | — | — | — | 0.364 |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | base | 4/4 | 0.492 | 0.518 | 0.388 | 0.577 | — | — | — | 0.494 |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 4/4 | 0.693 | 0.286 | 0.573 | 0.571 | — | — | — | 0.531 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 4/4 | 0.750 | 0.469 | 0.578 | 0.660 | — | — | — | 0.614 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43 | 3/4 | 0.770 +/- 0.006 | 0.646 +/- 0.107 | — | 0.412 +/- 0.583 | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43 | 4/4 | 0.735 +/- 0.004 | 0.726 +/- 0.001 | 0.701 +/- 0.008 | 0.672 +/- 0.019 | — | — | — | 0.708 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43 | 4/4 | 0.764 +/- 0.005 | 0.686 +/- 0.049 | 0.669 +/- 0.002 | 0.659 +/- 0.046 | — | — | — | 0.694 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43 | 4/4 | 0.742 +/- 0.003 | 0.691 +/- 0.010 | 0.693 +/- 0.002 | 0.701 +/- 0.036 | — | — | — | 0.707 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43 | 4/4 | <u>0.781 +/- 0.004</u> | <u>0.748 +/- 0.011</u> | <u>0.727 +/- 0.019</u> | **0.770 +/- 0.002** | — | — | — | <u>0.757</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43 | 4/4 | 0.732 +/- 0.009 | 0.701 +/- 0.015 | 0.682 +/- 0.047 | 0.662 +/- 0.013 | — | — | — | 0.694 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 4/4 | **0.786 +/- 0.002** | **0.749 +/- 0.002** | **0.727 +/- 0.004** | <u>0.764 +/- 0.003</u> | — | — | — | **0.757** |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43 | 4/4 | 0.764 +/- 0.003 | 0.724 +/- 0.004 | 0.726 +/- 0.004 | 0.729 +/- 0.011 | — | — | — | 0.736 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/4 | — | 0.739 | — | — | — | — | — | — |

*Table 2. QWK by training/inference configuration and task.*
## 4. Accuracy (%)

Accuracy (%) applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | base | 7/7 | 5.6 | 3.2 | 2.4 | 6.6 | 23.2 | 43.0 | 35.3 | 17.0 |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | base | 7/7 | 36.0 | 39.1 | 34.9 | 47.8 | 63.5 | 90.9 | 58.3 | 52.9 |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 53.0 | 50.7 | 52.2 | 56.3 | 61.8 | 86.9 | 59.3 | 60.0 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | <u>54.3</u> | 53.7 | 51.1 | **61.0** | 67.3 | 86.1 | 68.1 | 63.1 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43 | 7/7 | 52.5 +/- 2.0 | 55.6 +/- 10.9 | 0.0 +/- 0.0 | 11.3 +/- 15.6 | 47.9 +/- 10.0 | 55.1 +/- 17.6 | 52.5 +/- 1.4 | 39.3 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43 | 7/7 | 50.7 +/- 0.3 | 63.5 +/- 0.8 | 60.3 +/- 0.1 | 56.1 +/- 0.9 | 77.8 +/- 0.5 | <u>99.8 +/- 0.4</u> | 99.3 +/- 0.3 | 72.5 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43 | 7/7 | 51.8 +/- 0.6 | 56.0 +/- 0.1 | 55.8 +/- 1.7 | 44.5 +/- 3.2 | 71.8 +/- 0.5 | 87.0 +/- 15.8 | 84.8 +/- 4.2 | 64.5 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43 | 7/7 | 50.8 +/- 0.5 | 69.5 +/- 0.3 | 58.7 +/- 0.1 | 57.7 +/- 1.3 | 76.9 +/- 0.4 | 99.3 +/- 0.0 | <u>99.8 +/- 0.3</u> | 73.2 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43 | 7/7 | 53.4 +/- 0.2 | **71.2 +/- 0.6** | **61.1 +/- 0.7** | <u>61.0 +/- 1.3</u> | **80.6 +/- 2.5** | **99.8 +/- 0.0** | **100.0 +/- 0.0** | **75.3** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43 | 7/7 | 49.4 +/- 1.9 | 69.3 +/- 1.5 | 58.2 +/- 3.1 | 52.3 +/- 0.1 | 79.5 +/- 0.5 | 99.7 +/- 0.0 | **100.0 +/- 0.0** | 72.6 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 7/7 | **54.6 +/- 0.8** | <u>70.5 +/- 0.6</u> | <u>60.7 +/- 0.2</u> | 60.5 +/- 0.5 | <u>79.7 +/- 0.3</u> | 99.7 +/- 0.0 | **100.0 +/- 0.0** | <u>75.1</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43 | 7/7 | 52.8 +/- 0.1 | 70.4 +/- 0.5 | 60.3 +/- 0.6 | 59.8 +/- 1.5 | 77.9 +/- 0.1 | 99.6 +/- 0.1 | **100.0 +/- 0.0** | 74.4 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/7 | — | 66.5 | — | — | — | — | — | — |

*Table 3. Accuracy (%) by training/inference configuration and task.*
## 5. Pearson

Pearson applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | base | 6/7 | 0.532 | 0.382 | 0.429 | 0.556 | 0.266 | — | 0.158 | — |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | base | 7/7 | 0.546 | 0.571 | 0.451 | 0.582 | 0.284 | 0.841 | 0.425 | 0.528 |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 0.745 | 0.350 | 0.583 | 0.609 | 0.301 | 0.767 | 0.398 | 0.536 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 0.769 | 0.495 | 0.597 | 0.670 | 0.377 | 0.756 | 0.510 | 0.596 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43 | 6/7 | 0.773 +/- 0.005 | 0.656 +/- 0.109 | — | **0.831** | 0.297 +/- 0.084 | 0.867 +/- 0.144 | 0.573 +/- 0.024 | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43 | 7/7 | 0.753 +/- 0.001 | 0.732 +/- 0.000 | 0.710 +/- 0.006 | 0.695 +/- 0.009 | 0.556 +/- 0.010 | <u>0.995 +/- 0.007</u> | 0.994 +/- 0.008 | 0.776 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43 | 7/7 | 0.767 +/- 0.009 | 0.705 +/- 0.029 | 0.681 +/- 0.001 | 0.691 +/- 0.028 | 0.491 +/- 0.011 | 0.971 +/- 0.010 | 0.730 +/- 0.061 | 0.719 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43 | 7/7 | 0.747 +/- 0.002 | 0.721 +/- 0.004 | 0.710 +/- 0.004 | 0.723 +/- 0.031 | 0.538 +/- 0.008 | 0.987 +/- 0.000 | <u>0.995 +/- 0.008</u> | 0.774 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43 | 7/7 | <u>0.790 +/- 0.002</u> | <u>0.773 +/- 0.005</u> | 0.737 +/- 0.017 | <u>0.781 +/- 0.003</u> | **0.615 +/- 0.048** | **0.997 +/- 0.000** | **1.000 +/- 0.000** | **0.813** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43 | 7/7 | 0.738 +/- 0.008 | 0.736 +/- 0.019 | 0.689 +/- 0.047 | 0.684 +/- 0.012 | 0.590 +/- 0.009 | 0.993 +/- 0.000 | **1.000 +/- 0.000** | 0.776 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 7/7 | **0.793 +/- 0.001** | **0.776 +/- 0.001** | **0.742 +/- 0.000** | 0.774 +/- 0.001 | <u>0.595 +/- 0.006</u> | 0.993 +/- 0.000 | **1.000 +/- 0.000** | <u>0.811</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43 | 7/7 | 0.768 +/- 0.003 | 0.752 +/- 0.006 | <u>0.738 +/- 0.003</u> | 0.742 +/- 0.006 | 0.559 +/- 0.003 | 0.992 +/- 0.002 | **1.000 +/- 0.000** | 0.793 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/7 | — | 0.769 | — | — | — | — | — | — |

*Table 4. Pearson by training/inference configuration and task.*
## 6. Macro-F1

Macro-F1 applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | base | 7/7 | 0.082 | 0.048 | 0.044 | 0.090 | 0.339 | 0.423 | 0.351 | 0.196 |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | base | 7/7 | 0.333 | 0.317 | 0.286 | 0.426 | 0.629 | 0.911 | 0.589 | 0.499 |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 0.456 | 0.236 | 0.322 | 0.398 | 0.576 | 0.869 | 0.594 | 0.493 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 0.474 | 0.319 | 0.392 | 0.427 | 0.660 | 0.861 | 0.683 | 0.545 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43 | 7/7 | 0.449 +/- 0.014 | 0.341 +/- 0.085 | 0.000 +/- 0.000 | 0.166 +/- 0.230 | 0.508 +/- 0.097 | 0.650 +/- 0.202 | 0.615 +/- 0.017 | 0.390 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43 | 7/7 | 0.423 +/- 0.007 | 0.468 +/- 0.024 | 0.500 +/- 0.006 | 0.487 +/- 0.000 | 0.778 +/- 0.005 | <u>0.997 +/- 0.004</u> | 0.995 +/- 0.001 | 0.664 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43 | 7/7 | 0.462 +/- 0.008 | 0.360 +/- 0.008 | 0.459 +/- 0.014 | 0.404 +/- 0.029 | 0.726 +/- 0.001 | 0.920 +/- 0.087 | 0.843 +/- 0.041 | 0.596 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43 | 7/7 | 0.467 +/- 0.003 | 0.494 +/- 0.002 | 0.497 +/- 0.001 | 0.482 +/- 0.012 | 0.769 +/- 0.004 | 0.993 +/- 0.000 | <u>0.997 +/- 0.004</u> | 0.671 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43 | 7/7 | <u>0.501 +/- 0.005</u> | **0.518 +/- 0.024** | 0.503 +/- 0.044 | **0.556 +/- 0.015** | **0.806 +/- 0.025** | **0.998 +/- 0.000** | **1.000 +/- 0.000** | <u>0.698</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43 | 7/7 | 0.443 +/- 0.013 | 0.466 +/- 0.014 | 0.469 +/- 0.035 | 0.442 +/- 0.007 | 0.795 +/- 0.005 | 0.997 +/- 0.000 | **1.000 +/- 0.000** | 0.659 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 7/7 | **0.516 +/- 0.009** | 0.508 +/- 0.003 | **0.536 +/- 0.008** | <u>0.544 +/- 0.015</u> | <u>0.797 +/- 0.003</u> | 0.997 +/- 0.000 | **1.000 +/- 0.000** | **0.700** |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43 | 7/7 | 0.476 +/- 0.005 | <u>0.509 +/- 0.005</u> | <u>0.528 +/- 0.003</u> | 0.514 +/- 0.016 | 0.779 +/- 0.001 | 0.996 +/- 0.001 | **1.000 +/- 0.000** | 0.686 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/7 | — | 0.472 | — | — | — | — | — | — |

*Table 5. Macro-F1 by training/inference configuration and task.*
## 7. MAE

MAE applies to four ordinal tasks; lower is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | base | 4/4 | 1.069 | 1.043 | 1.000 | 0.767 | — | — | — | 0.970 |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | base | 4/4 | 0.905 | 0.997 | 0.830 | 0.675 | — | — | — | 0.852 |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 4/4 | 0.677 | 0.893 | 0.513 | 0.572 | — | — | — | 0.664 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 4/4 | 0.635 | 0.794 | 0.535 | 0.519 | — | — | — | 0.621 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43 | 3/4 | 0.607 +/- 0.007 | 0.529 +/- 0.019 | — | 0.903 +/- 0.609 | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43 | 4/4 | 0.688 +/- 0.002 | 0.603 +/- 0.002 | 0.423 +/- 0.002 | 0.588 +/- 0.005 | — | — | — | 0.575 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43 | 4/4 | 0.613 +/- 0.007 | 0.623 +/- 0.031 | 0.464 +/- 0.019 | 0.604 +/- 0.029 | — | — | — | 0.576 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43 | 4/4 | 0.633 +/- 0.008 | 0.535 +/- 0.011 | 0.434 +/- 0.001 | 0.542 +/- 0.043 | — | — | — | 0.536 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43 | 4/4 | <u>0.577 +/- 0.015</u> | **0.469 +/- 0.012** | **0.400 +/- 0.008** | **0.461 +/- 0.013** | — | — | — | **0.477** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43 | 4/4 | 0.653 +/- 0.037 | 0.528 +/- 0.025 | 0.440 +/- 0.044 | 0.617 +/- 0.003 | — | — | — | 0.560 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 4/4 | **0.563 +/- 0.007** | <u>0.472 +/- 0.005</u> | <u>0.408 +/- 0.001</u> | <u>0.468 +/- 0.007</u> | — | — | — | <u>0.478</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43 | 4/4 | 0.593 +/- 0.005 | 0.501 +/- 0.004 | 0.409 +/- 0.008 | 0.499 +/- 0.019 | — | — | — | 0.500 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/4 | — | 0.503 | — | — | — | — | — | — |

*Table 6. MAE by training/inference configuration and task.*

## 8. Cross-task method averages

| Id | Model | CoT | Train | Prompt | Inf. | Coverage | Average QWK | Average Accuracy (%) | Average Pearson | Average Macro-F1 | Average MAE |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | 7/7 | 0.364 (n=4) | 17.0 (n=7) | 0.387 (n=6) | 0.196 (n=7) | 0.970 (n=4) |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | 7/7 | 0.494 (n=4) | 52.9 (n=7) | 0.528 (n=7) | 0.499 (n=7) | 0.852 (n=4) |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | 7/7 | 0.531 (n=4) | 60.0 (n=7) | 0.536 (n=7) | 0.493 (n=7) | 0.664 (n=4) |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | 7/7 | 0.614 (n=4) | 63.1 (n=7) | 0.596 (n=7) | 0.545 (n=7) | 0.621 (n=4) |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 7/7 | 0.609 (n=3) | 39.3 (n=7) | 0.666 (n=6) | 0.390 (n=7) | 0.680 (n=3) |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 7/7 | 0.708 (n=4) | 72.5 (n=7) | 0.776 (n=7) | 0.664 (n=7) | 0.575 (n=4) |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 7/7 | 0.694 (n=4) | 64.5 (n=7) | 0.719 (n=7) | 0.596 (n=7) | 0.576 (n=4) |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 7/7 | 0.707 (n=4) | 73.2 (n=7) | 0.774 (n=7) | 0.671 (n=7) | 0.536 (n=4) |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 7/7 | <u>0.757 (n=4)</u> | **75.3 (n=7)** | **0.813 (n=7)** | <u>0.698 (n=7)</u> | **0.477 (n=4)** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 7/7 | 0.694 (n=4) | 72.6 (n=7) | 0.776 (n=7) | 0.659 (n=7) | 0.560 (n=4) |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 7/7 | **0.757 (n=4)** | <u>75.1 (n=7)</u> | <u>0.811 (n=7)</u> | **0.700 (n=7)** | <u>0.478 (n=4)</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 7/7 | 0.736 (n=4) | 74.4 (n=7) | 0.793 (n=7) | 0.686 (n=7) | 0.500 (n=4) |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 1/7 | 0.739 (n=1) | 66.5 (n=1) | 0.769 (n=1) | 0.472 (n=1) | 0.503 (n=1) |


## 9. Rebuild report

```bash
python training/evaluate.py --refresh-analysis-only --output_path eval_output/results
```
