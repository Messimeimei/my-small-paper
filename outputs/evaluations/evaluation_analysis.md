# Qwen3-4B and SciRM-7B evaluation results

> Generated at 2026-09-03T00:45:08.622209+00:00; 293 deduplicated task/configuration/seed records are included.
> This file is rebuilt by `scripts/evaluate.py` after evaluation.

## 1. Reporting protocol

Included configurations: B-L, B-C, SciRM-L, SciRM-C, LL, LC, CL, CC, PAL, PAC, MIX-L, MIX-C, SSAL, SSAC, SSA2L, SSA2C, SCL, SCC, SCAL, SCAC, LL-R.

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
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.775 +/- 0.010 | 0.672 +/- 0.088 | 0.717 | 0.526 +/- 0.457 | 0.600 +/- 0.174 | 0.766 +/- 0.247 | 0.743 +/- 0.223 | 0.686 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.742 +/- 0.013 | 0.724 +/- 0.004 | 0.704 +/- 0.007 | 0.680 +/- 0.020 | 0.773 +/- 0.008 | 0.997 +/- 0.003 | 0.997 +/- 0.003 | 0.802 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.763 +/- 0.004 | 0.687 +/- 0.034 | 0.676 +/- 0.013 | 0.659 +/- 0.032 | 0.732 +/- 0.011 | 0.942 +/- 0.072 | 0.854 +/- 0.035 | 0.759 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.730 +/- 0.021 | 0.681 +/- 0.019 | 0.683 +/- 0.017 | 0.690 +/- 0.031 | 0.769 +/- 0.003 | 0.994 +/- 0.001 | <u>0.998 +/- 0.003</u> | 0.792 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.783 +/- 0.004 | <u>0.749 +/- 0.008</u> | <u>0.727 +/- 0.013</u> | <u>0.768 +/- 0.004</u> | **0.804 +/- 0.018** | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | **0.833** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.735 +/- 0.008 | 0.706 +/- 0.013 | 0.682 +/- 0.033 | 0.667 +/- 0.013 | 0.792 +/- 0.006 | 0.996 +/- 0.001 | **1.000 +/- 0.000** | 0.797 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.790 +/- 0.009** | 0.723 +/- 0.008 | 0.714 +/- 0.005 | 0.740 +/- 0.003 | 0.776 +/- 0.008 | 0.998 +/- 0.002 | **1.000 +/- 0.000** | 0.820 |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 0.728 +/- 0.007 | 0.678 +/- 0.003 | 0.678 +/- 0.012 | 0.674 +/- 0.009 | 0.781 +/- 0.003 | 0.996 +/- 0.002 | **1.000 +/- 0.000** | 0.791 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 0.759 +/- 0.003 | 0.611 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 0.721 +/- 0.002 | 0.674 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 0.719 +/- 0.108 | 0.748 +/- 0.007 | 0.627 +/- 0.129 | 0.745 +/- 0.052 | 0.739 +/- 0.066 | **1.000 +/- 0.000** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | 0.767 +/- 0.011 | 0.742 +/- 0.003 | 0.712 +/- 0.007 | 0.714 +/- 0.030 | <u>0.799 +/- 0.006</u> | 0.998 +/- 0.000 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 44 | 6/7 | 0.661 | 0.662 | — | **0.934** | 0.517 | 0.499 | 0.400 | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 44 | 7/7 | 0.724 | 0.699 | 0.688 | 0.721 | 0.773 | 0.995 | **1.000** | 0.800 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.786 +/- 0.002</u> | **0.751 +/- 0.004** | **0.729 +/- 0.003** | 0.764 +/- 0.002 | 0.798 +/- 0.002 | 0.996 +/- 0.001 | **1.000 +/- 0.000** | <u>0.832</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.757 +/- 0.012 | 0.726 +/- 0.005 | 0.724 +/- 0.005 | 0.732 +/- 0.010 | 0.783 +/- 0.006 | 0.995 +/- 0.002 | **1.000 +/- 0.000** | 0.817 |
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
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.775 +/- 0.010 | 0.672 +/- 0.088 | 0.717 | 0.526 +/- 0.457 | — | — | — | 0.673 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.742 +/- 0.013 | 0.724 +/- 0.004 | 0.704 +/- 0.007 | 0.680 +/- 0.020 | — | — | — | 0.712 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.763 +/- 0.004 | 0.687 +/- 0.034 | 0.676 +/- 0.013 | 0.659 +/- 0.032 | — | — | — | 0.696 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.730 +/- 0.021 | 0.681 +/- 0.019 | 0.683 +/- 0.017 | 0.690 +/- 0.031 | — | — | — | 0.696 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.783 +/- 0.004 | <u>0.749 +/- 0.008</u> | <u>0.727 +/- 0.013</u> | <u>0.768 +/- 0.004</u> | — | — | — | <u>0.757</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.735 +/- 0.008 | 0.706 +/- 0.013 | 0.682 +/- 0.033 | 0.667 +/- 0.013 | — | — | — | 0.698 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 4/4 | **0.790 +/- 0.009** | 0.723 +/- 0.008 | 0.714 +/- 0.005 | 0.740 +/- 0.003 | — | — | — | 0.742 |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 4/4 | 0.728 +/- 0.007 | 0.678 +/- 0.003 | 0.678 +/- 0.012 | 0.674 +/- 0.009 | — | — | — | 0.690 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/4 | 0.759 +/- 0.003 | 0.611 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/4 | 0.721 +/- 0.002 | 0.674 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.719 +/- 0.108 | 0.748 +/- 0.007 | 0.627 +/- 0.129 | 0.745 +/- 0.052 | — | — | — | 0.710 |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.767 +/- 0.011 | 0.742 +/- 0.003 | 0.712 +/- 0.007 | 0.714 +/- 0.030 | — | — | — | 0.734 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 44 | 3/4 | 0.661 | 0.662 | — | **0.934** | — | — | — | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 44 | 4/4 | 0.724 | 0.699 | 0.688 | 0.721 | — | — | — | 0.708 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | <u>0.786 +/- 0.002</u> | **0.751 +/- 0.004** | **0.729 +/- 0.003** | 0.764 +/- 0.002 | — | — | — | **0.758** |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.757 +/- 0.012 | 0.726 +/- 0.005 | 0.724 +/- 0.005 | 0.732 +/- 0.010 | — | — | — | 0.735 |
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
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 54.3 | 53.7 | 51.1 | <u>61.0</u> | 67.3 | 86.1 | 68.1 | 63.1 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 53.0 +/- 1.6 | 60.4 +/- 11.4 | 20.2 +/- 35.0 | 27.9 +/- 30.8 | 58.1 +/- 19.0 | 70.0 +/- 28.7 | 68.3 +/- 27.5 | 51.1 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 51.9 +/- 2.1 | 64.0 +/- 1.0 | 60.0 +/- 0.5 | 56.0 +/- 0.7 | 77.3 +/- 0.9 | 99.7 +/- 0.3 | 99.5 +/- 0.5 | 72.6 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 52.1 +/- 0.6 | 57.2 +/- 2.0 | 56.1 +/- 1.3 | 45.1 +/- 2.5 | 72.7 +/- 1.4 | 90.8 +/- 13.0 | 85.9 +/- 3.5 | 65.7 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 49.7 +/- 2.0 | 68.8 +/- 1.3 | 57.6 +/- 2.0 | 55.8 +/- 3.6 | 76.9 +/- 0.3 | 99.4 +/- 0.1 | <u>99.8 +/- 0.3</u> | 72.6 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 53.6 +/- 0.5 | **70.9 +/- 0.6** | <u>61.1 +/- 0.5</u> | **61.3 +/- 1.1** | **80.4 +/- 1.8** | <u>99.9 +/- 0.1</u> | **100.0 +/- 0.0** | **75.3** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 49.6 +/- 1.4 | 69.7 +/- 1.2 | 57.7 +/- 2.3 | 53.1 +/- 1.4 | 79.2 +/- 0.6 | 99.6 +/- 0.1 | **100.0 +/- 0.0** | 72.7 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **54.6 +/- 1.2** | 70.3 +/- 0.5 | 59.2 +/- 0.4 | 59.7 +/- 0.7 | 77.6 +/- 0.8 | 99.8 +/- 0.2 | **100.0 +/- 0.0** | 74.4 |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 49.2 +/- 1.3 | 68.6 +/- 0.4 | 56.6 +/- 1.1 | 53.4 +/- 0.9 | 78.1 +/- 0.3 | 99.6 +/- 0.2 | **100.0 +/- 0.0** | 72.2 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 51.4 +/- 0.5 | 44.0 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 48.5 +/- 1.5 | 68.6 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 41.4 +/- 13.6 | 70.1 +/- 0.6 | 46.5 +/- 15.8 | 48.1 +/- 22.4 | 70.5 +/- 10.1 | **100.0 +/- 0.0** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | 52.4 +/- 1.2 | 70.5 +/- 0.5 | 59.8 +/- 0.7 | 55.9 +/- 2.2 | <u>79.9 +/- 0.6</u> | 99.8 +/- 0.0 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 44 | 7/7 | 29.2 | 51.6 | 0.0 | 1.8 | 50.3 | 40.1 | 33.8 | 29.5 |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 44 | 7/7 | 49.3 | 70.2 | 58.1 | 60.8 | 77.3 | 99.5 | **100.0** | 73.6 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>54.4 +/- 0.7</u> | <u>70.6 +/- 0.5</u> | **61.1 +/- 0.7** | 61.0 +/- 0.9 | 79.8 +/- 0.2 | 99.6 +/- 0.1 | **100.0 +/- 0.0** | <u>75.2</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 52.4 +/- 0.8 | 70.5 +/- 0.4 | 60.4 +/- 0.5 | 59.7 +/- 1.1 | 78.3 +/- 0.6 | 99.5 +/- 0.2 | **100.0 +/- 0.0** | 74.4 |
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
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.780 +/- 0.013 | 0.689 +/- 0.097 | 0.733 | <u>0.796 +/- 0.049</u> | 0.388 +/- 0.168 | 0.910 +/- 0.126 | 0.715 +/- 0.247 | 0.716 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.759 +/- 0.011 | 0.730 +/- 0.004 | 0.711 +/- 0.005 | 0.700 +/- 0.011 | 0.548 +/- 0.017 | 0.994 +/- 0.005 | 0.996 +/- 0.006 | 0.777 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.766 +/- 0.006 | 0.706 +/- 0.021 | 0.685 +/- 0.006 | 0.695 +/- 0.021 | 0.498 +/- 0.014 | 0.971 +/- 0.007 | 0.746 +/- 0.052 | 0.724 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.735 +/- 0.021 | 0.715 +/- 0.010 | 0.697 +/- 0.023 | 0.712 +/- 0.029 | 0.538 +/- 0.006 | 0.988 +/- 0.002 | <u>0.996 +/- 0.006</u> | 0.768 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.792 +/- 0.003 | <u>0.774 +/- 0.004</u> | <u>0.739 +/- 0.013</u> | 0.779 +/- 0.004 | **0.609 +/- 0.035** | <u>0.998 +/- 0.002</u> | **1.000 +/- 0.000** | **0.813** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.741 +/- 0.007 | 0.741 +/- 0.015 | 0.690 +/- 0.033 | 0.689 +/- 0.012 | 0.584 +/- 0.012 | 0.992 +/- 0.002 | **1.000 +/- 0.000** | 0.777 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.796 +/- 0.007** | 0.748 +/- 0.008 | 0.728 +/- 0.005 | 0.755 +/- 0.004 | 0.552 +/- 0.016 | 0.996 +/- 0.004 | **1.000 +/- 0.000** | 0.796 |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 0.733 +/- 0.006 | 0.716 +/- 0.002 | 0.687 +/- 0.011 | 0.695 +/- 0.010 | 0.563 +/- 0.006 | 0.992 +/- 0.004 | **1.000 +/- 0.000** | 0.769 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 0.764 +/- 0.004 | 0.675 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 0.728 +/- 0.005 | 0.715 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 0.729 +/- 0.108 | 0.774 +/- 0.007 | 0.641 +/- 0.128 | 0.750 +/- 0.055 | <u>0.599 +/- 0.004</u> | **1.000 +/- 0.000** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | 0.774 +/- 0.009 | 0.769 +/- 0.004 | 0.725 +/- 0.004 | 0.727 +/- 0.029 | 0.598 +/- 0.011 | 0.997 +/- 0.000 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 44 | 6/7 | 0.671 | 0.672 | — | **0.935** | 0.311 | 0.867 | 0.131 | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 44 | 7/7 | 0.729 | 0.731 | 0.706 | 0.737 | 0.547 | 0.990 | **1.000** | 0.777 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.794 +/- 0.001</u> | **0.778 +/- 0.004** | **0.743 +/- 0.001** | 0.774 +/- 0.001 | 0.596 +/- 0.005 | 0.992 +/- 0.002 | **1.000 +/- 0.000** | <u>0.811</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.761 +/- 0.012 | 0.754 +/- 0.005 | 0.737 +/- 0.004 | 0.744 +/- 0.005 | 0.565 +/- 0.012 | 0.990 +/- 0.003 | **1.000 +/- 0.000** | 0.793 |
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
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.466 +/- 0.032 | 0.390 +/- 0.104 | 0.174 +/- 0.302 | 0.294 +/- 0.275 | 0.600 +/- 0.174 | 0.766 +/- 0.247 | 0.743 +/- 0.223 | 0.490 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.433 +/- 0.019 | 0.469 +/- 0.017 | 0.503 +/- 0.007 | 0.487 +/- 0.001 | 0.773 +/- 0.008 | 0.997 +/- 0.003 | 0.997 +/- 0.003 | 0.666 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.460 +/- 0.007 | 0.368 +/- 0.014 | 0.458 +/- 0.010 | 0.407 +/- 0.021 | 0.732 +/- 0.011 | 0.942 +/- 0.072 | 0.854 +/- 0.035 | 0.603 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.457 +/- 0.017 | 0.479 +/- 0.025 | 0.481 +/- 0.028 | 0.469 +/- 0.023 | 0.769 +/- 0.003 | 0.994 +/- 0.001 | <u>0.998 +/- 0.003</u> | 0.664 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.506 +/- 0.008 | <u>0.517 +/- 0.017</u> | 0.507 +/- 0.032 | **0.557 +/- 0.011** | **0.804 +/- 0.018** | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | <u>0.698</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.449 +/- 0.013 | 0.464 +/- 0.011 | 0.467 +/- 0.025 | 0.446 +/- 0.008 | 0.792 +/- 0.006 | 0.996 +/- 0.001 | **1.000 +/- 0.000** | 0.659 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.510 +/- 0.014</u> | **0.518 +/- 0.007** | 0.511 +/- 0.008 | 0.507 +/- 0.005 | 0.776 +/- 0.008 | 0.998 +/- 0.002 | **1.000 +/- 0.000** | 0.688 |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 0.454 +/- 0.016 | 0.460 +/- 0.017 | 0.461 +/- 0.017 | 0.466 +/- 0.016 | 0.781 +/- 0.003 | 0.996 +/- 0.002 | **1.000 +/- 0.000** | 0.660 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 0.457 +/- 0.006 | 0.314 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 0.449 +/- 0.015 | 0.473 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 0.424 +/- 0.101 | 0.506 +/- 0.011 | 0.396 +/- 0.142 | 0.483 +/- 0.129 | 0.739 +/- 0.066 | **1.000 +/- 0.000** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | 0.491 +/- 0.017 | 0.505 +/- 0.010 | 0.517 +/- 0.007 | 0.507 +/- 0.024 | <u>0.799 +/- 0.006</u> | 0.998 +/- 0.000 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 44 | 7/7 | 0.286 | 0.350 | 0.000 | 0.070 | 0.517 | 0.499 | 0.400 | 0.303 |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 44 | 7/7 | 0.453 | 0.487 | 0.487 | 0.500 | 0.773 | 0.995 | **1.000** | 0.671 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.514 +/- 0.007** | 0.513 +/- 0.009 | **0.539 +/- 0.007** | <u>0.546 +/- 0.011</u> | 0.798 +/- 0.002 | 0.996 +/- 0.001 | **1.000 +/- 0.000** | **0.701** |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.469 +/- 0.012 | 0.511 +/- 0.004 | <u>0.524 +/- 0.007</u> | 0.514 +/- 0.012 | 0.783 +/- 0.006 | 0.995 +/- 0.002 | **1.000 +/- 0.000** | 0.685 |
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
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.594 +/- 0.024 | 0.519 +/- 0.022 | 0.409 | 0.758 +/- 0.498 | — | — | — | 0.570 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.664 +/- 0.040 | 0.597 +/- 0.010 | 0.423 +/- 0.001 | 0.579 +/- 0.015 | — | — | — | 0.566 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.615 +/- 0.006 | 0.615 +/- 0.026 | 0.459 +/- 0.016 | 0.601 +/- 0.021 | — | — | — | 0.572 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.653 +/- 0.036 | 0.551 +/- 0.028 | 0.446 +/- 0.020 | 0.569 +/- 0.056 | — | — | — | 0.555 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.573 +/- 0.013 | 0.470 +/- 0.009 | **0.400 +/- 0.006** | <u>0.460 +/- 0.009</u> | — | — | — | **0.476** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.649 +/- 0.028 | 0.521 +/- 0.021 | 0.444 +/- 0.032 | 0.607 +/- 0.017 | — | — | — | 0.555 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 4/4 | **0.563 +/- 0.019** | 0.503 +/- 0.011 | 0.425 +/- 0.003 | 0.500 +/- 0.002 | — | — | — | 0.498 |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 4/4 | 0.666 +/- 0.018 | 0.555 +/- 0.005 | 0.457 +/- 0.011 | 0.607 +/- 0.015 | — | — | — | 0.571 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/4 | 0.637 +/- 0.003 | 0.768 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/4 | 0.677 +/- 0.013 | 0.561 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.584 +/- 0.027 | **0.470 +/- 0.006** | 0.410 +/- 0.005 | 0.471 +/- 0.036 | — | — | — | 0.484 |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.602 +/- 0.024 | 0.479 +/- 0.004 | 0.419 +/- 0.008 | 0.546 +/- 0.042 | — | — | — | 0.511 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 44 | 3/4 | 0.670 | 0.507 | — | **0.300** | — | — | — | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 44 | 4/4 | 0.657 | 0.523 | 0.439 | 0.501 | — | — | — | 0.530 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | <u>0.564 +/- 0.006</u> | <u>0.470 +/- 0.004</u> | <u>0.404 +/- 0.008</u> | 0.466 +/- 0.007 | — | — | — | <u>0.476</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.601 +/- 0.016 | 0.498 +/- 0.006 | 0.409 +/- 0.006 | 0.497 +/- 0.014 | — | — | — | 0.502 |
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
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 7/7 | 0.673 (n=4) | 51.1 (n=7) | 0.716 (n=7) | 0.490 (n=7) | 0.570 (n=4) |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 7/7 | 0.712 (n=4) | 72.6 (n=7) | 0.777 (n=7) | 0.666 (n=7) | 0.566 (n=4) |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 7/7 | 0.696 (n=4) | 65.7 (n=7) | 0.724 (n=7) | 0.603 (n=7) | 0.572 (n=4) |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 7/7 | 0.696 (n=4) | 72.6 (n=7) | 0.768 (n=7) | 0.664 (n=7) | 0.555 (n=4) |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 7/7 | <u>0.757 (n=4)</u> | **75.3 (n=7)** | **0.813 (n=7)** | <u>0.698 (n=7)</u> | **0.476 (n=4)** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 7/7 | 0.698 (n=4) | 72.7 (n=7) | 0.777 (n=7) | 0.659 (n=7) | 0.555 (n=4) |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 7/7 | 0.742 (n=4) | 74.4 (n=7) | 0.796 (n=7) | 0.688 (n=7) | 0.498 (n=4) |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 7/7 | 0.690 (n=4) | 72.2 (n=7) | 0.769 (n=7) | 0.660 (n=7) | 0.571 (n=4) |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 2/7 | 0.685 (n=2) | 47.7 (n=2) | 0.719 (n=2) | 0.386 (n=2) | 0.703 (n=2) |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 2/7 | 0.698 (n=2) | 58.6 (n=2) | 0.721 (n=2) | 0.461 (n=2) | 0.619 (n=2) |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 6/7 | 0.710 (n=4) | 62.8 (n=6) | 0.749 (n=6) | 0.591 (n=6) | 0.484 (n=4) |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 6/7 | 0.734 (n=4) | 69.7 (n=6) | 0.765 (n=6) | 0.636 (n=6) | 0.511 (n=4) |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 7/7 | 0.752 (n=3) | 29.5 (n=7) | 0.598 (n=6) | 0.303 (n=7) | 0.493 (n=3) |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 7/7 | 0.708 (n=4) | 73.6 (n=7) | 0.777 (n=7) | 0.671 (n=7) | 0.530 (n=4) |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 7/7 | **0.758 (n=4)** | <u>75.2 (n=7)</u> | <u>0.811 (n=7)</u> | **0.701 (n=7)** | <u>0.476 (n=4)</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 7/7 | 0.735 (n=4) | 74.4 (n=7) | 0.793 (n=7) | 0.685 (n=7) | 0.502 (n=4) |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 1/7 | 0.739 (n=1) | 66.5 (n=1) | 0.769 (n=1) | 0.472 (n=1) | 0.503 (n=1) |


## 9. Rebuild report

```bash
python scripts/evaluate.py --refresh-analysis-only --output_path outputs/evaluations
```
