# Qwen3-4B, SciRM-7B, and SciRM-Ref-7B evaluation results

> Generated at 2026-09-29T07:17:58.743629+00:00; 363 deduplicated task/configuration/seed records are included.
> This file is rebuilt by `scripts/evaluate.py` after evaluation.

## 1. Reporting protocol

Included configurations: B-L, B-C, SciRM-L, SciRM-C, SciRM-Ref-L, SciRM-Ref-C, LL, LC, CL, CC, PAL, PAC, MIX-L, MIX-C, SSAL, SSAC, SSA2L, SSA2C, SCL, SCC, SCAL, SCAC, LL-R.

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
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 0.694 | 0.284 | 0.580 | 0.564 | 0.575 | 0.867 | 0.573 | 0.591 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 0.746 | 0.471 | 0.580 | 0.665 | 0.661 | 0.862 | 0.672 | 0.665 |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.643 +/- 0.001 | -0.004 +/- 0.002 | 0.578 +/- 0.001 | 0.422 +/- 0.001 | 0.610 +/- 0.001 | 0.941 +/- 0.001 | 0.565 +/- 0.005 | 0.536 |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 42, 43, 44 | 7/7 | 0.685 +/- 0.000 | 0.248 +/- 0.009 | 0.594 +/- 0.005 | 0.605 +/- 0.012 | 0.637 +/- 0.001 | 0.948 +/- 0.001 | 0.660 +/- 0.012 | 0.625 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.758 +/- 0.027 | 0.699 +/- 0.031 | 0.683 +/- 0.043 | 0.731 +/- 0.005 | 0.743 +/- 0.004 | 0.999 +/- 0.001 | **1.000 +/- 0.000** | 0.802 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.706 +/- 0.064 | 0.677 +/- 0.019 | 0.656 +/- 0.040 | 0.646 +/- 0.013 | 0.688 +/- 0.029 | 0.994 +/- 0.005 | 0.996 +/- 0.007 | 0.766 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.344 +/- 0.382 | 0.512 +/- 0.177 | 0.484 +/- 0.164 | 0.321 +/- 0.103 | 0.450 +/- 0.073 | 0.552 +/- 0.101 | 0.446 +/- 0.054 | 0.444 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.649 +/- 0.078 | 0.623 +/- 0.056 | 0.605 +/- 0.070 | 0.611 +/- 0.007 | 0.704 +/- 0.009 | 0.996 +/- 0.001 | 0.969 +/- 0.006 | 0.737 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.760 +/- 0.022 | 0.713 +/- 0.018 | 0.695 +/- 0.030 | 0.740 +/- 0.001 | 0.752 +/- 0.009 | 0.999 +/- 0.001 | **1.000 +/- 0.000** | <u>0.808</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.696 +/- 0.038 | 0.669 +/- 0.025 | 0.651 +/- 0.040 | 0.643 +/- 0.012 | 0.727 +/- 0.016 | 0.997 +/- 0.003 | 0.996 +/- 0.003 | 0.768 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.790 +/- 0.009** | 0.723 +/- 0.008 | **0.714 +/- 0.005** | <u>0.740 +/- 0.003</u> | 0.776 +/- 0.008 | 0.998 +/- 0.002 | **1.000 +/- 0.000** | **0.820** |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 0.728 +/- 0.007 | 0.678 +/- 0.003 | 0.678 +/- 0.012 | 0.674 +/- 0.009 | <u>0.781 +/- 0.003</u> | 0.996 +/- 0.002 | **1.000 +/- 0.000** | 0.791 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 0.759 +/- 0.003 | 0.611 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 0.721 +/- 0.002 | 0.674 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 0.719 +/- 0.108 | **0.748 +/- 0.007** | 0.627 +/- 0.129 | **0.745 +/- 0.052** | 0.739 +/- 0.066 | **1.000 +/- 0.000** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | <u>0.767 +/- 0.011</u> | <u>0.742 +/- 0.003</u> | <u>0.712 +/- 0.007</u> | 0.714 +/- 0.030 | **0.799 +/- 0.006** | 0.998 +/- 0.000 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | — | 0.288 +/- 0.207 | 0.445 +/- 0.019 | 0.405 +/- 0.184 | 0.337 +/- 0.003 | 0.410 +/- 0.144 | 0.447 +/- 0.067 | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.595 +/- 0.022 | 0.588 +/- 0.006 | 0.557 +/- 0.011 | 0.616 +/- 0.007 | 0.709 +/- 0.003 | 0.990 +/- 0.002 | 0.989 +/- 0.000 | 0.721 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.750 +/- 0.003 | 0.715 +/- 0.006 | 0.669 +/- 0.004 | 0.738 +/- 0.009 | 0.749 +/- 0.010 | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | 0.803 |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.743 +/- 0.004 | 0.703 +/- 0.006 | 0.687 +/- 0.006 | 0.729 +/- 0.002 | 0.746 +/- 0.007 | 0.995 +/- 0.004 | <u>0.996 +/- 0.003</u> | 0.800 |
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
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 4/4 | 0.694 | 0.284 | 0.580 | 0.564 | — | — | — | 0.531 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 4/4 | 0.746 | 0.471 | 0.580 | 0.665 | — | — | — | 0.616 |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.643 +/- 0.001 | -0.004 +/- 0.002 | 0.578 +/- 0.001 | 0.422 +/- 0.001 | — | — | — | 0.410 |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 42, 43, 44 | 4/4 | 0.685 +/- 0.000 | 0.248 +/- 0.009 | 0.594 +/- 0.005 | 0.605 +/- 0.012 | — | — | — | 0.533 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.758 +/- 0.027 | 0.699 +/- 0.031 | 0.683 +/- 0.043 | 0.731 +/- 0.005 | — | — | — | 0.718 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.706 +/- 0.064 | 0.677 +/- 0.019 | 0.656 +/- 0.040 | 0.646 +/- 0.013 | — | — | — | 0.671 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.344 +/- 0.382 | 0.512 +/- 0.177 | 0.484 +/- 0.164 | 0.321 +/- 0.103 | — | — | — | 0.415 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.649 +/- 0.078 | 0.623 +/- 0.056 | 0.605 +/- 0.070 | 0.611 +/- 0.007 | — | — | — | 0.622 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.760 +/- 0.022 | 0.713 +/- 0.018 | 0.695 +/- 0.030 | 0.740 +/- 0.001 | — | — | — | 0.727 |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.696 +/- 0.038 | 0.669 +/- 0.025 | 0.651 +/- 0.040 | 0.643 +/- 0.012 | — | — | — | 0.665 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 4/4 | **0.790 +/- 0.009** | 0.723 +/- 0.008 | **0.714 +/- 0.005** | <u>0.740 +/- 0.003</u> | — | — | — | **0.742** |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 4/4 | 0.728 +/- 0.007 | 0.678 +/- 0.003 | 0.678 +/- 0.012 | 0.674 +/- 0.009 | — | — | — | 0.690 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/4 | 0.759 +/- 0.003 | 0.611 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/4 | 0.721 +/- 0.002 | 0.674 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.719 +/- 0.108 | **0.748 +/- 0.007** | 0.627 +/- 0.129 | **0.745 +/- 0.052** | — | — | — | 0.710 |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 4/4 | <u>0.767 +/- 0.011</u> | <u>0.742 +/- 0.003</u> | <u>0.712 +/- 0.007</u> | 0.714 +/- 0.030 | — | — | — | <u>0.734</u> |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 3/4 | — | 0.288 +/- 0.207 | 0.445 +/- 0.019 | 0.405 +/- 0.184 | — | — | — | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.595 +/- 0.022 | 0.588 +/- 0.006 | 0.557 +/- 0.011 | 0.616 +/- 0.007 | — | — | — | 0.589 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.750 +/- 0.003 | 0.715 +/- 0.006 | 0.669 +/- 0.004 | 0.738 +/- 0.009 | — | — | — | 0.718 |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.743 +/- 0.004 | 0.703 +/- 0.006 | 0.687 +/- 0.006 | 0.729 +/- 0.002 | — | — | — | 0.715 |
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
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 53.0 | 50.6 | 52.8 | 56.3 | 61.7 | 86.7 | 57.4 | 59.8 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | <u>53.8</u> | 54.4 | 50.9 | **61.2** | 67.4 | 86.2 | 67.2 | 63.0 |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 42, 43, 44 | 7/7 | 51.5 +/- 0.1 | 39.4 +/- 0.1 | 53.6 +/- 0.0 | 52.7 +/- 0.2 | 63.7 +/- 0.1 | 93.8 +/- 0.2 | 56.0 +/- 0.6 | 58.7 |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 42, 43, 44 | 7/7 | 52.1 +/- 0.9 | 46.2 +/- 0.9 | 50.5 +/- 0.3 | 59.7 +/- 0.8 | 65.6 +/- 0.1 | 94.8 +/- 0.1 | 66.0 +/- 1.1 | 62.1 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 52.6 +/- 1.4 | 69.1 +/- 1.5 | 58.7 +/- 2.6 | <u>60.2 +/- 1.2</u> | 74.3 +/- 0.4 | 99.9 +/- 0.1 | **100.0 +/- 0.0** | 73.5 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 51.3 +/- 5.6 | 64.4 +/- 2.1 | 53.6 +/- 5.0 | 46.7 +/- 3.4 | 66.2 +/- 4.7 | 99.0 +/- 0.8 | 99.5 +/- 0.8 | 68.7 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 26.1 +/- 25.7 | 54.4 +/- 5.8 | 50.8 +/- 2.8 | 19.0 +/- 5.6 | 55.6 +/- 3.7 | 59.1 +/- 7.7 | 46.9 +/- 4.2 | 44.6 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 44.6 +/- 5.2 | 66.0 +/- 2.9 | 51.8 +/- 4.6 | 50.0 +/- 0.8 | 70.4 +/- 0.9 | 99.6 +/- 0.1 | 97.2 +/- 0.6 | 68.5 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 53.7 +/- 1.8 | 69.6 +/- 0.9 | 58.8 +/- 2.7 | 59.5 +/- 0.4 | 75.2 +/- 0.9 | 99.9 +/- 0.1 | **100.0 +/- 0.0** | <u>73.8</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 47.8 +/- 2.1 | 69.1 +/- 1.6 | 55.1 +/- 3.0 | 52.7 +/- 1.0 | 72.7 +/- 1.6 | 99.7 +/- 0.3 | <u>99.7 +/- 0.3</u> | 71.0 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **54.6 +/- 1.2** | <u>70.3 +/- 0.5</u> | <u>59.2 +/- 0.4</u> | 59.7 +/- 0.7 | 77.6 +/- 0.8 | 99.8 +/- 0.2 | **100.0 +/- 0.0** | **74.4** |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 49.2 +/- 1.3 | 68.6 +/- 0.4 | 56.6 +/- 1.1 | 53.4 +/- 0.9 | <u>78.1 +/- 0.3</u> | 99.6 +/- 0.2 | **100.0 +/- 0.0** | 72.2 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 51.4 +/- 0.5 | 44.0 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 48.5 +/- 1.5 | 68.6 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 41.4 +/- 13.6 | 70.1 +/- 0.6 | 46.5 +/- 15.8 | 48.1 +/- 22.4 | 70.5 +/- 10.1 | **100.0 +/- 0.0** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | 52.4 +/- 1.2 | **70.5 +/- 0.5** | **59.8 +/- 0.7** | 55.9 +/- 2.2 | **79.9 +/- 0.6** | 99.8 +/- 0.0 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.0 +/- 0.0 | 1.5 +/- 2.1 | 40.3 +/- 14.2 | 11.7 +/- 7.0 | 50.2 +/- 0.1 | 50.0 +/- 9.1 | 47.1 +/- 5.1 | 28.7 |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 42.9 +/- 1.2 | 64.3 +/- 0.7 | 52.5 +/- 0.9 | 58.1 +/- 1.1 | 71.0 +/- 0.3 | 99.1 +/- 0.2 | 99.0 +/- 0.0 | 69.6 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 52.7 +/- 0.4 | 69.1 +/- 0.6 | 57.5 +/- 0.7 | 58.7 +/- 1.0 | 74.9 +/- 1.0 | <u>99.9 +/- 0.1</u> | **100.0 +/- 0.0** | 73.3 |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 51.5 +/- 0.5 | 69.2 +/- 0.5 | 58.4 +/- 0.1 | 59.3 +/- 0.4 | 74.6 +/- 0.7 | 99.6 +/- 0.3 | <u>99.7 +/- 0.3</u> | 73.2 |
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
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 0.745 | 0.349 | 0.591 | 0.603 | 0.299 | 0.764 | 0.375 | 0.532 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 0.764 | 0.494 | 0.602 | 0.673 | 0.378 | 0.756 | 0.506 | 0.596 |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.708 +/- 0.000 | -0.009 +/- 0.004 | 0.584 +/- 0.001 | 0.500 +/- 0.002 | 0.327 +/- 0.002 | 0.894 +/- 0.000 | 0.376 +/- 0.010 | 0.483 |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 42, 43, 44 | 7/7 | 0.715 +/- 0.001 | 0.320 +/- 0.009 | 0.599 +/- 0.006 | 0.617 +/- 0.011 | 0.348 +/- 0.004 | 0.901 +/- 0.002 | 0.466 +/- 0.009 | 0.566 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.764 +/- 0.027 | 0.728 +/- 0.025 | 0.697 +/- 0.039 | 0.744 +/- 0.001 | 0.487 +/- 0.007 | 0.998 +/- 0.002 | **1.000 +/- 0.000** | 0.774 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.746 +/- 0.032 | 0.696 +/- 0.027 | 0.670 +/- 0.037 | 0.670 +/- 0.010 | 0.445 +/- 0.017 | 0.998 +/- 0.002 | 0.993 +/- 0.013 | 0.745 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.427 +/- 0.316 | 0.615 +/- 0.104 | 0.538 +/- 0.127 | 0.403 +/- 0.081 | 0.219 +/- 0.070 | 0.374 +/- 0.109 | 0.279 +/- 0.051 | 0.408 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.655 +/- 0.078 | 0.677 +/- 0.033 | 0.608 +/- 0.074 | 0.630 +/- 0.001 | 0.409 +/- 0.017 | 0.992 +/- 0.002 | 0.941 +/- 0.011 | 0.702 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.764 +/- 0.022 | 0.740 +/- 0.017 | 0.709 +/- 0.028 | <u>0.751 +/- 0.002</u> | 0.504 +/- 0.017 | 0.998 +/- 0.002 | **1.000 +/- 0.000** | <u>0.781</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.701 +/- 0.039 | 0.710 +/- 0.014 | 0.656 +/- 0.041 | 0.667 +/- 0.013 | 0.456 +/- 0.031 | 0.993 +/- 0.006 | 0.993 +/- 0.006 | 0.739 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.796 +/- 0.007** | 0.748 +/- 0.008 | **0.728 +/- 0.005** | **0.755 +/- 0.004** | 0.552 +/- 0.016 | 0.996 +/- 0.004 | **1.000 +/- 0.000** | **0.796** |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 0.733 +/- 0.006 | 0.716 +/- 0.002 | 0.687 +/- 0.011 | 0.695 +/- 0.010 | 0.563 +/- 0.006 | 0.992 +/- 0.004 | **1.000 +/- 0.000** | 0.769 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 0.764 +/- 0.004 | 0.675 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 0.728 +/- 0.005 | 0.715 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 0.729 +/- 0.108 | **0.774 +/- 0.007** | 0.641 +/- 0.128 | 0.750 +/- 0.055 | **0.599 +/- 0.004** | **1.000 +/- 0.000** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | <u>0.774 +/- 0.009</u> | <u>0.769 +/- 0.004</u> | <u>0.725 +/- 0.004</u> | 0.727 +/- 0.029 | <u>0.598 +/- 0.011</u> | 0.997 +/- 0.000 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | — | 0.391 +/- 0.233 | 0.497 +/- 0.017 | 0.484 +/- 0.137 | 0.049 +/- 0.007 | 0.198 +/- 0.173 | 0.280 +/- 0.064 | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.598 +/- 0.022 | 0.644 +/- 0.005 | 0.566 +/- 0.011 | 0.635 +/- 0.010 | 0.423 +/- 0.004 | 0.981 +/- 0.004 | 0.978 +/- 0.000 | 0.689 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.757 +/- 0.004 | 0.742 +/- 0.010 | 0.683 +/- 0.003 | 0.748 +/- 0.007 | 0.499 +/- 0.020 | <u>0.999 +/- 0.002</u> | **1.000 +/- 0.000** | 0.775 |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.751 +/- 0.003 | 0.734 +/- 0.005 | 0.698 +/- 0.009 | 0.740 +/- 0.003 | 0.492 +/- 0.015 | 0.991 +/- 0.007 | <u>0.993 +/- 0.006</u> | 0.771 |
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
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 7/7 | 0.457 | 0.235 | 0.330 | 0.391 | 0.575 | 0.867 | 0.573 | 0.490 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 7/7 | 0.463 | 0.323 | 0.370 | 0.439 | 0.661 | 0.862 | 0.672 | 0.541 |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.435 +/- 0.001 | 0.132 +/- 0.000 | 0.332 +/- 0.000 | 0.333 +/- 0.002 | 0.610 +/- 0.001 | 0.941 +/- 0.001 | 0.565 +/- 0.005 | 0.478 |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 42, 43, 44 | 7/7 | 0.441 +/- 0.007 | 0.240 +/- 0.005 | 0.364 +/- 0.009 | 0.397 +/- 0.002 | 0.637 +/- 0.001 | 0.948 +/- 0.001 | 0.660 +/- 0.012 | 0.527 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.482 +/- 0.021 | 0.471 +/- 0.046 | 0.477 +/- 0.044 | <u>0.517 +/- 0.012</u> | 0.743 +/- 0.004 | 0.999 +/- 0.001 | **1.000 +/- 0.000** | 0.670 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.424 +/- 0.042 | 0.440 +/- 0.039 | 0.419 +/- 0.054 | 0.439 +/- 0.033 | 0.688 +/- 0.029 | 0.994 +/- 0.005 | 0.996 +/- 0.007 | 0.629 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.200 +/- 0.227 | 0.287 +/- 0.105 | 0.307 +/- 0.125 | 0.145 +/- 0.036 | 0.450 +/- 0.073 | 0.552 +/- 0.101 | 0.446 +/- 0.054 | 0.341 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.393 +/- 0.061 | 0.395 +/- 0.093 | 0.424 +/- 0.064 | 0.385 +/- 0.006 | 0.704 +/- 0.009 | 0.996 +/- 0.001 | 0.969 +/- 0.006 | 0.610 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.496 +/- 0.024</u> | 0.494 +/- 0.036 | 0.488 +/- 0.041 | 0.516 +/- 0.009 | 0.752 +/- 0.009 | 0.999 +/- 0.001 | **1.000 +/- 0.000** | <u>0.678</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.422 +/- 0.029 | 0.422 +/- 0.075 | 0.444 +/- 0.037 | 0.427 +/- 0.005 | 0.727 +/- 0.016 | 0.997 +/- 0.003 | 0.996 +/- 0.003 | 0.634 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.510 +/- 0.014** | **0.518 +/- 0.007** | <u>0.511 +/- 0.008</u> | 0.507 +/- 0.005 | 0.776 +/- 0.008 | 0.998 +/- 0.002 | **1.000 +/- 0.000** | **0.688** |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 7/7 | 0.454 +/- 0.016 | 0.460 +/- 0.017 | 0.461 +/- 0.017 | 0.466 +/- 0.016 | <u>0.781 +/- 0.003</u> | 0.996 +/- 0.002 | **1.000 +/- 0.000** | 0.660 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/7 | 0.457 +/- 0.006 | 0.314 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/7 | 0.449 +/- 0.015 | 0.473 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | 0.424 +/- 0.101 | <u>0.506 +/- 0.011</u> | 0.396 +/- 0.142 | 0.483 +/- 0.129 | 0.739 +/- 0.066 | **1.000 +/- 0.000** | — | — |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 6/7 | 0.491 +/- 0.017 | 0.505 +/- 0.010 | **0.517 +/- 0.007** | 0.507 +/- 0.024 | **0.799 +/- 0.006** | 0.998 +/- 0.000 | — | — |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.000 +/- 0.000 | 0.013 +/- 0.018 | 0.258 +/- 0.054 | 0.098 +/- 0.028 | 0.337 +/- 0.003 | 0.410 +/- 0.144 | 0.447 +/- 0.067 | 0.223 |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.371 +/- 0.014 | 0.334 +/- 0.007 | 0.362 +/- 0.009 | 0.349 +/- 0.037 | 0.709 +/- 0.003 | 0.990 +/- 0.002 | 0.989 +/- 0.000 | 0.587 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.485 +/- 0.006 | 0.483 +/- 0.006 | 0.470 +/- 0.010 | 0.506 +/- 0.016 | 0.749 +/- 0.010 | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | 0.670 |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.465 +/- 0.005 | 0.478 +/- 0.016 | 0.474 +/- 0.010 | **0.527 +/- 0.004** | 0.746 +/- 0.007 | 0.995 +/- 0.004 | <u>0.996 +/- 0.003</u> | 0.669 |
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
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | base | 4/4 | 0.676 | 0.895 | 0.506 | 0.575 | — | — | — | 0.663 |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | base | 4/4 | 0.645 | 0.792 | 0.531 | 0.516 | — | — | — | 0.621 |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.755 +/- 0.001 | 1.116 +/- 0.001 | 0.506 +/- 0.001 | 0.679 +/- 0.002 | — | — | — | 0.764 |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 42, 43, 44 | 4/4 | 0.729 +/- 0.008 | 0.979 +/- 0.013 | 0.533 +/- 0.003 | 0.557 +/- 0.014 | — | — | — | 0.700 |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.608 +/- 0.042 | 0.530 +/- 0.041 | 0.438 +/- 0.037 | 0.500 +/- 0.010 | — | — | — | 0.519 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.743 +/- 0.141 | 0.629 +/- 0.044 | 0.510 +/- 0.065 | 0.674 +/- 0.039 | — | — | — | 0.639 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 1.037 +/- 0.410 | 0.823 +/- 0.233 | 0.539 +/- 0.043 | 0.906 +/- 0.060 | — | — | — | 0.826 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.761 +/- 0.109 | 0.617 +/- 0.065 | 0.521 +/- 0.059 | 0.689 +/- 0.002 | — | — | — | 0.647 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.598 +/- 0.037 | 0.511 +/- 0.022 | 0.434 +/- 0.033 | <u>0.494 +/- 0.001</u> | — | — | — | 0.509 |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.698 +/- 0.050 | 0.553 +/- 0.032 | 0.476 +/- 0.038 | 0.631 +/- 0.010 | — | — | — | 0.590 |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 42, 43, 44 | 4/4 | **0.563 +/- 0.019** | 0.503 +/- 0.011 | 0.425 +/- 0.003 | 0.500 +/- 0.002 | — | — | — | <u>0.498</u> |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 42, 43, 44 | 4/4 | 0.666 +/- 0.018 | 0.555 +/- 0.005 | 0.457 +/- 0.011 | 0.607 +/- 0.015 | — | — | — | 0.571 |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 42, 43 | 2/4 | 0.637 +/- 0.003 | 0.768 | — | — | — | — | — | — |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 42, 43 | 2/4 | 0.677 +/- 0.013 | 0.561 | — | — | — | — | — | — |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | <u>0.584 +/- 0.027</u> | **0.470 +/- 0.006** | **0.410 +/- 0.005** | **0.471 +/- 0.036** | — | — | — | **0.484** |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.602 +/- 0.024 | <u>0.479 +/- 0.004</u> | <u>0.419 +/- 0.008</u> | 0.546 +/- 0.042 | — | — | — | 0.511 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 3/4 | — | 1.172 +/- 0.719 | 0.524 +/- 0.025 | 0.674 +/- 0.253 | — | — | — | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.828 +/- 0.027 | 0.655 +/- 0.013 | 0.521 +/- 0.010 | 0.604 +/- 0.010 | — | — | — | 0.652 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.618 +/- 0.003 | 0.515 +/- 0.013 | 0.454 +/- 0.008 | 0.500 +/- 0.016 | — | — | — | 0.522 |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.634 +/- 0.008 | 0.526 +/- 0.007 | 0.440 +/- 0.004 | 0.505 +/- 0.003 | — | — | — | 0.526 |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 42 | 1/4 | — | 0.503 | — | — | — | — | — | — |

*Table 6. MAE by training/inference configuration and task.*

## 8. Cross-task method averages

| Id | Model | CoT | Train | Prompt | Inf. | Coverage | Average QWK | Average Accuracy (%) | Average Pearson | Average Macro-F1 | Average MAE |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |  |  |  |  |  |  |
| B-L | Qwen3-4B Base | ✗ | Base | Label-only | Greedy | 7/7 | 0.364 (n=4) | 17.0 (n=7) | 0.387 (n=6) | 0.196 (n=7) | 0.970 (n=4) |
| B-C | Qwen3-4B Base | ✓ | Base | CoT | Greedy | 7/7 | 0.494 (n=4) | 52.9 (n=7) | 0.528 (n=7) | 0.499 (n=7) | 0.852 (n=4) |
| SciRM-L | SciRM-7B RL | ✗ | RL | Label-only | Greedy | 7/7 | 0.531 (n=4) | 59.8 (n=7) | 0.532 (n=7) | 0.490 (n=7) | 0.663 (n=4) |
| SciRM-C | SciRM-7B RL | ✓ | RL | CoT | Greedy | 7/7 | 0.616 (n=4) | 63.0 (n=7) | 0.596 (n=7) | 0.541 (n=7) | 0.621 (n=4) |
| SciRM-Ref-L | SciRM-Ref-7B RL | ✗ | RL | Label-only | Greedy | 7/7 | 0.410 (n=4) | 58.7 (n=7) | 0.483 (n=7) | 0.478 (n=7) | 0.764 (n=4) |
| SciRM-Ref-C | SciRM-Ref-7B RL | ✓ | RL | CoT | Greedy | 7/7 | 0.533 (n=4) | 62.1 (n=7) | 0.566 (n=7) | 0.527 (n=7) | 0.700 (n=4) |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 7/7 | 0.718 (n=4) | 73.5 (n=7) | 0.774 (n=7) | 0.670 (n=7) | 0.519 (n=4) |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 7/7 | 0.671 (n=4) | 68.7 (n=7) | 0.745 (n=7) | 0.629 (n=7) | 0.639 (n=4) |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 7/7 | 0.415 (n=4) | 44.6 (n=7) | 0.408 (n=7) | 0.341 (n=7) | 0.826 (n=4) |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 7/7 | 0.622 (n=4) | 68.5 (n=7) | 0.702 (n=7) | 0.610 (n=7) | 0.647 (n=4) |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 7/7 | 0.727 (n=4) | <u>73.8 (n=7)</u> | <u>0.781 (n=7)</u> | <u>0.678 (n=7)</u> | 0.509 (n=4) |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 7/7 | 0.665 (n=4) | 71.0 (n=7) | 0.739 (n=7) | 0.634 (n=7) | 0.590 (n=4) |
| **Paper Align w/o Loss Balance** |  |  |  |  |  |  |  |  |  |  |  |
| MIX-L | Qwen3-4B | ✗ | Paper Align w/o Loss Balance | Label-only | Greedy | 7/7 | **0.742 (n=4)** | **74.4 (n=7)** | **0.796 (n=7)** | **0.688 (n=7)** | <u>0.498 (n=4)</u> |
| MIX-C | Qwen3-4B | ✓ | Paper Align w/o Loss Balance | CoT | Greedy | 7/7 | 0.690 (n=4) | 72.2 (n=7) | 0.769 (n=7) | 0.660 (n=7) | 0.571 (n=4) |
| **Single Sample Align** |  |  |  |  |  |  |  |  |  |  |  |
| SSAL | Qwen3-4B | ✗ | Single Sample Align SFT | Label-only | Greedy | 2/7 | 0.685 (n=2) | 47.7 (n=2) | 0.719 (n=2) | 0.386 (n=2) | 0.703 (n=2) |
| SSAC | Qwen3-4B | ✓ | Single Sample Align SFT | CoT | Greedy | 2/7 | 0.698 (n=2) | 58.6 (n=2) | 0.721 (n=2) | 0.461 (n=2) | 0.619 (n=2) |
| **SSA v2** |  |  |  |  |  |  |  |  |  |  |  |
| SSA2L | Qwen3-4B | ✗ | SSA v2 SFT | Label-only | Greedy | 6/7 | 0.710 (n=4) | 62.8 (n=6) | 0.749 (n=6) | 0.591 (n=6) | **0.484 (n=4)** |
| SSA2C | Qwen3-4B | ✓ | SSA v2 SFT | CoT | Greedy | 6/7 | 0.734 (n=4) | 69.7 (n=6) | 0.765 (n=6) | 0.636 (n=6) | 0.511 (n=4) |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 7/7 | 0.379 (n=3) | 28.7 (n=7) | 0.316 (n=6) | 0.223 (n=7) | 0.790 (n=3) |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 7/7 | 0.589 (n=4) | 69.6 (n=7) | 0.689 (n=7) | 0.587 (n=7) | 0.652 (n=4) |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 7/7 | 0.718 (n=4) | 73.3 (n=7) | 0.775 (n=7) | 0.670 (n=7) | 0.522 (n=4) |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 7/7 | 0.715 (n=4) | 73.2 (n=7) | 0.771 (n=7) | 0.669 (n=7) | 0.526 (n=4) |
| **Regression-aware methods** |  |  |  |  |  |  |  |  |  |  |  |
| LL-R | Qwen3-4B | ✗ | Label-only CE | Label-only | RAIL | 1/7 | <u>0.739 (n=1)</u> | 66.5 (n=1) | 0.769 (n=1) | 0.472 (n=1) | 0.503 (n=1) |


## 9. Rebuild report

```bash
python scripts/evaluate.py --refresh-analysis-only --output_path outputs/evaluations
```
