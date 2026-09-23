# Qwen3-4B and SciRM-7B evaluation results

> Generated at 2026-09-22T04:22:09.982683+00:00; 210 deduplicated task/configuration/seed records are included.
> This file is rebuilt by `scripts/evaluate.py` after evaluation.

## 1. Reporting protocol

Included configurations: LL, LC, CL, CC, PAL, PAC, SCL, SCC, SCAL, SCAC.

Tables follow the main-result layout of TRACT: training and inference configurations are listed on the left, tasks are expanded across columns, and the strict macro average is reported last. **Bold** marks the best result and <u>underline</u> marks the second-best distinct result in each column. Ties share a rank. MAE is ranked in ascending order; all other metrics are ranked in descending order.

`CoT` indicates whether the evaluation prompt requests an explicit rationale; it does not indicate hidden/internal model reasoning. Multi-seed cells report `mean +/- sample standard deviation`. Variances remain available in `evaluation_analysis_records.json`. An average is shown only when a configuration covers every task to which that metric applies; `—` means not applicable or unavailable.

Task sample counts: Actionability=1000; Grounding Specificity=1000; Helpfulness=1000; Verifiability=788; Coherence=1046; Positioning Check=603; Positioning Type=204.

## 2. Main results by primary metric

The primary metric is QWK for the four ordinal review-utility tasks and Macro-F1 for the three binary writing-quality tasks.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.741 +/- 0.007 | 0.685 +/- 0.010 | 0.660 +/- 0.004 | 0.731 +/- 0.005 | 0.743 +/- 0.004 | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | 0.794 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.690 +/- 0.038 | 0.658 +/- 0.014 | 0.629 +/- 0.013 | 0.646 +/- 0.013 | 0.688 +/- 0.029 | 0.994 +/- 0.005 | 0.996 +/- 0.007 | 0.757 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.169 +/- 0.108 | 0.359 +/- 0.090 | 0.393 +/- 0.008 | 0.321 +/- 0.103 | 0.450 +/- 0.073 | 0.552 +/- 0.101 | 0.446 +/- 0.054 | 0.384 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.596 +/- 0.014 | 0.590 +/- 0.001 | 0.559 +/- 0.010 | 0.611 +/- 0.007 | 0.704 +/- 0.009 | 0.996 +/- 0.001 | 0.969 +/- 0.006 | 0.718 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.750 +/- 0.007** | <u>0.709 +/- 0.011</u> | <u>0.679 +/- 0.005</u> | **0.740 +/- 0.001** | **0.752 +/- 0.009** | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | **0.804** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.681 +/- 0.018 | 0.651 +/- 0.006 | 0.625 +/- 0.005 | 0.643 +/- 0.012 | 0.727 +/- 0.016 | 0.997 +/- 0.003 | 0.996 +/- 0.003 | 0.760 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | — | 0.288 +/- 0.207 | 0.445 +/- 0.019 | 0.405 +/- 0.184 | 0.337 +/- 0.003 | 0.410 +/- 0.144 | 0.447 +/- 0.067 | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.595 +/- 0.022 | 0.588 +/- 0.006 | 0.557 +/- 0.011 | 0.616 +/- 0.007 | 0.709 +/- 0.003 | 0.990 +/- 0.002 | 0.989 +/- 0.000 | 0.721 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.750 +/- 0.003</u> | **0.715 +/- 0.006** | 0.669 +/- 0.004 | <u>0.738 +/- 0.009</u> | <u>0.749 +/- 0.010</u> | **0.999 +/- 0.001** | **1.000 +/- 0.000** | <u>0.803</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.743 +/- 0.004 | 0.703 +/- 0.006 | **0.687 +/- 0.006** | 0.729 +/- 0.002 | 0.746 +/- 0.007 | 0.995 +/- 0.004 | <u>0.996 +/- 0.003</u> | 0.800 |

*Table 1. Primary-metric results. Average requires complete coverage of all seven tasks.*

## 3. QWK

QWK applies to four ordinal tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.741 +/- 0.007 | 0.685 +/- 0.010 | 0.660 +/- 0.004 | 0.731 +/- 0.005 | — | — | — | 0.704 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.690 +/- 0.038 | 0.658 +/- 0.014 | 0.629 +/- 0.013 | 0.646 +/- 0.013 | — | — | — | 0.656 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.169 +/- 0.108 | 0.359 +/- 0.090 | 0.393 +/- 0.008 | 0.321 +/- 0.103 | — | — | — | 0.310 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.596 +/- 0.014 | 0.590 +/- 0.001 | 0.559 +/- 0.010 | 0.611 +/- 0.007 | — | — | — | 0.589 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | **0.750 +/- 0.007** | <u>0.709 +/- 0.011</u> | <u>0.679 +/- 0.005</u> | **0.740 +/- 0.001** | — | — | — | **0.719** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.681 +/- 0.018 | 0.651 +/- 0.006 | 0.625 +/- 0.005 | 0.643 +/- 0.012 | — | — | — | 0.650 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 3/4 | — | 0.288 +/- 0.207 | 0.445 +/- 0.019 | 0.405 +/- 0.184 | — | — | — | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.595 +/- 0.022 | 0.588 +/- 0.006 | 0.557 +/- 0.011 | 0.616 +/- 0.007 | — | — | — | 0.589 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | <u>0.750 +/- 0.003</u> | **0.715 +/- 0.006** | 0.669 +/- 0.004 | <u>0.738 +/- 0.009</u> | — | — | — | <u>0.718</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.743 +/- 0.004 | 0.703 +/- 0.006 | **0.687 +/- 0.006** | 0.729 +/- 0.002 | — | — | — | 0.715 |

*Table 2. QWK by training/inference configuration and task.*
## 4. Accuracy (%)

Accuracy (%) applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 51.8 +/- 0.4 | 68.3 +/- 0.5 | 56.8 +/- 0.8 | **60.2 +/- 1.2** | 74.3 +/- 0.4 | <u>99.9 +/- 0.1</u> | **100.0 +/- 0.0** | 73.0 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 49.0 +/- 2.0 | 60.6 +/- 5.2 | 50.4 +/- 1.7 | 46.7 +/- 3.4 | 66.2 +/- 4.7 | 99.0 +/- 0.8 | 99.5 +/- 0.8 | 67.3 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 10.1 +/- 14.0 | 48.6 +/- 4.5 | 50.1 +/- 1.6 | 19.0 +/- 5.6 | 55.6 +/- 3.7 | 59.1 +/- 7.7 | 46.9 +/- 4.2 | 41.3 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 41.3 +/- 0.9 | 64.4 +/- 0.1 | 49.2 +/- 0.8 | 50.0 +/- 0.8 | 70.4 +/- 0.9 | 99.6 +/- 0.1 | 97.2 +/- 0.6 | 67.4 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>52.5 +/- 0.9</u> | **69.3 +/- 0.4** | <u>57.8 +/- 1.0</u> | <u>59.5 +/- 0.4</u> | **75.2 +/- 0.9** | <u>99.9 +/- 0.1</u> | **100.0 +/- 0.0** | **73.5** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 47.3 +/- 1.7 | 68.0 +/- 0.5 | 53.6 +/- 0.4 | 52.7 +/- 1.0 | 72.7 +/- 1.6 | 99.7 +/- 0.3 | <u>99.7 +/- 0.3</u> | 70.5 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.0 +/- 0.0 | 1.5 +/- 2.1 | 40.3 +/- 14.2 | 11.7 +/- 7.0 | 50.2 +/- 0.1 | 50.0 +/- 9.1 | 47.1 +/- 5.1 | 28.7 |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 42.9 +/- 1.2 | 64.3 +/- 0.7 | 52.5 +/- 0.9 | 58.1 +/- 1.1 | 71.0 +/- 0.3 | 99.1 +/- 0.2 | 99.0 +/- 0.0 | 69.6 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | **52.7 +/- 0.4** | 69.1 +/- 0.6 | 57.5 +/- 0.7 | 58.7 +/- 1.0 | <u>74.9 +/- 1.0</u> | **99.9 +/- 0.1** | **100.0 +/- 0.0** | <u>73.3</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 51.5 +/- 0.5 | <u>69.2 +/- 0.5</u> | **58.4 +/- 0.1** | 59.3 +/- 0.4 | 74.6 +/- 0.7 | 99.6 +/- 0.3 | <u>99.7 +/- 0.3</u> | 73.2 |

*Table 3. Accuracy (%) by training/inference configuration and task.*
## 5. Pearson

Pearson applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.747 +/- 0.007 | 0.715 +/- 0.007 | 0.677 +/- 0.004 | 0.744 +/- 0.001 | 0.487 +/- 0.007 | <u>0.998 +/- 0.002</u> | **1.000 +/- 0.000** | 0.767 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.739 +/- 0.020 | 0.672 +/- 0.018 | 0.641 +/- 0.017 | 0.670 +/- 0.010 | 0.445 +/- 0.017 | 0.998 +/- 0.002 | 0.993 +/- 0.013 | 0.737 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.316 +/- 0.146 | 0.526 +/- 0.049 | 0.462 +/- 0.007 | 0.403 +/- 0.081 | 0.219 +/- 0.070 | 0.374 +/- 0.109 | 0.279 +/- 0.051 | 0.369 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.603 +/- 0.012 | 0.655 +/- 0.006 | 0.560 +/- 0.010 | 0.630 +/- 0.001 | 0.409 +/- 0.017 | 0.992 +/- 0.002 | 0.941 +/- 0.011 | 0.684 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.755 +/- 0.007</u> | <u>0.734 +/- 0.006</u> | <u>0.693 +/- 0.004</u> | **0.751 +/- 0.002** | **0.504 +/- 0.017** | <u>0.998 +/- 0.002</u> | **1.000 +/- 0.000** | **0.776** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.685 +/- 0.017 | 0.698 +/- 0.008 | 0.629 +/- 0.005 | 0.667 +/- 0.013 | 0.456 +/- 0.031 | 0.993 +/- 0.006 | 0.993 +/- 0.006 | 0.732 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 6/7 | — | 0.391 +/- 0.233 | 0.497 +/- 0.017 | 0.484 +/- 0.137 | 0.049 +/- 0.007 | 0.198 +/- 0.173 | 0.280 +/- 0.064 | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.598 +/- 0.022 | 0.644 +/- 0.005 | 0.566 +/- 0.011 | 0.635 +/- 0.010 | 0.423 +/- 0.004 | 0.981 +/- 0.004 | 0.978 +/- 0.000 | 0.689 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.757 +/- 0.004** | **0.742 +/- 0.010** | 0.683 +/- 0.003 | <u>0.748 +/- 0.007</u> | <u>0.499 +/- 0.020</u> | **0.999 +/- 0.002** | **1.000 +/- 0.000** | <u>0.775</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.751 +/- 0.003 | 0.734 +/- 0.005 | **0.698 +/- 0.009** | 0.740 +/- 0.003 | 0.492 +/- 0.015 | 0.991 +/- 0.007 | <u>0.993 +/- 0.006</u> | 0.771 |

*Table 4. Pearson by training/inference configuration and task.*
## 6. Macro-F1

Macro-F1 applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.473 +/- 0.009 | 0.443 +/- 0.002 | 0.452 +/- 0.006 | <u>0.517 +/- 0.012</u> | 0.743 +/- 0.004 | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | 0.661 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.412 +/- 0.023 | 0.410 +/- 0.014 | 0.386 +/- 0.018 | 0.439 +/- 0.033 | 0.688 +/- 0.029 | 0.994 +/- 0.005 | 0.996 +/- 0.007 | 0.618 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.063 +/- 0.072 | 0.209 +/- 0.030 | 0.246 +/- 0.020 | 0.145 +/- 0.036 | 0.450 +/- 0.073 | 0.552 +/- 0.101 | 0.446 +/- 0.054 | 0.302 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.356 +/- 0.004 | 0.339 +/- 0.007 | 0.388 +/- 0.007 | 0.385 +/- 0.006 | 0.704 +/- 0.009 | 0.996 +/- 0.001 | 0.969 +/- 0.006 | 0.591 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | <u>0.480 +/- 0.006</u> | 0.477 +/- 0.008 | 0.467 +/- 0.006 | 0.516 +/- 0.009 | **0.752 +/- 0.009** | <u>0.999 +/- 0.001</u> | **1.000 +/- 0.000** | <u>0.670</u> |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.410 +/- 0.014 | 0.380 +/- 0.003 | 0.423 +/- 0.013 | 0.427 +/- 0.005 | 0.727 +/- 0.016 | 0.997 +/- 0.003 | 0.996 +/- 0.003 | 0.623 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | 0.000 +/- 0.000 | 0.013 +/- 0.018 | 0.258 +/- 0.054 | 0.098 +/- 0.028 | 0.337 +/- 0.003 | 0.410 +/- 0.144 | 0.447 +/- 0.067 | 0.223 |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.371 +/- 0.014 | 0.334 +/- 0.007 | 0.362 +/- 0.009 | 0.349 +/- 0.037 | 0.709 +/- 0.003 | 0.990 +/- 0.002 | 0.989 +/- 0.000 | 0.587 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 7/7 | **0.485 +/- 0.006** | **0.483 +/- 0.006** | <u>0.470 +/- 0.010</u> | 0.506 +/- 0.016 | <u>0.749 +/- 0.010</u> | **0.999 +/- 0.001** | **1.000 +/- 0.000** | **0.670** |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 7/7 | 0.465 +/- 0.005 | <u>0.478 +/- 0.016</u> | **0.474 +/- 0.010** | **0.527 +/- 0.004** | 0.746 +/- 0.007 | 0.995 +/- 0.004 | <u>0.996 +/- 0.003</u> | 0.669 |

*Table 5. Macro-F1 by training/inference configuration and task.*
## 7. MAE

MAE applies to four ordinal tasks; lower is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 0.633 +/- 0.014 | 0.550 +/- 0.013 | 0.463 +/- 0.007 | <u>0.500 +/- 0.010</u> | — | — | — | 0.536 |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.787 +/- 0.074 | 0.700 +/- 0.098 | 0.548 +/- 0.018 | 0.674 +/- 0.039 | — | — | — | 0.677 |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | 1.208 +/- 0.205 | 1.035 +/- 0.139 | 0.555 +/- 0.014 | 0.906 +/- 0.060 | — | — | — | 0.926 |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.829 +/- 0.010 | 0.652 +/- 0.005 | 0.556 +/- 0.008 | 0.689 +/- 0.002 | — | — | — | 0.681 |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | **0.616 +/- 0.011** | <u>0.518 +/- 0.010</u> | <u>0.447 +/- 0.011</u> | **0.494 +/- 0.001** | — | — | — | **0.519** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.715 +/- 0.030 | 0.575 +/- 0.006 | 0.497 +/- 0.002 | 0.631 +/- 0.010 | — | — | — | 0.604 |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 42, 43, 44 | 3/4 | — | 1.172 +/- 0.719 | 0.524 +/- 0.025 | 0.674 +/- 0.253 | — | — | — | — |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.828 +/- 0.027 | 0.655 +/- 0.013 | 0.521 +/- 0.010 | 0.604 +/- 0.010 | — | — | — | 0.652 |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 42, 43, 44 | 4/4 | <u>0.618 +/- 0.003</u> | **0.515 +/- 0.013** | 0.454 +/- 0.008 | <u>0.500 +/- 0.016</u> | — | — | — | <u>0.522</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 42, 43, 44 | 4/4 | 0.634 +/- 0.008 | 0.526 +/- 0.007 | **0.440 +/- 0.004** | 0.505 +/- 0.003 | — | — | — | 0.526 |

*Table 6. MAE by training/inference configuration and task.*

## 8. Cross-task method averages

| Id | Model | CoT | Train | Prompt | Inf. | Coverage | Average QWK | Average Accuracy (%) | Average Pearson | Average Macro-F1 | Average MAE |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 7/7 | 0.704 (n=4) | 73.0 (n=7) | 0.767 (n=7) | 0.661 (n=7) | 0.536 (n=4) |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 7/7 | 0.656 (n=4) | 67.3 (n=7) | 0.737 (n=7) | 0.618 (n=7) | 0.677 (n=4) |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 7/7 | 0.310 (n=4) | 41.3 (n=7) | 0.369 (n=7) | 0.302 (n=7) | 0.926 (n=4) |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 7/7 | 0.589 (n=4) | 67.4 (n=7) | 0.684 (n=7) | 0.591 (n=7) | 0.681 (n=4) |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 7/7 | **0.719 (n=4)** | **73.5 (n=7)** | **0.776 (n=7)** | <u>0.670 (n=7)</u> | **0.519 (n=4)** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 7/7 | 0.650 (n=4) | 70.5 (n=7) | 0.732 (n=7) | 0.623 (n=7) | 0.604 (n=4) |
| **Self-correct CoT** |  |  |  |  |  |  |  |  |  |  |  |
| SCL | Qwen3-4B | ✗ | Self-correct CoT SFT | Label-only | Greedy | 7/7 | 0.379 (n=3) | 28.7 (n=7) | 0.316 (n=6) | 0.223 (n=7) | 0.790 (n=3) |
| SCC | Qwen3-4B | ✓ | Self-correct CoT SFT | CoT | Greedy | 7/7 | 0.589 (n=4) | 69.6 (n=7) | 0.689 (n=7) | 0.587 (n=7) | 0.652 (n=4) |
| **Self-correct Align** |  |  |  |  |  |  |  |  |  |  |  |
| SCAL | Qwen3-4B | ✗ | Self-correct Align SFT | Label-only | Greedy | 7/7 | <u>0.718 (n=4)</u> | <u>73.3 (n=7)</u> | <u>0.775 (n=7)</u> | **0.670 (n=7)** | <u>0.522 (n=4)</u> |
| SCAC | Qwen3-4B | ✓ | Self-correct Align SFT | CoT | Greedy | 7/7 | 0.715 (n=4) | 73.2 (n=7) | 0.771 (n=7) | 0.669 (n=7) | 0.526 (n=4) |


## 9. Rebuild report

```bash
python scripts/evaluate.py --refresh-analysis-only --output_path outputs/evaluations
```
