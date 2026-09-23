# Qwen3-4B and SciRM-7B evaluation results

> Generated at 2026-09-22T19:49:47.110228+00:00; 18 deduplicated task/configuration/seed records are included.
> This file is rebuilt by `scripts/evaluate.py` after evaluation.

## 1. Reporting protocol

Included configurations: LL, LC, CL, CC, PAL, PAC.

Tables follow the main-result layout of TRACT: training and inference configurations are listed on the left, tasks are expanded across columns, and the strict macro average is reported last. **Bold** marks the best result and <u>underline</u> marks the second-best distinct result in each column. Ties share a rank. MAE is ranked in ascending order; all other metrics are ranked in descending order.

`CoT` indicates whether the evaluation prompt requests an explicit rationale; it does not indicate hidden/internal model reasoning. Multi-seed cells report `mean +/- sample standard deviation`. Variances remain available in `evaluation_analysis_records.json`. An average is shown only when a configuration covers every task to which that metric applies; `—` means not applicable or unavailable.

Task sample counts: Actionability=1000; Grounding Specificity=1000; Helpfulness=1000; Verifiability=—; Coherence=—; Positioning Check=—; Positioning Type=—.

## 2. Main results by primary metric

The primary metric is QWK for the four ordinal review-utility tasks and Macro-F1 for the three binary writing-quality tasks.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42 | 3/7 | **0.788** | <u>0.734</u> | **0.732** | — | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42 | 3/7 | 0.776 | 0.699 | 0.700 | — | — | — | — | — |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42 | 3/7 | 0.774 | 0.715 | 0.673 | — | — | — | — | — |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42 | 3/7 | 0.740 | 0.687 | 0.686 | — | — | — | — | — |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42 | 3/7 | <u>0.784</u> | **0.734** | <u>0.729</u> | — | — | — | — | — |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42 | 3/7 | 0.737 | 0.698 | 0.696 | — | — | — | — | — |

*Table 1. Primary-metric results. Average requires complete coverage of all seven tasks.*

## 3. QWK

QWK applies to four ordinal tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42 | 3/4 | **0.788** | <u>0.734</u> | **0.732** | — | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42 | 3/4 | 0.776 | 0.699 | 0.700 | — | — | — | — | — |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42 | 3/4 | 0.774 | 0.715 | 0.673 | — | — | — | — | — |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42 | 3/4 | 0.740 | 0.687 | 0.686 | — | — | — | — | — |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42 | 3/4 | <u>0.784</u> | **0.734** | <u>0.729</u> | — | — | — | — | — |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42 | 3/4 | 0.737 | 0.698 | 0.696 | — | — | — | — | — |

*Table 2. QWK by training/inference configuration and task.*
## 4. Accuracy (%)

Accuracy (%) applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42 | 3/7 | 54.2 | <u>70.8</u> | <u>61.7</u> | — | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42 | 3/7 | **57.5** | 66.1 | 59.1 | — | — | — | — | — |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42 | 3/7 | 51.8 | 61.0 | 54.1 | — | — | — | — | — |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42 | 3/7 | 50.5 | 69.3 | 57.0 | — | — | — | — | — |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42 | 3/7 | <u>55.5</u> | 70.6 | **61.9** | — | — | — | — | — |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42 | 3/7 | 49.3 | **70.9** | 58.6 | — | — | — | — | — |

*Table 3. Accuracy (%) by training/inference configuration and task.*
## 5. Pearson

Pearson applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42 | 3/7 | **0.794** | <u>0.755</u> | **0.743** | — | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42 | 3/7 | 0.782 | 0.726 | 0.711 | — | — | — | — | — |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42 | 3/7 | 0.774 | 0.735 | 0.685 | — | — | — | — | — |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42 | 3/7 | 0.744 | 0.715 | 0.694 | — | — | — | — | — |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42 | 3/7 | <u>0.789</u> | **0.760** | <u>0.741</u> | — | — | — | — | — |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42 | 3/7 | 0.743 | 0.726 | 0.703 | — | — | — | — | — |

*Table 4. Pearson by training/inference configuration and task.*
## 6. Macro-F1

Macro-F1 applies to all seven tasks; higher is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42 | 3/7 | <u>0.504</u> | <u>0.525</u> | <u>0.527</u> | — | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42 | 3/7 | 0.470 | 0.484 | 0.478 | — | — | — | — | — |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42 | 3/7 | 0.449 | 0.409 | 0.451 | — | — | — | — | — |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42 | 3/7 | 0.464 | 0.503 | 0.498 | — | — | — | — | — |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42 | 3/7 | **0.522** | **0.535** | **0.536** | — | — | — | — | — |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42 | 3/7 | 0.452 | 0.508 | 0.484 | — | — | — | — | — |

*Table 5. Macro-F1 by training/inference configuration and task.*
## 7. MAE

MAE applies to four ordinal tasks; lower is better.

| Id | Model | CoT | Train | Prompt | Inf. | Seed | Coverage | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | Average |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 42 | 3/4 | <u>0.562</u> | **0.484** | **0.395** | — | — | — | — | — |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 42 | 3/4 | 0.590 | 0.595 | 0.437 | — | — | — | — | — |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 42 | 3/4 | 0.620 | 0.556 | 0.489 | — | — | — | — | — |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 42 | 3/4 | 0.635 | 0.542 | 0.454 | — | — | — | — | — |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 42 | 3/4 | **0.557** | <u>0.486</u> | <u>0.396</u> | — | — | — | — | — |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 42 | 3/4 | 0.649 | 0.517 | 0.432 | — | — | — | — | — |

*Table 6. MAE by training/inference configuration and task.*

## 8. Cross-task method averages

| Id | Model | CoT | Train | Prompt | Inf. | Coverage | Average QWK | Average Accuracy (%) | Average Pearson | Average Macro-F1 | Average MAE |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Standard fine-tuning** |  |  |  |  |  |  |  |  |  |  |  |
| LL | Qwen3-4B | ✗ | Label-only SFT | Label-only | Greedy | 3/7 | **0.751 (n=3)** | <u>62.2 (n=3)</u> | **0.764 (n=3)** | <u>0.519 (n=3)</u> | <u>0.480 (n=3)</u> |
| LC | Qwen3-4B | ✓ | Label-only SFT | CoT | Greedy | 3/7 | 0.725 (n=3) | 60.9 (n=3) | 0.740 (n=3) | 0.478 (n=3) | 0.541 (n=3) |
| CL | Qwen3-4B | ✗ | CoT SFT | Label-only | Greedy | 3/7 | 0.721 (n=3) | 55.6 (n=3) | 0.731 (n=3) | 0.437 (n=3) | 0.555 (n=3) |
| CC | Qwen3-4B | ✓ | CoT SFT | CoT | Greedy | 3/7 | 0.704 (n=3) | 58.9 (n=3) | 0.718 (n=3) | 0.488 (n=3) | 0.544 (n=3) |
| **Paper Align** |  |  |  |  |  |  |  |  |  |  |  |
| PAL | Qwen3-4B | ✗ | Paper Align SFT | Label-only | Greedy | 3/7 | <u>0.749 (n=3)</u> | **62.7 (n=3)** | <u>0.763 (n=3)</u> | **0.531 (n=3)** | **0.480 (n=3)** |
| PAC | Qwen3-4B | ✓ | Paper Align SFT | CoT | Greedy | 3/7 | 0.710 (n=3) | 59.6 (n=3) | 0.724 (n=3) | 0.481 (n=3) | 0.533 (n=3) |


## 9. Rebuild report

```bash
python scripts/evaluate.py --refresh-analysis-only --output_path outputs/evaluations
```
