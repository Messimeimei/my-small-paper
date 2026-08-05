# 固定版本 API 基线与已有结果对照

> 自动生成于 2026-08-05T08:50:25.424891+00:00。现有 `eval_output/results` 只读，未被修改。

## Actionability

| 来源 | 模型/条件 | Prompt | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 1000 | 0.492 | 36.0 | 0.546 | 0.333 | 0.905 |
| Local | B-L: Qwen3-4B Base | Label-only | 1000 | 0.413 | 5.6 | 0.532 | 0.082 | 1.069 |
| Local | CC: CoT SFT | CoT | 1000 | 0.722 | 49.6 | 0.728 | 0.455 | 0.661 |
| Local | CL: CoT SFT | Label-only | 1000 | 0.764 | 51.8 | 0.767 | 0.462 | 0.613 |
| Local | LC: Label-only SFT | CoT | 1000 | 0.735 | 50.7 | 0.753 | 0.423 | 0.688 |
| Local | LL: Label-only SFT | Label-only | 1000 | 0.781 | 54.2 | 0.788 | 0.486 | 0.576 |
| Local | PAC: Paper Align SFT | CoT | 1000 | 0.732 | 49.4 | 0.738 | 0.443 | 0.653 |
| Local | PAL: Paper Align SFT | Label-only | 1000 | 0.781 | 53.4 | 0.790 | 0.501 | 0.577 |
| Local | SciRM-C: SciRM-7B RL | CoT | 1000 | 0.750 | 54.3 | 0.769 | 0.474 | 0.635 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 1000 | 0.693 | 53.0 | 0.745 | 0.456 | 0.677 |
| API | doubao-seed-2-0-pro-260215 | label_only | 1000 | 0.665 | 46.1 | 0.682 | 0.389 | 0.716 |
| API | gpt-4.1-2025-04-14 | label_only | 1000 | 0.660 | 40.2 | 0.684 | 0.325 | 0.762 |

## Grounding Specificity

| 来源 | 模型/条件 | Prompt | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 1000 | 0.518 | 39.1 | 0.571 | 0.317 | 0.997 |
| Local | B-L: Qwen3-4B Base | Label-only | 1000 | 0.282 | 3.2 | 0.382 | 0.048 | 1.043 |
| Local | CC: CoT SFT | CoT | 1000 | 0.680 | 68.8 | 0.718 | 0.468 | 0.548 |
| Local | CL: CoT SFT | Label-only | 1000 | 0.686 | 56.0 | 0.705 | 0.360 | 0.623 |
| Local | COT-RAFT-G: CoT-RAFT | CoT | 1000 | 0.662 | 67.2 | 0.701 | 0.447 | 0.576 |
| Local | COT-RAFT-R: CoT-RAFT | CoT | 1000 | 0.672 | 67.6 | 0.709 | 0.450 | 0.565 |
| Local | LC: Label-only SFT | CoT | 1000 | 0.726 | 63.5 | 0.732 | 0.468 | 0.603 |
| Local | LL: Label-only SFT | Label-only | 1000 | 0.751 | 71.4 | 0.770 | 0.522 | 0.462 |
| Local | LL-R: Label-only CE | Label-only | 1000 | 0.739 | 66.5 | 0.769 | 0.472 | 0.503 |
| Local | PAC: Paper Align SFT | CoT | 1000 | 0.701 | 69.3 | 0.736 | 0.466 | 0.528 |
| Local | PAL: Paper Align SFT | Label-only | 1000 | 0.748 | 71.2 | 0.773 | 0.518 | 0.469 |
| Local | RAFT-G: RAFT without CoT | Label-only | 1000 | 0.697 | 69.3 | 0.730 | 0.464 | 0.527 |
| Local | RAFT-R: RAFT without CoT | Label-only | 1000 | 0.704 | 67.7 | 0.740 | 0.445 | 0.526 |
| Local | SciRM-C: SciRM-7B RL | CoT | 1000 | 0.469 | 53.7 | 0.495 | 0.319 | 0.794 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 1000 | 0.286 | 50.7 | 0.350 | 0.236 | 0.893 |

## Helpfulness

| 来源 | 模型/条件 | Prompt | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 1000 | 0.388 | 34.9 | 0.451 | 0.286 | 0.830 |
| Local | B-L: Qwen3-4B Base | Label-only | 1000 | 0.289 | 2.4 | 0.429 | 0.044 | 1.000 |
| Local | CC: CoT SFT | CoT | 1000 | 0.670 | 56.4 | 0.677 | 0.456 | 0.461 |
| Local | CL: CoT SFT | Label-only | 1000 | 0.669 | 55.8 | 0.681 | 0.459 | 0.464 |
| Local | LC: Label-only SFT | CoT | 1000 | 0.701 | 60.3 | 0.710 | 0.500 | 0.423 |
| Local | LL: Label-only SFT | Label-only | 1000 | 0.723 | 60.3 | 0.737 | 0.521 | 0.412 |
| Local | PAC: Paper Align SFT | CoT | 1000 | 0.682 | 58.2 | 0.689 | 0.469 | 0.440 |
| Local | PAL: Paper Align SFT | Label-only | 1000 | 0.727 | 61.1 | 0.737 | 0.503 | 0.400 |
| Local | SciRM-C: SciRM-7B RL | CoT | 1000 | 0.578 | 51.1 | 0.597 | 0.392 | 0.535 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 1000 | 0.573 | 52.2 | 0.583 | 0.322 | 0.513 |

## Verifiability

| 来源 | 模型/条件 | Prompt | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 788 | 0.577 | 47.8 | 0.582 | 0.426 | 0.675 |
| Local | B-L: Qwen3-4B Base | Label-only | 788 | 0.474 | 6.6 | 0.556 | 0.090 | 0.767 |
| Local | CC: CoT SFT | CoT | 788 | 0.652 | 50.8 | 0.672 | 0.441 | 0.641 |
| Local | CL: CoT SFT | Label-only | 788 | 0.659 | 44.5 | 0.691 | 0.404 | 0.604 |
| Local | LC: Label-only SFT | CoT | 788 | 0.672 | 56.1 | 0.695 | 0.487 | 0.588 |
| Local | LL: Label-only SFT | Label-only | 788 | 0.739 | 60.3 | 0.763 | 0.536 | 0.484 |
| Local | PAC: Paper Align SFT | CoT | 788 | 0.662 | 52.3 | 0.684 | 0.442 | 0.617 |
| Local | PAL: Paper Align SFT | Label-only | 788 | 0.770 | 61.0 | 0.781 | 0.556 | 0.461 |
| Local | SciRM-C: SciRM-7B RL | CoT | 788 | 0.660 | 61.0 | 0.670 | 0.427 | 0.519 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 788 | 0.571 | 56.3 | 0.609 | 0.398 | 0.572 |

## Coherence

| 来源 | 模型/条件 | Prompt | N | Accuracy (%) | Macro-F1 | Pearson |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 1046 | 63.5 | 0.629 | 0.284 |
| Local | B-L: Qwen3-4B Base | Label-only | 1046 | 23.2 | 0.339 | 0.266 |
| Local | CC: CoT SFT | CoT | 1046 | 77.3 | 0.773 | 0.546 |
| Local | CL: CoT SFT | Label-only | 1046 | 71.8 | 0.726 | 0.491 |
| Local | LC: Label-only SFT | CoT | 1046 | 77.8 | 0.778 | 0.556 |
| Local | LL: Label-only SFT | Label-only | 1046 | 78.9 | 0.789 | 0.579 |
| Local | PAC: Paper Align SFT | CoT | 1046 | 79.5 | 0.795 | 0.590 |
| Local | PAL: Paper Align SFT | Label-only | 1046 | 80.6 | 0.806 | 0.615 |
| Local | RAFT-G: RAFT without CoT | Label-only | 1046 | 0.0 | 0.000 | — |
| Local | RAFT-R: RAFT without CoT | Label-only | 1046 | 77.2 | 0.772 | 0.546 |
| Local | SciRM-C: SciRM-7B RL | CoT | 1046 | 67.3 | 0.660 | 0.377 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 1046 | 61.8 | 0.576 | 0.301 |

## Positioning Check

| 来源 | 模型/条件 | Prompt | N | Accuracy (%) | Macro-F1 | Pearson |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 603 | 90.9 | 0.911 | 0.841 |
| Local | B-L: Qwen3-4B Base | Label-only | 603 | 43.0 | 0.423 | — |
| Local | CC: CoT SFT | CoT | 603 | 99.3 | 0.993 | 0.987 |
| Local | CL: CoT SFT | Label-only | 603 | 87.0 | 0.920 | 0.971 |
| Local | LC: Label-only SFT | CoT | 603 | 99.8 | 0.997 | 0.995 |
| Local | LL: Label-only SFT | Label-only | 603 | 99.7 | 0.997 | 0.993 |
| Local | PAC: Paper Align SFT | CoT | 603 | 99.7 | 0.997 | 0.993 |
| Local | PAL: Paper Align SFT | Label-only | 603 | 99.8 | 0.998 | 0.997 |
| Local | SciRM-C: SciRM-7B RL | CoT | 603 | 86.1 | 0.861 | 0.756 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 603 | 86.9 | 0.869 | 0.767 |

## Positioning Type

| 来源 | 模型/条件 | Prompt | N | Accuracy (%) | Macro-F1 | Pearson |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Local | B-C: Qwen3-4B Base | CoT | 204 | 58.3 | 0.589 | 0.425 |
| Local | B-L: Qwen3-4B Base | Label-only | 204 | 35.3 | 0.351 | 0.158 |
| Local | CC: CoT SFT | CoT | 204 | 100.0 | 1.000 | 1.000 |
| Local | CL: CoT SFT | Label-only | 204 | 84.8 | 0.843 | 0.730 |
| Local | LC: Label-only SFT | CoT | 204 | 99.3 | 0.995 | 0.994 |
| Local | LL: Label-only SFT | Label-only | 204 | 100.0 | 1.000 | 1.000 |
| Local | PAC: Paper Align SFT | CoT | 204 | 100.0 | 1.000 | 1.000 |
| Local | PAL: Paper Align SFT | Label-only | 204 | 100.0 | 1.000 | 1.000 |
| Local | SciRM-C: SciRM-7B RL | CoT | 204 | 68.1 | 0.683 | 0.510 |
| Local | SciRM-L: SciRM-7B RL | Label-only | 204 | 59.3 | 0.594 | 0.398 |
