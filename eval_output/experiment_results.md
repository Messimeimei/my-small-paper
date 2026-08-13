# 实验结果汇总

> 生成时间：2026-08-13T09:09:25.328415+00:00。统计范围为 `eval_output/results` 和 `eval_output/api_results` 中已发现的完整 `metrics.json`。
> 本地实验 141 条，API 实验 32 条，共 173 条。
> 仅聚合同一模型配置的不同 seed，并报告 `均值 ± 样本标准差`；单 seed 报告单值。`CoT 2048` 作为单独模型配置展示。

## 各数据集的模型结果

数据集名称保持英文。同一模型配置的多个 seed 报告为 `均值 ± 样本标准差`；单 seed 仅报告单值。下划线表示该数据集、该指标的最优均值。

### Actionability

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 41.5 | 0.612 | 0.653 | 0.381 | 0.768 |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 41.5 | 0.618 | 0.621 | 0.397 | 0.812 |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 39.0 | 0.662 | 0.666 | 0.367 | 0.745 |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 46.1 | 0.665 | 0.682 | 0.389 | 0.716 |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 1 | 40.0 | 0.615 | 0.636 | 0.358 | 0.816 |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 1 | 40.2 | 0.660 | 0.684 | 0.325 | 0.762 |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 36.0 | 0.492 | 0.546 | 0.333 | 0.905 |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 5.6 | 0.413 | 0.532 | 0.082 | 1.069 |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | 49.6 ± 0.5 | 0.722 ± 0.003 | 0.728 ± 0.000 | 0.455 ± 0.008 | 0.661 ± 0.010 |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 51.8 ± 0.6 | 0.764 ± 0.005 | 0.767 ± 0.009 | 0.462 ± 0.008 | 0.613 ± 0.007 |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 50.7 ± 0.3 | 0.735 ± 0.004 | 0.753 ± 0.001 | 0.423 ± 0.007 | 0.688 ± 0.002 |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | 54.2 ± 0.2 | 0.781 ± 0.000 | 0.788 ± 0.001 | 0.486 ± 0.016 | 0.576 ± 0.000 |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | 49.4 ± 1.9 | 0.732 ± 0.009 | 0.738 ± 0.008 | 0.443 ± 0.013 | 0.653 ± 0.037 |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | 53.4 ± 0.2 | 0.781 ± 0.004 | 0.790 ± 0.002 | 0.501 ± 0.005 | 0.577 ± 0.015 |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | 52.8 ± 0.1 | 0.764 ± 0.003 | 0.768 ± 0.003 | 0.476 ± 0.005 | 0.593 ± 0.005 |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | <u>54.6 ± 0.8</u> | <u>0.786 ± 0.002</u> | <u>0.793 ± 0.001</u> | <u>0.516 ± 0.009</u> | <u>0.563 ± 0.007</u> |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | 54.3 | 0.750 | 0.769 | 0.474 | 0.635 |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 53.0 | 0.693 | 0.745 | 0.456 | 0.677 |

### Grounding Specificity

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 56.5 | 0.656 | 0.676 | 0.457 | 0.646 |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 55.8 | 0.661 | 0.668 | 0.422 | 0.708 |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 62.4 | 0.703 | 0.704 | 0.485 | 0.551 |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 50.0 | 0.666 | 0.707 | 0.408 | 0.768 |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 1 | 52.2 | 0.685 | 0.691 | 0.391 | 0.678 |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 39.1 | 0.518 | 0.571 | 0.317 | 0.997 |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 3.2 | 0.282 | 0.382 | 0.048 | 1.043 |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | 68.8 ± 0.8 | 0.680 ± 0.017 | 0.718 ± 0.018 | 0.468 ± 0.002 | 0.548 ± 0.022 |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 56.0 ± 0.1 | 0.686 ± 0.049 | 0.705 ± 0.029 | 0.360 ± 0.008 | 0.623 ± 0.031 |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 63.5 ± 0.8 | 0.726 ± 0.001 | 0.732 ± 0.000 | 0.468 ± 0.024 | 0.603 ± 0.002 |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | <u>71.4 ± 1.7</u> | <u>0.751 ± 0.026</u> | 0.770 ± 0.017 | <u>0.522 ± 0.013</u> | <u>0.462 ± 0.037</u> |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | 1 | 66.5 | 0.739 | 0.769 | 0.472 | 0.503 |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | 69.3 ± 1.5 | 0.701 ± 0.015 | 0.736 ± 0.019 | 0.466 ± 0.014 | 0.528 ± 0.025 |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | 71.2 ± 0.6 | 0.748 ± 0.011 | 0.773 ± 0.005 | 0.518 ± 0.024 | 0.469 ± 0.012 |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | 70.4 ± 0.5 | 0.724 ± 0.004 | 0.752 ± 0.006 | 0.509 ± 0.005 | 0.501 ± 0.004 |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | 70.5 ± 0.6 | 0.749 ± 0.002 | <u>0.776 ± 0.001</u> | 0.508 ± 0.003 | 0.472 ± 0.005 |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | 53.7 | 0.469 | 0.495 | 0.319 | 0.794 |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 50.7 | 0.286 | 0.350 | 0.236 | 0.893 |

### Helpfulness

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 36.5 | 0.431 | 0.527 | 0.320 | 0.787 |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 30.5 | 0.395 | 0.532 | 0.276 | 0.880 |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 35.8 | 0.539 | 0.612 | 0.338 | 0.720 |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 42.3 | 0.495 | 0.570 | 0.402 | 0.682 |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 34.9 | 0.388 | 0.451 | 0.286 | 0.830 |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 2.4 | 0.289 | 0.429 | 0.044 | 1.000 |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | 56.4 ± 0.5 | 0.670 ± 0.010 | 0.677 ± 0.007 | 0.456 ± 0.003 | 0.461 ± 0.010 |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 55.8 ± 1.7 | 0.669 ± 0.002 | 0.681 ± 0.001 | 0.459 ± 0.014 | 0.464 ± 0.019 |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 60.3 ± 0.1 | 0.701 ± 0.008 | 0.710 ± 0.006 | 0.500 ± 0.006 | 0.423 ± 0.002 |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | 60.3 ± 1.6 | 0.723 ± 0.011 | 0.737 ± 0.006 | 0.521 ± 0.027 | 0.412 ± 0.014 |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | 58.2 ± 3.1 | 0.682 ± 0.047 | 0.689 ± 0.047 | 0.469 ± 0.035 | 0.440 ± 0.044 |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | <u>61.1 ± 0.7</u> | 0.727 ± 0.019 | 0.737 ± 0.017 | 0.503 ± 0.044 | <u>0.400 ± 0.008</u> |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | 60.3 ± 0.6 | 0.726 ± 0.004 | 0.738 ± 0.003 | 0.528 ± 0.003 | 0.409 ± 0.008 |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | 60.7 ± 0.2 | <u>0.727 ± 0.004</u> | <u>0.742 ± 0.000</u> | <u>0.536 ± 0.008</u> | 0.408 ± 0.001 |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | 51.1 | 0.578 | 0.597 | 0.392 | 0.535 |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 52.2 | 0.573 | 0.583 | 0.322 | 0.513 |

### Verifiability

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 35.8 | 0.463 | 0.549 | 0.299 | 0.917 |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 44.3 | 0.601 | 0.626 | 0.384 | 0.746 |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | 1 | 32.7 | 0.446 | 0.475 | 0.259 | 0.972 |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 40.7 | 0.525 | 0.580 | 0.319 | 0.868 |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 39.3 | 0.552 | 0.645 | 0.330 | 0.771 |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 47.8 | 0.577 | 0.582 | 0.426 | 0.675 |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 6.6 | 0.474 | 0.556 | 0.090 | 0.767 |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | 50.8 ± 0.1 | 0.652 ± 0.005 | 0.672 ± 0.001 | 0.441 ± 0.003 | 0.641 ± 0.000 |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 44.5 ± 3.2 | 0.659 ± 0.046 | 0.691 ± 0.028 | 0.404 ± 0.029 | 0.604 ± 0.029 |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 56.1 ± 0.9 | 0.672 ± 0.019 | 0.695 ± 0.009 | 0.487 ± 0.000 | 0.588 ± 0.005 |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | 60.3 ± 0.9 | 0.739 ± 0.018 | 0.763 ± 0.002 | 0.536 ± 0.015 | 0.484 ± 0.012 |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | 52.3 ± 0.1 | 0.662 ± 0.013 | 0.684 ± 0.012 | 0.442 ± 0.007 | 0.617 ± 0.003 |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | 61.0 ± 1.3 | <u>0.770 ± 0.002</u> | <u>0.781 ± 0.003</u> | <u>0.556 ± 0.015</u> | <u>0.461 ± 0.013</u> |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | 59.8 ± 1.5 | 0.729 ± 0.011 | 0.742 ± 0.006 | 0.514 ± 0.016 | 0.499 ± 0.019 |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | 60.5 ± 0.5 | 0.764 ± 0.003 | 0.774 ± 0.001 | 0.544 ± 0.015 | 0.468 ± 0.007 |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | <u>61.0</u> | 0.660 | 0.670 | 0.427 | 0.519 |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 56.3 | 0.571 | 0.609 | 0.398 | 0.572 |

### Coherence

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 69.4 | — | 0.522 | 0.725 | — |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 63.8 | — | 0.427 | 0.671 | — |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 72.0 | — | 0.508 | 0.717 | — |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 69.6 | — | 0.447 | 0.678 | — |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 63.5 | — | 0.284 | 0.629 | — |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 23.2 | — | 0.266 | 0.339 | — |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | 77.3 ± 0.2 | — | 0.546 ± 0.004 | 0.773 ± 0.002 | — |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 71.8 ± 0.5 | — | 0.491 ± 0.011 | 0.726 ± 0.001 | — |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 77.8 ± 0.5 | — | 0.556 ± 0.010 | 0.778 ± 0.005 | — |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | 78.9 ± 0.5 | — | 0.579 ± 0.010 | 0.789 ± 0.005 | — |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | 79.5 ± 0.5 | — | 0.590 ± 0.009 | 0.795 ± 0.005 | — |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | <u>80.6 ± 2.5</u> | — | <u>0.615 ± 0.048</u> | <u>0.806 ± 0.025</u> | — |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | 77.9 ± 0.1 | — | 0.559 ± 0.003 | 0.779 ± 0.001 | — |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | 79.7 ± 0.3 | — | 0.595 ± 0.006 | 0.797 ± 0.003 | — |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | 67.3 | — | 0.377 | 0.660 | — |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 61.8 | — | 0.301 | 0.576 | — |

### Positioning Check

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 86.9 | — | 0.760 | 0.872 | — |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 66.8 | — | 0.474 | 0.666 | — |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 93.7 | — | 0.920 | 0.947 | — |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 95.9 | — | 0.916 | 0.958 | — |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 90.9 | — | 0.841 | 0.911 | — |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 43.0 | — | — | 0.423 | — |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | 99.3 ± 0.0 | — | 0.987 ± 0.000 | 0.993 ± 0.000 | — |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 87.0 ± 15.8 | — | 0.971 ± 0.010 | 0.920 ± 0.087 | — |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 99.8 ± 0.4 | — | 0.995 ± 0.007 | 0.997 ± 0.004 | — |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | 99.7 ± 0.2 | — | 0.993 ± 0.005 | 0.997 ± 0.002 | — |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | 99.7 ± 0.0 | — | 0.993 ± 0.000 | 0.997 ± 0.000 | — |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | <u>99.8 ± 0.0</u> | — | <u>0.997 ± 0.000</u> | <u>0.998 ± 0.000</u> | — |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | 99.6 ± 0.1 | — | 0.992 ± 0.002 | 0.996 ± 0.001 | — |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | 99.7 ± 0.0 | — | 0.993 ± 0.000 | 0.997 ± 0.000 | — |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | 86.1 | — | 0.756 | 0.861 | — |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 86.9 | — | 0.767 | 0.869 | — |

### Positioning Type

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 运行数 | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 1 | 89.2 | — | 0.887 | 0.907 | — |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 1 | 56.9 | — | 0.392 | 0.564 | — |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 1 | 93.6 | — | 0.965 | 0.946 | — |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 1 | 99.5 | — | 0.989 | 0.995 | — |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 1 | 58.3 | — | 0.425 | 0.589 | — |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1 | 35.3 | — | 0.158 | 0.351 | — |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 2 | <u>100.0 ± 0.0</u> | — | <u>1.000 ± 0.000</u> | <u>1.000 ± 0.000</u> | — |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 2 | 84.8 ± 4.2 | — | 0.730 ± 0.061 | 0.843 ± 0.041 | — |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 2 | 99.3 ± 0.3 | — | 0.994 ± 0.008 | 0.995 ± 0.001 | — |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 2 | <u>100.0 ± 0.0</u> | — | <u>1.000 ± 0.000</u> | <u>1.000 ± 0.000</u> | — |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 2 | <u>100.0 ± 0.0</u> | — | <u>1.000 ± 0.000</u> | <u>1.000 ± 0.000</u> | — |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 2 | <u>100.0 ± 0.0</u> | — | <u>1.000 ± 0.000</u> | <u>1.000 ± 0.000</u> | — |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 2 | <u>100.0 ± 0.0</u> | — | <u>1.000 ± 0.000</u> | <u>1.000 ± 0.000</u> | — |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 2 | <u>100.0 ± 0.0</u> | — | <u>1.000 ± 0.000</u> | <u>1.000 ± 0.000</u> | — |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 1 | 68.1 | — | 0.510 | 0.683 | — |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 1 | 59.3 | — | 0.398 | 0.594 | — |

## 按指标比较所有数据集

每张表固定一个指标，列出全部七个数据集及模型配置。`平均值（覆盖）`先对每个 seed 做跨数据集宏平均，再报告 seed 间的均值与样本标准差。

### 准确率 (%)

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | 平均值（覆盖） |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 41.5 | 56.5 | 36.5 | 35.8 | 69.4 | 86.9 | 89.2 | 59.4 (7/7) |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 41.5 | 55.8 | 30.5 | 44.3 | 63.8 | 66.8 | 56.9 | 51.4 (7/7) |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | — | — | — | 32.7 | — | — | — | 32.7 (1/7) |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 39.0 | 62.4 | 35.8 | 40.7 | 72.0 | 93.7 | 93.6 | 62.5 (7/7) |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 46.1 | 50.0 | 42.3 | 39.3 | 69.6 | 95.9 | 99.5 | 63.2 (7/7) |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 40.0 | — | — | — | — | — | — | 40.0 (1/7) |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 40.2 | 52.2 | — | — | — | — | — | 46.2 (2/7) |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 36.0 | 39.1 | 34.9 | 47.8 | 63.5 | 90.9 | 58.3 | 52.9 (7/7) |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 5.6 | 3.2 | 2.4 | 6.6 | 23.2 | 43.0 | 35.3 | 17.0 (7/7) |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 49.6 ± 0.5 | 68.8 ± 0.8 | 56.4 ± 0.5 | 50.8 ± 0.1 | 77.3 ± 0.2 | 99.3 ± 0.0 | <u>100.0 ± 0.0</u> | 71.7 ± 0.2 (7/7) |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 51.8 ± 0.6 | 56.0 ± 0.1 | 55.8 ± 1.7 | 44.5 ± 3.2 | 71.8 ± 0.5 | 87.0 ± 15.8 | 84.8 ± 4.2 | 64.5 ± 2.7 (7/7) |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 50.7 ± 0.3 | 63.5 ± 0.8 | 60.3 ± 0.1 | 56.1 ± 0.9 | 77.8 ± 0.5 | 99.8 ± 0.4 | 99.3 ± 0.3 | 72.5 ± 0.4 (7/7) |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 54.2 ± 0.2 | <u>71.4 ± 1.7</u> | 60.3 ± 1.6 | 60.3 ± 0.9 | 78.9 ± 0.5 | 99.7 ± 0.2 | <u>100.0 ± 0.0</u> | 75.0 ± 0.0 (7/7) |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | — | 66.5 | — | — | — | — | — | 66.5 (1/7) |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 49.4 ± 1.9 | 69.3 ± 1.5 | 58.2 ± 3.1 | 52.3 ± 0.1 | 79.5 ± 0.5 | 99.7 ± 0.0 | <u>100.0 ± 0.0</u> | 72.6 ± 0.4 (7/7) |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 53.4 ± 0.2 | 71.2 ± 0.6 | <u>61.1 ± 0.7</u> | 61.0 ± 1.3 | <u>80.6 ± 2.5</u> | <u>99.8 ± 0.0</u> | <u>100.0 ± 0.0</u> | <u>75.3 ± 0.3</u> (7/7) |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 52.8 ± 0.1 | 70.4 ± 0.5 | 60.3 ± 0.6 | 59.8 ± 1.5 | 77.9 ± 0.1 | 99.6 ± 0.1 | <u>100.0 ± 0.0</u> | 74.4 ± 0.3 (7/7) |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | <u>54.6 ± 0.8</u> | 70.5 ± 0.6 | 60.7 ± 0.2 | 60.5 ± 0.5 | 79.7 ± 0.3 | 99.7 ± 0.0 | <u>100.0 ± 0.0</u> | 75.1 ± 0.0 (7/7) |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 54.3 | 53.7 | 51.1 | <u>61.0</u> | 67.3 | 86.1 | 68.1 | 63.1 (7/7) |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 53.0 | 50.7 | 52.2 | 56.3 | 61.8 | 86.9 | 59.3 | 60.0 (7/7) |

### QWK

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | 平均值（覆盖） |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 0.612 | 0.656 | 0.431 | 0.463 | — | — | — | 0.541 (4/4) |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 0.618 | 0.661 | 0.395 | 0.601 | — | — | — | 0.569 (4/4) |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | — | — | — | 0.446 | — | — | — | 0.446 (1/4) |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 0.662 | 0.703 | 0.539 | 0.525 | — | — | — | 0.607 (4/4) |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 0.665 | 0.666 | 0.495 | 0.552 | — | — | — | 0.594 (4/4) |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 0.615 | — | — | — | — | — | — | 0.615 (1/4) |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 0.660 | 0.685 | — | — | — | — | — | 0.673 (2/4) |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 0.492 | 0.518 | 0.388 | 0.577 | — | — | — | 0.494 (4/4) |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 0.413 | 0.282 | 0.289 | 0.474 | — | — | — | 0.364 (4/4) |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 0.722 ± 0.003 | 0.680 ± 0.017 | 0.670 ± 0.010 | 0.652 ± 0.005 | — | — | — | 0.681 ± 0.001 (4/4) |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 0.764 ± 0.005 | 0.686 ± 0.049 | 0.669 ± 0.002 | 0.659 ± 0.046 | — | — | — | 0.694 ± 0.023 (4/4) |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 0.735 ± 0.004 | 0.726 ± 0.001 | 0.701 ± 0.008 | 0.672 ± 0.019 | — | — | — | 0.708 ± 0.008 (4/4) |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 0.781 ± 0.000 | <u>0.751 ± 0.026</u> | 0.723 ± 0.011 | 0.739 ± 0.018 | — | — | — | 0.749 ± 0.001 (4/4) |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | — | 0.739 | — | — | — | — | — | 0.739 (1/4) |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 0.732 ± 0.009 | 0.701 ± 0.015 | 0.682 ± 0.047 | 0.662 ± 0.013 | — | — | — | 0.694 ± 0.007 (4/4) |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 0.781 ± 0.004 | 0.748 ± 0.011 | 0.727 ± 0.019 | <u>0.770 ± 0.002</u> | — | — | — | 0.757 ± 0.008 (4/4) |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 0.764 ± 0.003 | 0.724 ± 0.004 | 0.726 ± 0.004 | 0.729 ± 0.011 | — | — | — | 0.736 ± 0.004 (4/4) |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | <u>0.786 ± 0.002</u> | 0.749 ± 0.002 | <u>0.727 ± 0.004</u> | 0.764 ± 0.003 | — | — | — | <u>0.757 ± 0.003</u> (4/4) |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 0.750 | 0.469 | 0.578 | 0.660 | — | — | — | 0.614 (4/4) |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 0.693 | 0.286 | 0.573 | 0.571 | — | — | — | 0.531 (4/4) |

### Pearson

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | 平均值（覆盖） |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 0.653 | 0.676 | 0.527 | 0.549 | 0.522 | 0.760 | 0.887 | 0.653 (7/7) |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 0.621 | 0.668 | 0.532 | 0.626 | 0.427 | 0.474 | 0.392 | 0.534 (7/7) |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | — | — | — | 0.475 | — | — | — | 0.475 (1/7) |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 0.666 | 0.704 | 0.612 | 0.580 | 0.508 | 0.920 | 0.965 | 0.708 (7/7) |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 0.682 | 0.707 | 0.570 | 0.645 | 0.447 | 0.916 | 0.989 | 0.708 (7/7) |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 0.636 | — | — | — | — | — | — | 0.636 (1/7) |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 0.684 | 0.691 | — | — | — | — | — | 0.687 (2/7) |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 0.546 | 0.571 | 0.451 | 0.582 | 0.284 | 0.841 | 0.425 | 0.528 (7/7) |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 0.532 | 0.382 | 0.429 | 0.556 | 0.266 | — | 0.158 | 0.387 (6/7) |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 0.728 ± 0.000 | 0.718 ± 0.018 | 0.677 ± 0.007 | 0.672 ± 0.001 | 0.546 ± 0.004 | 0.987 ± 0.000 | <u>1.000 ± 0.000</u> | 0.761 ± 0.002 (7/7) |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 0.767 ± 0.009 | 0.705 ± 0.029 | 0.681 ± 0.001 | 0.691 ± 0.028 | 0.491 ± 0.011 | 0.971 ± 0.010 | 0.730 ± 0.061 | 0.719 ± 0.002 (7/7) |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 0.753 ± 0.001 | 0.732 ± 0.000 | 0.710 ± 0.006 | 0.695 ± 0.009 | 0.556 ± 0.010 | 0.995 ± 0.007 | 0.994 ± 0.008 | 0.776 ± 0.001 (7/7) |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 0.788 ± 0.001 | 0.770 ± 0.017 | 0.737 ± 0.006 | 0.763 ± 0.002 | 0.579 ± 0.010 | 0.993 ± 0.005 | <u>1.000 ± 0.000</u> | 0.804 ± 0.003 (7/7) |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | — | 0.769 | — | — | — | — | — | 0.769 (1/7) |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 0.738 ± 0.008 | 0.736 ± 0.019 | 0.689 ± 0.047 | 0.684 ± 0.012 | 0.590 ± 0.009 | 0.993 ± 0.000 | <u>1.000 ± 0.000</u> | 0.776 ± 0.002 (7/7) |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 0.790 ± 0.002 | 0.773 ± 0.005 | 0.737 ± 0.017 | <u>0.781 ± 0.003</u> | <u>0.615 ± 0.048</u> | <u>0.997 ± 0.000</u> | <u>1.000 ± 0.000</u> | <u>0.813 ± 0.004</u> (7/7) |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 0.768 ± 0.003 | 0.752 ± 0.006 | 0.738 ± 0.003 | 0.742 ± 0.006 | 0.559 ± 0.003 | 0.992 ± 0.002 | <u>1.000 ± 0.000</u> | 0.793 ± 0.001 (7/7) |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | <u>0.793 ± 0.001</u> | <u>0.776 ± 0.001</u> | <u>0.742 ± 0.000</u> | 0.774 ± 0.001 | 0.595 ± 0.006 | 0.993 ± 0.000 | <u>1.000 ± 0.000</u> | 0.811 ± 0.000 (7/7) |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 0.769 | 0.495 | 0.597 | 0.670 | 0.377 | 0.756 | 0.510 | 0.596 (7/7) |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 0.745 | 0.350 | 0.583 | 0.609 | 0.301 | 0.767 | 0.398 | 0.536 (7/7) |

### Macro-F1

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | 平均值（覆盖） |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 0.381 | 0.457 | 0.320 | 0.299 | 0.725 | 0.872 | 0.907 | 0.566 (7/7) |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 0.397 | 0.422 | 0.276 | 0.384 | 0.671 | 0.666 | 0.564 | 0.483 (7/7) |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | — | — | — | 0.259 | — | — | — | 0.259 (1/7) |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 0.367 | 0.485 | 0.338 | 0.319 | 0.717 | 0.947 | 0.946 | 0.589 (7/7) |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 0.389 | 0.408 | 0.402 | 0.330 | 0.678 | 0.958 | 0.995 | 0.594 (7/7) |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 0.358 | — | — | — | — | — | — | 0.358 (1/7) |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 0.325 | 0.391 | — | — | — | — | — | 0.358 (2/7) |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 0.333 | 0.317 | 0.286 | 0.426 | 0.629 | 0.911 | 0.589 | 0.499 (7/7) |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 0.082 | 0.048 | 0.044 | 0.090 | 0.339 | 0.423 | 0.351 | 0.196 (7/7) |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 0.455 ± 0.008 | 0.468 ± 0.002 | 0.456 ± 0.003 | 0.441 ± 0.003 | 0.773 ± 0.002 | 0.993 ± 0.000 | <u>1.000 ± 0.000</u> | 0.655 ± 0.001 (7/7) |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 0.462 ± 0.008 | 0.360 ± 0.008 | 0.459 ± 0.014 | 0.404 ± 0.029 | 0.726 ± 0.001 | 0.920 ± 0.087 | 0.843 ± 0.041 | 0.596 ± 0.014 (7/7) |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 0.423 ± 0.007 | 0.468 ± 0.024 | 0.500 ± 0.006 | 0.487 ± 0.000 | 0.778 ± 0.005 | 0.997 ± 0.004 | 0.995 ± 0.001 | 0.664 ± 0.004 (7/7) |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 0.486 ± 0.016 | <u>0.522 ± 0.013</u> | 0.521 ± 0.027 | 0.536 ± 0.015 | 0.789 ± 0.005 | 0.997 ± 0.002 | <u>1.000 ± 0.000</u> | 0.693 ± 0.005 (7/7) |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | — | 0.472 | — | — | — | — | — | 0.472 (1/7) |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 0.443 ± 0.013 | 0.466 ± 0.014 | 0.469 ± 0.035 | 0.442 ± 0.007 | 0.795 ± 0.005 | 0.997 ± 0.000 | <u>1.000 ± 0.000</u> | 0.659 ± 0.005 (7/7) |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 0.501 ± 0.005 | 0.518 ± 0.024 | 0.503 ± 0.044 | <u>0.556 ± 0.015</u> | <u>0.806 ± 0.025</u> | <u>0.998 ± 0.000</u> | <u>1.000 ± 0.000</u> | 0.698 ± 0.005 (7/7) |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 0.476 ± 0.005 | 0.509 ± 0.005 | 0.528 ± 0.003 | 0.514 ± 0.016 | 0.779 ± 0.001 | 0.996 ± 0.001 | <u>1.000 ± 0.000</u> | 0.686 ± 0.002 (7/7) |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | <u>0.516 ± 0.009</u> | 0.508 ± 0.003 | <u>0.536 ± 0.008</u> | 0.544 ± 0.015 | 0.797 ± 0.003 | 0.997 ± 0.000 | <u>1.000 ± 0.000</u> | <u>0.700 ± 0.004</u> (7/7) |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 0.474 | 0.319 | 0.392 | 0.427 | 0.660 | 0.861 | 0.683 | 0.545 (7/7) |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 0.456 | 0.236 | 0.322 | 0.398 | 0.576 | 0.869 | 0.594 | 0.493 (7/7) |

### MAE

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | Actionability | Grounding Specificity | Helpfulness | Verifiability | Coherence | Positioning Check | Positioning Type | 平均值（覆盖） |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 0.768 | 0.646 | 0.787 | 0.917 | — | — | — | 0.779 (4/4) |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 0.812 | 0.708 | 0.880 | 0.746 | — | — | — | 0.786 (4/4) |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | — | — | — | 0.972 | — | — | — | 0.972 (1/4) |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 0.745 | 0.551 | 0.720 | 0.868 | — | — | — | 0.721 (4/4) |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 0.716 | 0.768 | 0.682 | 0.771 | — | — | — | 0.734 (4/4) |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 0.816 | — | — | — | — | — | — | 0.816 (1/4) |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 0.762 | 0.678 | — | — | — | — | — | 0.720 (2/4) |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 0.905 | 0.997 | 0.830 | 0.675 | — | — | — | 0.852 (4/4) |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 1.069 | 1.043 | 1.000 | 0.767 | — | — | — | 0.970 (4/4) |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 0.661 ± 0.010 | 0.548 ± 0.022 | 0.461 ± 0.010 | 0.641 ± 0.000 | — | — | — | 0.578 ± 0.005 (4/4) |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 0.613 ± 0.007 | 0.623 ± 0.031 | 0.464 ± 0.019 | 0.604 ± 0.029 | — | — | — | 0.576 ± 0.012 (4/4) |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 0.688 ± 0.002 | 0.603 ± 0.002 | 0.423 ± 0.002 | 0.588 ± 0.005 | — | — | — | 0.575 ± 0.002 (4/4) |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 0.576 ± 0.000 | <u>0.462 ± 0.037</u> | 0.412 ± 0.014 | 0.484 ± 0.012 | — | — | — | 0.483 ± 0.003 (4/4) |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | — | 0.503 | — | — | — | — | — | 0.503 (1/4) |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 0.653 ± 0.037 | 0.528 ± 0.025 | 0.440 ± 0.044 | 0.617 ± 0.003 | — | — | — | 0.560 ± 0.013 (4/4) |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | 0.577 ± 0.015 | 0.469 ± 0.012 | <u>0.400 ± 0.008</u> | <u>0.461 ± 0.013</u> | — | — | — | <u>0.477 ± 0.006</u> (4/4) |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 0.593 ± 0.005 | 0.501 ± 0.004 | 0.409 ± 0.008 | 0.499 ± 0.019 | — | — | — | 0.500 ± 0.007 (4/4) |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | <u>0.563 ± 0.007</u> | 0.472 ± 0.005 | 0.408 ± 0.001 | 0.468 ± 0.007 | — | — | — | 0.478 ± 0.003 (4/4) |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 0.635 | 0.794 | 0.535 | 0.519 | — | — | — | 0.621 (4/4) |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 0.677 | 0.893 | 0.513 | 0.572 | — | — | — | 0.664 (4/4) |

## 模型 / 配置平均结果汇总

括号内为数据集覆盖数。最佳值只在完整覆盖适用数据集的配置中比较；MAE 越低越好，其余指标越高越好。`—` 表示缺失或不适用。

| 来源 | 模型 / 配置 | 训练 | Prompt | 推理 | Seed | 准确率 (%) | QWK | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| API | deepseek-v4-pro | API baseline | cot | greedy | — | 59.4 (7/7) | 0.541 (4/4) | 0.653 (7/7) | 0.566 (7/7) | 0.779 (4/4) |
| API | deepseek-v4-pro | API baseline | label_only | greedy | — | 51.4 (7/7) | 0.569 (4/4) | 0.534 (7/7) | 0.483 (7/7) | 0.786 (4/4) |
| API | deepseek-v4-pro (CoT 2048) | API baseline | cot | greedy | — | 32.7 (1/7) | 0.446 (1/4) | 0.475 (1/7) | 0.259 (1/7) | 0.972 (1/4) |
| API | doubao-seed-2-0-pro-260215 | API baseline | cot | greedy | — | 62.5 (7/7) | 0.607 (4/4) | 0.708 (7/7) | 0.589 (7/7) | 0.721 (4/4) |
| API | doubao-seed-2-0-pro-260215 | API baseline | label_only | greedy | — | 63.2 (7/7) | 0.594 (4/4) | 0.708 (7/7) | 0.594 (7/7) | 0.734 (4/4) |
| API | gpt-4.1-2025-04-14 | API baseline | cot | greedy | — | 40.0 (1/7) | 0.615 (1/4) | 0.636 (1/7) | 0.358 (1/7) | 0.816 (1/4) |
| API | gpt-4.1-2025-04-14 | API baseline | label_only | greedy | — | 46.2 (2/7) | 0.673 (2/4) | 0.687 (2/7) | 0.358 (2/7) | 0.720 (2/4) |
| 本地 | Qwen3-4B / B-C | Qwen3-4B Base | CoT | Greedy | base | 52.9 (7/7) | 0.494 (4/4) | 0.528 (7/7) | 0.499 (7/7) | 0.852 (4/4) |
| 本地 | Qwen3-4B / B-L | Qwen3-4B Base | Label-only | Greedy | base | 17.0 (7/7) | 0.364 (4/4) | 0.387 (6/7) | 0.196 (7/7) | 0.970 (4/4) |
| 本地 | Qwen3-4B / CC | CoT SFT | CoT | Greedy | 42, 43 | 71.7 ± 0.2 (7/7) | 0.681 ± 0.001 (4/4) | 0.761 ± 0.002 (7/7) | 0.655 ± 0.001 (7/7) | 0.578 ± 0.005 (4/4) |
| 本地 | Qwen3-4B / CL | CoT SFT | Label-only | Greedy | 42, 43 | 64.5 ± 2.7 (7/7) | 0.694 ± 0.023 (4/4) | 0.719 ± 0.002 (7/7) | 0.596 ± 0.014 (7/7) | 0.576 ± 0.012 (4/4) |
| 本地 | Qwen3-4B / LC | Label-only SFT | CoT | Greedy | 42, 43 | 72.5 ± 0.4 (7/7) | 0.708 ± 0.008 (4/4) | 0.776 ± 0.001 (7/7) | 0.664 ± 0.004 (7/7) | 0.575 ± 0.002 (4/4) |
| 本地 | Qwen3-4B / LL | Label-only SFT | Label-only | Greedy | 42, 43 | 75.0 ± 0.0 (7/7) | 0.749 ± 0.001 (4/4) | 0.804 ± 0.003 (7/7) | 0.693 ± 0.005 (7/7) | 0.483 ± 0.003 (4/4) |
| 本地 | Qwen3-4B / LL-R | Label-only CE | Label-only | RAIL | 42 | 66.5 (1/7) | 0.739 (1/4) | 0.769 (1/7) | 0.472 (1/7) | 0.503 (1/4) |
| 本地 | Qwen3-4B / PAC | Paper Align SFT | CoT | Greedy | 42, 43 | 72.6 ± 0.4 (7/7) | 0.694 ± 0.007 (4/4) | 0.776 ± 0.002 (7/7) | 0.659 ± 0.005 (7/7) | 0.560 ± 0.013 (4/4) |
| 本地 | Qwen3-4B / PAL | Paper Align SFT | Label-only | Greedy | 42, 43 | <u>75.3 ± 0.3</u> (7/7) | 0.757 ± 0.008 (4/4) | <u>0.813 ± 0.004</u> (7/7) | 0.698 ± 0.005 (7/7) | <u>0.477 ± 0.006</u> (4/4) |
| 本地 | Qwen3-4B / SCAC | Self-correct Align SFT | CoT | Greedy | 42, 43 | 74.4 ± 0.3 (7/7) | 0.736 ± 0.004 (4/4) | 0.793 ± 0.001 (7/7) | 0.686 ± 0.002 (7/7) | 0.500 ± 0.007 (4/4) |
| 本地 | Qwen3-4B / SCAL | Self-correct Align SFT | Label-only | Greedy | 42, 43 | 75.1 ± 0.0 (7/7) | <u>0.757 ± 0.003</u> (4/4) | 0.811 ± 0.000 (7/7) | <u>0.700 ± 0.004</u> (7/7) | 0.478 ± 0.003 (4/4) |
| 本地 | SciRM-7B / SciRM-C | SciRM-7B RL | CoT | Greedy | base | 63.1 (7/7) | 0.614 (4/4) | 0.596 (7/7) | 0.545 (7/7) | 0.621 (4/4) |
| 本地 | SciRM-7B / SciRM-L | SciRM-7B RL | Label-only | Greedy | base | 60.0 (7/7) | 0.531 (4/4) | 0.536 (7/7) | 0.493 (7/7) | 0.664 (4/4) |
