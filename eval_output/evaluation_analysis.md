# Qwen3-4B 评测结果分析

> 自动生成于 2026-07-30T06:51:21.935442+00:00；扫描到 42 条有效 `metrics.json` 记录。
> 本文件由 `training/evaluate.py` 在每次评测后重建。

## 1. 分析范围与记号

本报告直接读取 `eval_output/<task>/<exp_name>/metrics.json`，按实验目录归档，避免不同训练方式互相覆盖。

当前覆盖条件：B-L、B-C、L-L、C-C、C→L、L→C。

配置文件命名规则：`{train}_on_{test}.yaml`，例如 `base_on_cot.yaml`、`ft_cot_on_label_only.yaml`。

| 记号 | 训练方式 | 测试 prompt | 含义 |
| --- | --- | --- | --- |
| B-L | Base | Label-only | 基座模型直接输出标签 |
| B-C | Base | CoT | 基座模型先输出推理再输出标签 |
| L-L | Label-only SFT | Label-only | 同格式 Label-only 微调与测试 |
| C-C | CoT SFT | CoT | 同格式 CoT 微调与测试 |
| C→L | CoT SFT | Label-only | CoT adapter 交叉测试 Label-only prompt |
| L→C | Label-only SFT | CoT | Label-only adapter 交叉测试 CoT prompt |
| A-C | Align SFT | CoT | Align adapter 在 CoT 测试 prompt 上评测 |
| A→L | Align SFT | Label-only | Align adapter 交叉测试 Label-only prompt |

Label-only 与 CoT 测试集按任务逐 ID、逐标签配对，因此交叉评测差值不受测试样本变化影响。
> Align 列（A-C / A→L）会在运行 `eval_output/configs/<task>/ft_align_on_*.yaml` 后自动填入。

## 2. 完整结果

下表单元格均为 `Accuracy / Macro-F1`，单位为 `%`。最后一行为 7 个任务的非加权宏平均。

| 任务 | N | B-L | B-C | L-L | C-C | C→L | L→C | A-C | A→L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actionability | 1000 | 5.6 / 8.2 | 36.0 / 33.3 | 54.0 / 49.7 | 50.2 / 46.9 | 52.6 / 47.4 | 50.5 / 42.7 | — | — |
| Grounding Specificity | 1000 | 3.2 / 4.8 | 39.1 / 31.7 | 70.2 / 51.3 | 69.3 / 46.9 | 56.1 / 36.6 | 62.9 / 48.5 | — | — |
| Helpfulness | 1000 | 2.4 / 4.4 | 34.9 / 28.6 | 61.5 / 54.0 | 56.0 / 45.4 | 57.0 / 44.9 | 60.2 / 50.5 | — | — |
| Verifiability | 788 | 6.6 / 9.0 | 47.8 / 42.6 | 60.9 / 54.6 | 50.9 / 43.9 | 42.3 / 38.4 | 55.5 / 48.6 | — | — |
| Coherence | 1046 | 23.2 / 33.9 | 63.5 / 62.9 | 78.6 / 78.6 | 77.4 / 77.4 | 71.5 / 72.6 | 77.4 / 77.4 | — | — |
| Positioning Check | 603 | 43.0 / 42.3 | 90.9 / 91.1 | 99.5 / 99.5 | 99.3 / 99.3 | 98.2 / 98.2 | 99.5 / 99.5 | — | — |
| Positioning Type | 204 | 35.3 / 35.1 | 58.3 / 58.9 | 100.0 / 100.0 | 100.0 / 100.0 | 87.7 / 87.2 | 99.5 / 99.4 | — | — |
| **任务宏平均** |  | 17.0 / 19.6 | 52.9 / 49.9 | 75.0 / 69.7 | 71.9 / 65.7 | 66.5 / 60.7 | 72.2 / 66.7 | — | — |

## 3. 跨格式迁移

这里比较同一个测试 prompt 下不同训练格式 adapter 的差异（单位：Accuracy 百分点）。

| 任务 | C→L 相对 L-L | L→C 相对 C-C | A→L 相对 L-L |
| --- | ---: | ---: | ---: |
| Actionability | -1.4 pp | +0.3 pp | — |
| Grounding Specificity | -14.1 pp | -6.4 pp | — |
| Helpfulness | -4.5 pp | +4.2 pp | — |
| Verifiability | -18.7 pp | +4.6 pp | — |
| Coherence | -7.1 pp | +0.0 pp | — |
| Positioning Check | -1.3 pp | +0.2 pp | — |
| Positioning Type | -12.3 pp | -0.5 pp | — |
| **平均** | -8.5 pp | +0.3 pp | — |

## 4. 有序评分指标

四个评审意见任务是 1–5 分有序分类。除 Accuracy 和 Macro-F1 外，使用 MAE 与 QWK 衡量距离和顺序质量。MAE 越低越好，QWK 越高越好。

| 任务 | L-L MAE / QWK | C-C MAE / QWK | C→L MAE / QWK | L→C MAE / QWK | A-C MAE / QWK | A→L MAE / QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actionability | 0.576 / 0.781 | 0.643 / 0.734 | 0.611 / 0.769 | 0.686 / 0.737 | — | — |
| Grounding Specificity | 0.488 / 0.733 | 0.533 / 0.692 | 0.645 / 0.651 | 0.604 / 0.726 | — | — |
| Helpfulness | 0.402 / 0.731 | 0.468 / 0.663 | 0.450 / 0.667 | 0.421 / 0.707 | — | — |
| Verifiability | 0.476 / 0.752 | 0.641 / 0.649 | 0.624 / 0.626 | 0.584 / 0.686 | — | — |

## 5. 格式稳定性与效率

下表为 7 个任务的非加权宏平均。`samples/s` 基于各结果中的 GPU 推理时间计算。

| 条件 | 格式有效率 | 平均输出 token | 平均 reasoning token | 平均 samples/s |
| --- | ---: | ---: | ---: | ---: |
| B-L | 33.0 | 3.0 | 0.0 | 112.6 |
| B-C | 99.4 | 150.7 | 135.3 | 16.5 |
| L-L | 100.0 | 7.0 | 0.0 | 107.6 |
| C-C | 100.0 | 131.5 | 116.5 | 17.8 |
| C→L | 99.4 | 7.0 | 0.0 | 109.3 |
| L→C | 100.0 | 123.6 | 106.2 | 19.1 |
| A-C | — | — | — | — |
| A→L | — | — | — | — |

## 6. 使用说明

| 配置文件 | 含义 |
| --- | --- |
| `base_on_cot.yaml` | Base × CoT 测试 |
| `base_on_label_only.yaml` | Base × Label-only 测试 |
| `ft_cot_on_cot.yaml` | CoT SFT × CoT 测试 |
| `ft_label_only_on_label_only.yaml` | Label-only SFT × Label-only 测试 |
| `ft_cot_on_label_only.yaml` | CoT SFT × Label-only 测试（交叉） |
| `ft_label_only_on_cot.yaml` | Label-only SFT × CoT 测试（交叉） |
| `ft_align_on_cot.yaml` | Align SFT × CoT 测试 |
| `ft_align_on_label_only.yaml` | Align SFT × Label-only 测试（交叉） |

```bash
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \
  --config eval_output/configs/rev_util_actionability/ft_align_on_cot.yaml
```

每次评测完成后，本文件会自动刷新；详细逐样本结果仍在对应实验目录的 `metrics.json` 与 `predictions.jsonl`。
