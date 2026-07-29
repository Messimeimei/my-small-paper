# Qwen3-4B 评测结果分析

## 1. 分析范围与记号

本报告基于 `evaloutput` 下现有的 42 份 `metrics.json`，覆盖 7 个任务、每个任务 6 种评测条件。

| 记号 | 训练方式 | 测试 prompt | 含义 |
| --- | --- | --- | --- |
| B-S | Base | Score-only | 基座模型直接输出分数 |
| B-C | Base | CoT | 基座模型先输出推理再输出分数 |
| S-S | Score-only SFT | Score-only | 同格式直接分数微调与测试 |
| C-C | CoT SFT | CoT | 同格式 CoT 微调与测试 |
| C→S | CoT SFT | Score-only | CoT adapter 交叉测试直接打分 prompt |
| S→C | Score-only SFT | CoT | Score-only adapter 交叉测试 CoT prompt |

Score-only 与 CoT 测试集已经核实为逐 ID、逐标签配对，因此交叉评测差值不受测试样本变化影响。

> 注意：现有 `evaloutput/comparison_table.md` 的 FT 槽位只按测试 prompt 类型归档。运行交叉评测后，原来的同分布 FT 数据被交叉结果覆盖，因此该表当前不能用于区分 S-S、C-C、C→S 和 S→C。本报告直接读取各实验目录中的原始 `metrics.json` 重建结果。

## 2. 完整结果

下表单元格均为 `Accuracy / Macro-F1`，单位为 `%`。最后一行为 7 个任务的非加权宏平均。

| 任务 | N | B-S | B-C | S-S | C-C | C→S | S→C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actionability | 1000 | 5.6 / 8.2 | 36.0 / 33.3 | **54.0 / 49.7** | 50.2 / 46.9 | 52.6 / 47.4 | 50.5 / 42.7 |
| Grounding Specificity | 1000 | 3.2 / 4.8 | 39.1 / 31.7 | **70.2 / 51.3** | 69.3 / 46.9 | 56.1 / 36.6 | 62.9 / 48.5 |
| Helpfulness | 1000 | 2.4 / 4.4 | 34.9 / 28.6 | 61.5 / 54.0 | 56.0 / 45.4 | 57.0 / 44.9 | **60.2 / 50.5** |
| Verifiability | 788 | 6.6 / 9.0 | 47.8 / 42.6 | **60.9 / 54.6** | 50.9 / 43.9 | 42.3 / 38.4 | 55.5 / 48.6 |
| Coherence | 1046 | 23.2 / 33.9 | 63.5 / 62.9 | **78.6 / 78.6** | 77.4 / 77.4 | 71.5 / 72.6 | 77.4 / 77.4 |
| Positioning Check | 603 | 43.0 / 42.3 | 90.9 / 91.1 | 99.5 / 99.5 | 99.3 / 99.3 | 98.2 / 98.2 | **99.5 / 99.5** |
| Positioning Type | 204 | 35.3 / 35.1 | 58.3 / 58.9 | **100.0 / 100.0** | **100.0 / 100.0** | 87.7 / 87.2 | 99.5 / 99.4 |
| **任务宏平均** | | 17.0 / 19.6 | 52.9 / 49.9 | **75.0 / 69.7** | 71.9 / 65.7 | 66.5 / 60.7 | 72.2 / 66.7 |

## 3. 跨格式迁移

这里比较同一个测试 prompt 下不同训练格式 adapter 的差异：

- `C→S - S-S`：Score-only 测试集上，CoT adapter 相对 Score-only adapter 的变化。
- `S→C - C-C`：CoT 测试集上，Score-only adapter 相对 CoT adapter 的变化。

| 任务 | C→S 相对 S-S | S→C 相对 C-C |
| --- | ---: | ---: |
| Actionability | -1.4 pp | +0.3 pp |
| Grounding Specificity | -14.1 pp | -6.4 pp |
| Helpfulness | -4.5 pp | +4.2 pp |
| Verifiability | -18.7 pp | +4.6 pp |
| Coherence | -7.1 pp | 0.0 pp |
| Positioning Check | -1.3 pp | +0.2 pp |
| Positioning Type | -12.3 pp | -0.5 pp |
| **平均** | **-8.5 pp** | **+0.3 pp** |

迁移具有明显的不对称性：

1. CoT adapter 改用直接打分 prompt 时，Accuracy 平均下降 8.5 个百分点。Grounding Specificity、Verifiability 和 Positioning Type 的下降尤其明显。
2. Score-only adapter 改用 CoT prompt 时，Accuracy 平均变化为 +0.3 个百分点，整体上没有迁移损失。
3. 这表明 Score-only 训练获得的判别能力对 prompt 格式更鲁棒；CoT 训练则更容易与特定的推理和输出流程绑定。
4. C→S 的格式有效率仍接近 100%，因此其主要退化不是分数解析失败，而是实际分类结果退化。唯一较明显的例外是 Coherence，C→S 的格式有效率为 95.7%。

## 4. 有序评分指标

四个评审意见任务是 1–5 分有序分类。除 Accuracy 和 Macro-F1 外，还应使用 MAE 与 QWK 衡量距离和顺序质量。MAE 越低越好，QWK 越高越好。

| 任务 | S-S MAE / QWK | C-C MAE / QWK | C→S MAE / QWK | S→C MAE / QWK |
| --- | ---: | ---: | ---: | ---: |
| Actionability | **0.576 / 0.781** | 0.643 / 0.734 | 0.611 / 0.769 | 0.686 / 0.737 |
| Grounding Specificity | **0.488 / 0.733** | 0.533 / 0.692 | 0.645 / 0.651 | 0.604 / 0.726 |
| Helpfulness | **0.402 / 0.731** | 0.468 / 0.663 | 0.450 / 0.667 | 0.421 / 0.707 |
| Verifiability | **0.476 / 0.752** | 0.641 / 0.649 | 0.624 / 0.626 | 0.584 / 0.686 |

四个任务均由 S-S 取得最低 MAE 和最高 QWK。这说明 Score-only SFT 不仅精确命中率更好，对错误等级距离和整体序关系的建模也更好。

## 5. 格式稳定性与效率

下表为 7 个任务的非加权宏平均。`samples/s` 基于各结果中的 GPU 推理时间计算。

| 条件 | 格式有效率 | 平均输出 token | 平均 reasoning token | 平均 samples/s |
| --- | ---: | ---: | ---: | ---: |
| B-S | 33.0% | 3.0 | 0.0 | 112.6 |
| B-C | 99.4% | 150.7 | 135.3 | 16.5 |
| S-S | **100.0%** | **7.0** | **0.0** | **107.6** |
| C-C | 100.0% | 131.5 | 116.5 | 17.8 |
| C→S | 99.4% | 7.0 | 0.0 | 109.3 |
| S→C | 100.0% | 123.6 | 106.2 | 19.1 |

主要结论如下：

1. 基座模型 B-S 的低分不能直接解释为分类能力差。其平均格式有效率只有 33.0%，大量样本因为没有生成可解析分数而被判错。B-C 相对 B-S 的提升同时混合了格式遵循提升和分类能力提升，不能全部归因于 CoT 推理。
2. 微调后，Score-only 是当前明显的 Pareto 最优方案。S-S 相对 C-C 的平均 Accuracy 高 3.1 个百分点，Macro-F1 高 4.0 个百分点，输出 token 少约 18.8 倍，吞吐约高 6 倍。
3. Score-only adapter 在 CoT prompt 下仍会生成平均约 106 个 reasoning token，并保持分类性能。这些 reasoning 没有经过正确性、忠实性或可用性评估，因此不能据此认定其推理过程可靠。

## 6. 类别不平衡与任务难度

四个五分类任务存在明显类别不平衡，因此 Accuracy 可能高估模型对少数等级的能力。

| 任务 | 主要分布特征 | S-S 中较弱类别 |
| --- | --- | --- |
| Actionability | 3 分占 31.2%，4 分仅占 9.5% | 4 分 F1 = 18.3% |
| Grounding Specificity | 3/5 分合计占 77.2%，4 分仅占 4.1% | 2 分 F1 = 22.2%，4 分 F1 = 34.4% |
| Helpfulness | 3/4 分合计占 77.3%，1 分仅占 3.4% | 5 分 F1 = 42.1%，2 分 F1 = 45.6% |
| Verifiability | 3 分占 43.9%，5 分仅占 3.0% | 2 分 F1 = 28.2% |

因此，后续优化应优先改善少数中间等级，尤其是 2 分和 4 分，而不是只优化总体 Accuracy。

二分类任务的结果则表现出明显的难度分层：

- Positioning Check 和 Positioning Type 已接近饱和，不适合作为区分当前模型方案的主要基准。
- Positioning Type 只有 204 条，100.0% 与 99.5% 实际仅相差约一个样本，不应过度解读。
- Coherence 测试集正负样本各 523 条，类别平衡，S-S 的 78.6% Accuracy / Macro-F1 更能反映真实判别能力，也是当前二分类任务中更有区分度的基准。

## 7. 总结

当前结果支持以下判断：

1. SFT 对所有任务都有效，但基座 Score-only 与其他条件的差距受到严重的输出格式失败影响。
2. 对最终评分指标而言，没有证据表明 CoT SFT 优于 Score-only SFT。Score-only SFT 的准确率、Macro-F1、有序指标和推理效率整体更优。
3. Score-only adapter 可以较好地适应 CoT prompt；反向迁移则明显不稳定，说明 CoT adapter 对输出格式的依赖更强。
4. Positioning 系列接近饱和，后续模型比较应主要关注 Coherence 和四个五分类评审任务。
5. 五分类任务的主要问题集中在少数中间等级，Macro-F1、MAE、QWK 和逐类指标比单独的 Accuracy 更有诊断价值。
6. 当前所有评测均为 seed 42、单次确定性 rollout，没有跨 seed 方差或置信区间。小幅差异不能视为统计显著，但 C→S 的大幅下降和 Score-only 的效率优势在各任务间较为一致。
