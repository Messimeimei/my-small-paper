# 接口切换 rationale 支持性总分析

目录按任务划分；每个任务下再分 label_only_sft（LL→LC）和 cot_sft（CL→CC）。原始 API 响应、筛选样本、裁判结果、失败记录和任务分析均保存在对应子目录。

## 完成情况

| 配对 | 任务 | 已选 | 已完成 | 未完成 | 覆盖率 | 原始 API 文件 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| label_only_sft | rev_util_actionability | 50 | 48 | 2 | 96.0% | 107 |
| cot_sft | rev_util_actionability | 50 | 47 | 3 | 94.0% | 111 |
| label_only_sft | rev_util_grounding_specificity | 50 | 43 | 7 | 86.0% | 127 |
| cot_sft | rev_util_grounding_specificity | 50 | 50 | 0 | 100.0% | 161 |
| label_only_sft | rev_util_helpfulness | 0 | 0 | 0 | — | 0 |
| cot_sft | rev_util_helpfulness | 8 | 8 | 0 | 100.0% | 16 |
| label_only_sft | rev_util_verifiability | 45 | 40 | 5 | 88.9% | 96 |
| cot_sft | rev_util_verifiability | 43 | 42 | 1 | 97.7% | 94 |

## 方向与共识结论

| 配对 | 方向 | 已完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | --- | ---: | ---: | ---: | ---: |
| label_only_sft | 有害 | 73 | 72 | 1 | 0 |
| label_only_sft | 有益 | 58 | 0 | 58 | 0 |
| cot_sft | 有害 | 86 | 86 | 0 | 0 |
| cot_sft | 有益 | 61 | 0 | 61 | 0 |

## 总结

- LL→LC 有害样本中，形成共识的 rationale 几乎都支持错误分数；有益样本全部支持正确分数。
- CL→CC 有害样本中，形成共识的 rationale 全部支持错误分数；有益样本全部支持正确分数。
- 主要错误类型是 rubric_misapplication、score_mapping_error、evidence_misread 和 factual_error。
- 未完成项均保留在各任务 judge_failures.jsonl 中，不计入共识比例。

## 错误定义与统计口径

本节只统计已完成共识的有害样本，即 Label-only 接口预测正确、CoT 接口变成严重错误的样本。只采用与最终共识标签一致的裁判标注；同一裁判句子在同一样本内去重。一个样本可以同时包含多种错误类型，所以错误类型比例之和可以超过 100%。句序统计按照英文句末的 .、!、? 边界切分 rationale，再把裁判返回的原始错误片段映射回对应句子；未匹配片段不进入句序计数。

| 错误类型 | 定义 |
| --- | --- |
| factual_error | 事实错误：陈述与待评价文本或评分标准中的事实不符。 |
| evidence_misread | 证据误读：忽略、曲解或错误归因于待评价文本中的证据。 |
| rubric_misapplication | 标准误用：没有按评分标准的维度或门槛判断。 |
| score_mapping_error | 分数映射错误：理由对应的等级与分数档位含义不匹配。 |
| unsupported_inference | 无依据推断：从材料中推不出的结论，或凭空添加信息。 |
| internal_contradiction | 内部矛盾：理由中的不同句子彼此冲突。 |
| irrelevant_or_missing_reasoning | 无关或关键缺失：理由与评分无关，或漏掉决定分数的关键证据。 |
| other | 其他明确影响分数判断、但不属于以上类别的错误。 |

## 有害样本中的错误类型分布

下表的“样本数”表示至少出现一次该错误类型的有害样本数，不是错误句子总条数。

| 错误类型 | LL→LC 样本数 | LL→LC 比例 | CL→CC 样本数 | CL→CC 比例 |
| --- | ---: | ---: | ---: | ---: |
| factual_error | 40 | 54.8% | 41 | 47.7% |
| evidence_misread | 43 | 58.9% | 52 | 60.5% |
| rubric_misapplication | 68 | 93.2% | 83 | 96.5% |
| score_mapping_error | 45 | 61.6% | 58 | 67.4% |
| unsupported_inference | 15 | 20.5% | 17 | 19.8% |
| internal_contradiction | 2 | 2.7% | 1 | 1.2% |
| irrelevant_or_missing_reasoning | 1 | 1.4% | 1 | 1.2% |
| other | 0 | 0.0% | 0 | 0.0% |

## 错误从 rationale 的第几句开始

| 配对 | 已完成有害样本 | rationale 平均句数 | 首错平均句序 | 首错中位句序 | 错误原句匹配率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LL→LC | 73 | 4.85 | 1.96 | 2 | 98.1% |
| CL→CC | 86 | 5.16 | 1.95 | 2 | 97.8% |

### 首个错误句的绝对位置

| 配对 | 第1句 | 第2句 | 第3句 | 第4句及以后 | 前两句合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LL→LC | 30 (41.1%) | 22 (30.1%) | 15 (20.5%) | 6 (8.2%) | 52 (71.2%) |
| CL→CC | 38 (44.2%) | 22 (25.6%) | 18 (20.9%) | 8 (9.3%) | 60 (69.8%) |

### 首个错误句在 rationale 中的相对位置

| 配对 | 前25% | 25%–50% | 50%–75% | 后25% |
| --- | ---: | ---: | ---: | ---: |
| LL→LC | 43 (58.9%) | 23 (31.5%) | 6 (8.2%) | 1 (1.4%) |
| CL→CC | 51 (59.3%) | 30 (34.9%) | 5 (5.8%) | 0 (0.0%) |

### 所有错误句的句序分布

该表统计去重后、能够映射回原 rationale 的错误原句。一个样本可以贡献多个位置。

| 配对 | 第1句 | 第2句 | 第3句 | 第4句 | 第5句 | 第6句 | 第7句及以后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LL→LC | 31 | 49 | 64 | 68 | 39 | 13 | 1 |
| CL→CC | 41 | 51 | 71 | 64 | 53 | 20 | 6 |

### 句序结论

- 两组的首个错误平均都约出现在第 2 句，且约 70% 的样本在前两句已经出现错误，说明错误通常不是只发生在最终分数映射处。
- 第 3–4 句聚集了最多的具体错误标注：早期对证据或 rubric 的理解偏差，会在中段推导中扩散，并最终形成错误分数。
- LL→LC 与 CL→CC 的句序分布接近，说明是否经过 CoT 训练没有明显改变错误首次出现的位置；主要差异仍在具体错误内容。

## 如何阅读结果：两个具体例子

下面各取一个已形成共识的有害样本。完整样本材料位于对应的 selected_samples.jsonl，裁判汇总位于 judge_results.jsonl，未经整理的 API 返回位于 raw_responses/。

### LL→LC（Label-only SFT） 示例

- 任务：rev_util_actionability
- source_key：rev_util_actionability:42:actionability_test_0826:label_only_correct_to_cot_severe
- 真实分数：3
- Label-only 接口预测：3
- CoT 接口预测：5
- 样本方向：有害（正确→严重错误）
- 裁判共识：supports_wrong_score
- 下表采用的共识裁判：MiniMax-M3

**待评价文本**

```text
- Can you provide empirical evidence or metrics that demonstrate the stability of your training process over time.
```

**CoT rationale**

```text
The answer is a direct request for empirical evidence or metrics to demonstrate the stability of the training process. This is an explicit action that the authors can directly apply by providing the requested evidence. The comment is clear and specific, indicating exactly what the authors should do. Therefore, it meets the criteria for a score of 5.
```

**裁判指出的错误句子**

| 原始错误句子 | 错误类型 | 解释 |
| --- | --- | --- |
| The answer is a direct request for empirical evidence or metrics to demonstrate the stability of the training process. | evidence_misread | 该评论仅要求提供'经验证据或指标'，但未具体说明需要哪些指标、如何衡量稳定性等细节，因此'明确且具体'的判断不准确。 |
| This is an explicit action that the authors can directly apply by providing the requested evidence. | rubric_misapplication | 虽然行动是明确的，但评论没有说明如何执行（哪些具体指标、什么形式呈现），按评分标准应对应3分而非5分。 |
| The comment is clear and specific, indicating exactly what the authors should do. | unsupported_inference | 'exactly what the authors should do'属于无依据推断，评论只说要'经验证据或指标'，但没有具体说明哪些指标或如何测量稳定性。 |
| Therefore, it meets the criteria for a score of 5. | score_mapping_error | 根据评分标准，明确陈述行动但执行方式模糊应对应3分，给5分属于分数映射错误。 |

**如何理解**：Label-only 接口原本预测正确；切换到 CoT 后，rationale 对评分标准或证据作出了错误解释，并且该错误推理继续支持最终的严重错误分数。这就是 supports_wrong_score。

### CL→CC（CoT SFT） 示例

- 任务：rev_util_verifiability
- source_key：rev_util_verifiability:43:verifiability_test_0206:label_only_correct_to_cot_severe
- 真实分数：3
- Label-only 接口预测：3
- CoT 接口预测：1
- 样本方向：有害（正确→严重错误）
- 裁判共识：supports_wrong_score
- 下表采用的共识裁判：MiniMax-M3

**待评价文本**

```text
1. The authors run their network S times repeatedly and collect S latent regions as suggestions each iteration. They set S to 2 to achieve a good balance between accuracy and efficiency. More details are missing to support this choice.
```

**CoT rationale**

```text
The comment claims that more details are missing to support the choice of S=2. However, it provides no reasoning, evidence, or justification for why the choice is questionable or what specific details are needed. The statement is merely an assertion without any supporting argument, making it unverifiable. Therefore, the evaluation score is 1.
```

**裁判指出的错误句子**

| 原始错误句子 | 错误类型 | 解释 |
| --- | --- | --- |
| it provides no reasoning, evidence, or justification for why the choice is questionable or what specific details are needed | evidence_misread | 待评价文本明确指出了S=2的选择并说明了'more details are missing to support this choice'，即评论者指出了具体缺失的信息（支撑S=2选择的细节），并非完全没有说明需要什么支持。 |
| The statement is merely an assertion without any supporting argument, making it unverifiable | rubric_misapplication | 评论提供了部分支持（指出了具体参数选择和缺失的内容），根据评分标准应对应2-3分档，而非1分（完全无证据）。 |
| Therefore, the evaluation score is 1. | score_mapping_error | 评论并非'完全无支撑'，而是'some support but vague/insufficient'，应映射到2-3分而非1分。 |

**如何理解**：Label-only 接口原本预测正确；切换到 CoT 后，rationale 对评分标准或证据作出了错误解释，并且该错误推理继续支持最终的严重错误分数。这就是 supports_wrong_score。
