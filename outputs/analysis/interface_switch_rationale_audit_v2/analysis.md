# 接口切换 rationale 支持性总分析

目录按任务划分；每个任务下再分 label_only_sft（LL→LC）和 cot_sft（CL→CC）。原始 API 响应、筛选样本、裁判结果、失败记录和任务分析均保存在对应子目录。

## 完成情况

| 配对 | 任务 | 已选 | 已完成 | 未完成 | 覆盖率 | 原始 API 文件 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| label_only_sft | rev_util_actionability | 176 | 171 | 5 | 97.2% | 401 |
| cot_sft | rev_util_actionability | 300 | 296 | 4 | 98.7% | 809 |
| label_only_sft | rev_util_grounding_specificity | 307 | 303 | 4 | 98.7% | 714 |
| cot_sft | rev_util_grounding_specificity | 256 | 255 | 1 | 99.6% | 586 |
| label_only_sft | rev_util_helpfulness | 0 | 0 | 0 | — | 0 |
| cot_sft | rev_util_helpfulness | 8 | 8 | 0 | 100.0% | 23 |
| label_only_sft | rev_util_verifiability | 45 | 43 | 2 | 95.6% | 140 |
| cot_sft | rev_util_verifiability | 43 | 43 | 0 | 100.0% | 133 |

## 方向与共识结论

| 配对 | 方向 | 已完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | --- | ---: | ---: | ---: | ---: |
| label_only_sft | 有害 | 368 | 367 | 1 | 0 |
| label_only_sft | 有益 | 149 | 0 | 149 | 0 |
| cot_sft | 有害 | 333 | 333 | 0 | 0 |
| cot_sft | 有益 | 269 | 0 | 269 | 0 |

## 总结

- LL→LC 有害样本中，形成共识的 rationale 几乎都支持错误分数；有益样本全部支持正确分数。
- CL→CC 有害样本中，形成共识的 rationale 全部支持错误分数；有益样本全部支持正确分数。
- 主要错误类型是 rubric_misapplication、score_mapping_error、evidence_misread 和 factual_error。
- 未完成项均保留在各任务 judge_failures.jsonl 中，不计入共识比例。

## 错误定义与统计口径

本节只统计已完成共识的有害样本，即 Label-only 接口预测正确、CoT 接口变成严重错误的样本。只采用与最终共识标签一致的裁判标注；同一裁判句子在同一样本内去重。一个样本可以同时包含多种错误类型，所以错误类型比例之和可以超过 100%。句序统计按照英文句末的 .、!、? 边界切分 rationale，再把裁判返回的原始错误片段映射回对应句子；未匹配片段不进入句序计数。

| 错误类型 | 定义 |
| --- | --- |
| factual_error | A claim contradicts the evaluated text or scoring criteria. |
| evidence_misread | Evidence in the evaluated text is ignored, distorted, or misattributed. |
| rubric_misapplication | The rationale applies the wrong rubric dimension or threshold. |
| score_mapping_error | The reasoning supports a different rubric level than the assigned score. |
| unsupported_inference | A conclusion is not warranted by the evaluation material. |
| internal_contradiction | Statements within the rationale conflict with one another. |
| irrelevant_or_missing_reasoning | The rationale is irrelevant or omits score-determining evidence. |
| other | Another explicit error materially affecting the score judgment. |

## 有害样本中的错误类型分布

下表的“样本数”表示至少出现一次该错误类型的有害样本数，不是错误句子总条数。

| 错误类型 | LL→LC 样本数 | LL→LC 比例 | CL→CC 样本数 | CL→CC 比例 |
| --- | ---: | ---: | ---: | ---: |
| factual_error | 233 | 63.3% | 195 | 58.6% |
| evidence_misread | 172 | 46.7% | 144 | 43.2% |
| rubric_misapplication | 333 | 90.5% | 328 | 98.5% |
| score_mapping_error | 287 | 78.0% | 287 | 86.2% |
| unsupported_inference | 90 | 24.5% | 179 | 53.8% |
| internal_contradiction | 26 | 7.1% | 17 | 5.1% |
| irrelevant_or_missing_reasoning | 15 | 4.1% | 1 | 0.3% |
| other | 0 | 0.0% | 0 | 0.0% |

## 错误从 rationale 的第几句开始

| 配对 | 已完成有害样本 | rationale 平均句数 | 首错平均句序 | 首错中位句序 | 错误原句匹配率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LL→LC | 368 | 4.49 | 2.48 | 3 | 98.2% |
| CL→CC | 333 | 5.04 | 2.20 | 2 | 98.5% |

### 首个错误句的绝对位置

| 配对 | 第1句 | 第2句 | 第3句 | 第4句及以后 | 前两句合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LL→LC | 76 (20.7%) | 104 (28.3%) | 134 (36.4%) | 54 (14.7%) | 180 (48.9%) |
| CL→CC | 121 (36.3%) | 84 (25.2%) | 87 (26.1%) | 41 (12.3%) | 205 (61.6%) |

### 首个错误句在 rationale 中的相对位置

| 配对 | 前25% | 25%–50% | 50%–75% | 后25% |
| --- | ---: | ---: | ---: | ---: |
| LL→LC | 108 (29.3%) | 131 (35.6%) | 113 (30.7%) | 16 (4.3%) |
| CL→CC | 164 (49.2%) | 122 (36.6%) | 39 (11.7%) | 8 (2.4%) |

### 所有错误句的句序分布

该表统计去重后、能够映射回原 rationale 的错误原句。一个样本可以贡献多个位置。

| 配对 | 第1句 | 第2句 | 第3句 | 第4句 | 第5句 | 第6句 | 第7句及以后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LL→LC | 76 | 138 | 276 | 302 | 142 | 37 | 8 |
| CL→CC | 130 | 157 | 184 | 246 | 214 | 84 | 35 |

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
- 下表采用的共识裁判：glm-5.3-flash

**待评价文本**

~~~text
- Can you provide empirical evidence or metrics that demonstrate the stability of your training process over time.
~~~

**CoT rationale**

~~~text
The answer is a direct request for empirical evidence or metrics to demonstrate the stability of the training process. This is an explicit action that the authors can directly apply by providing the requested evidence. The comment is clear and specific, indicating exactly what the authors should do. Therefore, it meets the criteria for a score of 5.
~~~

**裁判指出的错误句子**

| 原始错误句子 | 错误类型 | 解释 |
| --- | --- | --- |
| The comment is clear and specific, indicating exactly what the authors should do. | rubric_misapplication | This treats mere explicitness as the concreteness required for score 5. The comment does not specify which metrics, what evidence, or how stability should be demonstrated, so authors do not know exactly how to apply the action; the claim contradicts the score-5 criterion of 'concrete details on how to implement it.' |
| Therefore, it meets the criteria for a score of 5. | score_mapping_error | The reasoning describes an explicit request lacking implementation detail, which maps to score 3 (explicit action, vague on execution), not score 5; the conclusion assigns the wrong rubric level. |

**如何理解**：Label-only 接口原本预测正确；切换到 CoT 后，rationale 对评分标准或证据作出了错误解释，并且该错误推理继续支持最终的严重错误分数。这就是 supports_wrong_score。

### CL→CC（CoT SFT） 示例

- 任务：rev_util_verifiability
- source_key：rev_util_verifiability:43:verifiability_test_0206:label_only_correct_to_cot_severe
- 真实分数：3
- Label-only 接口预测：3
- CoT 接口预测：1
- 样本方向：有害（正确→严重错误）
- 裁判共识：supports_wrong_score
- 下表采用的共识裁判：doubao-seed-2.0-lite

**待评价文本**

~~~text
1. The authors run their network S times repeatedly and collect S latent regions as suggestions each iteration. They set S to 2 to achieve a good balance between accuracy and efficiency. More details are missing to support this choice.
~~~

**CoT rationale**

~~~text
The comment claims that more details are missing to support the choice of S=2. However, it provides no reasoning, evidence, or justification for why the choice is questionable or what specific details are needed. The statement is merely an assertion without any supporting argument, making it unverifiable. Therefore, the evaluation score is 1.
~~~

**裁判指出的错误句子**

| 原始错误句子 | 错误类型 | 解释 |
| --- | --- | --- |
| However, it provides no reasoning, evidence, or justification for why the choice is questionable or what specific details are needed. | factual_error | The evaluated comment does provide minimal reasoning for its claim: it notes the authors set S to 2 for a balance of accuracy and efficiency, then claims more details are missing to support this choice. It is not completely without reasoning or justification, so the rationale's claim here is factually incorrect. |
| The statement is merely an assertion without any supporting argument, making it unverifiable. | rubric_misapplication | The comment does have some minimal supporting context, it just lacks key elements (specific details on what information is missing and why it is needed), which aligns with rubric level 3, not level 1. The rationale incorrectly applies the rubric threshold for a score of 1. |
| Therefore, the evaluation score is 1. | score_mapping_error | The reasoning in the rationale incorrectly maps the comment's attributes to a score of 1 instead of the correct gold score of 3. |

**如何理解**：Label-only 接口原本预测正确；切换到 CoT 后，rationale 对评分标准或证据作出了错误解释，并且该错误推理继续支持最终的严重错误分数。这就是 supports_wrong_score。
