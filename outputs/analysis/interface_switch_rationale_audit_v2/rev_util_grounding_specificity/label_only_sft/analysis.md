# rev_util_grounding_specificity / label_only_sft

- 已选样本：307
- 已完成共识：303
- 未完成：4
- 原始 API 响应：714 个文件

## 按样本方向

| 方向 | 已选 | 已完成 | 未完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 有害 | 212 | 208 | 4 | 207 | 1 | 0 |
| 有益 | 95 | 95 | 0 | 0 | 95 | 0 |

## 有害样本错误类型

同一样本可包含多种错误类型，因此比例之和可以超过 100%。

| 错误类型 | 样本数 | 占已完成有害样本比例 |
| --- | ---: | ---: |
| rubric_misapplication | 177 | 85.1% |
| score_mapping_error | 156 | 75.0% |
| factual_error | 134 | 64.4% |
| evidence_misread | 95 | 45.7% |
| unsupported_inference | 40 | 19.2% |
| internal_contradiction | 18 | 8.7% |
| irrelevant_or_missing_reasoning | 15 | 7.2% |
