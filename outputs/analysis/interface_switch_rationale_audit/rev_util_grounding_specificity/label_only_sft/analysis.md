# rev_util_grounding_specificity / label_only_sft

- 已选样本：50
- 已完成共识：45
- 未完成：5
- 原始 API 响应：129 个文件

## 按样本方向

| 方向 | 已选 | 已完成 | 未完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 有害 | 25 | 20 | 5 | 20 | 0 | 0 |
| 有益 | 25 | 25 | 0 | 0 | 25 | 0 |

## 有害样本错误类型

同一样本可包含多种错误类型，因此比例之和可以超过 100%。

| 错误类型 | 样本数 | 占已完成有害样本比例 |
| --- | ---: | ---: |
| rubric_misapplication | 19 | 95.0% |
| factual_error | 14 | 70.0% |
| evidence_misread | 12 | 60.0% |
| score_mapping_error | 12 | 60.0% |
| internal_contradiction | 1 | 5.0% |
| irrelevant_or_missing_reasoning | 1 | 5.0% |
| unsupported_inference | 1 | 5.0% |
