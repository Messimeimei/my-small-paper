# rev_util_verifiability / label_only_sft

- 已选样本：45
- 已完成共识：40
- 未完成：5
- 原始 API 响应：96 个文件

## 按样本方向

| 方向 | 已选 | 已完成 | 未完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 有害 | 35 | 30 | 5 | 29 | 1 | 0 |
| 有益 | 10 | 10 | 0 | 0 | 10 | 0 |

## 有害样本错误类型

同一样本可包含多种错误类型，因此比例之和可以超过 100%。

| 错误类型 | 样本数 | 占已完成有害样本比例 |
| --- | ---: | ---: |
| rubric_misapplication | 26 | 86.7% |
| evidence_misread | 24 | 80.0% |
| score_mapping_error | 18 | 60.0% |
| factual_error | 15 | 50.0% |
| unsupported_inference | 7 | 23.3% |
| internal_contradiction | 1 | 3.3% |
