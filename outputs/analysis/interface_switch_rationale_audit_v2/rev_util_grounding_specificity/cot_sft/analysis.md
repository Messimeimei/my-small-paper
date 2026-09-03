# rev_util_grounding_specificity / cot_sft

- 已选样本：256
- 已完成共识：255
- 未完成：1
- 原始 API 响应：586 个文件

## 按样本方向

| 方向 | 已选 | 已完成 | 未完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 有害 | 128 | 127 | 1 | 127 | 0 | 0 |
| 有益 | 128 | 128 | 0 | 0 | 128 | 0 |

## 有害样本错误类型

同一样本可包含多种错误类型，因此比例之和可以超过 100%。

| 错误类型 | 样本数 | 占已完成有害样本比例 |
| --- | ---: | ---: |
| rubric_misapplication | 123 | 96.9% |
| score_mapping_error | 111 | 87.4% |
| factual_error | 94 | 74.0% |
| unsupported_inference | 56 | 44.1% |
| evidence_misread | 37 | 29.1% |
| internal_contradiction | 6 | 4.7% |
