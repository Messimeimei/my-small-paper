# rev_util_verifiability / label_only_sft

- 已选样本：45
- 已完成共识：43
- 未完成：2
- 原始 API 响应：99 个文件

## 按样本方向

| 方向 | 已选 | 已完成 | 未完成 | 支持错误分数 | 支持正确分数 | 无法判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 有害 | 35 | 33 | 2 | 32 | 1 | 0 |
| 有益 | 10 | 10 | 0 | 0 | 10 | 0 |

## 有害样本错误类型

同一样本可包含多种错误类型，因此比例之和可以超过 100%。

| 错误类型 | 样本数 | 占已完成有害样本比例 |
| --- | ---: | ---: |
| evidence_misread | 27 | 81.8% |
| rubric_misapplication | 27 | 81.8% |
| score_mapping_error | 20 | 60.6% |
| factual_error | 17 | 51.5% |
| unsupported_inference | 7 | 21.2% |
| internal_contradiction | 1 | 3.0% |
