# Cleaned Data

本目录存放清洗后的训练数据。原始样本正文和金标签保持不变；若原始 prompt 已可用，则仅做字段规范化（如补 `id`），不再改写任务提示。

## 当前文件

| 文件 | 数量 | 用途 |
|---|---:|---|
| [`rw_gen_coherence_4811.json`](rw_gen_coherence_4811.json) | 4811 | `rw_gen/coherence` 全量训练数据 |
| [`preview/rw_gen_coherence_4811_preview.json`](preview/rw_gen_coherence_4811_preview.json) | 2 | 人工检查与蒸馏流程测试 |
| [`rw_gen_positioning_type_954.json`](rw_gen_positioning_type_954.json) | 954 | `rw_gen/positioning_type` 全量训练数据 |
| [`preview/rw_gen_positioning_type_954_preview.json`](preview/rw_gen_positioning_type_954_preview.json) | 2 | 人工检查与蒸馏流程测试 |

## `rw_gen/coherence`

数据来源：[`rw_gen__coherence__exact_user_deduplicated__train__n4811.json`](../../rw_gen__coherence__exact_user_deduplicated__train__n4811.json)。Prompt 模板与最新测试集 [`rw_gen__coherence__exact_user_deduplicated__test__n1046.json`](../../rw_gen__coherence__exact_user_deduplicated__test__n1046.json) 一致。

### 清洗方式

- 将错误的“生成引用句”任务改为“判断候选引用句是否受到论文上下文支持”。
- 保留原始数据的宽松判定边界：语义兼容即可；多引用句不要求论文支持整句。
- 使用 `<task_description>`、`<evaluation_criteria>`、`<few_shot_examples>` 和 `<evaluation_instance>` 等 XML 字段。
- 统一输出为 `<reasoning>...</reasoning>` 和 `<score>0或1</score>`。
- 不使用“最小主张”“必须明确支持”“比上下文更具体即为负例”等会改变原标签边界的严格规则。

### 未修改内容

- 4811 条样本的顺序、任务、维度、标签和候选分数集合。
- 每条样本的论文上下文、目标引用编号和候选引用句。
- two-shot 示例的论文内容、候选句、参考推理和分数。

标签分布为 `0: 2404`、`1: 2407`。Preview 使用前两条训练样本，标签分别为 `1` 和 `0`。

## `rw_gen/positioning_type`

数据来源：[`../origin_data/rw_gen__positioning_type__n954.json`](../origin_data/rw_gen__positioning_type__n954.json)。

经检查，原始 prompt（`[QUERY]` / `[CRITERIA]` / `[EXAMPLES]` / `[ANSWER]`）与评分边界已可用，**不做内容清洗**。本目录仅做字段规范化：

- 按原始顺序补全稳定 `id`（`train_0001` …）；
- 保留 `task`、`aspect`、`labels`、`score_sets`、`prompt` 原样。

标签分布为 `0: 636`、`1: 318`。Preview 各取标签出现的标签 `0`、`1` 各一条（`train_0001`、`train_0002`）。
