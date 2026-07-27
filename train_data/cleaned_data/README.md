# Cleaned Data

本目录存放清洗后的训练数据。原始样本正文和金标签保持不变；若原始 prompt 已可用，则仅做字段规范化（如补 `id`），不再改写任务提示。

## 当前文件

| 文件 | 数量 | 用途 |
|---|---:|---|
| [`rw_gen_coherence_4811.json`](rw_gen_coherence_4811.json) | 4811 | `rw_gen/coherence` 全量训练数据 |
| [`preview/rw_gen_coherence_4811_preview.json`](preview/rw_gen_coherence_4811_preview.json) | 2 | 人工检查与蒸馏流程测试 |
| [`rw_gen_positioning_type_954.json`](rw_gen_positioning_type_954.json) | 954 | `rw_gen/positioning_type` 全量训练数据 |
| [`preview/rw_gen_positioning_type_954_preview.json`](preview/rw_gen_positioning_type_954_preview.json) | 2 | 人工检查与蒸馏流程测试 |
| [`rw_gen_positioning_check_2822.json`](rw_gen_positioning_check_2822.json) | 2822 | `rw_gen/positioning_check` 全量训练数据 |
| [`preview/rw_gen_positioning_check_2822_preview.json`](preview/rw_gen_positioning_check_2822_preview.json) | 2 | 人工检查与蒸馏流程测试 |
| [`rev_util_actionability_4800.json`](rev_util_actionability_4800.json) | 4800 | `rev_util/actionability` 分层抽样训练数据 |
| [`preview/rev_util_actionability_4800_preview.json`](preview/rev_util_actionability_4800_preview.json) | 5 | 标签 1–5 各一条 |
| [`rev_util_grounding_specificity_4800.json`](rev_util_grounding_specificity_4800.json) | 4800 | `rev_util/grounding_specificity` 分层抽样训练数据 |
| [`preview/rev_util_grounding_specificity_4800_preview.json`](preview/rev_util_grounding_specificity_4800_preview.json) | 5 | 标签 1–5 各一条 |
| [`rev_util_helpfulness_4800.json`](rev_util_helpfulness_4800.json) | 4800 | `rev_util/helpfulness` 分层抽样训练数据 |
| [`preview/rev_util_helpfulness_4800_preview.json`](preview/rev_util_helpfulness_4800_preview.json) | 5 | 标签 1–5 各一条 |
| [`rev_util_verifiability_4800.json`](rev_util_verifiability_4800.json) | 4800 | `rev_util/verifiability` 分层抽样训练数据 |
| [`preview/rev_util_verifiability_4800_preview.json`](preview/rev_util_verifiability_4800_preview.json) | 5 | 标签 1–5 各一条 |
| [`rev_util_verifiability_extraction_4800.json`](rev_util_verifiability_extraction_4800.json) | 4800 | `rev_util/verifiability_extraction` 分层抽样训练数据 |
| [`preview/rev_util_verifiability_extraction_4800_preview.json`](preview/rev_util_verifiability_extraction_4800_preview.json) | 2 | 标签 0、1 各一条 |

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

## `rw_gen/positioning_check`

数据来源：[`../origin_data/rw_gen__positioning_check__n2822.json`](../origin_data/rw_gen__positioning_check__n2822.json)。

经检查，原始 prompt 已可用，**不做内容清洗**。本目录仅做字段规范化：

- 按原始顺序补全稳定 `id`（`train_0001` … `train_2822`）；
- 保留 `task`、`aspect`、`labels`、`score_sets`、`prompt` 原样。

标签分布为 `0: 1598`、`1: 1224`。Preview 各取标签 `0`、`1` 的首条样本（`train_0001`、`train_0002`）。

## `rev_util` 五个维度

数据来源：[`../origin_data/`](../origin_data/) 下对应的官方 train 分文件。原始 prompt 经检查可用（结构完整、无空 `[ANSWER]`、无完整 user 重复、标签均在 `score_sets` 内），因此不改写样本内容，只筛选满足条件的原始行。

可复现脚本：[`rebuild_rev_util_4800.py`](rebuild_rev_util_4800.py)。当前文件由该脚本使用 seed=`20260721` 重新生成，处理策略如下：

- 只保留能同时与对应官方 `RevUtil_synthetic/*/train` 的评论正文和标签精确对应的 origin 行，从而排除 human `hard` 样本以及与 human eval 重叠的数据；
- 排除含 C1 控制字符、替换字符或典型 UTF-8 误解码字符的评论，不修复或改写原文；
- 排除与 [`../../test_data/prompted_rev_util_data.json`](../../test_data/prompted_rev_util_data.json) 任一 RevUtil aspect 测试样本正文重复的行，保证多 aspect 联合训练时也不存在测试正文交叉暴露；
- 在过滤后的候选池内按标签比例分层抽取各 4800 条，配额使用最大余数法计算；
- 使用 `seed + aspect + label + 原答案` 的 SHA-256 稳定排序抽样，保证重复运行结果一致；
- 按抽样后顺序补稳定 `id`（`train_0001` … `train_4800`）；
- 保留 origin 中的 `task`、`aspect`、`labels`、`score_sets`、`prompt` 原样。

### Prompt 审计结论

- 五个 aspect 的 cleaned prompt 与 [`../../test_data/prompted_rev_util_data.json`](../../test_data/prompted_rev_util_data.json) 中对应官方测试 prompt 完全一致，不存在训练/测试模板偏移；
- `[QUERY]`、`[CRITERIA]`、`[EXAMPLES]`、`[ANSWER]` 结构完整，few-shot 的 `<reasoning>` 和 `<score>` 标签均闭合且覆盖对应标签空间；
- 最终 `[ANSWER]` 后不存在金标签、`<reasoning>` 或 `<score>` 泄漏；评论开头的数字是原始审稿意见编号，不是标签；
- system 英文语法、`verifiability` 示例中的多余引号以及 `verifiability_extraction` 的边界表述存在轻微瑕疵，但官方测试使用相同模板，当前不修改；
- 蒸馏时直接沿用当前 prompt，不额外强制固定推理步骤；输出格式异常通过校验和重试处理，预测标签与金标签不一致的轨迹不进入专家训练数据；
- 若后续研究 prompt 优化，应建立独立版本和实验名，不覆盖当前数据。

| aspect | 合格候选 | cleaned | 标签分布 |
|---|---:|---|---|
| `actionability` | 6537 | `rev_util_actionability_4800.json` | 1: 1121；2: 803；3: 1455；4: 468；5: 953 |
| `grounding_specificity` | 6537 | `rev_util_grounding_specificity_4800.json` | 1: 549；2: 367；3: 1914；4: 222；5: 1748 |
| `helpfulness` | 6537 | `rev_util_helpfulness_4800.json` | 1: 195；2: 539；3: 2157；4: 1521；5: 388 |
| `verifiability` | 5265 | `rev_util_verifiability_4800.json` | 1: 1375；2: 549；3: 2114；4: 628；5: 134 |
| `verifiability_extraction` | 6537 | `rev_util_verifiability_extraction_4800.json` | 0: 934；1: 3866 |

Preview：五级任务各取标签 1–5 各一条；`verifiability_extraction` 取标签 0、1 各一条。完整大文件请用终端查看，Cursor 中优先打开 `preview/`。

## 更新说明

### 2026-07-21

#### `rev_util` 五个维度

- 从 origin 全量 train 中重新筛选并分层抽样各 4800 条，替换此前的 human/synthetic 混合抽样版本。
- 当前版本只包含能回溯到官方 synthetic train 的原始行，并排除了乱码以及与任一 RevUtil 官方测试 aspect 正文重叠的样本。
- 未修改原始 prompt、评论正文、标签或候选分数集合；蒸馏时沿用 cleaned prompt，并按格式与金标签一致性过滤轨迹。

#### `rw_gen/coherence`

- 当前 cleaned 文件为 [`rw_gen_coherence_4811.json`](rw_gen_coherence_4811.json)。
- 该版本沿用当前已训练使用的 XML 风格输入，并保留 `<reasoning>` / `<score>` 输出格式。
- 这一版应视作独立实验分支；若后续要做 prompt 回退或“最小修改版”对照实验，建议新开文件和新实验名，不覆盖当前文件。

#### `rw_gen/positioning_type`

- 当前 cleaned 文件为 [`rw_gen_positioning_type_954.json`](rw_gen_positioning_type_954.json)。
- 处理策略是“最小修改”：只补稳定 `id`，保留原始 `task/aspect/labels/score_sets/prompt` 原样。
- 原始 prompt 与测试任务定义一致，虽然有轻微英文语法问题，但含义清楚，因此不建议为了形式统一而重写 prompt。

#### `rw_gen/positioning_check`

- 原始数据文件：[`../origin_data/rw_gen__positioning_check__n2822.json`](../origin_data/rw_gen__positioning_check__n2822.json)。
- 当前 cleaned 文件为 [`rw_gen_positioning_check_2822.json`](rw_gen_positioning_check_2822.json)。
- 审计结论：
  - 顶层结构正常，样本数 `2822`，字段统一为 `task/aspect/labels/score_sets/prompt`。
  - 所有样本均为二分类，`task == rw_gen`，`aspect == positioning_check`，`score_sets == [0, 1]`。
  - 标签分布：`0: 1598`，`1: 1224`。
  - prompt 结构稳定，均为 `system + user` 两条消息。
  - 未发现完整 prompt 直接对应多个冲突标签的错误。
  - `<reasoning>` / `<score>` 仅出现在 few-shot 示例中，不构成标签泄漏。
  - 任务同时包含“单段 paragraph 判断”和“给定 context 的 final paragraph 判断”，因此 prompt 形态比 `positioning_type` 更多，这是任务设计差异，不是脏数据。
- 当前建议：
  - 暂不改写原始 prompt。
  - cleaned 版本沿用 `positioning_type` 的最小修改策略：补稳定 `id`，保留原始 `prompt`，只做训练格式适配，不做 XML 化重写。

## 维护规则

- 已经进入训练的 cleaned 文件不做覆盖式修改；若 prompt 策略变化，应新建文件并在文件名或实验名中体现版本差异。
- 对于原始 prompt 与测试分布一致的数据，优先采用“最小修改”原则，只做字段规范化。
- 若某个任务需要较大幅度的 prompt 结构调整，应在本 README 中记录修改原因、修改范围、是否已用于训练，以及是否需要单独的对照实验。
