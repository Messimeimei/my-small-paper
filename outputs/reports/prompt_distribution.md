# Evaluation Prompt Distribution

统计脚本：[`scripts/analyze_prompt_distribution.py`](../../scripts/analyze_prompt_distribution.py)。统计对象是当前发布版 `data/`；没有把 `outputs/evaluations/` 中模型生成的自由文本当作 prompt 模板。`origin_data/` 只用于注明未纳入当前训练/测试的 Verifiability Extraction。

## 结论

- 每个评测样本的输入都是 `system + user`。当前测试文件没有 `assistant` 前缀；训练文件每条样本恰有一个 `assistant` completion（见 [`src/data/datasets.py`](../../src/data/datasets.py) 的校验逻辑）。
- 每个任务的 user prompt **结构模板为 1 种**，但由于其中嵌入了待评价文本，实际字符串通常是“每条样本 1 种”。CoT / label-only 两个视图的 user 内容按 ID 对齐，不变更任务内容。
- 当前发布版有两种输出接口：RF/CoT（reasoning 后 score）和 DS/label-only（只输出 score）。精确 system 文本共有 **3 种**：RF 在 Related Work 任务使用 `Then output`，其余任务使用 `Then, output`；DS system 在所有任务共享。
- assistant 的**结构**有 2 种：`<reasoning>...</reasoning><score>N</score>` 与 `<score>N</score>`。assistant 的**具体文本**不是固定模板：CoT 训练集几乎每条 rationale 都不同；label-only 只由任务的标签集合决定（0/1 任务为 2 种，1--5 任务为 5 种）。
- 9 个训练配置方法只重用这两类 prompt 视图，没有引入第三种 system 或 user 模板。`paper_align*` 和 `self_correct_align` 同时训练两种 view；其余方法只训练 RF 或只训练 DS，详见方法表。

## 按评测任务

“user 实例（去重）”是完整 user 字符串的 SHA-256 去重数；“user 结构”按字段结构计数，不把每条不同的论文文本误当成新模板。`assistant 精确种类`来自 canonical `train_*.jsonl`，测试集没有 completion。

| 评测类型 | 数据角色 | 测试样本 | Train pool | System 精确种类（RF / DS） | User 结构 | User 实例（去重，测试） | Assistant 结构 | Assistant 精确种类（CoT / DS，训练） | Prompt 字段结构 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| RW - Coherence | 域内 | 1,046 | 1,526 | 1 / 1 | 1 | 1,046 | 2 | 1,526 / 2 | QUERY + CRITERIA + CONTEXT + CITATION NUMBER + ANSWER |
| RW - Pos. Cons. (`positioning_check`) | 域内 | 603 | 2,666 | 1 / 1 | 1 | 603 | 2 | 2,666 / 2 | QUERY + CRITERIA + ANSWER |
| RW - Pos. Type (`positioning_type`) | 域内 | 204 | 944 | 1 / 1 | 1 | 204 | 2 | 944 / 2 | QUERY + CRITERIA + ANSWER |
| RU - Actionability | 域内 | 1,000 | 1,788 | 1 / 1 | 1 | 1,000 | 2 | 1,788 / 5 | QUERY + CRITERIA + ANSWER |
| RU - Grounding Spec. | 域内 | 1,000 | 2,652 | 1 / 1 | 1 | 1,000 | 2 | 2,652 / 5 | QUERY + CRITERIA + ANSWER |
| RU - Verifiability | 域内 | 788 | 1,825 | 1 / 1 | 1 | 788 | 2 | 1,825 / 5 | QUERY + CRITERIA + ANSWER |
| RU - Helpfulness | 域内 | 1,000 | 2,279 | 1 / 1 | 1 | 1,000 | 2 | 2,279 / 5 | QUERY + CRITERIA + ANSWER |
| RU - Verifiability Ext. | 原始数据，仅保存于 `origin_data` | 1,000* | — | 1（旧版 RF） / 0 | 1,000* | — | 1（旧版 CoT） | — | QUERY + CRITERIA + EXAMPLES + ANSWER |
| Nov. Alignment | 未见任务 | 66 | — | 1 / 1 | 1 | 66 | 2 | — | QUERY + CRITERIA + ANSWER |
| Rev. - Relatedness | 未见任务 | 3,026 | — | 1 / 1 | 1 | 3,026 | 2 | — | QUERY + CRITERIA + ORIGINAL TEXT + INSTRUCTION + ANSWER |
| Rev. - Correctness | 未见任务 | 3,026 | — | 1 / 1 | 1 | 3,026 | 2 | — | QUERY + CRITERIA + ORIGINAL TEXT + INSTRUCTION + ANSWER |

\* `Verifiability Ext.` 的当前项目说明明确指出它没有进入最终训练/测试目录；上表的 1,000 是作者原始测试规模。原始完整文件 `data/origin_data/rev_util__verifiability_extraction__n10430.jsonl` 有 10,430 条、1 个旧版 system、10,430 个 user 实例，且保留 `[EXAMPLES]`。因此若表格只描述本项目实际运行的评测，应将该行标为“未纳入”，而不是把 1,000 当作当前结果。

## 方法与 prompt 覆盖

| 训练方法（配置名） | 训练时出现的 system view | assistant supervision | 评测接口覆盖 | 备注 |
| --- | --- | --- | --- | --- |
| `cot` | RF/CoT | 教师 rationale + score | RF、DS | 训练只看 RF completion，DS 仅作为交叉评测接口 |
| `label_only` | DS/label-only | score-only | RF、DS | score-only completion |
| `paper_align` | RF + DS | RF rationale+score；DS score-only | RF、DS | 配对双 view |
| `paper_align_without_loss_balance` | RF + DS | RF rationale+score；DS score-only | RF、DS | 配对双 view，普通 token CE |
| `self_correct_cot` | RF/CoT | 学生生成 rationale + score | RF、DS | user/system 与 CoT 视图相同，仅 assistant 文本换为自生成轨迹 |
| `self_correct_align` | RF + DS | 自生成 rationale+score；DS score-only | RF、DS | 配对双 view |
| `ssa` | RF/CoT | rationale + score，区域平衡 loss | RF、DS | 不新增 prompt |
| `ssa_v2` | RF/CoT | rationale + score，score attention 隔离 | RF、DS | 不新增 prompt |
| `ssa_v3` | RF/CoT | rationale + score，assistant attention prompt-only | RF、DS | 不新增 prompt |

以上方法均复用任务自己的 1 个 user 结构模板。方法差异只体现在训练是否包含 RF/DS view、assistant rationale 来源及 loss/attention 处理；不会产生额外的 prompt 类型。

## 可直接用于论文表格的简版

| Evaluation Type | # System templates | # User templates | # Assistant formats | Test assistant prefix | Train assistant formats |
| --- | ---: | ---: | ---: | --- | --- |
| RW - Coherence | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| RW - Pos. Type | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| RW - Pos. Cons. | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| RU - Verifiability Ext. | 1 (legacy) | 1 | 1 (legacy RF) | N/A | Not in current training |
| RU - Verifiability | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| RU - Actionability | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| RU - Grounding Spec. | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| RU - Helpfulness | 2 | 1 | 2 | No | RF rationale+score / DS score-only |
| Nov. Alignment | 2 | 1 | 2 | No | No training set |
| Rev. - Relatedness | 2 | 1 | 2 | No | No training set |
| Rev. - Correctness | 2 | 1 | 2 | No | No training set |

这里的 `# User templates` 是“任务结构模板”，不是完整 user 字符串的 distinct count；后者等于测试样本数（当前清洗后的测试文件）。

## 可复核依据

- 测试样本只保留 `prompt`，没有 `completion`；`evaluation_mode` 区分 CoT 与 label-only（[`data/README.md`](../../data/README.md)）。
- 训练加载器要求 `completion` 恰为一个 `assistant` message，并检查其中含合法 score（[`src/data/datasets.py`](../../src/data/datasets.py)）。
- 当前发布版的 train/test 规模、去重规则、以及 Verifiability Extraction 未纳入最终训练/测试的说明见 [`data/README.md`](../../data/README.md)。
