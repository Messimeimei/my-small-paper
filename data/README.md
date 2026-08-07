---
license: apache-2.0
task_categories:
  - text-classification
  - text-generation
language:
  - en
tags:
  - scientific-writing
  - peer-review
  - related-work
  - reward-modeling
  - chain-of-thought
  - distillation
pretty_name: SciLoom-MoDE
size_categories:
  - 10K<n<100K
---

# SciLoom-MoDE 数据说明

本目录包含 SciLoom 的科学写作评价训练集与测试集。数据主体来自论文
[*Reward Modeling for Scientific Writing Evaluation*](../paper/Reward%20Modeling%20for%20Scientific%20Writing%20Evaluation.pdf)
（Sahinuc, Dutta, and Gurevych, 2026；以下简称“原论文”）。本项目使用的是**向原论文作者直接申请取得的官方数据文件**，而不是根据论文表格或附录重新构造的数据。

在作者数据的基础上，本项目进一步完成了样本审计、去重和防泄漏、训练子采样、教师 CoT 蒸馏、金标签过滤、few-shot 示例移除，以及 CoT / label-only 双视图转换。因此，本目录中的发布版不是原论文数据的原样镜像，实验中应以本文档的“当前发布版统计”为准。

## 1. 原论文中的数据

原论文把科学写作评价统一表述为：给定任务说明（query）、评价标准及标签规则（criteria / scoring rubric）和待评价文本，模型先给出判断理由，再输出最终分数。论文的 SCIRM 使用 Related Work 与 Review Utility 进行训练和域内测试，并把 Novelty Alignment 与 Scientific Revision 仅用于未见任务测试。

### 1.1 数据来源和任务定义

- **Related Work Evaluation（`rw_gen`）**：源自相关工作生成评价数据，并由原论文作者扩展到更多机器学习会议论文。`coherence` 判断引用句是否受到被引论文上下文支持；`positioning_type` 判断 Related Work 是否采用指定的论文定位方式；论文中的 positioning consistency 在本仓库对应 `positioning_check`，判断相关工作内容是否恰当地定位本文贡献。
- **Review Utility Evaluation（`rev_util`）**：源自 RevUtil。原论文合并人工标注与合成标签，从可操作性、具体落地程度、可验证性和整体帮助程度等维度评价单条审稿意见。`verifiability_extraction` 先判断评论是否含有需要核验的主张，`verifiability` 再评价这些主张的可验证程度。
- **Novelty Alignment**：比较人工与 LLM 生成的两份新颖性评估是否得出相同的 `novel` / `not novel` 结论。原数据中的 LLM 评估没有显式结论，原论文使用 GPT-5.1 推断该结论。
- **Scientific Revision**：给定原段落、修改指令和候选修订，分别判断修订是否落实指令（`relatedness`）以及是否形成可替换原文的更好版本（`correctness`）。

### 1.2 原论文报告的规模

下表来自原论文第 4.2 节和附录 A。Related Work 与 RevUtil 合计 58,712 条训练样本和 6,645 条域内测试样本。

| 原论文任务 / 维度 | Train | Test | 标签 |
| --- | ---: | ---: | --- |
| Related Work / Coherence | 4,890 | 1,048 | 0/1 |
| Related Work / Positioning Consistency | 2,822 | 605 | 0/1 |
| Related Work / Positioning Type | 954 | 204 | 0/1 |
| Review Utility / Actionability | 10,432 | 1,000 | 1-5 |
| Review Utility / Grounding Specificity | 10,431 | 1,000 | 1-5 |
| Review Utility / Verifiability Extraction | 10,430 | 1,000 | 0/1 |
| Review Utility / Verifiability | 8,323 | 788 | 1-5 |
| Review Utility / Helpfulness | 10,430 | 1,000 | 1-5 |
| **域内合计** | **58,712** | **6,645** | - |
| Novelty Alignment（未见任务） | - | 76 | 0/1 |
| Revision / Relatedness（未见任务） | - | 3,092 | 0/1 |
| Revision / Correctness（未见任务） | - | 3,092 | 0/1 |

原论文使用 8 个域内维度联合训练。当前项目选择其中 7 个维度训练，未把二分类的 `verifiability_extraction` 纳入最终训练 / 测试目录；其作者原始训练数据仍保留在 `origin_data/` 中。

## 2. 本项目做出的修改

### 2.1 保存作者原始训练数据

`origin_data/` 保存从作者提供文件的 `train` 字段按任务和维度拆出的 8 份 JSONL。这里保留作者数据的 `task`、`aspect`、`labels`、`score_sets` 和完整 prompt，不包含本项目生成的推理轨迹。文件名中的 `n` 是样本数，例如 `rw_gen__coherence__n4890.jsonl`。

这些文件用于来源追踪和重新处理，不应直接当作当前实验训练集。`preview/` 只为人工浏览提供每个标签一条样例，也不用于训练。

### 2.2 清洗和训练子采样

本项目为样本补充稳定 ID，并在构造测试集时按完整 prompt 去重、检查训练 / 测试重叠。与原论文测试规模相比：

- `coherence` 删除 2 条完全重复的测试 prompt：1,048 -> 1,046；
- `positioning_check` 删除 2 条与训练数据完全重叠的 prompt：605 -> 603；
- `positioning_type` 和保留的 4 个 RevUtil 测试维度数量不变；
- Novelty 删除 10 条重复 prompt：76 -> 66；
- Revision 的两个维度分别删除 66 条重复 prompt：3,092 -> 3,026。

RevUtil 各训练维度的作者原始数据约为 8K-10K 条。为控制不同专家的数据量和蒸馏成本，本项目先为每个维度构造 4,800 条候选集（seed=`20260721`）：

1. 只保留可与官方 RevUtil synthetic train 中评论正文和标签精确对应的样本；
2. 排除 human `hard` 样本、与任一 RevUtil 测试正文重叠的样本，以及含控制字符、替换字符或典型 UTF-8 误解码字符的样本；
3. 按原候选池的标签比例使用最大余数法分配配额，再以 `seed + aspect + label + answer` 的 SHA-256 稳定排序抽样；
4. 不改写入选样本的评论、金标签或评分规则。

Related Work 的 `positioning_check` 和 `positioning_type` 使用全部作者训练样本进入蒸馏；当前 `coherence` 原始蒸馏日志只覆盖 4,890 条候选中的一部分，详见第 5 节。

### 2.3 移除 prompt 中的 few-shot 示例

原论文 prompt 在 `[EXAMPLES]` 中为每个分数提供了带理由和标签的示例。当前发布版统一删除这部分示例，只保留原始措辞的 `[QUERY]`、`[CRITERIA]`、评分规则和真正的评价实例；system prompt 中关于“将提供示例”的句子也同步删除。所有最终文件用 `prompt_version="original_wording_no_examples_v1"` 标识这一变化。

该处理使训练和测试采用一致的无示例输入，也避免固定示例的推理文本成为监督混杂因素。它与原论文附录 C 展示的完整 few-shot prompt 不同，复现实验时不能把两种设置视为同一输入条件。

### 2.4 生成并筛选教师 CoT

训练候选使用教师模型生成 `<reasoning>...</reasoning><score>N</score>`。当前最终训练集统一采用 `deepseek-v4-pro` 轨迹；原始生成参数为 temperature=`0`、top-p=`1`、max tokens=`2048`。只有同时满足以下条件的轨迹才进入训练：

- 输出含有合法且非空的 `<reasoning>` 和整数 `<score>`；
- 分数属于当前任务的 `score_sets`；
- 教师分数与作者数据中的金标签完全一致。

这种筛选保证了最终标签正确，但也会改变原始标签分布：较难或教师易错的类别保留率更低。因此，当前训练集不是对作者训练集的无偏随机子集。

`positioning_check` 和 `positioning_type` 还保留了 GLM-5.2 单教师轨迹及 DeepSeek / GLM 共识轨迹，供消融研究使用；它们没有用于当前各任务目录下的默认 `train_*.jsonl`。

### 2.5 构造 CoT / label-only 双视图

每个最终样本有两种严格对齐的视图：

- `cot`：system 要求先输出理由再输出分数；训练 completion 为教师 `<reasoning>...</reasoning><score>N</score>`；
- `label_only`：system 只要求输出分数；训练 completion 为 `<score>N</score>`。

同一任务、同一 split 的两种视图具有相同的 ID、输入实例和金标签，只在输出要求及训练 completion 上不同。测试文件均不含 `completion`，避免标签泄漏。

## 3. 当前发布版统计

### 3.1 训练、验证和测试规模

`Train pool` 是 `train_cot.jsonl` / `train_label_only.jsonl` 的总行数。训练程序再使用固定 seed=`20260720` 按 9:1 划分为 `Optimizer train` 和 `Validation`；测试集始终独立，不参与该切分。

| 当前任务 | 评价目标 | 标签 | Train pool | Optimizer train | Validation | Test | 实验角色 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `rw_gen_coherence` | 引用句是否受被引论文上下文支持 | 0/1 | 1,526 | 1,373 | 153 | 1,046 | 域内 |
| `rw_gen_positioning_check` | Related Work 是否定位本文贡献 | 0/1 | 2,666 | 2,399 | 267 | 603 | 域内 |
| `rw_gen_positioning_type` | 是否只在末段汇总本文定位 | 0/1 | 944 | 850 | 94 | 204 | 域内 |
| `rev_util_actionability` | 审稿意见是否明确且可执行 | 1-5 | 1,788 | 1,609 | 179 | 1,000 | 域内 |
| `rev_util_grounding_specificity` | 是否定位具体内容并说清问题 | 1-5 | 2,652 | 2,387 | 265 | 1,000 | 域内 |
| `rev_util_helpfulness` | 意见整体能否帮助作者改稿 | 1-5 | 2,279 | 2,051 | 228 | 1,000 | 域内 |
| `rev_util_verifiability` | 评论主张是否有充分依据、可核验 | 1-5 | 1,825 | 1,643 | 182 | 788 | 域内 |
| **域内合计** | - | - | **13,680** | **12,312** | **1,368** | **5,641** | - |
| `novelty` | 两份新颖性评估的最终结论是否一致 | 0/1 | - | - | - | 66 | 未见任务 |
| `revision_relatedness` | 修订是否落实给定指令 | 0/1 | - | - | - | 3,026 | 未见任务 |
| `revision_correctness` | 修订是否优于原文并可直接替换 | 0/1 | - | - | - | 3,026 | 未见任务 |
| **未见任务合计** | - | - | - | - | - | **6,118** | - |

CoT 与 label-only 是相同样本的两种监督 / 推理形式，不能相加后声称样本量翻倍。

### 3.2 当前标签分布

| 任务 | Train pool 标签分布 | Test 标签分布 |
| --- | --- | --- |
| `rw_gen_coherence` | 0: 770；1: 756 | 0: 523；1: 523 |
| `rw_gen_positioning_check` | 0: 1,499；1: 1,167 | 0: 341；1: 262 |
| `rw_gen_positioning_type` | 0: 628；1: 316 | 0: 136；1: 68 |
| `rev_util_actionability` | 1: 148；2: 361；3: 521；4: 86；5: 672 | 1: 234；2: 164；3: 312；4: 95；5: 195 |
| `rev_util_grounding_specificity` | 1: 104；2: 78；3: 839；4: 66；5: 1,565 | 1: 123；2: 64；3: 389；4: 41；5: 383 |
| `rev_util_helpfulness` | 1: 122；2: 358；3: 853；4: 858；5: 88 | 1: 34；2: 107；3: 462；4: 311；5: 86 |
| `rev_util_verifiability` | 1: 964；2: 247；3: 373；4: 172；5: 69 | 1: 221；2: 77；3: 346；4: 120；5: 24 |
| `novelty` | - | 0: 15；1: 51 |
| `revision_relatedness` | - | 0: 747；1: 2,279 |
| `revision_correctness` | - | 0: 1,133；1: 1,893 |

## 4. 目录与样本格式

~~~text
data/
  README.md
  <task>/
    cot/
      train_cot.jsonl             # 仅域内训练任务存在
      test_cot.jsonl
      splits/*.json               # 固定 train/validation ID
    label_only/
      train_label_only.jsonl      # 仅域内训练任务存在
      test_label_only.jsonl
      splits/*.json
  distill_data/                    # 教师原始日志与 accepted / consensus 派生轨迹
  origin_data/                     # 从作者文件拆出的原始训练数据与 preview
~~~

### 4.1 训练样本

~~~json
{
  "id": "train_0002",
  "task": "rw_gen",
  "aspect": "coherence",
  "label": 0,
  "score_sets": [0, 1],
  "teacher_models": ["deepseek-v4-pro"],
  "supervision_mode": "cot",
  "prompt_version": "original_wording_no_examples_v1",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "content": "<reasoning>...</reasoning><score>0</score>"}
  ]
}
~~~

训练文件使用单数 `label`；`completion` 仅参与 assistant 输出监督。label-only 样本的 `completion` 只含 `<score>N</score>`。

### 4.2 测试样本

~~~json
{
  "id": "coherence_0000",
  "task": "rw_gen",
  "aspect": "coherence",
  "labels": 1,
  "score_sets": [0, 1],
  "evaluation_mode": "cot",
  "prompt_version": "original_wording_no_examples_v1",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
~~~

测试文件沿用复数 `labels`，且没有 `completion`。`evaluation_mode` 标识评测时要求 CoT 还是只输出标签。

### 4.3 固定划分文件

`splits/train_*_seed20260720.json` 保存 `train_ids`、`validation_ids`、`validation_ratio=0.1`、数据文件哈希和 ID / 标签内容哈希。CoT 与 label-only 的 ID 切分相同，可用于配对比较；文件内容哈希因 completion 和 system prompt 不同而分别记录。

## 5. `distill_data/` 说明

`*_N_distill.jsonl` 是追加式生成日志，包含 `run_start`、`distillation` 和 `run_end`；`*_distill_<teacher>.jsonl` 只保存已接受轨迹；`*_consensus_<teacher>.jsonl` 保存两位教师都预测正确且相互一致的同 ID 子集，并选择指定教师的完整推理轨迹。

| 任务 | 原始尝试 | DeepSeek accepted 发布快照 | GLM accepted | 双教师共识 | 默认 Train pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rw_gen_coherence` | 1,938 / 4,890（运行中断） | 1,526 | - | - | 1,526 |
| `rw_gen_positioning_check` | 2,822 / 教师 | 2,666 | 2,693 | 2,613 | 2,666 |
| `rw_gen_positioning_type` | 954 / 教师 | 944 | 953 | 943 | 944 |
| `rev_util_actionability` | 4,800 | 1,788 | - | - | 1,788 |
| `rev_util_grounding_specificity` | 4,800 | 2,652 | - | - | 2,652 |
| `rev_util_helpfulness` | 4,800 | 2,279 | - | - | 2,279 |
| `rev_util_verifiability` | 4,800 | 1,825 | - | - | 1,825 |

注意：`rw_gen_coherence_4890_distill.jsonl` 是以 4,890 条输入命名的追加日志，但当前只完成 1,938 条尝试，不能称为全量蒸馏结果。其日志在 accepted 派生文件生成后又追加了少量记录；为了保持已完成实验的数据版本不变，默认训练仍使用已发布的 1,526 条快照。重新抽取该日志可能得到不同数量，必须作为新的数据版本记录。

更详细的日志字段和生成方式见 [`distill_data/README.md`](distill_data/README.md)。其中部分历史命令仍使用重组目录前的 `train_data/` 路径，实际运行时应改为当前 `data/` 路径。

## 6. 加载和使用

~~~python
from datasets import load_dataset

train = load_dataset(
    "XXLEO/SciLoom-MoDE",
    data_files="rw_gen_coherence/cot/train_cot.jsonl",
    split="train",
)

test = load_dataset(
    "XXLEO/SciLoom-MoDE",
    data_files="rw_gen_coherence/cot/test_cot.jsonl",
    split="train",
)
~~~

建议：

- 正式训练和评测优先使用各任务目录下的 `cot` / `label_only` 文件，不直接使用 `origin_data/` 或原始追加日志；
- 训练时读取对应 `splits/`，不要重新随机切分，以保证不同监督方法可比；
- 报告结果时明确说明使用了无 few-shot 的 `original_wording_no_examples_v1`，并区分 Train pool、实际优化训练集和验证集；
- 使用本数据时引用原论文及各上游任务数据论文，并说明本项目额外执行了清洗和教师标签一致性过滤。

## 7. 相关资源与版本记录

- 配套模型：[`XXLEO/SciLoom-4B`](https://huggingface.co/XXLEO/SciLoom-4B)
- 数据集：[`XXLEO/SciLoom-MoDE`](https://huggingface.co/datasets/XXLEO/SciLoom-MoDE)
- 原论文：Sahinuc, Furkan, Subhabrata Dutta, and Iryna Gurevych. *Reward Modeling for Scientific Writing Evaluation*. arXiv:2601.11374, 2026.

版本记录：

- **2026-07-29**：将 train/test、distillation 和 origin 数据重组到 `data/`，并上传 SciLoom-MoDE。
- **2026-07-30**：将 score-only 统一更名为 label-only，明确其监督目标。
- **2026-08-06**：补充作者数据来源、原论文任务和规模、本项目清洗 / 蒸馏 / 双视图修改、最终划分及标签分布。
