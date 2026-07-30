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

# SciLoom-MoDE

科学写作评价相关数据，用于训练 / 评测 **SciLoom** 多专家（MoDE）LoRA。  
配套模型：[`XXLEO/SciLoom-4B`](https://huggingface.co/XXLEO/SciLoom-4B)。

数据来源主要基于 *Reward Modeling for Scientific Writing Evaluation* 任务族，经清洗、蒸馏与划分后整理。

## 目录结构

```text
SciLoom-MoDE/
  README.md
  <task>/                         # 训练或评测用 JSONL
    cot/
      train_cot.jsonl             # 有则用于训练（含 completion）
      test_cot.jsonl              # 测试集
      splits/*.json               # train/val 划分（若有）
    label_only/
      train_label_only.jsonl
      test_label_only.jsonl
      splits/*.json
  distill_data/                   # 教师模型原始 / 可接受蒸馏轨迹
  origin_data/                    # 清洗前的原始任务 JSONL（含 preview）
```

## 任务一览

| 数据集 | 任务简介 | 标签 | CoT train | CoT test | Label-only train | Label-only test |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `rev_util_actionability` | 评审意见是否给出作者可执行的修改建议 | 1–5 | 1788 | 1000 | 1788 | 1000 |
| `rev_util_grounding_specificity` | 意见是否落到具体章节/图表，并说清问题 | 1–5 | 2652 | 1000 | 2652 | 1000 |
| `rev_util_helpfulness` | 意见综合是否有用（可执行、有依据、有针对性） | 1–5 | 2279 | 1000 | 2279 | 1000 |
| `rev_util_verifiability` | 意见中的论断是否有可核验的依据 | 1–5 | 1825 | 788 | 1825 | 788 |
| `rw_gen_coherence` | 引用句是否被论文上下文支持（蕴含） | 0/1 | 1526 | 1046 | 1526 | 1046 |
| `rw_gen_positioning_check` | 段落是否体现本文贡献或在文献中的定位 | 0/1 | 2666 | 603 | 2666 | 603 |
| `rw_gen_positioning_type` | Related Work 是否先综述再在末段定位本文 | 0/1 | 944 | 204 | 944 | 204 |
| `novelty` | 两份新颖性评估的最终结论是否一致 | 0/1 | — | 66 | — | 66 |
| `revision_correctness` | 修订文本是否优于原文并可直接替换 | 0/1 | — | 3026 | — | 3026 |
| `revision_relatedness` | 修订文本是否正确落实给定修改指令 | 0/1 | — | 3026 | — | 3026 |

说明：

- 前 7 个任务有 train + test，用于垂域微调与 in-domain 评测。
- `novelty` / `revision_*` 仅有 test，用作未训练维度 / OOD 评测。
- **cot** 与 **label_only** 样本对齐：同一 `id`，仅 system 要求与 `completion` 不同。
  - cot：输出 `<reasoning>...</reasoning><score>N</score>`
  - label_only：输出 `<score>N</score>`

## 样本格式

### 训练集（`train_*.jsonl`）

```json
{
  "id": "train_0002",
  "task": "rw_gen",
  "aspect": "coherence",
  "label": 0,
  "score_sets": [0, 1],
  "teacher_models": ["deepseek-v4-pro"],
  "supervision_mode": "cot",
  "prompt_version": "original_wording_no_examples_v1",
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "<reasoning>...</reasoning><score>0</score>"}]
}
```

### 测试集（`test_*.jsonl`）

字段与训练类似，金标签为 `labels`（整数），无 `completion`；`evaluation_mode` 为 `cot` 或 `label_only`。

### 划分文件（`splits/`）

`train_*_seed20260720.json` 记录从 train 再切 validation 的样本 id（seed=`20260720`，默认 val 比例 0.1）。

## `distill_data/`（教师蒸馏）

教师模型在原始 prompt 上生成 CoT；仅格式合法且分数等于金标签的轨迹记为 `accepted=true`。

- `*_N_distill.jsonl`：原始追加式蒸馏日志（含 `run_start` / `distillation` / `run_end`）
- `*_distill_<teacher>.jsonl`：该教师全部可接受轨迹
- `*_consensus_<teacher>.jsonl`：两位教师标签一致时的共识轨迹

主要派生规模：

| 任务 | DeepSeek 可接受 | GLM 可接受 | 共识 |
| --- | ---: | ---: | ---: |
| rw_gen_coherence | 1526* | — | — |
| rw_gen_positioning_check | 2666 | 2693 | 2613 |
| rw_gen_positioning_type | 944 | 953 | 943 |
| rev_util_actionability | 1788 | — | — |
| rev_util_grounding_specificity | 2652 | — | — |
| rev_util_helpfulness | 2279 | — | — |
| rev_util_verifiability | 1825 | — | — |

\*coherence 当前 train 使用 DeepSeek 可接受子集；原始全量蒸馏日志见对应 `*_4890_distill.jsonl`。

## `origin_data/`

清洗前 / 全量原始 JSONL（文件名含样本量，如 `rw_gen__coherence__n4890.jsonl`），以及 `preview/` 小样。  
正式训练请优先使用各任务下的 `cot` / `label_only` 划分数据。

## 快速加载

```python
from datasets import load_dataset

# 单个 JSONL 文件
ds = load_dataset(
    "XXLEO/SciLoom-MoDE",
    data_files="rw_gen_coherence/cot/train_cot.jsonl",
    split="train",
)

# 或先下载再本地读
# hf download XXLEO/SciLoom-MoDE --repo-type dataset --local-dir ./SciLoom-MoDE
```

## 相关资源

- 模型（LoRA 专家）：https://huggingface.co/XXLEO/SciLoom-4B
- 原始任务背景：Reward Modeling for Scientific Writing Evaluation

## 更新

- **2026-07-29**：用本地 `data/` 全量替换本仓库（train/test、distill、origin），并更新说明。
