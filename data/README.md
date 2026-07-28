- 基于蒸馏数据划分得到 cot / score-only 两套 train、test
- cot：system 要求输出 `<reasoning>`；score-only：只要求输出 `<score>`
- 两套样本对齐，仅 completion / system 不同

| 数据集 | 任务简介 | 标签 | CoT train | CoT test | Score-only train | Score-only test |
|---|---|---|---:|---:|---:|---:|
| `rev_util_actionability` | 评审意见是否给出作者可执行的修改建议 | 1–5 | 1788 | 1000 | 1788 | 1000 |
| `rev_util_grounding_specificity` | 意见是否落到具体章节/图表，并说清问题 | 1–5 | 2652 | 1000 | 2652 | 1000 |
| `rev_util_helpfulness` | 意见综合是否有用（可执行、有依据、有针对性） | 1–5 | 2279 | 1000 | 2279 | 1000 |
| `rev_util_verifiability` | 意见中的论断是否有可核验的依据 | 1–5 | 1825 | 788 | 1825 | 788 |
| `rw_gen_coherence` | 引用句是否被论文上下文支持（蕴含） | 0/1 | 1526 | 1046 | 1526 | 1046 |
| `rw_gen_positioning_check` | 段落是否体现本文贡献或在文献中的定位 | 0/1 | 2666 | 603 | 2666 | 603 |
| `rw_gen_positioning_type` | Related Work 是否先综述再在末段定位本文 | 0/1 | 944 | 204 | 944 | 204 |
| `novelty` | 两份新颖性评估的最终结论是否一致 | 0/1 | - | 66 | - | 66 |
| `revision_correctness` | 修订文本是否优于原文并可直接替换 | 0/1 | - | 3026 | - | 3026 |
| `revision_relatedness` | 修订文本是否正确落实给定修改指令 | 0/1 | - | 3026 | - | 3026 |
