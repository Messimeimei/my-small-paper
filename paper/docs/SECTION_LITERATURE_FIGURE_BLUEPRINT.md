# 论文逐节文献、表达与图表蓝图

> 更新日期：2026-08-05  
> 目标论文：*When Chain-of-Thought Supervision Hurts Scientific-Writing Judges*  
> 当前定位：New Problem/Setting + Empirical Analysis，附带受控缓解与诊断  
> 目标读者：ACL/EMNLP 的 LLM evaluation、scientific NLP 与 reasoning 研究者

## 1. 先冻结整篇论文的一句话

建议用下面这句话控制所有 section：

> In seven scientific-writing evaluation tasks, we show that chain-of-thought
> supervision produces task-dependent specialization to the reasoning
> interface, reflected in both output-schema failures and valid-score drift;
> paired Direct/Reason supervision restores much of the direct-scoring ability,
> while regression-aware scoring alone does not remove the interface mismatch.

这句话包含五个必要元素：场景、发现、行为解释、两类证据和结论边界。目前不要把
`gradient conflict`、`representation alignment` 或“Paper Align 的 balanced loss
导致恢复”写进主论点。前两项没有直接证据，最后一项仍缺 matched Mix baseline。

## 2. 核心文献池：每篇承担什么写作任务

### 2.1 第一层：必须精读并进入正文

| 文献 | 它能支撑什么 | 最值得模仿的写法或图表 | 放入本文哪里 |
|---|---|---|---|
| [TRACT](https://aclanthology.org/2025.acl-long.147/) | CE 忽略分数距离；RAFT/RAIL、CoT-RAFT/CoT-RAIL 与两阶段 self-CoT | Table 1 统一训练/推理方案；Method 先 notation，再 objective、inference；主结果后做 component ablation | Related Work 2.4；Setup；Results RQ3；RAFT diagnostics |
| [Investigating the Impact of Rationales for LLMs on NLU](https://arxiv.org/abs/2510.16686) | Label-only、Reason、Mix、Align；训练方式与推理方式必须分开 | Figure 3 的训练条件；Table 2 的 cross-condition 比较；Direct-correct/CoT-wrong taxonomy | Introduction gap；factorial design；Mechanism |
| [To CoT or not to CoT?](https://arxiv.org/abs/2409.12183) | CoT 收益具有任务结构依赖，主要集中于 math/symbolic reasoning | Figure 1 先交代核心异质性；Figure 2/3 用类别级 effect plot；解析失败单独报告 | Introduction tension；RQ1；Discussion |
| [Synthetic Rationale SFT Hurts Disease Prediction](https://arxiv.org/abs/2606.10279) | 推理时 rationale 可能有用，但作为长目标进行 SFT 可能伤害判别 | Figure 2 给稳定负结果；Figure 3 做 factor diagnostics；Figure 4 做 matched error analysis | Counter-evidence；Mechanism；Discussion |
| [Reward Modeling for Scientific Writing Evaluation](https://arxiv.org/abs/2601.11374) | 与本文任务域最接近，覆盖 related work 和 review utility 多方面评价 | Figure 1 用错误案例建立需求；Figure 2 展示 pipeline；Tables 5-6 交代数据与 rubric；Figure 3-4 按任务族画结果 | Introduction；Related Work 2.1；Task table |
| [G-Eval](https://aclanthology.org/2023.emnlp-main.153/) | CoT-style evaluation steps 与 score probability aggregation 是经典 judge 设计 | Figure 1 用简单 pipeline 解释 evaluator；实验以 human correlation 为中心 | Related Work 2.2；主流假设 |
| [Prometheus](https://arxiv.org/abs/2310.08491) | rubric-conditioned fine-grained judge training；feedback 与 score 联合输出 | Figure 1 比较 coarse/fine-grained evaluation；组件消融检验 rubric/reference/feedback | Related Work 2.2；rubric-conditioned positioning |
| [Improving LLM-as-a-Judge Inference with the Judgment Distribution](https://arxiv.org/abs/2503.03064) | 利用 judge 的分数分布改进 inference readout | 把生成标签与 probability readout 分开比较 | Related Work 2.4；RAFT/RAIL positioning |

### 2.2 第二层：用于领域覆盖、边界或审稿防御

| 文献 | 使用目的 | 注意事项 |
|---|---|---|
| [PRISM](https://arxiv.org/abs/2605.26730) | 说明 peer-review quality 是多维的；参考维度分解和跨系统 profile 图 | 它评价 reviewer，不等于训练 point-wise score judge |
| [Can LLMs Provide Useful Feedback on Research Papers?](https://doi.org/10.1056/AIoa2400196) | 建立自动科研反馈的现实价值和人类评价背景 | 用于动机，不用于支持训练机制 |
| [JudgeLM](https://arxiv.org/abs/2310.17631) | judge fine-tuning、bias 与格式控制背景 | 一般 judge，不是科学写作专域 |
| [Quantitative LLM Judges](https://arxiv.org/abs/2506.02945) | point-wise quantitative scoring 与连续/离散 readout | 核验其具体 objective 后再写公式段落 |
| [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) | rationale supervision 的正面证据，避免 Related Work 单边化 | 主要研究知识/推理蒸馏，不能直接外推到 soft semantic judgment |
| [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) | rationale plausibility 不等于 causal faithfulness | 只支持解释边界，不能证明本文 rationale 不忠实 |
| [Measuring Faithfulness in CoT Reasoning](https://arxiv.org/abs/2307.13702) | rationale intervention 或 causal use 的方法标准 | 当前无 intervention 时只在 Discussion/Limitations 使用 |

### 2.3 证据平衡规则

Related Work 不能只引用“CoT 有害”的论文。应在同一段形成张力：

> Rationale supervision can improve smaller models on tasks that benefit from
> explicit intermediate computation, but its gains do not transfer uniformly
> to discriminative or soft-semantic judgments. Recent controlled studies
> report that the effect depends on task structure, model scale, and whether
> rationales are elicited at inference time or optimized as long training
> targets.

这不是逐篇摘要，而是先给共同问题，再把正面证据与边界证据放在一个比较句中。

## 3. 建议的最终 section 结构

```text
1 Introduction
2 Related Work
  2.1 LLM Evaluation of Scientific Writing and Peer Review
  2.2 Training LLM-as-a-Judge
  2.3 Chain-of-Thought Elicitation and Rationale Supervision
  2.4 Ordinal and Distribution-Aware Scoring
3 Tasks and Experimental Design
  3.1 Scientific-Writing Evaluation Tasks
  3.2 Training Conditions
  3.3 Inference Protocols
  3.4 Metrics and Statistical Analysis
4 Main Results
  4.1 CoT Effects Are Task-Dependent
  4.2 Training and Inference Interfaces Interact
  4.3 Paper Align Restores Direct Scoring, Whereas RAFT Does Not Remove the Gap
5 Behavioral Mechanism Analysis
  5.1 Output-Schema Failure versus Valid-Score Drift
  5.2 Paired Prediction Transitions
  5.3 Task Characteristics and Failure Profiles
  5.4 Probability Geometry under RAFT and RAIL
  5.5 Matched Mix Attribution
6 Discussion
7 Conclusion
Limitations
Ethics Statement
Appendix
```

与当前主稿相比，最重要的变化是：Results 先报告 factorial interaction，再进入
机制分解；Mechanism 新增 task characteristics，但只写关联，不写因果。

## 4. Introduction：六段怎么写

### P1：现实场景与重要性

说明 scientific-writing judge 做的不是单一“好/坏”判断，而是跨 related-work
coherence、positioning、review actionability、grounding、helpfulness 和
verifiability 的异质决策。参考 SciRM Introduction/Figure 1 和 PRISM Introduction。

句子顺序：评价需求 → construct 异质性 → judge 可靠性的下游影响 → 不能只看
aggregate score。推荐表达动作：`range from X to Y`、`must distinguish`、
`reliability matters because`。

### P2：主流假设

说明 G-Eval、Prometheus、reasoning reward models 经常让 judge 先产生 evaluation
steps、feedback 或 rationale，再给 score。不要写“所有工作都认为 CoT 有益”，应写：

> This design is often motivated by the expectation that an explicit rationale
> helps the judge apply a rubric before committing to a score.

### P3：反常现象与文献张力

将 To-CoT、NLU Rationales 与 disease rationale paper 放在一起，说明
inference-time elicitation 和 training-time supervision 不是同一操作。推荐结构：
`Although A, B shows ...; moreover, C reports ...; together, these findings
leave open ...`。

### P4：本文缺口

指出现有工作至少混淆以下两项：training target、inference interface、score
objective 和 failure type。段末给精确问题：

> It therefore remains unclear whether rationale supervision improves the
> underlying judgment, merely specializes the model to a rationale-first
> interface, or changes how scores are emitted and read out.

### P5：本文设计与主要发现

按七任务 → paired Direct/Reason views → 训练×推理交叉 → 两 seed → paired error
decomposition → Paper Align/RAFT diagnostics 的顺序写。只放 2-3 个最强 summary
effect，不把七任务数字全部塞入 Introduction。

### P6：三项贡献

1. 七任务 factorial evidence，对应 Section 4；
2. schema failure 与 valid-score drift 的 paired decomposition，对应 Section 5；
3. Paper Align 与 RAFT 的互补诊断和边界，对应 Sections 4-5。

不要把“extensive experiments”单独列为贡献。

## 5. Related Work：建立四条比较轴

### 5.1 Scientific writing and peer review evaluation

第一段用 Liang et al.、RevUtil、PRISM 说明多维 construct；第二段比较 SciRM 与
本文：SciRM 研究多任务 reward model 和 reasoning refinement，本文研究 matched
training/inference interface 以及 CoT supervision 的负迁移。段末定位：

> We use these scientific-writing constructs as controlled judgment tasks,
> rather than proposing another review generator or an aggregate review-quality
> benchmark.

### 5.2 Training LLM-as-a-Judge

第一段写 prompting/API judge（G-Eval、MT-Bench），第二段写 trained open judge
（Prometheus、JudgeLM、reward reasoning models）。固定比较：输出形式、监督信号、
score readout、rubric conditioning、format failures。

### 5.3 CoT elicitation and rationale supervision

按正面路线 → task boundary → training risk 写三段。必须定义：

> We reserve *elicitation* for requesting a rationale at inference time and
> *supervision* for optimizing rationale tokens during fine-tuning.

### 5.4 Ordinal and distribution-aware scoring

先写 TRACT 的 ordinal objective，再写 judgment distribution 的 probability
readout，最后说明本文把它作为受控诊断，不声称复现完整两阶段 TRACT。

## 6. Tasks and Experimental Design：按可复现顺序写

### 6.1 Task table

保留当前 Table 1，建议增加 `Input unit` 和 `Construct source`。参考 SciRM Tables
5-6 与 PRISM Table 1。完整 license 与来源放 Appendix。

### 6.2 Factorial design figure

参考 NLU Rationales Figure 3，但重新画成本文自己的 2×2：

```text
                         Inference interface
                    Direct                 Reason
Training Direct     LL                     LC
Training Reason     CL                     CC

Paired training     PAL                    PAC
```

Base 的 B-L/B-C 放在矩阵上方，RAFT/RAIL 放右侧作为独立 score-objective
diagnostic。这样读者不会把 13 个缩写看成互不相关的模型。

### 6.3 Loss 与 inference 顺序

参考 TRACT：定义 legal label set 与 score token → Label-only/CoT/Paper Align →
RAFT expectation/MSE → CoT-RAFT token masking → greedy/RAIL/CoT-RAIL → parser
和 invalid policy。不要把 training objective 与 decoding method 混成一个段落。

### 6.4 Statistical analysis

补充两 seed 的解释边界、paired bootstrap 的 resampling unit、McNemar、ordinal
paired CI、multiplicity policy，以及 invalid outputs 始终保留在 denominator。

## 7. Main Results：统一证据句法

每个结果段落固定四步：Claim → Evidence → Interpretation → Boundary。

> CoT supervision reduced direct-scoring performance on all four ordinal review
> criteria. Relative to Label-only SFT under the same Direct interface, CoT SFT
> lowered QWK by [effects and CIs]. The consistent direction across seeds is
> compatible with specialization to the Reason interface. This comparison does
> not by itself identify whether the shift is caused by rationale length,
> rationale content, or score-token optimization.

### 7.1 RQ1：异质性，而非平均排名

主图用 per-task effect plot：ordinal tasks 画 `CL - LL` QWK，binary tasks 画
`CL - LL` Macro-F1；同时画 `CC - LC` 表示 Reason inference 下的训练效应。点为
两 seed mean，细线为 seed range 或 paired bootstrap CI，0 线表示无效应。

### 7.2 RQ2：接口 interaction

报告 training effect under Direct (`CL - LL`)、under Reason (`CC - LC`) 和
interaction (`(CC - LC) - (CL - LL)`)。先逐任务，再按 task family 做描述性汇总；
不要把七任务当作独立同分布样本做夸大的总体显著性结论。

### 7.3 RQ3：Paper Align 与 RAFT

可写：paired Direct/Reason supervision 恢复 direct scoring。不可写：恢复由
separate normalization 或 0.5/0.5 loss 本身造成。

可写：ordinal objective 与 probability readout 改变 score representation，但
没有消除当前 CoT/interface gap。不可写：RAFT 一般无效，或完整 TRACT 无效；你的
CoT-RAFT 是单阶段 teacher-CoT 版本，不是两阶段 self-CoT TRACT。

## 8. Behavioral Mechanism Analysis

### 8.1 Schema failure versus valid-score drift

使用 mutually exclusive decomposition：invalid/malformed；valid-correct；valid
adjacent error；valid severe error。推荐 100% stacked bars，每个 task 只画 LL、CL、
PAL 三个 Direct-interface 条件。

### 8.2 Paired transitions

参考 NLU Rationales 的 error taxonomy 和 disease paper 的 matched confusion
analysis。定义 harmed、helped、rescued、lost。正文画 transition matrix 或 signed
bars；完整表放 Appendix。类别少时矩阵比 Sankey 更精确。

### 8.3 Task characteristics

只分析预先定义的 binary/ordinal、rubric complexity、input length、evidence
locality、label imbalance、base ceiling。可以写 `degradation was larger on`，不能写
`rubric complexity caused`。七个任务不足以支持复杂 meta-regression。

### 8.4 RAFT probability geometry

四面板：legal score mass；entropy/top margin；expected-minus-argmax 分布；expected
score vs gold。若 entropy 近零且 expected score 贴近整数，应写 probability
saturation，而不是 calibration improvement。

### 8.5 Matched Mix attribution

Mix 与 Paper Align 使用完全相同的 Direct/Reason examples、训练步数和 seed；Mix
使用 token-averaged CE，Paper Align 使用 view-wise normalization。预先解释：

- `Mix ≈ PAL`：主要收益来自 Direct-view exposure；
- `Mix < PAL`：view balancing 提供额外收益；
- `Mix > PAL`：0.5/0.5 weighting 可能过强约束 Reason view。

## 9. Discussion：只讨论三件事

1. Scientific-writing criteria are heterogeneous：连接 SciRM 与 PRISM；
2. Reasoning supervision can change the usable interface：连接 To-CoT、NLU
   Rationales 与 disease paper；
3. Ordinal objectives do not replace interface alignment：连接 TRACT 与 judgment
   distribution。

最后给部署建议：高吞吐 scoring 默认 Direct；audit mode 单独请求 rationale；两种
mode 分别验证，不假设 audit rationale faithful 地解释另一次 Direct score。

## 10. Abstract 与 Conclusion

Abstract 最后写，采用六句：需求；常见假设；training/inference conflation；七任务
factorial design；两个最强结果加机制分解；bounded implication。

Conclusion 只回答 RQ1-RQ3，不加入新数字。建议末句：

> Reasoning supervision and score-aware objectives should be selected according
> to the evidence and output interface required by the evaluation task, rather
> than treated as interchangeable upgrades to a judge.

## 11. 最终图表包

| 编号 | 图表 | 核心结论 | 形式 | 位置 |
|---|---|---|---|---|
| Fig. 1 | Study design + motivating result | training target 与 inference interface 必须交叉分析 | 左：2×2 design；右：七任务 effect overview | 第一或第二页顶部 |
| Table 1 | Task inventory | construct、label space 和规模不同 | `booktabs`，无竖线 | Setup |
| Fig. 2 | Per-task CoT effects | CoT effect is task-dependent | paired forest/effect plots | Results |
| Table 2 | Main results | 关键条件的主 metric、valid rate 与 cost | 按 task family 分块 | Results |
| Fig. 3 | Behavioral decomposition | schema failure 与 valid-score drift | stacked bars + transition matrix | Mechanism |
| Fig. 4 | RAFT probability geometry | soft ordinal readout 是否被利用 | mass/entropy/delta/scatter | 主文或附录 |
| Table 3 | Attribution diagnostics | Mix、PAL、RAFT 分别解决什么 | controlled ablation table | 主文/附录 |
| Appendix | Complete per-seed results | 完整透明度 | 每任务完整表 | 附录 |

视觉规则：条件颜色跨图固定；Direct/Reason 同时用形状或线型编码；effect plot 有
零线；明确 error bar；ordinal 与 binary metric 不共用数值轴；PDF/SVG 矢量输出；
缩放后字体至少 8 pt；caption 说明 finding、panel、metric、n、error bar 和 test；
避免 radar、3D、双 y 轴和装饰色。

## 12. 写作表达词典

| 证据强度 | 推荐动词 |
|---|---|
| 两 seed 同方向 + paired CI | `show`, `consistently reduced`, `recovered` |
| 单 seed RAFT diagnostic | `suggests`, `is consistent with`, `in the available run` |
| 行为机制 | `manifests as`, `can be decomposed into`, `is compatible with` |
| 未做因果干预 | `may reflect`, `does not distinguish`, `cannot establish` |

避免：`proves that CoT is harmful`、`universally`、`Paper Align solves`、`RAFT
fails`、未经 intervention 的 `faithful rationale`、协议不一致时的 `state of the
art`、未完成 novelty search 时的 `the first study`。

## 13. 现在开始写的顺序

1. 冻结 Table 1 的来源与 license；
2. 生成 Table 2 和 Fig. 2 的 derived CSV，不手抄数字；
3. 写 Section 4 Main Results；
4. 生成 transition、error decomposition、RAFT diagnostics；
5. 写 Section 5 Mechanism Analysis；
6. 回写 Section 3 统计协议；
7. 写 Discussion 和 Limitations；
8. 写 Introduction、Related Work、Conclusion；
9. 最后写 Abstract 和 Title。

## 14. 当前仍缺的高价值工作

必须补：matched Mix、paired bootstrap CI/McNemar、统一 parser、dataset
source/license/split/overlap 核验、references.bib 元数据验证。

强烈建议：第二模型家族的代表性任务复现、rationale utility 小规模 blinded
audit、Direct/Reason cost 与 invalid rate。

暂不需要：为了“有方法”新增复杂 optimizer；无干预证据时做 gradient/representation
因果叙事；第三个 Qwen seed。后者优先级低于第二模型家族和 matched Mix。

## 15. 本次探索的完整性说明

本蓝图使用了多源题名检索、本地 PDF 全文结构提取和现有实验记录。TRACT 的 ACL
2025 元数据已核验为 `2025.acl-long.147`。2026 年的 SciRM、PRISM、NLU Rationales
与 disease rationale 工作目前按 arXiv 版本引用，投稿前应再次核验正式会议信息。

`nature-paper-card` 的 PyMuPDF 依赖下载超时，因此没有声称完成 PDF 页码级 Paper
Card 审计。本次采用 structure-grounded 精读：结论定位到 section、figure、table
与 appendix，而不生成不可靠页码。
