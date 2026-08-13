# 具体论文方案：科学写作评价模型真的需要 Rationale 吗？

## 结论先说

不需要避开已有研究。把 NLU、医学预测中已经出现的现象迁移到科学写作评价，只要新领域具有独特需求、实验控制更严格、并能产生新的机制结论，就是成立的论文路线。

但只写“别人发现在医学中 CoT 微调有害，我发现在科学写作中也有害”通常只够领域复现。你现在可以把它推进成一篇更完整的论文：

> 在 scientific-writing evaluator 中，rationale 在基础模型推理时似乎有用，作为微调监督却没有超过 score-only。本文分离 rationale elicitation、explanation learning 和 score readout，判断性能损失来自 token dilution、decision--explanation conflict，还是两种 score readout 的路径冲突，并用 score-priority 的非对称梯度投影验证冲突是否可以被干预。

推荐标题：

> **Do Scientific Evaluators Need Rationales? Separating Elicitation, Explanation Learning, and Score Readout**

这篇论文的主体是“科学评价领域的新问题与机制研究”，方法是机制验证和修复工具，不是为了凑创新点而另造一套复杂系统。

---

## 1. 论文到底研究什么

### 核心问题

科学写作评价既要求最终评分准确，又希望模型给出可核查理由。现有工作通常默认“能生成 rationale”有利于评价，或者只用最终 score reward 衡量训练是否成功，却没有回答：

1. 推理时要求模型先写 rationale，与训练时把 rationale 当监督，是否是同一种收益？
2. Rationale SFT 弱于 score-only，究竟只是因为 rationale token 太多，还是 explanation learning 与 score decision 在优化方向上冲突？
3. 一个模型能同时保持直接评分能力和有用理由吗？

### 三个研究对象必须分开

- **Elicitation：** 推理时是否要求先生成 rationale；
- **Supervision：** 训练时 target 是 score-only 还是 rationale + score；
- **Utility：** 生成的 rationale 是否真的包含相关、可核查的证据。

你的现有实验已经证明，前两者不能混为一谈；下一步论文负责解释为什么，并检查第三者。

---

## 2. 当前结果应该怎样理解

### 总体结果

| 训练方式       | Direct score inference | Rationale-first inference |
| -------------- | ---------------------: | ------------------------: |
| Base           |            17.0 / 19.6 |               52.9 / 49.9 |
| Score-only SFT |  **75.0 / 69.7** |     **72.2 / 66.7** |
| Rationale SFT  |            66.5 / 60.7 |               71.9 / 65.7 |

表中为 Accuracy / Macro-F1。六个 arm 实际由“两种 base inference”以及微调后的 `training target x inference protocol` 2x2 组成。

### 发现一：基础模型受益于 rationale-first protocol

`B-C - B-S = +35.9 Acc / +30.3 F1`。

这说明不能写“CoT 在科学评价中完全无用”。准确说法是：对未做任务适配的 base model，rationale-first 的整套推理协议显著改善可解析评分结果。

这个差值同时包含 prompt、额外计算和生成轨迹，因此暂时不能进一步声称增益一定来自 faithful reasoning。

### 发现二：控制 inference protocol 后，rationale supervision 没有超过 score-only

| 受控比较                              |       Accuracy |       Macro-F1 |
| ------------------------------------- | -------------: | -------------: |
| Direct 下：`S-S - C->S`             | **+8.5** | **+9.0** |
| Rationale-first 下：`S->C - C-C`    |           +0.3 |           +1.0 |
| Difference-in-differences interaction | **+8.2** | **+8.0** |

目前最稳妥的结论是：

> Rationale SFT 没有学出 score-only SFT 不具备的新判别能力；它主要让模型更加依赖 rationale-first decision path。Score-only SFT 不仅直接评分更强，在要求生成 rationale 时也与 rationale SFT 持平或略好。

### 发现三：这种交互并非只在 RevUtil 出现

| Task group   | Direct training effect | Rationale-first training effect | Interaction |
| ------------ | ---------------------: | ------------------------------: | ----------: |
| Four RevUtil |              +9.65 Acc |                       +0.68 Acc |   +8.98 Acc |
| Three RW-Gen |              +6.90 Acc |                       -0.10 Acc |   +7.00 Acc |

所以不能根据 `S-S` 与 `C-C` 的对角比较，直接声称只有“主观、序数任务”出现冲突。现有结果支持的是跨两组任务都存在 rationale-path dependence；RevUtil 的最终分数损失更明显，但其中还混有标签数、难度和 ceiling effect。

---

## 3. 为什么这不是简单重复已有论文

### 你的数据设置更严格

当前 rationale 数据具有以下性质：

1. Teacher 只看到原始评价输入和 rubric，没有看到 gold label；
2. Teacher 独立生成 rationale 和 score；
3. 生成后才按 `teacher score == gold score` 过滤；
4. Score-only target 与 CoT target 来自同一个 accepted completion；
5. 两种数据在样本、输入、标签和 teacher decision 上严格配对。

因此，这些 rationale 是 **label-unconditioned、score-correct、correctness-filtered**，排除了“teacher 因为提前看到 gold label 而倒推解释”这一直接混杂。

但只能称 teacher 的最终 score 正确，不能称 rationale 本身正确或 faithful；理由质量仍需独立评价。

### 与最接近工作的区别

| 文献                                                         | 已经做了什么                                                          | 本文仍然可以解决什么                                          |
| ------------------------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------- |
| [To CoT or not to CoT?](https://arxiv.org/abs/2409.12183)     | 研究 inference-time CoT 的跨任务边界                                  | 科学评价中的训练监督、推理协议和解释效用分离                  |
| [NLU Rationales](https://arxiv.org/abs/2510.16686)            | 比较 label-only、rationale、Mix、Align；Align 已覆盖分离归一化损失    | Answer-blind scientific rationale、完整交叉设计和梯度方向冲突 |
| [Medical rationale failure](https://arxiv.org/abs/2606.10279) | 发现 synthetic rationale SFT 伤害真实疾病预测，并提出 token dilution  | 在科学评价中直接区分并干预 dilution 与 conflict               |
| [Judgment Distribution](https://arxiv.org/abs/2503.03064)     | 改进现成 judge 的 inference-time score readout                        | 研究经过专门微调的 evaluator 及训练目标作用                   |
| [TRACT](https://arxiv.org/abs/2503.04381)                     | CoT CE + score-aware regression，并用 self-generated CoT 缩小分布偏移 | 检验固定联合损失未回答的梯度方向冲突与训练/推理解耦           |
| [SciRM](https://arxiv.org/abs/2601.11374)                     | 在同类科学评价任务上用 outcome-based RL 优化最终 score                | 验证 rationale target 的独立贡献及其实际可核查性              |
| [SFTKey](https://arxiv.org/abs/2512.21017)                    | 第二阶段重点优化 answer tokens                                        | 检验保留非冲突 rationale gradient 是否优于完全忽略它          |
| [PCGrad](https://arxiv.org/abs/2001.06782)                    | 提出通用多任务梯度手术                                                | 将其作为科学 evaluator 的机制干预，不声称发明新优化器         |

这里最重要的 novelty 边界是：

- “给 score token 加权”不是你的创新，Align、SFTKey、TRACT 都已经覆盖相近思想；
- “CoT 在另一个领域也可能有害”可以成为领域贡献，但单独不够强；
- 真正能把论文推高的是：**严格控制的领域发现** **+ 三目标机制拆分 + 冲突干预 + rationale utility 审计**。

---

## 4. 具体机制：把三个损失彻底拆开

同一条配对样本定义：

\[
L_d=\mathrm{CE}(y\mid p_{\mathrm{direct}}(x)),
\]

\[
L_r=\frac{1}{|r|}\sum_{t\in r}\mathrm{CE}(r_t\mid p_{\mathrm{reason}}(x),r_{<t}),
\]

\[
L_c=\mathrm{CE}(y\mid p_{\mathrm{reason}}(x),r).
\]

- `L_d`：从原输入直接预测 score 的 decision objective；
- `L_r`：只计算 rationale tokens 的 explanation objective；
- `L_c`：在 teacher-forced rationale 后预测 score 的 conditional readout objective。

分别计算 `g_d`、`g_r` 和 `g_c` 后，三种解释可以被区分：

1. **Token dilution：** 标准 sequence CE 中，rationale token 占据绝大部分有效训练权重；span balancing 后差距消失。
2. **Decision--explanation conflict：** `g_d` 与纯 rationale gradient `g_r` 经常负向，且 balancing 后性能差距仍存在。
3. **Readout-path conflict：** 主要是 `g_d` 与 `g_c` 冲突；此时问题不是理由语义本身，而是直接评分与 rationale-conditioned 评分学习了不同路径。

如果实验只发现第三种，就应把论文结论改为 decision-path conflict，不能仍然写 explanation conflict。

---

## 5. 方法：Decision-Preserving Rationale Tuning

方法暂称 **DPRT**。它不是新优化器，而是 score-primary asymmetric PCGrad 在本问题上的透明实现。

对辅助梯度定义：

\[
P(g_a;g_d)=
\begin{cases}
g_a-\dfrac{g_a^Tg_d}{\|g_d\|^2+\epsilon}g_d,&g_a^Tg_d<0,\\
g_a,&g_a^Tg_d\ge0.
\end{cases}
\]

最终更新：

\[
g_{update}=g_d+\lambda_rP(g_r;g_d)+\lambda_cP(g_c;g_d).
\]

含义很简单：

- 保留 direct score 作为主任务；
- explanation 和 conditional-score 中有帮助或中性的梯度继续学习；
- 只删除会抵消 direct decision 的辅助梯度分量。

必须设置一个完全同构的 two-view baseline：

\[
g_{joint}=g_d+\lambda_rg_r+\lambda_cg_c.
\]

它与 DPRT 使用相同 forward passes、权重、训练步数和计算量；另加 norm-matched baseline，才能证明收益来自“处理方向冲突”，而不是多看了一个 score-only view 或简单缩小了 rationale gradient。

### 两种部署模式

- **Decision mode：** 直接输出 score，是主要 accuracy/F1 endpoint；
- **Audit mode：** rationale-first 输出 rationale 和 score，rationale 只解释 audit-mode prediction，不声称它 faithful 地解释另一次 direct prediction。

这样不会把两个不同 readout 的 score 和 rationale 硬拼成一个含糊的 Pareto 指标。

---

## 6. 一套闭合的实验，而不是继续堆方向

### 第一组：复现并固定现象

- 保留现有 Qwen3-4B 七 criterion 六臂结果；
- 在第二个模型家族的 7B/8B 模型上复现训练目标乘推理协议的 2x2；
- 主要报告同列训练效应、推理协议效应和 interaction，不再用 `S-S` 对 `C-C` 单独证明 CoT supervision 有害。

### 第二组：机制区分

比较：

| 方法                       | 要检验的解释                         |
| -------------------------- | ------------------------------------ |
| Score-only                 | decision-only reference              |
| Standard rationale SFT     | 当前失败基线                         |
| Align / span-balanced      | 是否只是 token dilution              |
| SFTKey                     | 只重点训练 score 是否足够            |
| TRACT-style objective      | 固定 score-aware joint loss 是否足够 |
| Unprojected two-view joint | 加入 direct view 本身是否足够        |
| Norm-matched joint         | 梯度缩小是否足够                     |
| DPRT                       | 梯度方向冲突是否可干预               |

再用 same-length shuffled/template rationale 保持 criterion、label 和长度匹配，只破坏 rationale 语义，从而区分“长序列影响”和“解释内容影响”。

### 第三组：理由是否真的有用

在同一盲评子集上比较 Standard CoT、Align/SFTKey、DPRT 和 score-only 模型的事后理由：

- Evidence 是否真实存在于输入；
- Evidence 是否对应指定 criterion；
- Rationale 是否与 audit-mode score 一致；
- Rationale 是否帮助第三方复核评分。

这些指标只能支持 groundedness、consistency 和 utility，不要把它们写成 rationale faithfulness。

### Discovery / confirmation 划分

- **Discovery：** Qwen3-4B 的四个 RevUtil，确定主要机制和候选干预；
- **Confirmation：** 第二模型家族的全部七 criterion，用固定配置验证；
- **统计单位：** `criterion x backbone x seed` 的 run-level summary，不把逐 token 或逐样本梯度伪装成大量独立实验。

---

## 7. 预计得到什么，以及每种结果怎样成文

### 情况 A：主要是 token dilution

如果 Align/SFTKey 追平 score-only，`g_d/g_r` 冲突弱，DPRT 不再增加收益：

> 论文结论是科学 evaluator 中的 rationale SFT failure 主要来自监督分配，而不是 rationale 本身无价值。

删除 DPRT 的方法贡献，以严格领域研究投 ACL/EMNLP Findings。这个结果依然有价值，因为它直接约束 SciRM 类方法何时值得生成长 rationale，并给科学评价系统一个低成本训练结论。

### 情况 B：存在 decision--explanation conflict

如果 span balancing 仍未追平，负冲突跨模型稳定，DPRT 超过 Align、SFTKey、TRACT-style、同构 joint 和 norm-matched baseline，同时理由 utility 不下降：

> 论文结论是解释学习与评分边界存在可测量、可干预的优化冲突；保留非冲突解释梯度可以改善 score--rationale trade-off。

这是最强版本，可以冲 ACL/EMNLP Main。

### 情况 C：主要是 readout-path conflict

如果 `g_d/g_c` 冲突明显，而 `g_d/g_r` 并不冲突：

> 论文改写为 rationale-conditioned 与 direct score readout 学习了不同决策路径，解释内容不是主要问题。

这仍是一篇具体论文，但标题和贡献应转向 scientific evaluator 的 decision-path dependence。

### 情况 D：准确率与理由效用不能兼得

如果所有保护 score 的方法都会降低 rationale utility：

> 论文转向 scientific evaluator 的 accuracy--auditability frontier，指出单一生成目标无法同时优化判决与审计。

不要把这种结果包装成方法成功。

---

## 8. 最终贡献应怎样写

1. **Controlled domain finding：** 在 answer-blind、score-correct、严格配对的数据上，首次系统分离 scientific evaluator 的 rationale-first protocol effect 与 rationale-supervision effect。
2. **Mechanism decomposition：** 将 rationale-bearing SFT 拆为 direct decision、explanation token 和 conditional score readout，区分 token dilution、decision--explanation conflict 与 readout-path conflict。
3. **Companion intervention：** 如果结果支持，用 score-primary asymmetric projection 证明冲突可被干预，并在不放弃可核查理由的情况下保护评分能力。

---

## 9. 投稿判断

| 完成度                                                         | 合理目标                                                                    |
| -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 只有当前 Qwen3-4B 六臂结果                                     | Workshop / short paper，尚不足以投好会议全文                                |
| 第二模型复现 + 强基线 + 机制拆分 + rationale 盲评              | **ACL/EMNLP Findings**，现实主目标                                    |
| DPRT 跨模型超过强基线，并形成稳定 score--rationale Pareto 改善 | **ACL/EMNLP Main**；也可考虑 ICLR 的 optimization/representation 取向 |

这篇论文当前最合理的目标是先按 ACL/EMNLP Findings 的证据标准完成；Main 由实验结果决定，不应在方法有效前预设。

---

## 10. 现在只做什么

下一阶段只围绕一个机制闭环：

1. 在四个 RevUtil 上实现 `L_d/L_r/L_c` 的分离，先跑 span-balanced、SFTKey、同构 two-view joint 和 DPRT；
2. 根据 `g_d/g_r/g_c` 的实际关系，锁定 dilution、explanation conflict 或 readout conflict 中的一条；
3. 用第二模型和全部七 criterion 确认，不再同时扩展 specialist/joint、RL 或 MoE；
4. 对保留下来的方法做 rationale utility 盲评，决定论文能否声称兼顾准确性与可审计性。

这四步会直接产出论文的四张核心结果：factorial interaction、gradient mechanism、method comparison，以及 score--rationale utility 图。除此之外的实验暂时都不进入本文。

---

## 最终判断

你完全可以在已有研究基础上找一个小而具体的问题，也可以把医学或 NLU 的发现迁移到科学写作评价。问题不在于“别人是否做过相似方法”，而在于新论文是否增加了可验证知识。

你现在最值得做的，不是重新发明一种庞大训练框架，而是回答这一句：

> **为什么 rationale-first generation 能帮助未适配 evaluator，却不能作为更好的监督来训练 scientific evaluator？**

现有结果已经给了这篇论文的第一项发现；损失拆分、冲突干预和理由审计负责把它从领域复现推进为一篇完整的科学评价研究。
