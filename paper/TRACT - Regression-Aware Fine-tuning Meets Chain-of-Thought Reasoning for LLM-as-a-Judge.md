# TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge

PDF 原文：[TRACT - Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge](./TRACT - Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge.pdf)

## 核心结论

这篇论文提出了一套面向 **1–5 分制 LLM-as-a-Judge** 的训练方法：既让模型生成评价理由，又让最终分数真正按照回归目标学习。

> 普通 rationale SFT 只用交叉熵学习“解释文本+分数”，没有利用 1–5 分之间的数值距离；TRACT 用 CE 学习解释，同时用回归损失学习分数，再通过自生成 CoT 缩小训练与推理的解释分布差异。

## 为什么普通 CE 不适合评分

假设真实分数为 1：

- 模型预测 2，误差较小；
- 模型预测 5，误差很大。

但普通 token CE 只关心正确 token 是不是 `1`，错误预测 `2` 和 `5` 都被视为“错一个 token”，没有显式利用评分的顺序和距离。

因此，作者先使用模型对五个分数的概率计算期望分数：


\hat y=\sum_{k=1}^{5}k\cdot p(k\mid x,s)


这里的 x 是评价对象和评分标准，s 是模型生成的 CoT。得到的预测可以是 3.7、4.2 这样的连续值。

然后用平方误差训练：


L_{\text{RAFT}}=(\hat y-y^*)^2


这就是 regression-aware fine-tuning 的核心。

## CoT-RAFT：同时学习解释和分数

TRACT 的训练目标近似为：


L=L_{\text{CE}}(s,y^*)+\lambda(\hat y-y^*)^2


其中：

- CE 负责学习评价解释及输出格式；
- RAFT 负责让期望分数接近真实分数；
- \lambda 控制回归目标权重，论文默认设为 1。

需要注意：论文文字有时说“CE 只学习 CoT”，但公式中的 CE 是 -\log p([s,y^*]\mid x)，实际上也包含最终分数 token。因此它不是用 RAFT 替代 CE，而是 **完整序列 CE 加回归损失**。

推理时，模型先生成 CoT，遇到固定结束语 `So the overall score is` 后，不直接贪心生成一个整数，而是读取 1–5 五个分数的概率并计算期望值。这称为 **CoT-RAIL**。

## TRACT 的两阶段训练



### 第一阶段：学习 GPT-4 CoT

从原始模型 p_0 开始，用 GPT-4 生成的评价理由和真实分数训练，目标是 CoT-RAFT，得到模型 p_s。

### 生成自有 CoT

让 p_s 为全部训练样本生成自己的评价理由。模型同时会生成分数，但作者丢弃这个预测分数，只保留：

- p_s 自己生成的 CoT；
- 原数据中的真实分数。



### 第二阶段：从原始模型重新训练

第二阶段不是在 p_s 上继续训练，而是重新从 p_0 初始化，用自生成 CoT 和真实分数再次执行 CoT-RAFT，得到最终模型 p_{\text{TRACT}}。

这个细节非常关键。作者尝试过直接从第一阶段模型继续训练，平均 Pearson 相关只有 0.515；重新从 base model 训练则达到 0.650。

## 为什么要换成自生成 CoT

训练时使用 GPT-4 的 CoT，推理时模型只能使用自己的 CoT，两者存在分布偏移。

作者发现，第一阶段模型：

- 条件在 GPT-4 CoT 上预测训练分数，RMSE 为 0.12；
- 条件在自己生成的 CoT 上预测，RMSE 上升到 0.63。

第二阶段使用自生成 CoT 后：

- 条件在训练 CoT 上，RMSE 为 0.45；
- 条件在自己生成的 CoT 上，RMSE 也是 0.45。

作者据此认为，第二阶段缩小了训练和推理时 CoT 的分布差异。不过这只是通过分数 RMSE 间接反映，并没有直接计算两个 CoT 文本分布的距离。

## 主要实验结果

训练数据是约 10 万条 GPT-4 合成的 Feedback Collection，模型为 Mistral-7B 和 Llama-3.1-8B。

Mistral-7B 在四个 point-wise Judge 数据集上的平均 Pearson 相关：


| 方法              | CoT | 分数目标    | 平均 Pearson |
| --------------- | --- | ------- | ---------- |
| Label-only CE   | 否   | CE      | 0.488      |
| 普通 CoT SFT      | 是   | CE      | 0.557      |
| RAFT            | 否   | 回归      | 0.623      |
| Prometheus-2-7B | 是   | CE      | 0.591      |
| **TRACT**       | 是   | CE + 回归 | **0.650**  |


Llama-3.1-8B 上：

- 普通 CoT SFT：0.561；
- RAFT：0.639；
- TRACT：**0.674**。

这说明最强的简单基线其实不是普通 CoT，而是没有 CoT 的 RAFT。完整 TRACT 相对 RAFT 的增益约为 0.027–0.035。

## 消融实验说明了什么

Mistral-7B 平均 Pearson：


| 变体                        | 结果        |
| ------------------------- | --------- |
| CoT-RAFT，只用 GPT-4 CoT     | 0.556     |
| 自生成 CoT，只用 CE，CoT-RAIL 推理 | 0.617     |
| 自生成 CoT，只用 CE，普通解码        | 0.521     |
| **完整 TRACT**              | **0.650** |
| 第二阶段从第一阶段模型继续训练           | 0.515     |


可以看出：

- 回归感知推理本身贡献很大；
- 自生成 CoT 必须与回归目标结合；
- 只用 CE 训练自生成 CoT，效果并不好；
- 第二阶段必须从原始 base model 重启；
- CE 与回归目标必须联合训练，先 CE 后单独 RAFT 会破坏 CoT 输出格式。



## RewardBench 上并非全面领先

在 pairwise RewardBench 上：

- RAFT：**0.775**；
- CLoud Reward Model：0.759；
- TRACT-Llama：0.748；
- TRACT-Mistral：0.736；
- Prometheus-2 pairwise：0.720。

因此，TRACT 在 point-wise 评分任务上最强，但在 RewardBench 上没有超过纯 RAFT。论文更准确的结论是：它虽然没有用 pairwise 数据训练，仍具备不错的排序能力，而不是“全面优于奖励模型”。

## 需要谨慎看待的地方

- 主实验没有报告多训练随机种子的均值和方差，也没有给相关系数差异的显著性检验。
- 只测试 Mistral-7B 和 Llama-8B，以及固定的 1–5 分任务。
- 训练数据、分数和 CoT 大量来自 GPT-4，模型可能主要学习模仿 GPT-4 的评价标准。
- 自生成 CoT 质量仍由 GPT-4 评估，200 个样本中 GPT-4 CoT 得分 4.78，自生成 CoT 为 4.50，缺少独立人工验证。
- “分布偏移被消除”通过预测 RMSE 间接论证，没有直接比较 CoT 的语义或概率分布。
- 完整流程需要两次约 50 小时的训练，加上全训练集 CoT 生成，成本较高。
- TRACT 需要访问 score token 的 logits，无法直接用于只返回文本、不提供概率的闭源 API。



## 与当前研究的关系

当前理解有两个地方需要校正：

1. **TRACT 不是用 CoT-RAFT 取代 CE。**
  CoT-RAFT 本身就是“完整序列 CE + 期望分数平方误差”。
2. **第二阶段不是继续训练第一阶段模型。**
  第一阶段模型只负责生成学生自己的 CoT；最终训练必须重新从 base model 开始。

它和 Align 解决的问题也不同：

- Align 解决长 rationale token 淹没标签 loss；
- TRACT 解决 1–5 分之间存在顺序和数值距离，但 token CE 不理解这种距离；
- 自生成 CoT 解决教师解释与学生推理分布不一致。

对科学写作评分任务，最合理的实验链条应当是：

`Label-only CE → Rationale CE → Align → RAFT without CoT → CoT-RAFT → 完整 TRACT`

这样才能分别判断增益来自标签权重平衡、回归目标、期望分数解码，还是自生成 CoT。