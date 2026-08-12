# TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge

PDF 原文：[TRACT - Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge](<./TRACT - Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge.pdf>)

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

需要注意：论文公式把 CE 写成 `-\log p([s,y^*]\mid x)`，形式上包含最终分数 token，但正文又称其为 “CE over the CoT”。官方发布代码采用后一种语义：先把数字分数 token 从 CE labels 中屏蔽，再单独用 RAFT 回归损失训练该位置。因此它不是用 RAFT 取代全部 CE，而是 **解释与格式 CE 加分数回归损失**。

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



## 论文、官方代码与当前仓库实现对照

### 先区分训练和推理术语

- **RAFT**：不使用 CoT 的回归感知训练目标。
- **RAIL**：与 RAFT 对应的期望分数推理方法。
- **CoT-RAFT**：同时使用解释文本 CE 和分数 RAFT MSE 的训练目标。
- **CoT-RAIL**：先生成 CoT，再根据该 CoT 后面的分数概率计算期望分数。
- **TRACT**：包含两次独立 CoT-RAFT 训练和中间 self-CoT 数据生成的完整方法。

因此，RAFT/CoT-RAFT 描述训练，RAIL/CoT-RAIL 描述推理；greedy decoding 是另一种推理方式，不是一种训练方法。

### 三种实现共有的分数期望

当前仓库先在完整词表上计算 softmax：

$$
p=\operatorname{softmax}(z)
$$

再抽取合法分数 token 的概率，计算：

$$
\hat y=\sum_{k\in\mathcal Y}k\cdot p(\operatorname{token}(k))
$$

最后使用：

$$
L_{\text{RAFT}}=(\hat y-y)^2
$$

这里不会在合法分数集合内二次归一化。如果 `1` 到 `5` 的总概率质量只有 0.7，剩余 0.3 不会重新分给五个分数。这一点与 TRACT 官方推理代码一致，也是当前训练和 RAIL/CoT-RAIL 推理共同采用的定义。

当前实现位于 [`training/trainers/raft_trainer.py`](../../training/trainers/raft_trainer.py)，论文定义见 [§2.3 Regression-Aware Fine-Tuning](https://arxiv.org/html/2503.04381v2#S2.SS3)。

### 当前 RaftWithoutCot

训练数据形如：

```text
prompt
→ <score>5</score>
```

训练步骤如下：

1. 根据当前 tokenizer 动态解析每个合法分数的 token ID，并要求每个分数恰好占一个 token。
2. 在 completion 中定位唯一的数字分数 token。
3. 取数字前一个位置的 logits，即根据 `prompt + <score>` 预测数字。
4. 在完整词表上计算 softmax，再计算合法分数的原始概率加权和。
5. 只优化分数期望与 gold label 之间的 MSE。

其损失只有：

$$
L=L_{\text{RAFT}}
$$

数字后面的 `</score>`、EOS 以及 `<score>` 本身都不参加 CE，因为该 Trainer 完全没有 CE loss。它只学习“已经给定 `<score>` 前缀后，分数概率应该如何分布”，不负责学习自由生成解释或分数包装格式。

训练期间的验证也不做自由文本生成，而是强制提供 `<score>` 前缀，直接执行 RAIL。连续期望分数用于 MSE/MAE；为了计算 Accuracy、Macro-F1 和 QWK，当前评估代码额外把连续值映射到最近的合法标签。

它与论文的关系是：

- 方法上对应论文中的 **RAFT baseline**，不是 TRACT。
- 论文和官方 score-only RAIL 近似计算 $p(y\mid x)$；当前仓库计算 $p(y\mid x,\texttt{<score>})$，回归思想相同，但分数位置的条件上下文不同。
- TRACT 官方仓库没有发布独立、完整的 `RaftWithoutCotTrainer`；当前版本是本仓库依据论文公式实现的 score-only RAFT。

### 当前 CotRaft

训练数据形如：

```text
prompt
→ <reasoning>教师解释</reasoning><score>5</score>
```

当前 CoT 数据来自教师模型，例如 `deepseek-v4-pro`，不是当前 Qwen 模型在 Stage 1 后自生成的 CoT。

预处理会使用 `<score>...</score>` 的结构位置精确标记数字 token，并要求：

- 每条 completion 恰好有一个 `<score>` block；
- block 内数字等于 gold label；
- 数字恰好占一个 tokenizer token；
- 截断后该数字仍然存在。

训练目标是：

$$
L=L_{\text{LM}}+\lambda L_{\text{RAFT}}
$$

当前 `lambda = 1.0`。各 token 的监督关系为：

```text
<reasoning>...</reasoning><score>  5  </score><eos>
|-------------- CE -------------| MSE |---- CE ----|
```

- `L_LM` 覆盖解释、`<reasoning>`/`<score>` 包装、`</score>` 和 EOS，但屏蔽数字分数。
- `L_RAFT` 只作用在数字分数位置。
- 训练分数时使用 teacher forcing，实际优化的是 $p(y\mid x,s_{teacher},\texttt{<score>})$。

当前每个 epoch 内的验证仍然使用 greedy 完整生成并解析 `<score>`，不是 CoT-RAIL。独立运行 `../../training/evaluate.py --inference_mode cot_rail` 时，才会先生成解释直到 `<score>`，再执行一次单 token 概率探测。如果第一阶段没有生成 `<score>`，该样本的 CoT-RAIL 结果就是无效值。

当前 `cot_raft` 与论文的关系是：

- loss 结构与官方代码表达的 CoT-RAFT 一致；
- 数据上只对应使用外部教师 CoT 的单阶段训练；
- 更接近论文的 Stage 1 或 “CoT-RAFT with annotation CoT” 消融；
- 不能直接称为完整 TRACT。

### 完整 TRACT 缺少的 Stage 2

论文完整流程是：

```text
Stage 1
原始底模 p0
+ 教师标注 CoT
+ gold score
→ CoT-RAFT
→ p_s

Self-CoT 数据生成
p_s 为每个训练输入生成 CoT 和预测分数
→ 丢弃预测分数
→ 保留 self-CoT
→ 配回原始 gold score

Stage 2
重新加载原始底模 p0
+ self-CoT
+ 原始 gold score
→ 再做一次 CoT-RAFT
→ p_TRACT
```

Stage 2 必须重新从 `p0` 初始化，不能从 `p_s` checkpoint 继续训练。当前仓库没有把 self-CoT 生成、丢弃预测分数、配回 gold label、重新初始化底模和第二次训练串成这条流水线，所以当前 `cot_raft` adapter 不是 `p_TRACT`。

论文算法见 [§3.2 与 Algorithm 1](https://arxiv.org/html/2503.04381v2#S3.SS2)，官方仓库也把训练过程明确拆成 [Stage 1、self-CoT 生成和 Stage 2](https://github.com/d223302/TRACT#fine-tuning)。

### 官方发布代码的实际行为与问题

官方 [`custom_loss.py`](https://github.com/d223302/TRACT/blob/dcb7a649a899274f2c072f7884dc94ad77a6ea19/finetuning_utils/cot_with_raft/custom_loss.py) 表达的目标行为是：

1. 把 completion 倒数第二个有效 token 当成数字分数。
2. 从 CE labels 中屏蔽这个数字。
3. 在数字前一个位置取 logits。
4. 对完整词表做 softmax，不对分数候选重新归一化。
5. 计算 `LM loss + 1.0 * score MSE`。

但官方仓库当前发布版本不是可以直接照搬的干净参考实现：

- [`custom_processor.py`](https://github.com/d223302/TRACT/blob/dcb7a649a899274f2c072f7884dc94ad77a6ea19/finetuning_utils/cot_with_raft/custom_processor.py) 已提前屏蔽数字并单独保存 `score_labels`，但 `custom_loss.py` 又尝试从未屏蔽的 `labels` 中读取数字。
- [`workflow.py`](https://github.com/d223302/TRACT/blob/dcb7a649a899274f2c072f7884dc94ad77a6ea19/finetuning_utils/cot_with_raft/workflow.py) 导入了自定义 collator，却实际实例化普通 SFT collator。
- 分数 token ID 硬编码为 Mistral 或 Llama 的 `1` 到 `5`。
- 在 `num_items_in_batch` 存在时，代码会对已经取 mean 的 MSE 再除一次 batch size，源码本身也留有 TODO。

相比之下，当前仓库在单阶段 loss 的工程实现上更稳健：动态解析分数 token ID，使用 `<score>` 的显式结构 mask 定位数字，并采用正常的 batch mean。不过这些工程修正不等于补齐了 TRACT Stage 2。

### 训练配置差异

| 项目 | 当前仓库 | 论文报告设置 |
| --- | --- | --- |
| 底模 | Qwen3-4B | Mistral-7B / Llama-3.1-8B |
| CoT 教师 | DeepSeek 教师数据 | GPT-4 annotation CoT，随后 self-CoT |
| LoRA rank | 16 | 8 |
| Learning rate | `1e-4` | `1e-5` |
| Epoch | 3 | 2 |
| 有效 batch | 16 | 8 |
| CoT-RAFT 阶段数 | 1 | 2 |
| checkpoint | 每个 epoch 保存，只保留最后一个；adapter 来自最终 checkpoint | 官方配置按 steps 保存，论文不以当前仓库的保留策略为方法组成部分 |

因此，当前实验与论文不能被视为严格复现。模型、教师 CoT、训练超参数和最关键的 Stage 2 都不同。

### 最简洁的方法定位

| 当前名称 | CoT 来源 | 训练目标 | 训练期验证 | 准确定位 |
| --- | --- | --- | --- | --- |
| `raft_without_cot` | 无 | 仅分数期望 MSE | RAIL | 论文 RAFT baseline |
| `cot_raft` | DeepSeek 教师 CoT | 解释/格式 CE + 分数 MSE | Greedy | 单阶段 CoT-RAFT，接近 TRACT Stage 1 |
| 完整 TRACT | Stage 1 教师 CoT；Stage 2 self-CoT | 两次从 `p0` 开始的 CoT-RAFT | 最终使用 CoT-RAIL | 当前尚未实现 |

最终结论：当前 `raft_without_cot` 基本对齐论文 RAFT 基线，但使用了显式 `<score>` 条件前缀；当前 `cot_raft` 基本对齐官方代码意图中的单阶段 CoT-RAFT loss，但不等于完整 TRACT。

## 与当前研究的关系

当前理解有两个地方需要校正：

1. **TRACT 不是用 CoT-RAFT 取代 CE。**
  CoT-RAFT 本身就是“解释与格式 CE + 期望分数平方误差”；按官方代码行为，数字分数 token 本身从 CE 中屏蔽。
2. **第二阶段不是继续训练第一阶段模型。**
  第一阶段模型只负责生成学生自己的 CoT；最终训练必须重新从 base model 开始。

它和 Align 解决的问题也不同：

- Align 解决长 rationale token 淹没标签 loss；
- TRACT 解决 1–5 分之间存在顺序和数值距离，但 token CE 不理解这种距离；
- 自生成 CoT 解决教师解释与学生推理分布不一致。

对科学写作评分任务，最合理的实验链条应当是：

`Label-only CE → Rationale CE → Align → RAFT without CoT → CoT-RAFT → 完整 TRACT`

这样才能分别判断增益来自标签权重平衡、回归目标、期望分数解码，还是自生成 CoT。
