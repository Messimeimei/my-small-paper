# Investigating the Impact of Rationales for LLMs on Natural Language Understanding

PDF 原文：[Investigating the Impact of Rationales for LLMs on Natural Language Understanding](<./Investigating the Impact of Rationales for LLMs on Natural Language Understanding.pdf>)

## 核心结论

这篇论文研究的是：在 NLU 判别任务中，推理解释到底应该放在推理阶段还是训练阶段，以及怎样训练才不会让长解释压制短标签。

> 对 NLU 任务，直接把 rationale 串在标签前后做普通 SFT，通常不如 Label-only；但把直接标签学习和 rationale 学习拆成两个分支、分别归一化损失后再加权，可以稳定恢复并小幅超过 Label-only。

## 数据和任务

作者构建了中文数据集集合 **NLURC**：

- 34 个公开数据集；
- 11 类 NLU 任务；
- 252,507 条合成 rationale；
- 覆盖情感分析、立场检测、主题分类、复述检测、NLI、阅读理解、常识阅读理解、中文语法、NER 等任务；
- 主要使用 Hunyuan-Turbo，根据输入和真实标签生成解释；
- 实验模型为 Qwen1.5-Chat 0.5B、1.8B、7B、32B。

作者重点评测五类任务：阅读理解、立场检测、主题分类、中文语言学任务和常识阅读理解。

## 五种训练方法

| 方法 | 训练目标 | 含义 |
|---|---|---|
| Label-Only | 文本 → 标签 | 只学习分类 |
| Reason | 文本 → rationale → 标签 | 先解释再预测 |
| Explain | 文本 → 标签 → rationale | 先预测再解释 |
| Mix | 混合 Label-Only 和 Reason 样本 | 所有 token 一起平均 loss |
| Align | 配对 Label-Only 和 Reason 分支 | 两个分支分别归一化，再加权求和 |

`Mix` 的问题是：标签通常只有几个 token，rationale 平均有 200–500 个 token。普通 token 平均会让 rationale 几乎控制整个 loss。

`Align` 则近似为：

\[
L = \lambda L_{\text{label}} + (1-\lambda)L_{\text{rationale}}
\]

两个 loss 各自在自己的 token 范围内求平均，因此长 rationale 不再天然获得几百倍的权重。消融实验表明，标签系数约为 0.5 时通常最好。

## 结果一：普通 Rationale SFT 大多有害

以 Direct inference 的平均成绩为例：

| 模型 | Label-Only | Mix | Align |
|---|---:|---:|---:|
| 0.5B | 73.39 | 70.47 | **77.03** |
| 1.8B | 80.55 | 77.96 | **83.49** |
| 7B | 88.51 | 86.98 | **88.81** |
| 32B | 91.38 | 90.99 | **91.78** |

主要规律是：

- `Reason` 和 `Explain` 基本都不如 Label-only；
- `Mix` 仍然受到 rationale 长度支配，平均结果通常也更差；
- 只有 `Align` 在不同模型规模和任务上持续超过 Label-only；
- 但随着模型变大，Align 的增益明显缩小：0.5B、1.8B 提升较明显，7B、32B 只有约 0.3–0.4 分。

因此，论文并不是证明“rationale SFT 一定有效”，而是证明：**只有显式保护标签目标，rationale 才可能成为有用的辅助监督。**

## 结果二：CoT 推理是否有效取决于模型和训练状态

对于没有经过任务微调的原始模型：

| 模型 | Direct | CoT | 差值 |
|---|---:|---:|---:|
| 0.5B | 24.00 | 23.95 | -0.05 |
| 1.8B | 42.21 | 42.16 | -0.05 |
| 7B | 62.87 | 64.07 | +1.20 |
| 32B | 78.44 | 80.28 | +1.84 |

原始模型越大，CoT 越可能有帮助。

但对于已经使用 Align 训练过的模型，CoT 仍然低于 Direct：

- 0.5B：低 11.53 分；
- 7B：低约 7 分；
- 32B：低约 3.7 分。

所以论文标题式结论“CoT 随模型变大而有益”需要加限定：**它主要在原始模型上成立；微调模型到 32B 时，CoT 仍未超过直接输出，只是损害变小。**

作者对错误案例的解释是：NLU 分类通常只需要抓住少数关键证据，而 CoT 会迫使模型分析整段文本。模型分析得越多，误解某个局部信息的机会也越多，错误随后会沿推理链传播。人工分析中，CoT 独有错误主要是理解错误，而不是形式逻辑错误。

## 结果三：Align 提高未见任务泛化

作者还进行了 leave-one-task-out 实验：训练时排除整个目标任务，再测试模型能否泛化过去。

例如 7B 模型：

| 未见任务 | Label-Only | Align |
|---|---:|---:|
| 阅读理解 | 80.8 | **81.4** |
| 立场检测 | 87.0 | **89.4** |
| 中文语言学 | 52.4 | **53.2** |
| 常识阅读理解 | 72.6 | **73.1** |

Align 7B 在阅读理解和立场检测上略微超过原始 Qwen 72B。不过这不能简单理解为“7B 全面击败 72B”：7B 接受了大量 NLURC 多任务训练，而 72B 是未经该数据训练的原始模型；在另外两个任务上，72B 仍明显更强。

## 需要谨慎看待的地方

- 7B 和 32B 上 Align 相对 Label-only 只提高约 0.3–0.4 分，但论文没有报告标准差、置信区间或显著性检验，不能确定小增益是否稳定。
- Seen-task 实验有三个随机种子，但 unseen-task 只有一次训练。
- 所有主实验都来自中文数据和同一 Qwen1.5 模型家族，跨语言、跨模型泛化尚未验证。
- Rationale 是在已知真实标签的条件下生成的，本质上属于 post-hoc justification，不一定反映真实决策过程。
- 论文把生成解释的“合理、完整、符合标签”称为 interpretability，但没有做删除、替换、反事实干预，因此没有证明解释的因果忠实性。
- CoT 推理从温度 0 和温度 0.7 两种设置中报告更优结果，如果是在测试集上选择，会带来一定的选择偏差。
- 论文正文公式把 Align 写成标签和 rationale 两类目标，但附录又称实际组合 Label-only 与 Reason（rationale+label）样本，目标定义存在一定表述歧义。

## 与当前实验的直接关系

这篇论文的附录明确说明：`Mix` 和 `Align` 使用的是 **Label-only prompt 与 Reason prompt 两套不同指令**。

- Label 分支的 prompt 要求直接输出标签；
- Reason 分支的 prompt 要求先输出 reasoning，再输出标签；
- 两个分支在同一 batch 中分别计算 loss；
- 不能使用同一个“先 reasoning 再 label”的 system prompt，却让一个 completion 只输出 label、另一个只输出 reasoning。

因此，之前发现的 Align 数据构造问题确实成立。原来的做法同时制造了**提示—completion 冲突**和**训练—推理格式不一致**；在修复这个问题之前，旧的 Align 结果不能用来判断损失重加权方法是否有效。
