# STaR: Self-Taught Reasoner - Bootstrapping Reasoning With Reasoning

## 一句话概括

STaR 提出了一种迭代式自训练方法：**只给模型少量人工 rationale 示例，让模型为大量只有问题和答案的数据自行生成 rationale，再利用答案正确性筛选训练数据，不断提高推理能力。**

## 它要解决什么问题

当时让模型学会输出推理过程主要有两种方法：

- 人工标注大量 rationale：质量相对可控，但成本很高。
- Few-shot CoT：只需要少量示例，但通常不如在完整数据集上直接微调。

STaR 想结合两者的优点：只人工准备约 10 个 rationale 示例，然后让模型把自己的推理能力逐步扩展到整个带答案数据集。

需要注意，STaR 不是完全无监督方法。它不需要为全部样本标注 rationale，但仍然需要每道题的标准答案。

## STaR 的训练循环

假设训练数据只有问题 `x` 和正确答案 `y`，没有 rationale。每轮训练包括以下步骤：

1. 使用少量 CoT 示例提示当前模型，为每道题生成 `rationale + answer`。
2. 如果模型的最终答案正确，就保留这条 rationale。
3. 如果答案错误，就把正确答案作为 hint，让模型根据正确答案反向生成 rationale。
4. 去掉 hint，将筛选出的普通 rationale 和反向生成的 rationale 用于微调。
5. 使用新的模型重新生成下一轮训练数据，重复上述过程。

整体流程可以表示为：

```text
问题
  ↓
模型生成 rationale + answer
  ├─ 答案正确 → 保留
  └─ 答案错误 → 给出正确答案 → 重新生成 rationale
                              ↓
                    去掉答案提示后加入训练集
  ↓
从原始预训练模型重新微调
  ↓
下一轮生成
```

每轮训练都从原始预训练模型重新开始，而不是在上一轮模型上持续训练。作者认为，这样可以减少连续微调导致的过拟合和训练漂移。

## Rationalization 为什么重要

如果只保留模型原本就能答对的问题，训练集会越来越局限于简单题。模型始终无法从不会做的问题中获得训练信号，最终很快停止进步。

STaR 为此引入 **rationalization**：

> 对答错的问题提供正确答案，让模型根据答案反向构造一个能够支持正确答案的推理过程。

普通 rationale generation 从下面的分布中搜索：

```text
p(rationale | question)
```

Rationalization 则利用正确答案，把搜索空间变为：

```text
p(rationale | question, correct answer)
```

后者更容易生成能够到达正确答案的推理轨迹。训练时再移除答案提示，使模型学习在没有 hint 的情况下产生类似 rationale。

## 与强化学习的关系

作者将 rationale 看作潜变量，把最终答案正确与否看作二元奖励：

```text
reward = 1，最终答案正确
reward = 0，最终答案错误
```

因此，“只用答案正确的推理轨迹进行 SFT”可以理解为一种简化的 policy gradient 或 Expert Iteration。

STaR 不需要额外训练奖励模型，也不直接运行标准强化学习算法，而是通过生成、答案筛选和监督微调来近似这一优化过程。

## 实验设置

作者以 GPT-J 6B 为基础模型，在三种任务上评估 STaR：

- 多位数加法：结构明确的符号计算任务。
- CommonsenseQA：五选一常识推理任务。
- GSM8K：需要多步计算的小学数学文字题。

## 实验结果

| 任务 | Direct 微调 | STaR 无 Rationalization | 完整 STaR |
| --- | ---: | ---: | ---: |
| 多位数加法 | 76.3% | 未单独汇总 | **89.5%** |
| CommonsenseQA | 60.0% | 68.8% | **72.5%** |
| GSM8K | 5.8% | 10.1% | **10.7%** |

### 多位数加法

- Few-shot 条件下，两位数加法准确率不足 1%，更多位数接近 0%。
- 经过 16 轮 STaR 后，总体准确率达到 89.5%。
- Direct Answer 微调基线为 76.3%。
- Rationalization 明显加快了不同数字长度上的学习。
- 模型还能够解决部分训练中没有出现过的 9 位和 10 位加法问题，表现出一定长度外泛化能力。

### CommonsenseQA

| 方法 | Accuracy |
| --- | ---: |
| Few-shot Direct GPT-J | 20.9% |
| Few-shot CoT GPT-J | 36.6% |
| GPT-J Direct 微调 | 60.0% |
| STaR 无 Rationalization | 68.8% |
| 完整 STaR | **72.5%** |
| GPT-3 Direct 微调 | 73.0% |

STaR 让 GPT-J 6B 的结果接近参数量约大 30 倍的 GPT-3。不过，这并不意味着两个模型具有相同的整体能力，只说明它们在该数据集上的准确率接近。

Rationalization 将 CommonsenseQA 准确率从 68.8% 提高到 72.5%，说明它确实能让模型从原本答错的训练样本中获得信号。

### GSM8K

- Few-shot Direct：3.0%。
- Few-shot CoT：3.1%。
- Direct 微调：5.8%。
- STaR 无 Rationalization：10.1%。
- 完整 STaR：10.7%。

STaR 相对基线有明显提升，但绝对准确率仍然较低。Rationalization 只带来 0.6 个百分点提升，说明它的作用具有明显的任务依赖性。

## Rationale 人工评价

作者从 CommonsenseQA 中选择了 50 道题，并邀请 20 名参与者比较不同来源的 rationale。

- 参与者更倾向于选择 STaR rationale，而不是普通 few-shot rationale，差异达到统计显著。
- 参与者也更偏好 STaR rationale，而不是数据集中的人工 rationale。

作者明确指出，这不能证明 STaR 已经达到人类解释水平。一个重要原因是原始人工 rationale 数据本身存在重复、语病、简单复述答案等质量问题。

## 最大的问题：答对不等于推理正确

STaR 只检查最终答案是否正确，不直接验证 rationale 是否正确、相关或忠实。在多选题中，模型可能使用错误理由，但碰巧选中了正确答案。

作者在附录中展示了多种失败模式：

- 循环论证：把答案换一种说法，当作证明。
- 先假设答案成立，再声称答案正确。
- 给出与结论没有支持关系的事实。
- 编造题目中不存在的世界状态。
- 使用与问题有关但与结论无关的内容。
- 利用 rationalization 中的答案 hint 走捷径。
- 推理过程错误，但最终答案碰巧正确。

因此，STaR 学到的有时只是“能够通向正确标签的文本”，不一定是模型真实使用的、逻辑上可靠的推理过程。Rationalization 尤其容易产生事后合理化。

## 其他局限

- 初始模型的 few-shot 表现必须高于随机水平；作者发现 GPT-2 即使在加法任务上也无法成功自举。
- 二分类任务尤其危险，因为随机猜测就有 50% 的概率通过答案筛选。
- 方法依赖标准答案，不能直接用于真正无标签的数据。
- 多轮生成和重新微调的计算成本较高。
- 主要只验证了 GPT-J 6B 和三个任务，结论的模型与任务覆盖有限。
- 人工 rationale 评价规模较小。
- 没有可靠验证 rationale 的真实性、证据一致性和因果忠实性。
- 提高采样温度并不能简单替代 rationalization；高温采样更容易产生“答案正确但 reasoning 错误”的训练样本。
- 是否保留 few-shot prompt 会影响 rationale 风格漂移、计算成本和后续 rationalization 能力，是一个额外超参数。

## 对科学写作评价实验的意义

STaR 为科学写作评价提供了一条不同于普通 CoT SFT 的训练路线：

- 普通 CoT SFT：直接学习固定的 synthetic rationale。
- STaR：学习模型自己生成、并经过答案正确性筛选的 on-policy rationale。
- Rationalization：对失败样本利用金标签反向构造新的训练轨迹。

这种方法可能减少教师生成 rationale 与学生模型自身分布之间的不一致，但不能直接照搬到当前所有任务。

### 1. 标签正确性过滤不够可靠

四个五分类评价任务的随机正确率为 20%，仍然会保留一部分错误 rationale。Positioning Check 等二分类任务的随机正确率达到 50%，正是 STaR 明确认为容易失效的场景。

因此，如果应用 STaR，筛选条件不应只有“最终标签正确”，还需要考虑：

- 标签预测置信度或分类 margin。
- 多次采样的一致性。
- rationale 与输入证据的一致性。
- 对 rationale 进行反事实修改后，标签是否合理变化。
- 独立 rationale evaluator 的质量判断。

### 2. STaR 不证明 rationale 普遍有用

STaR 的结果不能推出所有 rationale SFT 都优于 Label-only SFT。它使用的是经过答案筛选、与当前模型分布相关的 rationale，而且实验任务本身具有较强的推理结构。

科学写作评价主要是软语义判断。即使 rationale 最终指向正确标签，也可能没有使用正确证据。因此，STaR 更适合作为一种需要单独验证的对照方法，而不是 rationale 有效性的直接证据。

### 3. 应区分三个评价目标

如果在科学写作评价中引入 STaR，应当分别评价：

1. 最终标签预测是否改善。
2. Rationale 是否正确、相关且基于输入证据。
3. Rationale 是否忠实反映模型得到标签的依据。

只观察 Accuracy 或 Macro-F1，不能判断后两个目标是否实现。

## 总结

STaR 的核心贡献不是证明 rationale 天然有用，而是提出了一种“生成推理轨迹、用最终答案筛选、再迭代训练”的自举框架。它显著减少了人工 rationale 数据需求，并证明模型可以通过学习自身成功轨迹提高部分推理任务的性能。

但它也留下了一个关键问题：**最终答案正确是否足以证明中间推理值得学习？** 对数学等可验证任务，这一假设相对可靠；对常识判断和科学写作评价等软语义任务，它可能保留大量合理化、循环论证或碰巧答对的 rationale。
