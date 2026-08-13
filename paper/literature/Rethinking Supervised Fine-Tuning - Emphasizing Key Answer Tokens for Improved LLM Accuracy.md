# Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy

PDF 原文：[Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy](<./Rethinking Supervised Fine-Tuning - Emphasizing Key Answer Tokens for Improved LLM Accuracy.pdf>)

> 整理说明：本文档由 AI 辅助根据本地 PDF 原文整理。方法、数字和作者结论均以论文为依据；标为“分析”或“证据边界”的内容是对论文论证力度的进一步判断，不代表作者原话。

## 1. 一句话概括

这篇论文认为，在包含长 Chain-of-Thought（CoT）和短 final answer 的训练样本中，标准 SFT 对所有输出 token 使用相同的优化方式，长 CoT 因 token 数量多而占据大部分训练信号，最终答案反而训练不足。作者因此提出 **SFTKey-Tag**：先对“推理 + 答案”做完整 SFT，再增加一个只对答案 token 计算 loss 的训练阶段，在保留输出格式的同时提高答案准确率。

论文最重要的结论不是“只训练答案就够了”，而是：

> 完整序列训练负责学习响应结构和推理形式，答案专项训练负责提高最终预测准确率；两个阶段具有互补作用。

## 2. 论文试图解决什么问题

### 2.1 典型 CoT 训练样本的长度不平衡

许多 reasoning SFT 样本具有如下结构：

```text
Prompt: 一个问题

Response:
  很长的 reasoning / chain-of-thought
  很短的 final answer
```

例如，CoT 可能包含数百个 token，而最终答案只有几个 token。标准 causal language modeling loss 通常在所有未被 mask 的输出 token 上求和或求平均，因此每个 token 都是一个监督位置。

设 reasoning 长度为 $R$，答案长度为 $A$。在 token-level 均匀优化的直觉下，答案 token 占全部监督位置的比例约为：

$$
\frac{A}{R+A}.
$$

当 $R \gg A$ 时，这一比例很小。论文据此提出以下问题：

1. 模型可能更擅长模仿冗长的推理文本，而不是输出正确答案。
2. Benchmark 通常按最终答案评分，而不是按 CoT 长度或语言流畅度评分。
3. 因而，训练目标中的 token 权重可能与任务的评价目标不一致。

这里需要澄清：论文所说的“模型把更多 attention 分配给 CoT”，更准确地说是 **CoT token 在 loss 和梯度中占据更多监督位置**，不是在测量 Transformer 内部的 attention weight。（PDF p. 1）

### 2.2 直接只训练答案又会产生新问题

最直接的解决方案是 mask 掉 CoT，只在答案位置计算 loss。但论文发现，这样虽然常常能提高答案准确率，却会严重破坏模型的输出结构：模型可能不能正确生成 `<Thinking>`、`</Thinking>`、`<Answer>` 和 `</Answer>`，甚至不能稳定地产生可读的“推理后回答”格式。（PDF pp. 7-8）

因此真正的问题不是简单的“CoT 与答案二选一”，而是如何同时保留：

- 完整响应的结构与可读性；
- CoT 作为答案生成的上下文；
- final answer token 的充分优化；
- 最终答案的准确率。

## 3. 作者的完整思路链条

论文的逻辑可以压缩成以下六步：

1. 标准 SFT 将所有输出 token 放入同一个训练目标。
2. 长 CoT 包含远多于 final answer 的 token。
3. 因此，标准 SFT 可能低估 final answer 在任务评价中的重要性。
4. 如果从头只训练答案，模型又学不好完整输出结构。
5. 所以应当先用全序列 SFT 学习格式和一般响应能力。
6. 再从该 checkpoint 出发，仅优化答案位置，强化最终预测。

对应的方法流程如下：

```mermaid
flowchart LR
    A[原始问题与响应] --> B[加入 Thinking / Answer 标签]
    B --> C[Stage 1: 全序列 SFT-Tag]
    C --> D[学会推理形式与输出结构]
    D --> E[Stage 2: mask Thinking loss]
    E --> F[仅 Answer 位置产生 loss]
    F --> G[保留格式并强化答案准确率]
```

## 4. 方法定义

### 4.1 标准 SFT

给定包含 $N$ 个 prompt-response 样本的数据集：

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N},
$$

标准 SFT 在整个 response 上最小化负对数似然：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=-
\sum_{i=1}^{N}
\sum_{t=1}^{L_i}
\log P_{\theta}(y_{i,t}\mid x_i,y_{i,<t}).
$$

其中，$L_i$ 是 response 的长度。该目标没有显式区分 reasoning token 和 answer token。（PDF p. 2, Eq. 1）

### 4.2 使用 Tag 分离 reasoning 和 answer

作者将每个训练目标重写为：

```text
<Thinking>
reasoning tokens
</Thinking>
<Answer>
answer tokens
</Answer>
```

形式化表示为：

$$
\hat y_i=
[\texttt{<Thinking>},
y_i^{(think)},
\texttt{</Thinking>},
\texttt{<Answer>},
y_i^{(answer)},
\texttt{</Answer>}].
$$

Tag 有两个作用：

1. 给模型提供清晰的结构边界；
2. 让训练代码可以准确地 mask reasoning 区间，只保留答案区间的 loss。

论文单独验证了 Tag 的作用，结果表明 **仅添加 Tag 并不能稳定提高 accuracy**，不同模型和数据集上的效果有正有负。因此，Tag 主要是结构工具，而不是稳定的性能增强方法。（PDF p. 6, Table 2）

### 4.3 Stage 1：SFT-Tag

第一阶段在完整的 tagged response 上训练：

$$
\theta_{\mathrm{SFT}}
=\arg\min_{\theta}\mathcal{L}_{\mathrm{SFT}}(\theta).
$$

此时 `<Thinking>` 和 `<Answer>` 两个区间的 token 都参与 loss。该阶段的职责是：

- 学习从问题生成推理过程；
- 学习 reasoning 在前、answer 在后的响应顺序；
- 学会正确生成开始和结束 Tag；
- 为第二阶段提供稳定的初始化。

### 4.4 Stage 2：Key-Tag

第二阶段从 $\theta_{\mathrm{SFT}}$ 继续训练，但 mask 掉 `<Thinking>` 区间的 label。reasoning token 仍以 teacher-forcing 上下文的形式输入模型，只是不再作为需要预测和计分的监督位置。

设 $T_i$ 表示 `</Thinking>` 结束位置，则论文将答案专项 loss 写为：

$$
\mathcal{L}_{\mathrm{Answer}}(\theta)
=-
\sum_{i=1}^{N}
\sum_{t=T_i+1}^{L_i}
\log P_{\theta}(y_{i,t}\mid x_i,\hat y_{i,<t}).
$$

随后从第一阶段 checkpoint 初始化并继续优化：

$$
\theta_{\mathrm{SFTKey}}
=\arg\min_{\theta}\mathcal{L}_{\mathrm{Answer}}(\theta),
\qquad
\theta\leftarrow\theta_{\mathrm{SFT}}.
$$

需要注意两个容易误解的点：

- “只训练答案”指只有答案位置产生 loss，并不是只更新某一小部分模型参数；答案 loss 仍会更新整个模型。
- CoT 没有从输入中删除。Stage 2 预测答案时仍然能够条件化于问题和已有 reasoning，只是 reasoning token 本身不再承担预测损失。（PDF p. 3, Eqs. 4-5）

### 4.5 等价的训练伪代码

```text
# Stage 1: learn full response and output structure
model = base_model
for prompt, tagged_response in dataset:
    labels = all_response_tokens
    loss = causal_lm_loss(model, prompt, tagged_response, labels)
    update_all_parameters(loss)

# Stage 2: specialize final answer generation
model = load(stage_1_checkpoint)
for prompt, tagged_response in dataset:
    labels = tagged_response
    labels[prompt_and_thinking_positions] = IGNORE_INDEX
    loss = causal_lm_loss(model, prompt, tagged_response, labels)
    update_all_parameters(loss)
```

## 5. 四种训练策略分别检验什么

| 方法 | 是否使用 Tag | 参与 loss 的 token | 是否两阶段 | 要检验的问题 |
|---|---|---|---|---|
| SFT | 否 | 整个 response | 否 | 普通全序列训练的基线表现 |
| SFT-Tag | 是 | Thinking + Answer | 否 | 仅增加结构标签是否有效 |
| Key-Tag | 是 | 仅 Answer | 否 | 从头只强调答案会发生什么 |
| SFTKey-Tag | 是 | Stage 1 全序列；Stage 2 仅 Answer | 是 | 两阶段能否兼得格式与准确率 |

这四个方法构成论文的主要论证：

- `SFT` 与 `SFT-Tag` 的差异用于观察 Tag 的影响；
- `SFT-Tag` 与 `Key-Tag` 的差异用于观察全序列 loss 和答案 loss 的取舍；
- `Key-Tag` 与 `SFTKey-Tag` 的差异用于观察第一阶段结构学习是否必要；
- `SFT` 与 `SFTKey-Tag` 的差异用于衡量最终方法的总体收益。

## 6. 实验设计

### 6.1 模型

论文覆盖五个 1.5B 至 8B 的模型：

- Qwen3-8B-Base；
- Qwen2.5-7B；
- Qwen2.5-3B；
- Qwen2.5-1.5B；
- SmolLM3-3B-Base。

### 6.2 数据集

| 数据集 | 类型 | 主要能力 |
|---|---|---|
| GSM8K | 数学文字题 | 小学数学、多步计算 |
| OpenR1-Math-220k | 大规模数学题 | 代数、几何、概率等 |
| OpenBookQA | 多选科学问答 | 科学知识与组合推理 |
| CoT-Collection | 多领域 CoT 数据 | 多任务 reasoning 样本 |

所有数据都被人工或规则化地拆分成 reasoning 和 answer，再包装为统一的 Tag 格式。（PDF p. 4）

### 6.3 训练配置

论文报告的主要训练配置为：

- learning rate：$5\times10^{-6}$；
- linear warmup：0.5 epoch；
- weight decay：0.1；
- precision：bfloat16；
- per-device batch size：32；
- 通常训练约 3 epochs；
- 硬件：8 张 NVIDIA A100 40GB；
- 正文实验估计不超过 1,000 GPU hours，连同探索实验不超过 2,000 GPU hours。（PDF pp. 4, 12）

### 6.4 评价指标

#### Answer accuracy

模型生成 CoT 和答案后，系统提取 final answer，并使用 Meta-Llama-3-70B-Instruct 判断预测答案与参考答案在语义上是否一致。（PDF p. 4；评测 prompt 见 PDF p. 12）

因此这里的 accuracy 不是所有任务都使用严格 exact match，而是部分依赖 LLM judge。

#### Format adherence

检查输出是否严格满足：

```text
<Thinking>Reasoning Content</Thinking>
<Answer>Answer Content</Answer>
```

格式正确率记为 $\mathrm{Fmt}$。（PDF p. 12）

#### Composite score

主表还定义了综合分数：

$$
\mathrm{Score}
=\alpha\cdot\mathrm{Acc}
+(1-\alpha)\cdot\mathrm{Fmt},
\qquad \alpha=0.7.
$$

即答案准确率占 70%，格式正确率占 30%。因此阅读论文中“提高约 5%”的表述时，需要区分它指的是纯 accuracy、绝对百分点，还是该 composite score 的相对提升。（PDF p. 5）

## 7. 实验结果如何支撑方法

### 7.1 Tag 本身不是稳定增益来源

`SFT-Tag` 相比普通 `SFT` 的平均 accuracy 有升有降：

| 模型 | SFT | SFT-Tag | 变化 |
|---|---:|---:|---:|
| Qwen3-8B-Base | 0.7069 | 0.7297 | +0.0228 |
| Qwen2.5-7B | 0.6722 | 0.6696 | -0.0026 |
| SmolLM3-3B-Base | 0.5758 | 0.5565 | -0.0193 |
| Qwen2.5-3B | 0.5966 | 0.5719 | -0.0247 |
| Qwen2.5-1.5B | 0.4955 | 0.5435 | +0.0480 |

这说明标签只是让 reasoning 与 answer 的边界显式化，不能单独解释最终方法的收益。（PDF p. 6, Table 2；PDF p. 13, Table 6）

### 7.2 Key-Tag 说明答案专项 loss 确实有用

在五个模型中，`Key-Tag` 相比 `SFT-Tag` 通常具有更高的答案 accuracy。论文还展示了 GSM8K 上的 answer loss 曲线：训练后期 `Key-Tag` 的答案 loss 下降到普通 SFT 和 SFT-Tag 以下。（PDF pp. 6-7, Figures 2-3）

这一结果直接支持一个较弱但可靠的结论：

> 如果评价目标主要取决于 final answer，增加答案 token 的专项优化可以改善答案预测。

但它还不能单独证明“标准 SFT 被长 CoT 干扰”是唯一原因，因为答案专项阶段也改变了训练时长和优化路径。

### 7.3 Key-Tag 的代价是格式崩溃

只在答案位置训练时，模型没有充分监督去学习完整响应结构。部分格式结果非常极端：

| 模型 | SFT-Tag Avg-Fmt | Key-Tag Avg-Fmt | SFTKey-Tag Avg-Fmt |
|---|---:|---:|---:|
| Qwen3-8B-Base | 0.9717 | 0.7512 | **0.9959** |
| Qwen2.5-7B | 0.9601 | 0.0000 | **0.9632** |
| SmolLM3-3B-Base | **0.9564** | 0.0679 | 0.9084 |

这组结果解释了为什么作者不能直接推荐 `Key-Tag`：它可能答得更准，却不能稳定生成要求的结构。（PDF p. 8, Tables 3 and 5；PDF p. 13, Table 7）

### 7.4 两阶段方法在模型平均 accuracy 上取得提升

附录给出的纯 accuracy 如下：

| 模型 | SFT Avg-Acc | SFTKey-Tag Avg-Acc | 绝对变化 |
|---|---:|---:|---:|
| Qwen3-8B-Base | 0.7069 | 0.7791 | +0.0722 |
| Qwen2.5-7B | 0.6722 | 0.7369 | +0.0647 |
| SmolLM3-3B-Base | 0.5758 | 0.6115 | +0.0357 |
| Qwen2.5-3B | 0.5966 | 0.6115 | +0.0149 |
| Qwen2.5-1.5B | 0.4955 | 0.5543 | +0.0588 |
| **五模型平均** | **0.6094** | **0.6587** | **+0.0493** |

按表中数值重新计算，五个模型的平均 accuracy 约提高 **4.93 个百分点**。提升并非在每个模型、每个数据集上都成立；它是跨四个数据集取平均后的模型级总体趋势。（PDF p. 13, Table 6）

### 7.5 SFTKey-Tag 不总是 accuracy 最高，但平衡最好

`Key-Tag` 在部分小模型上的纯 accuracy 高于 `SFTKey-Tag`，例如 SmolLM3-3B 和 Qwen2.5-3B。最终方法的优势不应表述为“任何情况下准确率最高”，而应表述为：

> 相比只训练答案，SFTKey-Tag 大幅恢复格式；相比普通 SFT，它在模型平均结果上提高答案 accuracy，因此在 accuracy-format trade-off 上更均衡。

这也是论文为什么把 composite score 作为主表指标。（PDF pp. 5, 8-9）

## 8. 论文直接支持了什么

根据实验，可以较有把握地得到以下结论：

1. reasoning 和 answer 的结构标签本身不会稳定提高准确率。
2. 只对答案位置计算 loss，通常能降低 answer loss，并经常提高 answer accuracy。
3. 从头只训练答案会损害完整输出格式。
4. 先进行全序列训练，再进行答案专项训练，可以显著缓解格式损害。
5. 在论文测试的 1.5B 至 8B 模型和四个数据集上，两阶段方法的模型平均 accuracy 高于标准 SFT。

## 9. 论文没有充分证明什么

### 9.1 没有直接证明 CoT token“抢走”了梯度

论文的核心动机是长 CoT 让 final answer 权重不足，但没有直接报告：

- CoT 与 answer 的 gradient norm；
- 不同位置的 token loss 或梯度占比；
- answer 长度与性能收益的相关性；
- CoT 越长时 SFTKey 收益是否越大。

因此，“token 长度失衡导致答案训练不足”是一个合理假设，但论文的实验主要证明了答案专项训练有效，并未完整验证这条因果机制。

### 9.2 缺少训练计算量匹配的对照组

SFTKey-Tag 比普通 SFT 多一个训练阶段。论文没有加入如下关键对照：

```text
Stage 1: 全序列 SFT
Stage 2: 再做等量的全序列 SFT
```

也没有比较“单阶段但提高 answer token 权重、总训练 step 不变”的版本。因此，当前实验不能完全排除以下替代解释：

- 性能提升来自更多优化 step；
- 性能提升来自 two-stage curriculum；
- 性能提升来自第二阶段较低 answer loss，而不一定来自 CoT 长度失衡。

这是验证论文中心机制时最重要的缺失对照。

### 9.3 保住了 CoT 格式，不等于保住了 CoT 质量

论文的 format metric 只检查标签和输出结构，没有评估：

- reasoning 是否事实正确；
- reasoning 是否逻辑有效；
- reasoning 是否真正支持 final answer；
- final answer 是否因该 reasoning 而产生；
- 第二阶段是否让模型更倾向于“答案正确但推理错误”。

所以论文能说明模型继续生成完整 CoT 格式，却不能说明 CoT 的 correctness 或 faithfulness 没有下降。

### 9.4 评测依赖 LLM judge

作者使用 Meta-Llama-3-70B-Instruct 判断预测答案与标准答案是否语义等价，但没有报告：

- judge 与人工判断的一致性；
- judge 的误判率；
- exact match 与 LLM judge 的结果差异；
- 对不同训练方法是否存在系统性偏好。

这会给 accuracy 引入额外测量误差。

### 9.5 缺少方差和显著性信息

论文没有报告多随机种子结果、标准差、置信区间或显著性检验。部分模型的平均增益较小，例如 Qwen2.5-3B 约为 1.49 个百分点，因此不能从现有结果判断小幅提升是否对随机种子稳定。

### 9.6 “约 5%”需要谨慎解释

论文在不同位置讨论 accuracy improvement 和 composite score improvement。主表的 Score 还取决于人为设定的 $\alpha=0.7$。此外，附录 Table 6 中 Qwen2.5-3B 和 Qwen2.5-1.5B 括号内的增量与表中两列直接相减并不完全一致。

因此，更稳妥的表述是：

> 按附录展示的五个模型平均 accuracy 重算，SFTKey-Tag 相对 SFT 平均提高约 4.93 个百分点；主表 composite score 的平均相对提升约为 5%，但该指标依赖 accuracy-format 权重。

## 10. 作者承认的局限

论文的 Limitations 部分主要承认：（PDF p. 9）

- 没有测试 14B、32B 等更大模型；
- 只覆盖 1.5B、3B、7B 和 8B 模型；
- Benchmark 主要集中于通用任务和数学任务；
- 第二个答案专项阶段增加了训练时间和计算成本；
- 尚未验证更多领域和不同 reasoning 风格下的泛化。

## 11. 如何更严格地验证这个想法

若要验证论文的核心机制，建议至少增加以下实验：

### 11.1 计算量匹配

| 对照 | Stage 1 | Stage 2 | 总 step |
|---|---|---|---:|
| Standard SFT | 全 token | 无 | $K$ |
| Longer SFT | 全 token | 全 token | $2K$ |
| SFTKey | 全 token | answer-only | $2K$ |
| Weighted SFT | 全 token，answer 加权 | 无 | $K$ |

只有 SFTKey 在计算量匹配后仍显著占优，才能更有力地说明收益来自 answer specialization，而不是额外训练。

### 11.2 长度分桶

按 $R/A$ 比例将样本分桶，检查 CoT 越长时 SFTKey 是否收益越大。如果论文的长度失衡解释正确，增益应与 reasoning-answer 长度比存在一定关联。

### 11.3 梯度与 loss 分析

分别记录：

- reasoning token loss；
- answer token loss；
- 两个区域的 gradient norm；
- Stage 2 前后 answer probability margin；
- 不同 token weighting 对训练稳定性的影响。

### 11.4 CoT 质量评估

除格式外，还应评价：

- rationale correctness；
- answer-rationale consistency；
- rationale perturbation 后答案是否改变；
- 将错误 reasoning 注入上下文时，模型是否盲目跟随；
- 使用外部 verifier 检查数学步骤。

### 11.5 多随机种子

至少报告 3 至 5 个随机种子的 mean、standard deviation 和置信区间，尤其需要验证 1 至 2 个百分点的提升是否稳定。

## 12. 与 SCOTT 的区别

这篇论文与 SCOTT 都关心 rationale/CoT 和 final answer 的关系，但目标不同：

| 论文 | 核心问题 | 主要方法 | 主要评价 |
|---|---|---|---|
| SCOTT | 学生是否真正根据 rationale 作答 | 教师 contrastive decoding + 学生 counterfactual reasoning | rationale-answer faithfulness、LAS、干预敏感性 |
| SFTKey-Tag | final answer token 是否在 SFT 中训练不足 | 全序列 SFT 后进行 answer-only SFT | answer accuracy、format adherence |

因此：

- SCOTT 更关心“答案是否依赖理由”；
- SFTKey-Tag 更关心“答案 token 是否获得足够优化权重”；
- SFTKey-Tag 的 accuracy 提升不能自动推出其 CoT 更忠实；
- 两者理论上可以组合：先构造更一致的 rationale，再使用显式分段和答案专项 loss，但需要同时监控 faithfulness，避免只提高答案而损伤 reasoning。

## 13. 最终评价

### 方法价值

SFTKey-Tag 的优点是实现简单，不需要额外 reward model、token importance judge 或复杂的数据重写模型。对于回答格式固定、评价主要由 final answer 决定、并且训练样本包含长 CoT 的任务，它提供了一个直接可实现的训练策略。

### 证据强度

论文较好地证明了“两阶段训练在其设置中能够改善 accuracy-format trade-off”，但对中心机制的证明仍不充分。特别是缺少计算量匹配对照、CoT 质量评价和多随机种子，使得结论更适合表述为：

> 在完整 SFT 后增加答案专项训练，是一个有前景且低复杂度的经验方法；现有实验尚不能确定其收益究竟来自 token 重要性重平衡、额外训练计算，还是两阶段 curriculum。

### 实践结论

如果要复现或采用这篇论文的思路，最值得保留的设计是：

1. 明确标记 reasoning 和 answer 边界；
2. 第一阶段学习完整响应，不从头只训答案；
3. 第二阶段 mask reasoning loss，但保留 reasoning 作为答案上下文；
4. 分开报告 accuracy、format 和 reasoning quality，不只报告加权综合分；
5. 加入等计算量基线，确认收益不是由额外训练 step 造成。
