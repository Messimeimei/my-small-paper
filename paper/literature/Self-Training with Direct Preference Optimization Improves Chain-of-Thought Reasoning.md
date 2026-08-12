# Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning

PDF 原文：[Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning](<./Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning.pdf>)

## 核心结论

这篇论文提出 **DPO-augmented Self-Training（DPO-ST）**：不用 GPT-4 等更大模型生成全部训练轨迹，而是让小模型从自己的数学推理轨迹中反复学习，并在每一轮普通自训练之前加入 Direct Preference Optimization（DPO），提高下一轮自生成 Chain-of-Thought（CoT）的正确率和多样性。

它的核心不是“用 DPO 替代 SFT”，而是：

> 用 DPO 改进每轮的数据生成器，使其产生更多可通过答案验证的 CoT；再把这些正确 CoT 加入训练集，从原始基础模型重新执行 SFT。

因此，论文最终交付的模型仍然是一个 SFT 模型。DPO 模型主要负责生成更好的伪标签数据。

## 研究动机

提高小模型数学推理能力通常有两条路线：

1. 使用 GPT-4、PaLM、Codex 等更大模型生成 CoT，再把这些轨迹蒸馏给小模型；
2. 让小模型从自己的正确输出中学习，即 self-training。

第一种方法的主要问题是调用和计算成本较高，并且依赖闭源模型。第二种方法不需要大模型，但存在明显的启动问题：早期小模型只能生成少量正确 CoT，而且生成轨迹可能高度重复，导致下一轮可用训练数据不足。

论文的思路是：数学题具有容易自动验证的最终答案，因此可以把模型自己生成的正确和错误 CoT 组成偏好对，使用 DPO 改进生成器，再让这个生成器为下一轮 SFT 生产数据。

## 普通 CoT 自训练

设：

- 基础模型为 $f_\theta$；
- 带人工 CoT 的有标注数据为 $L=\{(x_i,y_i,a_i)\}$；
- 只有问题和标准答案的数据为 $U=\{(x_i,a_i)\}$；
- $y_i$ 表示推理过程，$a_i$ 表示最终答案。

普通 self-training 的流程是：

```text
基础模型
  -> 在人工 CoT 数据上做 SFT
  -> 为训练问题生成 CoT 和答案
  -> 保留最终答案正确的 CoT
  -> 将这些 CoT 加入人工训练集
  -> 从基础模型重新执行 SFT
  -> 重复
```

形式上，模型先生成伪标签集合：

$$
S=\{(x_i,\hat y_i,\hat a_i)\}
$$

然后只保留满足 $\hat a_i=a_i$ 的样本：

$$
S_\alpha=\{(x_i,\hat y_i,\hat a_i)\in S:\hat a_i=a_i\}
$$

最后用 $L\cup S_\alpha$ 重新训练模型。

## DPO-ST 的完整流程

### 1. Warm-up SFT

首先用 GSM8K 中人工标注的 CoT 微调基础模型：

$$
f_\theta \xrightarrow{\mathrm{SFT}(L)} f'_\theta
$$

得到具备初步数学推理能力的 SFT 模型 $f'_\theta$。

### 2. 构造 DPO 偏好数据

对于每个问题，从 $f'_\theta$ 采样多条 CoT：

- 最终答案正确的 CoT 作为 winning/chosen completion；
- 最终答案错误的 CoT 作为 losing/rejected completion。

得到偏好数据：

$$
D=\{(x_i,y_i^w,y_i^l)\}
$$

其中 $y_i^w$ 是答案正确的轨迹，$y_i^l$ 是答案错误的轨迹。

然后使用 DPO 目标优化 $f'_\theta$，得到 DPO 模型 $f_\theta^d$：

$$
L_{\mathrm{DPO}}
=
-\mathbb{E}_{(x,y^w,y^l)\sim D}
\left[
\log\sigma\left(r(y^w\mid x)-r(y^l\mid x)\right)
\right]
$$

其中：

$$
r(y\mid x)=
\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

论文设置 $\beta=0.1$。

### 3. 使用 DPO 模型生成伪标签

让 $f_\theta^d$ 为训练问题生成新的 CoT：

$$
S=\{(x,\hat y):x\sim U,\hat y\sim f_\theta^d(\cdot\mid x)\}
$$

然后进行两种清洗：

1. 删除最终答案错误的 CoT；
2. 使用 Jaccard similarity 删除高度重复的 CoT，阈值设为 0.7。

清洗后的自生成数据与原始人工 CoT 数据合并。

### 4. 从基础模型重新执行 SFT

这里是论文非常关键的实现细节：最终 SFT 不是继续训练 DPO 模型，也不是继续训练上一轮的 SFT 模型，而是从原始基础模型 $f_\theta$ 重新初始化：

$$
f_\theta
\xrightarrow{\mathrm{SFT}(L\cup S_\alpha)}
f'_\theta
$$

作者认为，如果一直在上一轮模型上继续训练，模型容易对早期生成的伪标签过拟合。重新从基础权重训练遵循了 STaR 和 ReST-EM 等工作的做法。

### 5. 迭代

新的 $f'_\theta$ 再次生成 DPO 偏好数据，经过 DPO 得到新的生成器，然后生成下一轮 SFT 数据。论文实验执行三轮。

完整流程可以概括为：

```text
人工 CoT -> Warm-up SFT 模型
                   |
                   v
        采样多条 CoT，按答案正确性组成偏好对
                   |
                   v
                DPO 模型
                   |
                   v
         生成并筛选正确、非重复 CoT
                   |
                   v
     从原始基础模型重新 SFT -> 下一轮
```

## DPO 实际起了什么作用

论文比较了第一轮中 DPO 前后的 Flan-T5-Large。

### Pass@1

- DPO 前：36.1%；
- DPO 后：36.5%。

DPO 对贪心生成最优单条答案的提升非常小。

### Pass@10

- DPO 前：62.9%；
- DPO 后：64.8%。

当每道题采样十条轨迹时，DPO 的优势更明显。

### 可用伪标签数量

- DPO 前：2495 条；
- DPO 后：2940 条。

因此，DPO 的主要贡献不是直接大幅提高单次生成准确率，而是提高采样时找到正确解法的概率，并增加去重后可用于下一轮 SFT 的轨迹数量。

更准确地说：

> DPO 在该框架中主要是数据生成器优化方法，而不是最终模型的训练目标。

## 外部计算器

小模型经常理解了解题方法，却在简单算术上出错。论文使用 GSM8K 的计算标记，例如：

```text
3 * 2 = <<3*2=6>>6
```

当解码出现这种模式时，程序调用外部计算器，并用计算结果覆盖模型生成的数字。作者还通过自定义 `LogitsProcessor` 支持批量解码，避免工具调用只能使用 batch size 1。

计算器对结果影响非常大。Flan-T5-Large 在 GSM8K 开发集第三轮的准确率为：

- 使用计算器：44.8%；
- 不使用计算器：18.0%。

这说明论文的提升不能全部归因于 DPO。计算器不仅改善测试阶段的算术结果，也能减少训练数据生成阶段的错误伪标签。

## 实验设置

### 模型

- Flan-T5-Base；
- Flan-T5-Large；
- Llama-1-7B；
- Llama-2-7B；
- Llama-3-8B。

### 数据

训练只使用 GSM8K：

- 6705 条训练样本；
- 768 条验证样本；
- 1319 条测试样本。

另外在 MultiArith、ASDiv 和 SVAMP 上测试 OOD 泛化能力。

论文所说的“无标注数据”并不是完全没有标签：作者删除了 GSM8K 的人工 CoT，但仍然保留每道题的标准答案，因为答案是筛选 CoT 和构造 DPO 偏好对所必需的。

### 采样与训练

- DPO 数据：每题从 SFT 模型采样 5 条 CoT；
- SFT 伪标签：默认每题从 DPO 模型采样 3 条，也实验了更大的 $K$；
- temperature：0.7；
- DPO $\beta$：0.1；
- 最大生成长度：300 tokens；
- 总共迭代三轮。

## 主要结果

### Flan-T5

| 方法 | 基础模型 | GSM8K | MultiArith | ASDiv | SVAMP |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT | Flan-T5-Base | 18.1 | 54.2 | 26.2 | 19.5 |
| Self-Training | Flan-T5-Base | 25.9 | 73.8 | 28.2 | **24.2** |
| DPO-ST | Flan-T5-Base | **27.2** | **74.3** | **29.2** | 22.6 |
| SFT | Flan-T5-Large | 30.8 | 77.2 | 38.1 | 33.6 |
| Self-Training | Flan-T5-Large | 35.6 | 86.2 | 42.5 | 34.8 |
| DPO-ST | Flan-T5-Large | **37.4** | **89.0** | **42.8** | **36.8** |

结果表明：

- 普通 self-training 已经比一次 SFT 强很多；
- DPO-ST 通常进一步提高结果；
- DPO 并非在所有数据集上都稳定有效，例如 Flan-T5-Base 在 SVAMP 上低于普通 self-training；
- DPO 相对普通 self-training 的增益明显小于 self-training 相对一次 SFT 的增益。

### 增加每题采样数

Flan-T5-Large 在 GSM8K 上：

| 每题保留的采样规模 | Accuracy |
| --- | ---: |
| $K=3$ | 37.4% |
| $K=5$ | 39.1% |
| $K=10$ | 40.0% |

这进一步说明，多样化采样和获得更多可验证轨迹是方法有效的重要因素。

### Llama 系列

论文报告的 GSM8K 结果为：

- Llama-1-7B DPO-ST：44.7%；
- Llama-2-7B DPO-ST：54.7%；
- Llama-3-8B DPO-ST：68.8%。

这些模型使用了外部计算工具，因此与不使用工具或使用更大规模蒸馏数据的方法比较时需要谨慎。

## 与 STaR 的关系

DPO-ST 可以理解为在 STaR 式自训练循环中加入一个 DPO 数据生成阶段。

```text
STaR：
生成 CoT -> 用答案正确性筛选 -> 从基础模型重新训练

DPO-ST：
生成正确/错误 CoT 偏好对 -> DPO 改进生成器
-> 重新生成 CoT -> 用答案正确性筛选 -> 从基础模型重新训练
```

二者的共同点包括：

- 都使用模型自己的推理轨迹；
- 都依赖标准答案筛选轨迹；
- 都反复扩大正确 CoT 训练集；
- 每一轮都从原始基础模型重新训练，而不是无限继续微调上一轮模型。

区别是：STaR 对错误样本可使用带正确答案提示的 rationalization；DPO-ST 则将正确和错误轨迹构造成偏好对，用 DPO 提高下一轮生成器的探索能力。

## 与 TRACT 的关系

| 方法 | 自生成 CoT | CoT 处理方式 | 最终训练目标 | 是否从基础模型重训 |
| --- | --- | --- | --- | --- |
| STaR | 是 | 按最终答案筛选，可 rationalize | CE | 是 |
| DPO-ST | 是 | DPO 偏好优化后再按答案筛选 | CE | 是 |
| TRACT | 是 | 丢弃生成分数，CoT 基本不筛选 | CE + 回归损失 | 是 |

三者共享同一个元框架：

```text
先得到轨迹生成器
-> 生成模型自身分布下的推理轨迹
-> 使用任务监督重新标注或筛选
-> 从基础权重训练最终模型
```

DPO-ST 更关注如何提高轨迹生成器的探索质量；TRACT 更关注教师 CoT 与学生 CoT 的分布差距，以及 1-5 分任务的序数回归目标。

## 局限性

### 1. 只验证数学推理

论文只在具有明确、可自动验证答案的数学任务上实验。它不能证明 DPO-ST 对文本分类、科学评价或其他软语义任务有效。

### 2. 最终答案正确不代表 CoT 正确

模型可能在中间推理中犯错，但碰巧输出正确答案。作者也观察到这类 false positive，并主要通过外部计算器减少算术错误，但没有系统验证所有推理步骤。

### 3. “无标注数据”仍需要标准答案

训练问题可以没有人工 CoT，但必须具有标准答案。否则无法构造 DPO 偏好对，也无法筛选伪标签。

### 4. DPO 的独立增益有限

Pass@1 只从 36.1% 提高到 36.5%。DPO 的作用更多体现在 Pass@10 和伪标签数量，而不是单次解码能力。

### 5. 对外部计算器依赖很强

无计算器时准确率从 44.8% 降到 18.0%，说明工具使用是整体结果的重要来源。

### 6. 实验统计不充分

论文没有系统报告多随机种子的均值、方差和显著性检验，因此部分较小差异可能受到训练或采样波动影响。

## 对科学写作评价研究的启发

这篇论文不能直接迁移到科学写作评价，因为你的任务没有数学题那样可靠的答案验证器。

如果简单规定：

- 标签正确的 rationale 是 chosen；
- 标签错误的 rationale 是 rejected；

就会产生明显问题：

- 二分类任务随机命中正确标签的概率可能达到 50%；
- 1-5 分任务中，预测 3 和预测 5 相对真实分数 2 的错误程度不同，二元 chosen/rejected 会丢失序数距离；
- 标签正确的解释也可能引用了错误证据；
- DPO 可能只学习让最终标签看起来正确，而不是提高科学评价理由的忠实性。

如果借鉴 DPO-ST，更合理的偏好构造可以综合：

$$
R(r)=
-\alpha\left|\hat y-y^*\right|
+\beta R_{\mathrm{grounding}}(r)
+\gamma R_{\mathrm{rubric}}(r)
+\delta R_{\mathrm{utility}}(r)
$$

其中：

- $|\hat y-y^*|$ 衡量序数评分距离；
- $R_{\mathrm{grounding}}$ 衡量是否引用输入中的真实证据；
- $R_{\mathrm{rubric}}$ 衡量是否遵守对应评价标准；
- $R_{\mathrm{utility}}$ 衡量 rationale 是否真正提高正确评分概率。

然后用高奖励与低奖励的自生成 rationale 构造偏好对，再让 DPO 模型负责生成下一轮训练数据。

不过，DPO-ST 的实验也提醒我们：在科学写作评价中，必须分别验证增益来自：

1. 普通 self-training；
2. 更多采样轨迹；
3. DPO 偏好优化；
4. rationale 筛选；
5. 最终标签或序数回归目标。

否则不能把整体提升直接归因于 DPO。

## 总结

DPO-ST 是一个 STaR 风格的迭代自训练方法。它先在模型自己的正确和错误数学 CoT 上执行 DPO，提高生成器的采样质量和多样性；再用这个生成器产生通过答案验证的 CoT；最后从原始基础模型重新执行 SFT。

论文最值得借鉴的思想是：

> 偏好优化不一定直接用于最终模型，也可以用于改善自训练数据生成器，让后续 SFT 获得更多高质量的模型自身轨迹。

但它的有效性依赖数学任务的自动验证、标准答案和外部计算器。迁移到科学写作评价时，必须设计比“最终标签正确”更可靠的 rationale 偏好标准。
