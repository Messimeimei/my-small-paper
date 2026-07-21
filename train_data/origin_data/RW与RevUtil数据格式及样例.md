# RW 与 RevUtil 数据格式及样例

## 1. 数据来源与总体结构

本文档分析的数据来自：

`eval_data/Reward Modeling for Scientific Writing Evaluation/data/final_reward_data.json`

该文件包含原论文用于训练和域内测试的两个任务：

- `rw_gen`：Related Work Evaluation，评价生成的相关工作文本；
- `rev_util`：Review Utility Evaluation，评价审稿意见对作者是否有用。

两个任务合计包含 58,712 条训练样本和 6,645 条测试样本：

| 任务 | Train | Test | 总计 |
|---|---:|---:|---:|
| RW (`rw_gen`) | 8,666 | 1,857 | 10,523 |
| RevUtil (`rev_util`) | 50,046 | 4,788 | 54,834 |
| 合计 | 58,712 | 6,645 | 65,357 |

### 1.1 单条样本的原始 JSON 格式

```json
{
  "task": "rw_gen 或 rev_util",
  "aspect": "具体评价维度",
  "labels": 0,
  "score_sets": [0, 1],
  "prompt": [
    {
      "role": "system",
      "content": "统一的科学写作评价角色、推理和输出格式要求"
    },
    {
      "role": "user",
      "content": "[QUERY]... [CRITERIA]... [EXAMPLES]... [ANSWER]..."
    }
  ]
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `task` | 上层任务名称，取值为 `rw_gen` 或 `rev_util` |
| `aspect` | 当前样本评价的具体维度 |
| `labels` | 人工或原数据给出的金标签 |
| `score_sets` | 当前维度允许输出的全部分数 |
| `prompt[0]` | system prompt，要求先输出 `<reasoning>`，再输出 `<score>` |
| `prompt[1]` | user prompt，包含任务说明、rubric、few-shot 示例和真正待评价文本 |

`prompt[1].content` 的通用结构为：

```text
[QUERY]: 当前评价任务的目标

[CRITERIA]: 评价维度定义和各标签的评分标准

[EXAMPLES]:
若干已包含 <reasoning> 和 <score> 的演示样例

[ANSWER]: 真正需要模型评价的文本
```

注意：`[EXAMPLES]` 中出现的 `ANSWER / EVALUATION` 是 few-shot 演示，不是当前样本。最后一个 `[ANSWER]` 才是当前样本的待评对象。金标签保存在外层 `labels` 字段中，不直接出现在 user prompt 末尾。

### 1.2 各维度的 few-shot 数量和一致性

这里把每个 `<START OF EXAMPLE N> ... <END OF EXAMPLE N>` 计为一条 few-shot。对完整 train/test 数据逐条统计，并对整个 few-shot 区块进行精确哈希比较，结果如下：

| Task | Aspect | 每条样本的 few-shot 数 | Train 中模板数 | Test 中模板数 | Train/Test 是否共用模板 |
|---|---|---:|---:|---:|---|
| RW | `coherence` | 2 | 1 | 1 | 是，完全相同 |
| RW | `positioning_check` | 2 | 2 | 2 | 是，两套都同时出现 |
| RW | `positioning_type` | 3 | 2 | 2 | 是，两套都同时出现 |
| RevUtil | `actionability` | 5 | 1 | 1 | 是，完全相同 |
| RevUtil | `grounding_specificity` | 5 | 1 | 1 | 是，完全相同 |
| RevUtil | `helpfulness` | 5 | 1 | 1 | 是，完全相同 |
| RevUtil | `verifiability_extraction` | 2 | 1 | 1 | 是，完全相同 |
| RevUtil | `verifiability` | 5 | 1 | 1 | 是，完全相同 |

因此，“每个样本的 few-shot 是否一模一样”需要分三层回答：

1. **不同 aspect 之间不一样。** 每个评价维度有自己的 QUERY、rubric 和演示样例，不能跨维度共用。
2. **同一 aspect 内，大多数完全一样。** RW `coherence` 和 RevUtil 五个维度中，同一维度的所有训练/测试样本都重复使用同一套固定 few-shot；变化的只有最后真正待评价的输入。
3. **两个 positioning 维度是例外。** `positioning_check` 和 `positioning_type` 各存在两套模板，而不是全体样本共用一套。

#### 1.2.1 `coherence` 的 few-shot

- 每条样本固定包含 2 条演示；
- Example 1 是引用内容与论文上下文一致的正例，标签为 1；
- Example 2 是候选引用谈 NLI、但论文上下文谈对话生成的负例，标签为 0；
- 4,890 条训练样本和 1,048 条测试样本使用完全相同的两条演示。

也就是说，该维度中模型看到的前置任务说明和演示始终不变，只有 `[CONTEXT]`、`CITATION NUMBER` 和最后 `[ANSWER]` 变化。

#### 1.2.2 `positioning_check` 的两套 few-shot

两套模板都包含 2 条演示，标签语义也都是：

- 正例：段落明确说明当前论文相对于已有工作的贡献，标签 1；
- 负例：段落只罗列已有研究，没有定位当前论文，标签 0。

但两套模板的输入形式不同：

| 模板 | Train 数量 | Test 数量 | 演示格式 |
|---|---:|---:|---|
| Template A | 1,801 | 377 | 每条演示只有 `ANSWER` |
| Template B | 1,021 | 228 | 每条演示包含 `CONTEXT + ANSWER` |

Template A 的演示直接判断一个完整相关工作段落；Template B 先给一段已有工作背景 `CONTEXT`，再判断候选 `ANSWER` 是否针对该背景定位本文贡献。两者评价目标相近，但 prompt 输入结构并不完全相同。

后续蒸馏时建议先决定是否保留两种任务形式：

- 如果希望训练统一的单段定位专家，应把输入格式规范化；
- 如果保留两套模板，应在数据中增加明确的 `prompt_variant` 标记，并分别做性能统计，避免模型只学习模板差异。

#### 1.2.3 `positioning_type` 的两套 few-shot及标签错误

每条样本都包含 3 条演示，但存在两套模板：

| 模板 | Train 数量 | Test 数量 | 状态 |
|---|---:|---:|---|
| Template A | 483 | 95 | 前两个演示的标签与rubric相反 |
| Template B | 471 | 109 | 前两个演示标签与rubric一致 |

两套模板中的演示文本基本相同，关键差异是前两个演示的标签：

```text
Example 1：只有最后一段说明本文贡献，前面各段没有定位本文。
按照“每一段都必须定位”的rubric，正确标签应为 0。

Example 2：每个段落都在介绍已有工作后说明本文的区别或贡献。
按照rubric，正确标签应为 1。
```

Template B 使用正确的 `Example 1 = 0, Example 2 = 1`；Template A 却写成 `Example 1 = 1, Example 2 = 0`。更明显的是，Template A 的 reasoning 文字已经指出“只在最后一段说明贡献”或“每一段都说明了贡献”，但最终 `<score>` 与 reasoning 自身也不一致。

这是实际数据质量问题，不只是模板风格差异。若直接让教师基于原始 prompt 生成 CoT：

- 相同任务会收到互相冲突的演示监督；
- 教师可能复制错误标签；
- 后续 LoRA 专家可能同时学习正确和相反的判定规则。

因此，在生成教师 CoT 之前，应统一采用 Template B 的正确 few-shot，或者修正 Template A 的前两个 `<reasoning>/<score>`。训练集和测试集应使用同一份修正后的模板，但测试样本的金标签不能参与模板修正过程。

#### 1.2.4 RevUtil 的 few-shot

RevUtil 的模板最规则：

- `actionability`、`grounding_specificity`、`helpfulness`、`verifiability` 都是 5-shot；
- 五条演示分别对应标签 1、2、3、4、5，每个等级一条；
- `verifiability_extraction` 是 2-shot，分别对应标签 0 和 1；
- 同一 aspect 的全部 train/test 样本使用完全相同的演示文本、reasoning 和 score。

这意味着 RevUtil 的 few-shot 本身已经覆盖完整标签空间，但仍需人工检查演示推理是否准确。固定使用同一套演示也可能使模型过度依赖具体示例措辞，因此后续可以增加“保留原始固定 few-shot”和“移除 few-shot、仅保留 rubric”两种蒸馏设置作为消融实验。

## 2. Related Work Evaluation (`rw_gen`)

RW 用于评价生成的相关工作文本，共有三个维度：

| Aspect | 中文解释 | Train | Test | 标签空间 |
|---|---|---:|---:|---|
| `coherence` | 引用句与被引论文上下文是否一致 | 4,890 | 1,048 | 0/1 |
| `positioning_type` | 多段相关工作是否逐段定位本文贡献 | 954 | 204 | 0/1 |
| `positioning_check` | 单段相关工作是否定位本文贡献 | 2,822 | 605 | 0/1 |

### 2.1 标签分布

| Aspect | Train 标签分布 | Test 标签分布 |
|---|---|---|
| `coherence` | 0: 2,445；1: 2,445 | 0: 524；1: 524 |
| `positioning_type` | 0: 636；1: 318 | 0: 136；1: 68 |
| `positioning_check` | 0: 1,598；1: 1,224 | 0: 343；1: 262 |

`coherence` 完全平衡，另外两个定位维度存在一定类别不平衡，尤其 `positioning_type` 的 0/1 比例为 2:1。

### 2.2 `coherence`：引用上下文一致性

#### 任务定义

该名称容易被误解。它并不是判断整段相关工作是否语言连贯，而是判断：

> 给定被引论文的上下文和引用编号，生成的引用句是否正确表达了该论文的内容。

输入除了 `[ANSWER]` 外，还包含：

- `[CONTEXT]`：被引论文的标题、摘要或正文片段；
- `CITATION NUMBER`：引用编号；
- `[ANSWER]`：包含该编号的候选引用句。

标签定义：

- `0`：引用句中与该编号对应的说法不受给定上下文支持；
- `1`：引用句与该编号对应的上下文兼容。

#### 训练样例

```text
[CONTEXT]:
Tuning hyperparameters of learning algorithms is hard because gradients are
usually unavailable. We compute exact gradients of cross-validation performance
by chaining derivatives backwards through the entire training procedure...

CITATION NUMBER: 6

[ANSWER]:
Approximating the best-response Jacobian using the IFT as in [6] twice is
feasible, but requires changing the FT objective to include a proximal term [7],
and tuning two sets of interacting approximations.

Gold label: 0
```

#### 标签解释

给定上下文主要讨论通过反向传播训练过程计算超参数梯度，没有支持候选引用句中“使用 IFT 两次”“修改 FT 目标并加入 proximal term”等具体说法。因此引用编号 `[6]` 所处的论述与给定论文上下文不一致，标签为 `0`。

#### 可蒸馏的 CoT 结构

```text
1. 提取候选引用句中由指定引用编号支持的核心主张；
2. 概括被引论文上下文的核心方法和结论；
3. 比较主张与上下文是否蕴含、兼容或矛盾；
4. 根据 0/1 rubric 输出标签。
```

该维度更适合训练“引用—上下文核验”或“科学文本蕴含”专家，而不是一般语言连贯性专家。

### 2.3 `positioning_type`：多段相关工作的逐段定位

#### 任务定义

输入是一篇由多个段落组成的完整 Related Work。评价要求是：

> 每一个段落都应说明当前论文相对于该段已有工作的贡献或位置。

标签定义：

- `0`：至少一个段落只罗列已有工作，没有定位当前论文；
- `1`：每个段落都显式或隐式说明当前论文的贡献或差异。

#### 训练样例（节选）

```text
[ANSWER]:
The extensive body of work on defending classifiers against adversarial
perturbations demonstrates that most heuristic defenses have been defeated...
In contrast, our paper does not aim to propose a new defense; instead, it
provides a diagnostic tool that explains why existing defenses fall short...

Certified defenses based on linear programming, semidefinite programming,
interval bound propagation, and randomized smoothing have shown that provable
robustness can be attained for small networks...
Our contribution differs from these works in that we do not seek to certify
robustness for a particular model; rather, we introduce a
label-uncertainty-aware concentration framework...

Theoretical analyses that connect concentration of measure with intrinsic
robustness limits have established that adversarially robust classifiers cannot
exist... Our work builds directly on this line of research by identifying the
insufficiency of label-agnostic concentration...

Gold label: 1
```

原始样本还包含后续多个段落，此处只保留部分代表性内容。

#### 标签解释

每个段落都同时完成两件事：先概述某类已有研究，再使用 `In contrast`、`Our contribution differs`、`Our work builds directly` 等表达说明当前论文的区别或贡献。因此标签为 `1`。

#### 可蒸馏的 CoT 结构

```text
1. 将 Related Work 切分为段落；
2. 每段识别已有研究主题；
3. 每段定位当前论文的自我指称、贡献或比较语句；
4. 检查定位语句是否与该段主题一致；
5. 汇总全部段落，只要有一段不满足就输出 0。
```

这是一个“逐段判断后再聚合”的任务，推理结构比 `positioning_check` 多一层全局汇总。

### 2.4 `positioning_check`：单段贡献定位

#### 任务定义

输入是单个 Related Work 段落。评价要求是判断该段是否显式或隐式说明：

- 当前论文做了什么；
- 当前论文相对于已有工作有什么区别；
- 当前论文如何填补前述研究空缺。

标签定义：

- `0`：只总结已有工作，没有当前论文定位；
- `1`：出现与本段主题一致的当前论文贡献或定位。

#### 训练样例

```text
[ANSWER]:
A number of recent studies have investigated conformal prediction (CP) under
various forms of distribution shift. Works such as those by Vovk et al. and
subsequent extensions have formalized weighted exchangeability, allowing CP to
retain finite-sample coverage guarantees when the test distribution differs
from the training distribution. In the context of causal inference, CP has been
applied to counterfactual settings...

Gold label: 0
```

#### 标签解释

该段完整介绍了 conformal prediction 的已有研究，但没有出现 `our work`、`we propose`、`unlike previous work` 等显式定位，也没有隐式说明当前论文如何建立在这些研究之上，所以标签为 `0`。

#### 可蒸馏的 CoT 结构

```text
1. 概括本段已有研究主题；
2. 查找当前论文的贡献、差异或研究空缺陈述；
3. 检查该定位是否与本段主题相关；
4. 输出 0/1。
```

### 2.5 RW 三个维度之间的关系

| 维度 | 核心对象 | 主要推理操作 |
|---|---|---|
| `coherence` | 被引论文上下文 + 引用句 | 科学文本蕴含与引用核验 |
| `positioning_check` | 单段 Related Work | 单段贡献定位 |
| `positioning_type` | 多段 Related Work | 逐段定位 + 全局聚合 |

因此，不建议把三者笼统称为“连贯性推理”。其中 `positioning_type` 和 `positioning_check` 可以共享一个“论文定位”专家；`coherence` 更接近独立的引用核验专家。

## 3. Review Utility Evaluation (`rev_util`)

RevUtil 的输入都是单条审稿意见，但依据五种不同 rubric 评价：

| Aspect | 中文解释 | Train | Test | 标签空间 |
|---|---|---:|---:|---|
| `actionability` | 建议是否明确、具体、可执行 | 10,432 | 1,000 | 1-5 |
| `grounding_specificity` | 是否定位论文具体部分并指出具体问题 | 10,431 | 1,000 | 1-5 |
| `helpfulness` | 意见整体上能否帮助作者改进论文 | 10,430 | 1,000 | 1-5 |
| `verifiability_extraction` | 评论中是否包含需要论证的主张或意见 | 10,430 | 1,000 | 0/1 |
| `verifiability` | 评论中的主张是否获得充分论证 | 8,323 | 788 | 1-5 |

### 3.1 标签分布

| Aspect | Train 标签分布 | Test 标签分布 |
|---|---|---|
| `actionability` | 1: 2,264；2: 1,649；3: 2,856；4: 1,060；5: 2,603 | 1: 234；2: 164；3: 312；4: 95；5: 195 |
| `grounding_specificity` | 1: 1,081；2: 761；3: 3,821；4: 491；5: 4,277 | 1: 123；2: 64；3: 389；4: 41；5: 383 |
| `helpfulness` | 1: 407；2: 1,173；3: 4,365；4: 3,121；5: 1,364 | 1: 34；2: 107；3: 462；4: 311；5: 86 |
| `verifiability_extraction` | 0: 2,107；1: 8,323 | 0: 212；1: 788 |
| `verifiability` | 1: 2,190；2: 917；3: 3,391；4: 1,122；5: 703 | 1: 221；2: 77；3: 346；4: 120；5: 24 |

RevUtil 各维度不是均衡数据：

- `verifiability_extraction` 中约 80% 为标签 1；
- `grounding_specificity` 主要集中在 3 和 5；
- `helpfulness` 主要集中在 3 和 4；
- `verifiability` 的高分 5 很少。

生成 CoT 或训练专家时需要按标签分层采样，不能直接随机抽取相同数量，否则教师数据会进一步强化多数标签偏置。

### 3.2 `actionability`：可操作性

#### 任务定义

该维度同时判断两个因素：

1. 建议动作是显式提出，还是只能由作者隐含推断；
2. 建议是否具体到作者知道如何实施。

五级 rubric：

| 分数 | 含义 |
|---:|---|
| 1 | 没有有意义的改进信息 |
| 2 | 隐含动作，而且实施方式模糊 |
| 3 | 显式动作，但实施方式模糊 |
| 4 | 动作是隐含的，但实施方式具体 |
| 5 | 显式动作且实施方式具体 |

#### 训练样例

```text
[ANSWER]:
Incorporating QAT from scratch is not a good idea. QAT is generally performed
on a pretrained model and literature shows that it can improve results
significantly. You can significantly improve results if you perform quantization
after some iterations.

Gold label: 5
```

#### 标签解释

评论明确要求不要从头执行 QAT，并给出“先使用预训练模型、训练若干轮后再量化”的实施方式。动作和执行细节都明确，因此为 `5`。

#### 可蒸馏的 CoT 结构

```text
1. 提取评论要求作者采取的动作；
2. 判断动作是显式还是隐式；
3. 提取实施位置、方法和条件；
4. 判断作者是否知道具体怎么做；
5. 映射到 1-5 分。
```

### 3.3 `grounding_specificity`：定位与具体性

#### 任务定义

该维度包含两个轴：

- Grounding：评论是否能定位到论文的章节、表格、实验、方法或其他唯一部分；
- Specificity：评论是否明确指出该部分具体错了什么或缺少什么。

五级 rubric：

| 分数 | 定位 | 问题描述 |
|---:|---|---|
| 1 | 完全无法定位 | 高度不具体 |
| 2 | 只能弱推测 | 不具体 |
| 3 | 只能弱推测 | 具体 |
| 4 | 可以准确定位 | 不具体 |
| 5 | 可以准确定位 | 具体 |

#### 训练样例

```text
[ANSWER]:
The experimental results are slightly different from the paper
"Efficient Argument Structure Extraction with Transfer Learning and Active
Learning", and the author needs to give corresponding explanations.

Gold label: 3
```

#### 标签解释

评论明确指出问题是“实验结果与指定外部论文不同，并需要解释”，问题类型较具体；但没有指出当前论文中的表格、章节或具体实验位置，作者只能推测它所指的部分。因此属于“弱定位但具体”，标签为 `3`。

#### 可蒸馏的 CoT 结构

```text
1. 提取评论指向的论文位置；
2. 判断作者能否唯一定位该位置；
3. 提取评论指出的具体问题；
4. 分别判断 grounding 和 specificity；
5. 根据二维组合映射到 1-5 分。
```

### 3.4 `helpfulness`：整体有用性

#### 任务定义

Helpfulness 是一个综合维度。rubric 明确要求综合考虑：

- 是否指出有意义的弱点；
- 是否有具体改进建议；
- 是否可操作；
- 是否定位到论文内容；
- 是否对主张提供理由或证据；
- 处理该意见能否实质改善论文。

五级分数从“没有有用反馈”逐渐提高到“全面、具体、可操作且能显著帮助作者改进”。

#### 训练样例

```text
[ANSWER]:
The important results from Section 4 are hard to understand if one only reads
the main paper. In particular, the claim that constrained attention solves
parity is only supported in the appendix. Also, the logic behind Section 5 is
hard to comprehend without reading parts of Section A.4. The paper would be
made much stronger by moving a significant part of Appendices A.4 and A.5 into
Section 4. This can probably be done at the expense of Figure 1 and parts of
Section 5.

Gold label: 5
```

#### 标签解释

评论指出了明确问题及位置，解释了为什么主文难以理解，并提出把附录内容移入正文、压缩 Figure 1 和 Section 5 的具体修改方案。它同时具有 grounding、actionability 和改善价值，因此标签为 `5`。

#### 可蒸馏的 CoT 结构

```text
1. 识别评论指出的核心弱点；
2. 判断评论是否定位、具体、可操作且有依据；
3. 判断解决该问题对论文的预期改善；
4. 综合而非简单平均其他维度；
5. 映射到 1-5 分。
```

### 3.5 `verifiability_extraction`：待核验主张识别

#### 任务定义

该名称也容易被误解。它不要求模型从论文中抽取证据，而是判断：

> 当前评论中是否包含需要证据或理由支持的主张、意见、建议或推断。

标签定义：

- `0`：仅有客观描述、一般问题、澄清请求或可直接推导的信息，不包含需要额外论证的观点；
- `1`：包含主观判断、建议、论文质量判断、推断或其他需要论证的主张。

#### 训练样例

```text
[ANSWER]:
How dependent on the specific parametrization of the MLP (or NN more general)
is the performance?

Gold label: 0
```

#### 标签解释

该评论只是提出一个一般澄清问题，没有直接断言参数化会导致某种结果，也没有给出需要外部证据支撑的评价结论，所以标签为 `0`。

#### 可蒸馏的 CoT 结构

```text
1. 将评论拆分为陈述、问题、意见和建议；
2. 判断是否存在主观判断或需论证的推断；
3. 区分普通澄清问题与隐含评价主张；
4. 输出 0/1。
```

它更适合称为“claim detection”或“待核验主张识别”，不是 evidence extraction。

### 3.6 `verifiability`：论证充分性

#### 任务定义

该维度只评价已经包含待核验主张的评论，判断评论是否通过以下方式充分支持主张：

- 清楚的逻辑推理；
- 领域常识或公认实践；
- 具体案例；
- 外部文献、数据或引用。

五级 rubric：

| 分数 | 含义 |
|---:|---|
| 1 | 有主张但完全没有支持 |
| 2 | 有少量支持，但非常模糊或不完整 |
| 3 | 有一定支持，但缺少关键案例、细节或引用 |
| 4 | 支持基本充分，仅有小缺口 |
| 5 | 主张获得明确、充分且稳健的支持 |

#### 训练样例

```text
[ANSWER]:
Data collection and annotation are not clear. The Enforceable Annotation might
have ethical issues. What will be the reward for the 10 law experts? Why did
they volunteer? Does it count toward their study credit, or will they co-author
the paper? This is a serious issue, which might lead to the low quality of the
data.

Gold label: 3
```

#### 标签解释

评论提出“标注安排可能存在伦理问题并影响数据质量”的主张，也列举了报酬、志愿原因、学分和署名等具体问题作为支持；但它没有提供论文中的具体事实、伦理规范或外部依据来证明这些担忧，因此有一定论证但仍缺关键证据，标签为 `3`。

#### 可蒸馏的 CoT 结构

```text
1. 提取评论中的待核验主张；
2. 提取评论提供的逻辑、案例、常识或外部引用；
3. 检查支持内容与主张是否相关；
4. 判断缺少哪些关键论证；
5. 映射到 1-5 分。
```

### 3.7 `verifiability_extraction` 与 `verifiability` 的级联关系

两个维度不是完全独立的并行任务：

```text
先执行 verifiability_extraction
    ├── 0：没有待核验主张，不再评价论证充分性
    └── 1：存在待核验主张，继续执行 verifiability 1-5 分
```

数据数量直接反映了这一关系：

- 训练集中 `verifiability_extraction=1` 有 8,323 条；
- `verifiability` 训练集也正好有 8,323 条；
- 测试集中相应数量均为 788 条。

因此，后续生成 CoT 时可以把两者建模为一个两阶段“主张识别—论证充分性”专家，也可以分别训练后进行级联，但不能把所有 `verifiability_extraction=0` 样本强行赋予一个 `verifiability` 分数。

### 3.8 RevUtil 五个维度之间的关系

| 维度 | 主要问题 | 与其他维度关系 |
|---|---|---|
| `actionability` | 作者是否知道做什么、怎么做 | Helpfulness 的重要组成部分 |
| `grounding_specificity` | 是否定位具体位置并说清问题 | Helpfulness 的重要组成部分 |
| `verifiability_extraction` | 是否存在需论证主张 | Verifiability 的前置门控 |
| `verifiability` | 主张是否获得充分支持 | Helpfulness 的证据维度 |
| `helpfulness` | 评论整体是否能改善论文 | 综合 actionability、grounding、evidence 等因素 |

Helpfulness 不是其他几个分数的简单平均，但其定义明确依赖多个基础因素。这个结构天然适合后续检验：多个基础专家组合后，是否能提高综合 Helpfulness 判断。

## 4. 对后续 CoT 数据构建的直接启示

### 4.1 可以从现有训练数据直接构建的专家

根据真实任务格式，比较自然的专家划分是：

1. **引用—上下文一致性专家**：RW `coherence`；
2. **论文贡献定位专家**：RW `positioning_type` + `positioning_check`；
3. **可操作性专家**：RevUtil `actionability`；
4. **定位与证据充分性专家**：`grounding_specificity` + 两个 verifiability 维度；
5. **综合有用性专家**：RevUtil `helpfulness`。

这五组数据都来自官方训练部分，并且可以将 Novelty、Revision 和 Meta Reviewer 保持为未见复杂测试任务。

### 4.2 教师 CoT 生成应使用维度专属模板

不能给八个维度使用完全相同的“请一步步思考”。每个维度应使用本文件给出的专属 CoT 结构，否则教师只是生成通用解释，LoRA 专家未必形成不同推理能力。

推荐流程：

1. 教师只看到输入、rubric 和 few-shot 示例，不直接看到金标签；
2. 教师按照对应维度的固定步骤生成 CoT 和预测标签；
3. 只保留教师预测与 `labels` 一致的样本；
4. 自动检查输出格式、标签范围和必要推理字段；
5. 按 aspect 和标签分层进行人工校验；
6. 对少数标签优先补充和校验，避免类别不平衡被进一步放大。

### 4.3 人工校验重点

| 维度 | 人工校验重点 |
|---|---|
| RW coherence | 是否准确概括引用主张与原上下文，而非凭领域常识猜测 |
| RW positioning | 是否真的找到了当前论文贡献，而非把已有工作贡献误认成本文贡献 |
| Actionability | 是否分别判断了动作显式性和执行具体性 |
| Grounding | 是否区分“能定位位置”和“问题说得具体” |
| Claim detection | 是否区分一般问题、事实描述和需要论证的主张 |
| Verifiability | 是否只根据评论中实际提供的理由评分，没有自行补充证据 |
| Helpfulness | 是否综合评价改进价值，而不是只看到建议就给高分 |

## 5. 最重要的结论

RW 和 RevUtil 虽然只有两个上层任务，但实际包含八个不同评价维度，而且维度之间既有独立性，也有级联和组合关系：

- RW `coherence` 是引用核验；两个 positioning 维度是单段/多段贡献定位；
- RevUtil 的 claim detection 是 verifiability 的前置步骤；
- Helpfulness 是 actionability、grounding 和 evidence 等因素的综合评价。

因此，后续训练 LoRA 专家时，专家划分不应只依据 `task=rw_gen/rev_util`，而应依据 `aspect` 的推理结构。现有数据足以构造五组有明确语义的 CoT 专家训练数据，同时保留 Novelty、Revision 和 Meta Reviewer 作为未见复杂任务验证专家组合。
