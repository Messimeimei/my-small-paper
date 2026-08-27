# Rationale 错误修正与固定理由重评分分析

## 1. 实验目的

本实验检验：当其他条件保持不变、只修正错误 rationale 时，原评分模型的输出分数是否随之改变。

实验包含三个配对条件：

```text
A. 自然生成：原 rationale + 原分数（历史参考）
B. 固定原错误 rationale -> 同一个 Qwen3-4B 续写分数（控制组）
C. 固定修正后 rationale -> 同一个 Qwen3-4B 续写分数（处理组）
```

主要因果比较是 `B vs. C`；`A vs. C` 只作为补充描述。

其中：

- GLM-5.3-Flash 只负责修正已标注的 rationale 错误。
- MiniMax-M3 只负责检查修改是否合理、是否保留非错误内容、是否直接泄露分数。
- 真正重新评分的是每条样本原来对应的 Qwen3-4B adapter。
- Gold 不进入重评分 prompt，只在模型输出后用于比较。

## 2. 输入与控制变量

共处理 158 条 `label_only_correct_to_cot_severe` 有害样本，覆盖 4 个任务、2 种训练方法和 3 个 seed，共使用 20 个原始 adapter。

重评分时保持以下内容与原评测逐字或逐配置一致：

- system prompt；
- user prompt，包括 Query、Criteria 和 Answer；
- Qwen3-4B 基座模型；
- 每条样本对应的 adapter；
- chat template；
- `max_model_len`、`max_tokens`、temperature、top-p、seed 和 thinking 配置；
- greedy 推理方式。

控制组和处理组分别将 assistant 生成边界后的内容固定为：

```text
控制组 B：
<reasoning>{original_rationale}</reasoning>
<score>

处理组 C：
<reasoning>{corrected_rationale}</reasoning>
<score>
```

然后让原 Qwen3-4B 只续写分数，例如 `3</score>`。

逐条 provenance 校验确认：两个固定前缀组的 158 条 prompt 均与原测试集一致，模型、adapter 和采样配置相同；原自然预测均与原 `predictions.jsonl` 一致。Gold 不进入任何模型 prompt。

## 3. 总体结果

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 158 |
| B/C 两组格式有效 | 158 / 158（100.00%） |
| 固定原理由复现自然原分数 | 157 / 158（99.37%） |
| 固定原理由仍为错误分数 | 158 / 158（100.00%） |
| B -> C 分数发生变化 | 158 / 158（100.00%） |
| C 组修改后变为 Gold（全体） | 142 / 158（89.87%） |
| 严格可比较样本（B 成功复现） | 157 |
| 严格成功（B 复现且 C 变为 Gold） | 141 / 157（89.81%） |
| 自然生成总绝对误差 | 322 |
| 固定原理由 B 总绝对误差 / MAE | 323 / 2.044 |
| 固定修正理由 C 总绝对误差 / MAE | 16 / 0.101 |
| B -> C MAE 相对下降 | 95.05% |

固定原错误 rationale 在 157/158 条样本中精确复现了自然生成时的原错误分数，排除了“固定前缀续写方式本身普遍改变分数”的解释。唯一未复现样本从自然分数 3 变为固定原理由分数 4，但两者都错误；其固定修正理由分数为 Gold 1。

在主要的 `B vs. C` 配对比较中，158 条分数全部改变，控制组 158 条全部错误，处理组 142 条变为 Gold。在严格复现子集内，141/157（89.81%）变为 Gold。这构成“修正 rationale 内容导致 score 改变”的受控干预证据。

## 4. 按训练方法分析

| 训练方法 | 样本数 | B 复现 | 严格成功 | 严格成功率 | C 回到 Gold（全体） |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoT | 86 | 86 | 75 | 87.21% | 75 |
| Label-only | 72 | 71 | 66 | 92.96% | 67 |

两种训练方法中，rationale 修正都显著改变分数。Label-only 的一条样本未复现自然原分数，因此严格分母为 71。该差异来自有害样本子集，不能直接解释为完整测试集上的方法优劣。

## 5. 按任务分析

| 任务 | 样本数 | B 复现 | 严格成功 | 严格成功率 | C 回到 Gold（全体） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actionability | 47 | 47 | 45 | 95.74% | 45 |
| Grounding Specificity | 43 | 43 | 42 | 97.67% | 42 |
| Helpfulness | 4 | 4 | 4 | 100.00% | 4 |
| Verifiability | 64 | 63 | 50 | 79.37% | 51 |

Verifiability 是主要残余误差来源：16 条未回到 Gold 的样本中有 13 条来自该任务。唯一未复现控制分数的样本也来自 Verifiability。

进一步按任务与训练方法拆分：

| 任务 | 训练方法 | 严格成功 / B 复现 | 严格成功率 |
| --- | --- | ---: | ---: |
| Actionability | CoT | 21 / 22 | 95.45% |
| Actionability | Label-only | 24 / 25 | 96.00% |
| Grounding Specificity | CoT | 24 / 25 | 96.00% |
| Grounding Specificity | Label-only | 18 / 18 | 100.00% |
| Helpfulness | CoT | 4 / 4 | 100.00% |
| Verifiability | CoT | 26 / 35 | 74.29% |
| Verifiability | Label-only | 24 / 28 | 85.71% |

## 6. 按 Seed 分析

| Seed | 样本数 | B 复现 | 严格成功 | 严格成功率 | C 回到 Gold（全体） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 56 | 56 | 48 | 85.71% | 48 |
| 43 | 54 | 54 | 47 | 87.04% | 47 |
| 44 | 48 | 47 | 46 | 97.87% | 47 |

三个 seed 的方向一致：固定原理由几乎全部复现，修正理由后大多数变为 Gold。Seed 44 的一条样本未复现自然原分数，因此严格分母为 47。

## 7. 主要分数迁移

固定原理由 B 到固定修正理由 C、且 C 回到 Gold 的主要迁移为：

| B 固定原理由分数 -> C 固定修正理由分数 | 样本数 | 结果 |
| --- | ---: | --- |
| 5 -> 3 | 63 | 回到 Gold |
| 1 -> 3 | 30 | 回到 Gold |
| 3 -> 5 | 22 | 回到 Gold |
| 3 -> 1 | 13 | 回到 Gold |
| 4 -> 2 | 4 | 回到 Gold |
| 5 -> 2 | 3 | 回到 Gold |
| 2 -> 4 | 3 | 回到 Gold |
| 4 -> 1 | 3 | 回到 Gold |
| 2 -> 5 | 1 | 回到 Gold |

固定原理由通常产生严重的跨档错误；修正 rationale 后，绝大多数预测直接移动到正确档位，而不是只向 Gold 小幅靠近。

## 8. 未回到 Gold 的 16 条样本

所有残余样本都只差 1 分；没有残留 severe error。

| 任务 | 训练 | Seed | 样本 ID | B 固定原理由 -> C 固定修正理由 | Gold |
| --- | --- | ---: | --- | ---: | ---: |
| Actionability | CoT | 43 | `actionability_test_0157` | 3 -> 2 | 1 |
| Grounding Specificity | CoT | 43 | `grounding_specificity_test_0458` | 5 -> 2 | 3 |
| Verifiability | CoT | 42 | `verifiability_test_0206` | 1 -> 2 | 3 |
| Verifiability | CoT | 42 | `verifiability_test_0489` | 1 -> 2 | 3 |
| Verifiability | CoT | 42 | `verifiability_test_0491` | 5 -> 4 | 3 |
| Verifiability | CoT | 42 | `verifiability_test_0584` | 1 -> 2 | 3 |
| Verifiability | CoT | 42 | `verifiability_test_0596` | 1 -> 2 | 3 |
| Verifiability | CoT | 43 | `verifiability_test_0011` | 1 -> 2 | 3 |
| Verifiability | CoT | 43 | `verifiability_test_0206` | 1 -> 2 | 3 |
| Verifiability | CoT | 43 | `verifiability_test_0489` | 1 -> 2 | 3 |
| Verifiability | CoT | 44 | `verifiability_test_0206` | 1 -> 2 | 3 |
| Actionability | Label-only | 42 | `actionability_test_0261` | 5 -> 2 | 3 |
| Verifiability | Label-only | 42 | `verifiability_test_0297` | 1 -> 2 | 3 |
| Verifiability | Label-only | 42 | `verifiability_test_0342` | 1 -> 2 | 3 |
| Verifiability | Label-only | 43 | `verifiability_test_0491` | 5 -> 2 | 3 |
| Verifiability | Label-only | 43 | `verifiability_test_0590` | 1 -> 2 | 3 |

残余错误高度集中在 `2 <-> 3` 邻近档位：15 条输出为 2，其中 14 条 Gold 为 3；另有 1 条输出为 4、Gold 为 3。这说明错误 rationale 被修正后，严重偏差基本消失，剩余问题主要是相邻 rubric 档位的边界判断。

## 9. 结论与边界

### 可以支持的结论

1. 固定原错误 rationale 在 157/158 条样本中复现了自然原错误分数，控制组有效。
2. 在固定原模型、adapter、system/user prompt、chat template 和 greedy 配置后，`B vs. C` 的 158 条分数全部改变。
3. 严格复现子集内，141/157（89.81%）在修正 rationale 后回到 Gold；全体为 142/158（89.87%）。
4. 从固定原理由到固定修正理由，MAE 从 2.044 降至 0.101，下降 95.05%。
5. 因此，这批有害样本中的 rationale 内容对最终 score token 具有直接、可重复的条件因果影响。
6. Verifiability 的剩余困难主要来自相邻档位边界，而不是原先的严重方向性错误。

### 不能过度推出的结论

1. 这是对筛选出的有害样本进行的干预实验，不代表完整测试集上的总体发生率。
2. 一条样本未复现自然原分数；严格成功率已将其排除并单独报告。
3. 实验能证明“给定固定 rationale 前缀时，score 会受其内容影响”，但不能单独证明模型自然生成时的隐藏计算过程完全忠实于外显 CoT。
4. 修正理由由 GLM-5.3-Flash 生成并经 MiniMax-M3 审核，不是逐字人工最小编辑；后续可增加人工最小编辑或无关措辞改写作为额外对照。

## 10. 结果文件

- 原始有效有害样本：[`valid_harmful_samples.jsonl`](./valid_harmful_samples.jsonl)
- 修正并审核后的理由：[`corrected_harmful_samples.jsonl`](./corrected_harmful_samples.jsonl)
- 固定原错误 rationale 控制组：[`fixed_original_rescored_harmful_samples.jsonl`](./fixed_original_rescored_harmful_samples.jsonl)
- 固定修正 rationale 处理组：[`rescored_harmful_samples.jsonl`](./rescored_harmful_samples.jsonl)
- B/C 配对逐条结果：[`qwen_rescore_control_comparison/predictions.jsonl`](./qwen_rescore_control_comparison/predictions.jsonl)
- B/C 配对汇总：[`qwen_rescore_control_comparison/summary.json`](./qwen_rescore_control_comparison/summary.json)
- 控制组运行汇总：[`qwen_rescore_fixed_original/summary.json`](./qwen_rescore_fixed_original/summary.json)
- 处理组运行汇总：[`qwen_rescore/summary.json`](./qwen_rescore/summary.json)
- 控制组日志：[`qwen_rescore_fixed_original/run.log`](./qwen_rescore_fixed_original/run.log)
- 处理组日志：[`qwen_rescore/run.log`](./qwen_rescore/run.log)
