# 训练与推理逻辑记录

本文档记录当前训练和推理代码的实际行为。它描述的是仓库实现，不把当前 Align
变体写成论文中的 paired-batch Align。

## 逻辑 A：Label-only / CoT 标准 SFT

Label-only（旧目录名为 `score_only`）和 CoT 共用标准 SFT 逻辑，区别仅在 prompt
和 completion：

- Label-only：模型接收直接评分指令，目标为 `<score>N</score>`。
- CoT：模型接收先解释再评分的指令，目标为
  `<reasoning>...</reasoning><score>N</score>`。
- prompt token 和 padding token 的 label 为 `-100`，不参与 loss。
- completion 中所有有效 token 使用普通 next-token cross-entropy，并在当前
  micro-batch 内求平均。
- 每个 micro-batch 执行 `forward -> loss -> backward`。梯度累计达到配置的
  `gradient_accumulation_steps` 后，执行梯度裁剪、optimizer step、scheduler step
  和清空梯度。
- LoRA 是唯一被更新的参数；基座模型参数保持冻结。

## 逻辑 B：当前 Align 变体

每条 CoT 训练样本被拆成两个普通 Dataset row：

- label view：复用原 CoT prompt，completion 只有 `<score>` 块；
- rationale view：复用原 CoT prompt，completion 只有 `<reasoning>` 块。

这两个 view 会被 Trainer 独立随机打乱，不保证来自同一原始样本的两个 view
处于同一个 micro-batch 或同一个梯度累计窗口。

每个 micro-batch 内先计算所有 completion token 的逐 token CE，然后按 mask 分组：

```text
label_loss     = mean(label-view token CE)
rationale_loss = mean(rationale-view token CE)
loss = label_coeff * label_loss + rationale_coeff * rationale_loss
```

当前系数为 `0.5 / 0.5`。如果一个 micro-batch 缺少某类 view，该类 loss 记为 0。
随后立即 backward；达到梯度累计次数后才执行一次 optimizer step。因此不同
micro-batch 的两类梯度通常会在一次参数更新前汇合，但不是跨累计窗口统一做 token
归一化。

## 训练期验证与 checkpoint

- 数据从训练 JSONL 固定分层抽取 validation，划分文件和 dataset SHA256 一起保存。
- 每个 epoch 先做 teacher-forced completion loss 验证。
- 随后仅输入 validation prompt，使用 greedy `model.generate()` 生成结果。
- 解析输出中最后一个合法 `<score>N</score>`；缺失或越界记为 invalid 和错误。
- 最佳 checkpoint 按 `eval_generation_accuracy` 选择，而不是按 `eval_loss`。
- Align 配置来自 CoT 数据，因此训练期 checkpoint 选择使用 CoT validation prompt。

## 独立测试推理

- 无 adapter 时直接用 vLLM 加载基座模型。
- 有 adapter 时先把 LoRA 临时合并到基座模型，再由 vLLM 加载；测试后删除临时模型。
- prompt 来自测试数据自身，所以 `on_label_only` 使用直接评分指令，`on_cot` 使用
  解释后评分指令。
- 解码固定为 greedy：`temperature=0`、`top_p=1`。
- 输出仍通过最后一个合法 `<score>` 提取预测。
- 记录 Accuracy、Macro-F1、逐类指标、混淆矩阵、格式有效率和 token 统计；有序
  多分类任务额外记录 MAE 和 QWK。

## 自动留档机制

未来训练会在每个 `train_outputs/.../<run>/` 中自动生成：

```text
training_logic.json
training_logic.md
logic_snapshots/training_<logic_id>.json
logic_snapshots/training_<logic_id>.md
```

未来测试会在每个 `eval_output/.../<run>/` 中生成对应的 `inference_logic.*` 文件。

`logic_id` 根据逻辑类型、监督模式和关键源码文件的 SHA256 自动生成。仅修改数据、
seed 或普通超参数不会创建新逻辑 ID；修改 loss、数据组织、训练管线、验证、生成或
指标源码后，ID 会自动改变，并在 `training/logic_history/` 中创建一份新的长期记录。
run-local 文件同时保存实际超参数，因此同一逻辑下的不同实验仍可逐一追溯。
