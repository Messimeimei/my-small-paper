# Minimal LoRA Training

该目录提供可复现的 LoRA SFT 入口：读取 `data/<task>/{cot,label_only}/train_*.jsonl`（Label-only 数据在 `label_only/` 目录），
固定分层划分、每个 epoch 保存且仅保留最后一个 checkpoint，并写出完整
manifest / summary。生成式验证使用训练模型与同一张 GPU。

## 0. 数据与配置

训练数据在 `data/`：

```text
data/<task>/cot/train_cot.jsonl
data/<task>/label_only/train_label_only.jsonl   # Label-only 训练数据
```

对应 YAML 按任务分子目录（与 `data/` 对齐）：

```text
training/configs/<task>/cot.yaml
training/configs/<task>/label_only.yaml
training/configs/<task>/raft_without_cot.yaml
training/configs/<task>/cot_raft.yaml
training/configs/<task>/legacy_align.yaml
training/configs/<task>/paper_align.yaml
```

例如 `training/configs/rw_gen_coherence/cot.yaml`。`split_path` 可省略，默认写到
`<dataset_dir>/splits/<stem>_seed<split_seed>.json`。

### 监督方式

| 方式 | 配置 | 说明 |
|------|------|------|
| **standard**（默认） | 不写 `supervision` | 对一条完整 completion 计算平均 CE |
| **raft_without_cot** | `supervision.method: raft_without_cot` | Label-only 输入；合法分数 token 的概率期望与真实分数做 MSE |
| **cot_raft** | supervision.method: cot_raft | CoT 输入；解释与格式用 CE，最终数字用 RAFT MSE |
| **legacy_align** | `supervision.method: legacy_align` | 冻结保留的旧实现：同一 CoT prompt 对应 score-only / reasoning-only 两个独立 view |
| **paper_align** | `supervision.method: paper_align` | 按 ID 配对 Direct 与 Reason view；两个 view 强制同 batch，分别平均 CE 后加权 |

RAFT without CoT 读取与 standard label-only 完全相同的样本，但不计算 completion
交叉熵。设合法分值集合为 Y，在数值标签前一位置对完整词表 logits 做 softmax，
再取每个分数 token 的概率并计算：

```text
y_hat = sum(y in Y) p(token=str(y) | x) * y
L_RAFT = mean((gold - y_hat)^2)
```

这里按论文作者公开实现，分数 token 概率不在 Y 内二次归一化；也不加入预测方差项。
代码会从 tokenizer 动态解析分数 token ID，要求每个合法分数恰好对应一个 token。
验证采用配套的 RAIL：给 assistant 补上 `<score>` 前缀后做一次前向计算，记录连续
`expected_score`、各分数概率和概率总质量，再映射到最近的合法标签计算 Accuracy、
Macro-F1 以及有序任务的 QWK。指标名 `eval_generation_accuracy` 为兼容历史日志与结果而保留，
不再用于选择 checkpoint，RAFT 路径本身不调用 `model.generate()`。

CoT-RAFT 直接读取同一任务的 cot/train_cot.jsonl，不需要生成新数据。预处理器
按最终 <score>...</score> 的字符区间记录唯一 score mask，因此 reasoning
里出现其他 0–5 数字不会被误认成标签。训练目标为：

    L_CoT-RAFT = CE(completion tokens except the numeric score)
               + raft_weight * mean((gold - y_hat)^2)

默认 raft_weight 为 1.0。这里选择论文作者发布代码的实际行为：数字 score
不参与 CE，解释、reasoning/score 标签和 EOS 仍参与 CE；论文 Eq. 4 的字面
写法会让 score 同时参与 CE，与发布代码存在差异。回归期望与 RAFT without CoT
一样，训练时使用完整词表 softmax 后的合法数字原始概率，不做候选内二次归一化。

`paper_align` 必须同时配置：

```yaml
dataset_path: data/<task>/cot/train_cot.jsonl
label_dataset_path: data/<task>/label_only/train_label_only.jsonl
supervision:
  method: paper_align
  label_coeff: 0.5
  rationale_coeff: 0.5
```

每个 paper-align dataset item 是一个源样本 pair。collator 将其展开为
`label-only prompt -> score` 和 `CoT prompt -> reasoning + score` 两条序列，
保证同一 ID 的两条序列处于同一 micro-batch。验证使用 label-only Direct prompt。

不再提供含义不明确的 `align.yaml`；训练时必须显式选择 `legacy_align.yaml` 或 `paper_align.yaml`。

代码按职责分层，顶层只保留稳定的命令入口：

```text
training/
├── train.py / evaluate.py       # CLI 兼容入口
├── shared/                      # 训练和评测共同使用的无状态代码
│   ├── project_io.py            # 项目根路径、JSON IO、UTC 时间
│   └── metrics.py               # 分类指标、分数解析、任务识别
├── training_workflow/           # 一次训练从数据到产物的完整流程
│   ├── dataset_splits.py        # 数据读取和固定 train/validation 划分
│   ├── generation_validation.py # 训练期生成验证与 Trainer callbacks
│   ├── run_lifecycle.py         # run 目录、manifest、checkpoint 和断点续训
│   └── training_pipeline.py     # 训练总编排
├── training_methods/            # 可插拔训练方法
│   ├── interfaces.py            # 训练方法接口和上下文
│   ├── registry.py              # 方法名到实现类的延迟加载注册表
│   ├── sft_config.py            # 各方法共享的 SFT 参数
│   └── standard_sft.py / *_raft.py / *_align.py
├── custom_trainers/             # 自定义损失的 Trainer 实现
└── evaluation/                  # 评测流程
    ├── cli_config.py            # CLI/YAML 解析和参数校验
    ├── dataset_loading.py       # 测试数据读取与 rollout 指标聚合
    ├── model_loading.py         # adapter 选择、模型合并和 vLLM 初始化
    ├── methods/                 # greedy、RAIL、CoT-RAIL 方法及注册表
    ├── inference_loops.py       # 具体批量推理循环
    ├── result_records.py        # 历史结果发现、规范化与聚合
    └── report_generation.py     # Markdown 汇总报告生成
```

依赖方向保持为“入口 → workflow → training_methods/evaluation → shared”；
`training_workflow/` 与 `evaluation/` 不互相依赖。训练命令仍由
`training/train.py` 启动，评测命令仍由 `training/evaluate.py` 启动。

### 扩展训练与评测方法

新增训练方法时，公共流水线不应感知具体损失：

1. 在 `training_methods/<method>.py` 实现 `SupervisionStrategy`，负责数据构造、
   collator 与 Trainer 装配；复用 `training_methods/sft_config.py` 的公共训练参数。
   自定义损失放在 `custom_trainers/`。
2. 在 `training_methods/registry.py` 增加一个延迟加载的 `StrategySpec`，并声明允许的
   dataset supervision mode。注册表导入本身不会加载 torch、TRL 或模型代码。
3. YAML 使用 `supervision.method: <method>`；方法专属参数继续放在 `supervision`
   对象内。只要沿用单数据集契约，就不需要修改
   `training_workflow/training_pipeline.py`。

新增评测方法时，评测流水线同样保持不变：

1. 在 `evaluation/methods/` 新建 `EvaluationMethod` 子类；只需实现
   `run_rollout()`，按需覆盖 `prepare()`、记录构造、汇总指标和元数据方法。
2. 在 `evaluation/methods/registry.py` 注册实例。CLI 的 `inference_mode` 选项会
   自动来自注册表，`evaluation/evaluation_pipeline.py` 无需增加 `if/elif` 分支。
3. 方法专属参数可写入 YAML 的 `method_options` 对象，或通过
   `--method_options '{"key": "value"}'` 传入。

轻量架构回归测试不会加载 GPU 训练栈：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s training/tests -v
```

## 1. 数据检查

无需 GPU：

```bash
python \
  training/train.py \
  --config training/configs/rw_gen_coherence/cot.yaml \
  --seed 42 \
  --dry-run
```

首次运行会创建固定的 train/validation ID 划分。后续配置和训练 seed 共用该文件。
`training/configs/rev_util_actionability/cot.yaml` 使用同一入口训练 1–5 分五分类专家；
训练脚本会从数据推断分值集合，并据此完成分层划分和生成式验证。

## 2. 从头训练

在 GPU 节点执行。`--fresh` 明确表示从基座模型重新开始，并创建新的时间戳目录；
为了兼容旧命令，不传 `--fresh` 时也默认从头训练。

```bash
CUDA_VISIBLE_DEVICES=0 \
python \
  training/train.py \
  --config training/configs/rw_gen_coherence/cot.yaml \
  --seed 42 \
  --fresh
```

RAFT without CoT 的启动方式相同，例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
python \
  training/train.py \
  --config training/configs/rw_gen_coherence/raft_without_cot.yaml \
  --seed 42 \
  --fresh
```

每次运行会建立独立目录：

```text
train_outputs/<task>/<mode>/<experiment>__seed<seed>__<北京时间>/
├── manifest.json
├── resolved_config.json
├── data_summary.json
├── train.log
├── train_history.jsonl
├── tensorboard/
├── trainer_state.json
├── checkpoints/          # 每个 epoch 保存，始终只保留最后一个完整 checkpoint
├── adapter/              # 从最终训练状态导出的 LoRA
├── validation_metrics.json
├── validation_predictions.jsonl
└── summary.json          # 含 final_checkpoint_epoch / final_generation_accuracy
```

- `manifest.json`：运行状态、时间、命令、代码版本和依赖版本。
- `train_history.jsonl`：每个 logging step 的 loss、学习率和梯度范数。
- `tensorboard/`：训练/验证 loss、学习率、梯度范数及最终生成式验证指标。
- `checkpoints/`：每个 epoch 保存一次，`save_total_limit=1`，仅保留 step 最大的完整 checkpoint。
- `adapter/`：训练结束后直接从最终模型状态导出，不回载较早 checkpoint。
- `validation_metrics.json`：生成式 Accuracy、macro-F1、格式有效率；1–5 分任务另含 MAE/QWK；以及 token 统计。
- `validation_predictions.jsonl`：每条验证样本的标签、预测和原始输出。
- `summary.json`：记录最终 checkpoint、adapter 来源、完整配置与 seed。

## 3. 断点续训

`--resume` 可以接收旧 run 目录，也可以直接接收 `checkpoint-*` 目录。传入 run
目录时，脚本自动选择 step 最大且包含 LoRA、optimizer、scheduler、RNG 和 Trainer
state 的完整 checkpoint。路径包含 `#`，在 shell 中必须加引号。

先做无 GPU 预检，不会修改旧 run：

```bash
python \
  training/train.py \
  --config training/configs/rw_gen_coherence/cot.yaml \
  --seed 42 \
  --resume 'train_outputs/rw_gen_coherence/cot/rw_gen_coherence#qwen3_4b#cot__seed42__...' \
  --dry-run
```

确认后继续训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
python \
  training/train.py \
  --config training/configs/rw_gen_coherence/cot.yaml \
  --seed 42 \
  --resume 'train_outputs/rw_gen_coherence/cot/<run_dir>'
```

恢复时会：

- 复用原 run 目录，追加 `train.log`、`train_history.jsonl` 和 TensorBoard 事件；
- 恢复 LoRA 权重、optimizer、scheduler、RNG、global step 和已训练数据位置；
- 校验模型、数据 hash、固定划分、LoRA 和 optimizer 相关训练配置；
- 允许 run 目录移动，并兼容修复旧 run 的最佳 checkpoint 绝对路径；
- 要求模型、数据、固定划分、LoRA、目标 epoch 和 optimizer / scheduler 相关配置不变；
- 在 `manifest.json` 的 `attempt_history` 中分别保留失败和恢复记录。

若要放弃断点状态重新训练，使用上一节的 `--fresh`。新 run 使用独立目录，不会覆盖
旧 checkpoint。旧参数名 `--resume-from-checkpoint` 仍作为 `--resume` 的兼容别名。

每个 epoch 结束时，脚本先计算 validation loss。standard/Align/CoT-RAFT 路径在当前训练 GPU
逐批执行 `model.generate()`；RAFT without CoT 路径执行上面的 score-only RAIL 概率期望。两者都记录
Accuracy、macro-F1（1–5 分另含 MAE/QWK）和逐样本输出。RAFT 还记录连续分数的
`rail_mse`、`rail_mae` 与合法分数概率质量。**每个 epoch 都写 checkpoint**（`save_strategy=epoch`），并通过
`save_total_limit=1` 仅保留最后一个完整 checkpoint；训练结束不回载 best，直接把最终权重保存到 `adapter/`。

## 4. 评测

评测配置按任务和训练方式放在 `eval_output/configs/<task>/<train_method>/`，结果写到
`eval_output/results/<task>/<exp_name>/`：

```text
eval_output/configs/<task>/
  base/
    greedy_on_cot.yaml
    greedy_on_label_only.yaml
  cot/
    greedy_on_cot.yaml
    greedy_on_label_only.yaml
    rail_on_cot.yaml
  label_only/
    greedy_on_cot.yaml
    greedy_on_label_only.yaml
    rail_on_label_only.yaml
  align/
    greedy_on_cot.yaml
    greedy_on_label_only.yaml
  paper_align/
    greedy_on_cot.yaml
    greedy_on_label_only.yaml
  cot_raft/
    greedy_on_cot.yaml
    rail_on_cot.yaml
  raft_without_cot/
    greedy_on_label_only.yaml
    rail_on_label_only.yaml
eval_output/results/<task>/<exp_name>/
  resolved_config.json / metrics.json / predictions.jsonl
```

```bash
CUDA_VISIBLE_DEVICES=0 \
python training/evaluate.py \
  --config eval_output/configs/rw_gen_coherence/base/greedy_on_cot.yaml

CUDA_VISIBLE_DEVICES=0 \
python training/evaluate.py \
  --config eval_output/configs/rev_util_actionability/cot/greedy_on_cot.yaml
```

CoT-RAIL 先生成到 `<score>`，再用第二次单 token 调用读取所有合法数字的
log-prob；未生成该前缀的样本记为 invalid，不会强行补前缀。`inference_mode: rail` 的 score-only 路径与 `cot_rail` 的两阶段路径使用同一个官方 TRACT
scorer。vLLM 返回的是完整词表 softmax 下的 log-prob，脚本直接计算
`sum(score * exp(logprob_score))`，不在合法候选内二次归一化，也不再记录非官方
预测方差。结果明确记录 `probability_normalization: full_vocab_raw`、合法分数
token 的原始概率质量和连续期望。TRACT 官方只评测连续值；本项目为保留 QWK、
Macro-F1 与 Accuracy，额外把连续值映射到最近合法标签，平局取较小标签。

`metrics.json` 含：adapter 来源 checkpoint 及其 epoch（若能从训练 run 读到）、test Acc/Macro-F1、
MAE/QWK（1–5）、格式有效率、平均 reasoning/output token、GPU 时间、seed 与完整配置。
汇总分析写入 `eval_output/evaluation_analysis.md`（每次评测后自动重建）。

## 5. 查看训练曲线

TensorBoard 是默认后端，不需要账号或外网。在训练节点启动：

```bash
tensorboard \
  --logdir train_outputs/rw_gen_coherence \
  --host 127.0.0.1 \
  --port 6006
```

浏览器访问 `http://127.0.0.1:6006`；远程服务器可通过 SSH 将本机 6006 端口转发到
训练节点的 `127.0.0.1:6006`。同一 logdir 下的不同 seed/run 会自动显示为独立曲线。

当前环境也安装了 W&B。需要跨机器集中管理时，将 YAML 中的
`report_to: tensorboard` 改为 `report_to: wandb`，并设置 `WANDB_PROJECT` 和
`WANDB_MODE=online`；网络不稳定时可用 `WANDB_MODE=offline`。主实验默认继续使用
本地 TensorBoard，避免账号、联网和上传策略成为训练依赖。

当前环境包含旧版可选依赖 `torchao 0.9.0`，而 `peft 0.19.1` 要求
`torchao>=0.16.0`。本任务使用普通 BF16 权重，不使用 TorchAO 量化；训练脚本会
安全跳过这一可选后端并将处理记录在 `manifest.json`，不会修改 Python 环境。
