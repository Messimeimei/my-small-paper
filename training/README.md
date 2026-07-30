# Minimal LoRA Training

该目录提供可复现的 LoRA SFT 入口：读取 `data/<task>/{cot,score_only}/train_*.jsonl`（Label-only 数据在 `score_only/` 目录），
固定分层划分、按 **生成式 validation accuracy** 保存最佳 checkpoint，并写出完整
manifest / summary。生成式验证使用训练模型与同一张 GPU。

## 0. 数据与配置

训练数据在 `data/`：

```text
data/<task>/cot/train_cot.jsonl
data/<task>/score_only/train_score_only.jsonl   # Label-only 训练数据
```

对应 YAML 按任务分子目录（与 `data/` 对齐）：

```text
training/configs/<task>/cot.yaml
training/configs/<task>/score_only.yaml   # Label-only SFT（配置文件名保留 score_only）
training/configs/<task>/align.yaml   # 可选：Align 监督（需 CoT 数据）
```

例如 `training/configs/rw_gen_coherence/cot.yaml`。`split_path` 可省略，默认写到
`<dataset_dir>/splits/<stem>_seed<split_seed>.json`。

### 监督方式

| 方式 | 配置 | 说明 |
|------|------|------|
| **standard**（默认） | 不写 `supervision`，或 `cot` / Label-only 数据 | 与原先一致：`completion_only_loss=True`，整段 completion 平均 CE |
| **align** | `supervision.method: align` + CoT 数据 | 论文 Align：每条样本拆成 label-only / rationale-only 两视图，分别平均 loss 后加权组合 |

Align 示例（`training/configs/rw_gen_coherence/align.yaml`）：

```yaml
supervision:
  method: align
  label_coeff: 0.5      # α，对应 <score> 块
  rationale_coeff: 0.5  # 1−α，对应 <reasoning> 块
```

训练数据仍为 CoT JSONL；脚本会将 train 集扩成 2× 视图（1373 条 → 2746 视图）。
验证与 checkpoint 选择逻辑不变，仍按完整 CoT 生成式 `eval_generation_accuracy`。

代码结构：`train.py`（CLI）→ `pipeline.py`（编排）→ `supervision/{standard,align}.py`（策略）；
共用 `data_utils.py`、`run_utils.py`、`generative_trainer.py`。

日志前缀形如 `[rw_gen_coherence|cot|Qwen3-4B]`，含任务、监督模式与模型名。

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
├── checkpoints/          # save_strategy=best，仅在 generation acc 创新高时保存
├── adapter/              # 训练结束时的最佳 LoRA（按 eval_generation_accuracy）
├── validation_metrics.json
├── validation_predictions.jsonl
└── summary.json          # 含 best_checkpoint_epoch / best_generation_accuracy
```

- `manifest.json`：运行状态、时间、命令、代码版本和依赖版本。
- `train_history.jsonl`：每个 logging step 的 loss、学习率和梯度范数。
- `tensorboard/`：训练/验证 loss、学习率、梯度范数及最终生成式验证指标。
- `checkpoints/`：按 **accuracy** 变好时保存（不是按 eval_loss）；`save_total_limit` 限制数量。
- `adapter/`：`load_best_model_at_end` 后保存的最佳 LoRA adapter。
- `validation_metrics.json`：生成式 Accuracy、macro-F1、格式有效率；1–5 分任务另含 MAE/QWK；以及 token 统计。
- `validation_predictions.jsonl`：每条验证样本的标签、预测和原始输出。
- `summary.json`：记录 `best_checkpoint_epoch`、完整配置与 seed。

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
- 允许 run 目录移动，并修复旧的最佳 checkpoint 绝对路径；
- 要求模型、数据、固定划分、LoRA、目标 epoch 和 optimizer / scheduler 相关配置不变；
- 在 `manifest.json` 的 `attempt_history` 中分别保留失败和恢复记录。

若要放弃断点状态重新训练，使用上一节的 `--fresh`。新 run 使用独立目录，不会覆盖
旧 checkpoint。旧参数名 `--resume-from-checkpoint` 仍作为 `--resume` 的兼容别名。

每个 epoch 结束时，脚本先计算 validation loss，再在当前训练 GPU 上逐批执行
`model.generate()`，记录 Accuracy、macro-F1、格式有效率（1–5 分另含 MAE/QWK）
和逐样本输出。**仅当 `eval_generation_accuracy` 创新高时才写 checkpoint**
（`save_strategy=best`），训练结束按该指标加载最佳权重并保存到 `adapter/`。

## 4. 评测

评测配置按任务放在 `eval_output/configs/<task>/`（每个可训练任务 8 份），结果写到
`eval_output/<task>/<exp_name>/`：

```text
eval_output/configs/<task>/
  base_on_cot.yaml / base_on_label_only.yaml
  ft_cot_on_cot.yaml / ft_label_only_on_label_only.yaml
  ft_cot_on_label_only.yaml / ft_label_only_on_cot.yaml
  ft_align_on_cot.yaml / ft_align_on_label_only.yaml
eval_output/<task>/<exp_name>/
  resolved_config.json / metrics.json / predictions.jsonl
```

```bash
CUDA_VISIBLE_DEVICES=0 \
python training/evaluate.py \
  --config eval_output/configs/rw_gen_coherence/base_on_cot.yaml

CUDA_VISIBLE_DEVICES=0 \
python training/evaluate.py \
  --config eval_output/configs/rev_util_actionability/ft_cot_on_cot.yaml
```

`metrics.json` 含：best checkpoint epoch（若能从训练 run 读到）、test Acc/Macro-F1、
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
