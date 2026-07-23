# Minimal LoRA Training

该目录提供一个最小的可复现训练入口：固定分层数据划分、LoRA SFT、按 epoch
验证和保存 checkpoint、保存最佳 adapter。生成式验证直接使用训练中的模型和同一张
GPU，不启动额外推理进程。

## 1. 数据检查

无需 GPU：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/train.py \
  --config training/configs/rw_gen_coherence.yaml \
  --seed 42 \
  --dry-run
```

首次运行会创建固定的 train/validation ID 划分。后续配置和训练 seed 共用该文件。
`training/configs/rev_util_actionability.yaml` 使用同一入口训练 1–5 分五分类专家；
训练脚本会从 LoRA 数据推断分值集合，并据此完成分层划分和生成式验证。

## 2. 从头训练

在 GPU 节点执行。`--fresh` 明确表示从基座模型重新开始，并创建新的时间戳目录；
为了兼容旧命令，不传 `--fresh` 时也默认从头训练。

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/train.py \
  --config training/configs/rw_gen_coherence.yaml \
  --seed 42 \
  --fresh
```

每次运行会建立独立目录：

```text
train_outputs/rw_gen_coherence/<experiment>__seed<seed>__<UTC time>/
├── manifest.json
├── resolved_config.json
├── data_summary.json
├── train.log
├── train_history.jsonl
├── tensorboard/
├── trainer_state.json
├── checkpoints/
├── adapter/
├── validation_metrics.json
├── validation_predictions.jsonl
└── summary.json
```

- `manifest.json`：运行状态、时间、命令、代码版本和依赖版本。
- `train_history.jsonl`：每个 logging step 的 loss、学习率和梯度范数。
- `tensorboard/`：训练/验证 loss、学习率、梯度范数及最终生成式验证指标。
- `checkpoints/`：可恢复训练的 Trainer checkpoint，最多保留两个。
- `adapter/`：依据最高生成式 validation accuracy 自动恢复并保存的 LoRA adapter。
- `validation_metrics.json`：生成式 Accuracy、macro-F1 和格式有效率。
- `validation_predictions.jsonl`：每条验证样本的标签、预测和原始输出。

## 3. 断点续训

`--resume` 可以接收旧 run 目录，也可以直接接收 `checkpoint-*` 目录。传入 run
目录时，脚本自动选择 step 最大且包含 LoRA、optimizer、scheduler、RNG 和 Trainer
state 的完整 checkpoint。路径包含 `#`，在 shell 中必须加引号。

先做无 GPU 预检，不会修改旧 run：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/train.py \
  --config training/configs/rw_gen_coherence.yaml \
  --seed 42 \
  --resume '/data01/public/yangxin/small-paper/train_outputs/rw_gen_coherence/rw_gen_coherence#qwen3_4b#distill_glm-5.2__seed42__20260720T131544Z' \
  --dry-run
```

确认后继续训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/train.py \
  --config training/configs/rw_gen_coherence.yaml \
  --seed 42 \
  --resume '/data01/public/yangxin/small-paper/train_outputs/rw_gen_coherence/rw_gen_coherence#qwen3_4b#distill_glm-5.2__seed42__20260720T131544Z'
```

这个 GLM run 会从 `checkpoint-204`（epoch 1）继续到配置的 3 epochs，而不是重新
执行第 1 个 epoch。恢复时会：

- 复用原 run 目录，追加 `train.log`、`train_history.jsonl` 和 TensorBoard 事件；
- 恢复 LoRA 权重、optimizer、scheduler、RNG、global step 和已训练数据位置；
- 校验模型、数据 hash、固定划分、LoRA 和 optimizer 相关训练配置；
- 允许 run 目录移动，并修复旧的最佳 checkpoint 绝对路径；
- 要求模型、数据、固定划分、LoRA、目标 epoch 和 optimizer / scheduler 相关配置不变；
- 在 `manifest.json` 的 `attempt_history` 中分别保留失败和恢复记录。

若要放弃断点状态重新训练，使用上一节的 `--fresh`。新 run 使用独立目录，不会覆盖
旧 checkpoint。旧参数名 `--resume-from-checkpoint` 仍作为 `--resume` 的兼容别名。

每个 epoch 结束时，脚本先计算 validation loss，再在当前训练 GPU 上逐批执行
`model.generate()`，记录 Accuracy、macro-F1、格式有效率和逐样本输出，随后保存
checkpoint 并继续下一个 epoch。

## 4. 查看训练曲线

TensorBoard 是默认后端，不需要账号或外网。在训练节点启动：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/tensorboard \
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
