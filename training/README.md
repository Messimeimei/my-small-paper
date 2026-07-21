# Minimal LoRA Training

该目录提供一个最小的可复现训练入口：固定分层数据划分、LoRA SFT、按 epoch
验证和保存 checkpoint、保存最佳 adapter，并在训练结束后进行生成式验证。

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

## 2. 开始训练

在 GPU 节点执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/train.py \
  --config training/configs/rw_gen_coherence.yaml \
  --seed 42
```

每次运行会建立独立目录：

```text
train_outputs/lora/rw_gen_coherence/<experiment>__seed<seed>__<UTC time>/
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
- `adapter/`：依据最低 validation loss 自动恢复并保存的 LoRA adapter。
- `validation_metrics.json`：生成式 Accuracy、macro-F1 和格式有效率。
- `validation_predictions.jsonl`：每条验证样本的标签、预测和原始输出。

## 3. 测试集评测

用 vLLM 加载基座 + LoRA（`adapter/` 或 `checkpoints/checkpoint-*`）做生成式评测
（参数风格与根目录 `评测命令.txt` 一致）：

```bash
CUDA_VISIBLE_DEVICES=2 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/evaluate.py \
  --exp_name rw_gen_coherence_lora_ckpt205 \
  --model_name model/Qwen3-4B \
  --adapter train_outputs/lora/rw_gen_coherence/<run>/checkpoints/checkpoint-205 \
  --dataset_file test_data/prompted_rw_gen_coherence_data.json \
  --output_path eval_data/eval_outputs \
  --max_model_len 8192 \
  --max_tokens 512 \
  --temp 0 \
  --top_p 1.0 \
  --seed 42 \
  --rollout 1 \
  --batch_size 64 \
  --gpu_memory_utilization 0.9
```

结果写到 `eval_data/eval_outputs/<exp_name>/metrics.json` 与 `predictions.jsonl`。
首次评测会把 LoRA merge 成完整权重并缓存到 `eval_data/.merged_models/`（绕过
vLLM 0.8.4 的 enable_lora + cachetools bug）；之后同一 adapter 会直接复用。
显存紧张时降低 `--gpu_memory_utilization`（共享卡建议按空闲显存/总显存估算）；
训练结束后优先评 `adapter/`。

## 4. 查看训练曲线

TensorBoard 是默认后端，不需要账号或外网。在训练节点启动：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/tensorboard \
  --logdir train_outputs/lora/rw_gen_coherence \
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
