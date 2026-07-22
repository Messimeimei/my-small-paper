# MoDE factor-mix pilot

本目录实现当前三个 Qwen3-4B LoRA 专家的第一版 MoDE 闭环：

1. 从每个任务的官方测试集按标签分别抽取 1、3、5 条 calibration 样本；
2. 先剔除训练集精确 prompt 泄漏并对测试 prompt 去重，再将未抽中的样本保留为
   新的 held-out test；
3. 冻结基座和三个专家，用无梯度优化搜索三个任务级权重；
4. 按 MoDE 论文 Eq.(1) / 官方代码的 factor-mix 语义保存一个新 LoRA；
5. 退出优化进程后，使用现有 `training/evaluate.py` 评测新 LoRA。

## 1. 数据划分

运行：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/prepare_fewshot_splits.py \
  --seed 42 \
  --shots-per-class 1 3 5
```

`k` 表示每个标签抽取的条数，因此二分类任务的 calibration 总数是 `2k`。
每个输出 JSON 同时包含：

```text
train: MoDE 权重搜索使用的 calibration rows
test:  清洗后删除 calibration IDs 的 held-out test rows
```

文件名同时记录每类抽样数、calibration 总数和剩余测试数，例如：

```text
rw_gen_coherence_k5_per_class_cal10_test1036_seed42.json
```

抽样采用固定 SHA256 排序并且嵌套：1-shot 是 3-shot 的子集，3-shot 是
5-shot 的子集。完全相同 prompt 的重复组不会进入 calibration；held-out test
只保留重复组中的第一条。脚本还会和当前三个专家实际使用的 JSONL 逐 prompt
比对，并从候选池和测试集删除全部精确训练重合项。源文件和训练文件 hash、抽中
ID、标签分布、重复/泄漏剔除清单和输出 hash 都记录在
`data/split_manifest.json`。

当前审计只发现 `positioning_check_0178`、`positioning_check_0602` 与对应 LoRA
训练数据精确重合，二者均已删除。coherence 另有两个测试内重复组，各只保留
第一条。因此 coherence 的清洗池为 1046 条，positioning_check 为 603 条，
positioning_type 为 204 条。

每个 k 文件按用户要求保存“抽走该 k calibration 后”的剩余测试集，因此 k=1/3/5
的测试样本并不完全相同。比较 shot 数消融时，应将三个模型统一评测在 k=5 文件的
`test` 上；因为抽样嵌套，它是三者共同且不含任何 calibration 的测试集。

## 2. 已实现的方法

当前只实现 `factor_mix`：

```text
A_hat = sum_i(w_i * A_i)
B_hat = sum_i(w_i * B_i)
delta_hat = (alpha / r) * B_hat @ A_hat
```

这与论文 Eq.(1) 和官方仓库逐 key 混合 state dict 的代码一致，但不等于
`sum_i(w_i * delta_i)`。实现手工混合原始 A/B tensor；不能替换为 PEFT 0.19.1
的 `add_weighted_adapter(combination_type="linear")`，后者具有不同的权重语义。

测试集只提供 0/1 金标签，没有金标准 reasoning。因此第一版为每条 calibration
样本构造规范 completion：

```text
<score>0</score>
```

或：

```text
<score>1</score>
```

prompt 保持原样，并使用 Qwen3 `enable_thinking=False` chat template。目标函数只对
assistant completion token 计算 CE，再加 L1：

```text
objective = score_completion_token_CE + 0.05 * mean(abs(weights))
```

Qwen3 的 `logits_to_keep` 只计算 completion 对应位置的 logits，避免长 prompt
物化完整词表 logits。任何超过 `max_length=8192` 的 calibration 序列都会直接报错，
不会静默截断。

这是一个用于先跑通参数搜索、保存和独立评测闭环的 **pilot 目标**。真实 prompt
要求先生成 reasoning，而当前 calibration 没有可监督的 reasoning；因此直接 score
CE 与正式自由生成的分布不完全一致，不能将结果当作论文级 CoT 蒸馏复现。正式实验
应补齐教师 CoT 后改为完整 completion CE，或额外比较基于实际生成并解析 score 的
accuracy/F1 黑盒目标。该限制也会写入每次 run 的配置和 adapter 元数据。

## 3. 论文、官方代码与本实现

三者需要明确区分：

| 设置 | 论文文字 | 官方代码 | 默认配置 |
|---|---|---|---|
| 融合 | 正文同时出现 delta-sum 和 factor-mix | factor-mix | factor-mix |
| 优化器 | Shiwa | NGOpt | NGOpt |
| 权重范围 | `[-1.5, 1.5]` | `[-3, 3]` | `[-3, 3]` |
| 搜索预算 | 未报告 | 40 | 40 |
| L1 | `0.05 * sum(abs(w))` | `0.05 * mean(abs(w))` | mean |
| CE target | 完整任务输出 | 完整任务输出 | 规范 score completion（pilot） |

默认配置跟随官方代码的 NGOpt 设置。每个 run 的 `manifest.json` 和
`resolved_config.json` 会再次记录这些差异，不能将本实现不加说明地称为严格论文复现。

## 4. 运行优化

当前 `small-paper` 环境已安装并验证 `nevergrad==1.0.12`。在新环境中运行：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/pip install \
  -r MoDE/requirements.txt
```

先做不加载模型、不需要 CUDA 的检查：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target coherence \
  --shots-per-class 5 \
  --dry-run
```

在 GPU 上搜索 coherence 的 5-per-class 权重：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target coherence \
  --shots-per-class 5
```

同一个入口支持三个任务和 1/3/5-per-class。若暂时不安装 Nevergrad，可使用当前
环境已有的 SciPy gradient-free fallback：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target positioning_check \
  --shots-per-class 3 \
  --optimizer scipy_differential_evolution
```

每次运行建立独立目录：

```text
MoDE/outputs/factor_mix__target-.../
├── manifest.json
├── resolved_config.json
├── calibration_summary.json
├── search_history.jsonl
├── best_weights.json
├── summary.json
└── adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── mode_meta.json
```

记录包括三个源 adapter 的路径/hash、calibration 文件/hash/IDs、全部候选权重、
CE、L1、总目标、优化器设置、软件版本、最佳权重和最终 adapter hash。

## 5. 独立评测

优化脚本会复测优化器 recommendation，并最终保存全部已评估候选中目标最低的
权重；它只保存 adapter，不在同一进程启动 vLLM。以 coherence k=5 为例：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/evaluate.py \
  --exp_name mode_factor_mix_coherence_k5pc \
  --model_name model/Qwen3-4B \
  --adapter 'MoDE/outputs/<run-id>/adapter' \
  --dataset_file MoDE/data/rw_gen_coherence_k5_per_class_cal10_test1036_seed42.json \
  --output_path MoDE/eval_outputs \
  --max_model_len 8192 \
  --max_tokens 512 \
  --temp 0 \
  --seed 42
```

必须显式传 `--dataset_file`。现有 evaluator 会读取文件中的 `test`，不会评测
`train` calibration 部分。

## 6. 测试

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  -m unittest discover -s MoDE/tests -v
```

测试覆盖嵌套分层抽样、重复 prompt 排除、文件计数、零/one-hot/任意权重混合、
输入不变性、L1 语义、adapter 配置兼容、factor-mix 交叉项、训练泄漏清洗和
最优候选权重顺序恢复。
