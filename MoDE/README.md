# MoDE factor-mix

本目录实现可配置数量的 Qwen3-4B LoRA 专家 MoDE 闭环：

1. 从各专家训练时固定的 validation split 按标签抽取 1、3、5 条 calibration；
2. calibration 保留原始教师 `<reasoning>...<score>...` completion；
3. 原始 final test 不因抽取 calibration 而减少，所有 k 共用同一份测试集；
4. 冻结基座和一个或多个兼容专家，用无梯度优化搜索每个专家的任务级融合权重；
5. 保存融合 LoRA，退出优化进程后使用现有 evaluator 独立测试。

## 1. 数据来源

运行：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/prepare_fewshot_splits.py \
  --seed 42 \
  --shots-per-class 1 3 5
```

`k` 表示每个标签的样本数，因此 calibration 总数为 `标签数 * k`。抽样使用
固定 SHA256 排序，满足 1-shot ⊂ 3-shot ⊂ 5-shot。

当前 `training/configs/*.yaml` 已被后续实验改为其他教师版本，不能据此反推现有
示例 adapter 的数据。拆分脚本固定使用与 `configs/factor_mix-example.yaml` 中 adapter
实际 run hash 一致的数据和 split：

| 任务 | calibration 来源 | validation 数量 | 可抽样数量 |
|---|---|---:|---:|
| coherence | DeepSeek run validation | 363 | 363 |
| positioning_check | DeepSeek run validation | 267 | 264 |
| positioning_type | GLM run validation | 95 | 95 |
| actionability | DeepSeek run validation | 179 | 179 |

positioning_check 有 3 条 validation prompt 与 gradient-train 或 final test 精确重合，
已从 calibration 候选池排除。源文件、split、hash、候选数、排除原因和抽中 ID 均记录
在 `data/split_manifest.json`；该 manifest 只用于生成审计，不参与优化器的运行时选路。

优化器直接从 `configs/factor_mix-example.yaml` 的 `datasets` 读取数据路径：

```yaml
datasets:
  actionability:
    calibration_files:
      1: data/actionability/validation_k1_per_class_cal5_seed42.json
      3: data/actionability/validation_k3_per_class_cal15_seed42.json
      5: data/actionability/validation_k5_per_class_cal25_seed42.json
    test_file: data/actionability/clean_test1000.json
```

`--target` 选择 `datasets` 下的任务，`--shots-per-class` 选择对应 calibration 文件。

`data/` 按 target 分目录。每个目录中：

- `clean_test*.json` 和配置选中的 `*validation*.json` 是优化/评测运行时输入；
- `api_trajectories.jsonl` 是 API 生成缓存和审计记录，不参与优化运行，但应保留以便
  离线重建或中断续跑；
- 根目录的 `split_manifest.json` 与 `unseen_generation_summary.json` 是生成审计元数据，
  不参与运行时选路。

actionability 的 API calibration 与当前专家的固定 validation calibration 重复表达了两套
候选来源，配置只使用后者；前者可由 `api_trajectories.jsonl` 重建，因此未保留。

### Novelty、revision 与 meta-reviewer

四个 unseen target 可从 `test_data` 的原始评测文件中建立互斥的 calibration/test：

```text
novelty
revision_relatedness
revision_correctness
meta_reviewer
```

运行生成脚本调用 API 教师并生成数据：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/prepare_unseen_fewshot.py
```

脚本先按完整 prompt 去重，再按 prompt 哈希复用 `test_data/eval_outputs` 中已有的正确
历史生成，缺失时才请求 API 教师 CoT；只有输出包含 reasoning、合法 score 且分数与
gold label 一致的样本进入 calibration。最终 test 删除最大 calibration 中的样本，
因此不存在 calibration/test prompt 重叠，且同一 target 的所有 shot 共用同一 test。

novelty 和两个 revision target 支持 k=1/3/5。`meta_reviewer` 是 10 分类，但原始数据
中标签 2 只有 2 条、标签 5 只有 3 条，所以在不复制样本的前提下只支持 k=1。
meta prompt 为 9.5k--26k token。当前历史输出已覆盖 meta 的九个类别，只有标签 5
仍需 API 补充；现有目录只有可续跑 trajectory，尚无完整 calibration/test，所以示例配置
暂不暴露该 target。完成后可以把生成文件补入 `datasets`。单独续跑：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/prepare_unseen_fewshot.py \
  --targets meta_reviewer
```

生成后可检查：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target novelty \
  --shots-per-class 5 \
  --dry-run
```

每个任务生成三份 calibration，例如：

```text
data/coherence/validation_k1_per_class_cal2_seed42.json
data/coherence/validation_k3_per_class_cal6_seed42.json
data/coherence/validation_k5_per_class_cal10_seed42.json
```

文件中的 `train` 是 MoDE 搜索样本，每行都有 `prompt`、`label` 和教师
`completion`。它们不是 final test。

### RevUtil calibration

`actionability` 已有 LoRA 专家，其 calibration 由上面的
`prepare_fewshot_splits.py --tasks actionability` 从固定 validation split 抽取，不能再用
旧的 `_api_validation_` 文件。其余尚无 LoRA 专家的 RevUtil 维度使用独立流程：从清洗后的
人工标注数据按标签进行固定 SHA256 排序，调用 API 生成教师 CoT，并且只接受格式严格、
教师分数合法且与人工标签一致的轨迹。运行过程写入可续跑的 JSONL，不会重复请求已经由
同一教师处理过的样本：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/prepare_rev_util_fewshot.py
```

包装脚本读取 `train_data/distill_data/.env` 中的 `OPENBITFUN_API_KEY`、
`OPENBITFUN_BASE_URL` 和 `OPENBITFUN_MODEL`。可以中断后重复执行；已落盘的同模型
轨迹会直接复用。底层脚本的参数可以追加在命令后，例如仅离线检查已有 actionability：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/prepare_rev_util_fewshot.py \
  --aspects actionability \
  --offline
```

该 API 流程仍支持显式选择以下 target；其中 actionability 只用于复核历史产物：

```text
actionability
grounding_specificity
helpfulness
verifiability
verifiability_extraction
```

前四个维度是五分类，因此 k=1/3/5 分别包含 5/15/25 条 calibration；
`verifiability_extraction` 是二分类，分别包含 2/6/10 条。各维度从
`test_data/prompted_rev_util_data.json` 拆出独立 clean test，所有 k 共用同一测试文件。
除 actionability 外，这些 calibration 不是 LoRA 训练 validation，manifest 会明确记录其来源为
`cleaned_human_labeled_data_with_api_teacher_cot`。

## 2. 测试集关系

validation 没有参与 LoRA 的梯度训练，可以作为下游 MoDE 的 calibration/dev 数据。
不过训练代码启用了 `load_best_model_at_end=True`，它参与过 checkpoint 选择，因此
不是“模型完全没见过”的测试集。抽中后不需要从 final test 删除，因为二者本来就是
独立 split；calibration 只能用于调权重，不能再用于报告最终指标。

脚本仍会执行精确 prompt 审计。当前原始 positioning_check test 有 2 条 prompt 与
LoRA gradient-train 重合，coherence test 有 2 条内部重复，因此生成三份所有 k 共用的
clean test：

```text
data/coherence/clean_test1046.json
data/positioning_check/clean_test603.json
data/positioning_type/clean_test204.json
```

这里的清理与 k 无关，不是因为抽取 validation calibration 而删除测试数据。这样
k=1/3/5 的指标使用完全相同的测试样本，可以直接比较。

## 3. Factor Mix

当前实现遵循 MoDE 论文 Eq.(1) 和官方仓库的逐因子融合：

```text
A_hat = sum_i(w_i * A_i)
B_hat = sum_i(w_i * B_i)
delta_hat = (alpha / r) * B_hat @ A_hat
```

配置中的 `experts` 支持任意非零数量，并会拒绝重复的 adapter 路径或相同权重文件，
避免把同一专家重复计入专家数量消融。运行目录名和 manifest 均记录有序专家集合、
adapter 路径及权重 SHA256，便于复现实验组合。

它不等于 `sum_i(w_i * delta_i)`。代码手工混合原始 A/B tensor，不能替换为
PEFT 0.19.1 的 `add_weighted_adapter(combination_type="linear")`，后者的系数
语义不同。

目标函数只对 validation 的完整教师 assistant completion 计算 token CE：

```text
objective = teacher_CoT_completion_CE + 0.05 * mean(abs(weights))
```

prompt token 不计入 loss。任何完整序列超过 `max_length=8192` 都会报错，不会静默
截断。

## 4. 论文与默认设置

| 设置 | 论文文字 | 官方代码 | 默认配置 |
|---|---|---|---|
| 融合 | 正文同时出现 delta-sum 和 factor-mix | factor-mix | factor-mix |
| 优化器 | Shiwa | NGOpt | NGOpt |
| 权重范围 | `[-1.5, 1.5]` | `[-3, 3]` | `[-3, 3]` |
| 搜索预算 | 未报告 | 40 | 40 |
| L1 | `0.05 * sum(abs(w))` | `0.05 * mean(abs(w))` | mean |
| CE target | 完整任务输出 | 完整任务输出 | validation 教师 CoT |

默认采用官方代码的 NGOpt 设置。论文差异会再次写入每次 run 的 manifest。
正式比较不同专家数量时要固定 L1 的归一化方式：默认的 `l1_reduction: mean` 会让
正则项随维度定义变化，不适合直接做专家数量消融。此类实验应统一使用
`l1_reduction: sum`（或显式按专家数缩放 `l1_alpha`）；复现官方代码时才保留 `mean`。

为避免小预算下的专家顺序偏置，NGOpt 和 SciPy 都会先评估全零、每个单专家 one-hot
以及等权融合。NGOpt 会先用这些结果更新优化器，再执行剩余搜索；锚点评估包含在
`budget` 内。`budget` 必须至少为 `专家数 + 2`，每次运行会在 `best_weights.json` 的
优化器元数据中记录实际锚点和自由搜索次数。

## 5. 运行

当前 `small-paper` 环境已安装 `nevergrad==1.0.12`。先做静态检查：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target coherence \
  --shots-per-class 5 \
  --dry-run
```

在 GPU 上搜索：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target coherence \
  --shots-per-class 5
```

也可使用 manifest 中的任意 target，例如 `positioning_check`、`actionability` 或
`verifiability_extraction`，并配合 `--shots-per-class 1|3|5`。例如先检查 unseen task：

```bash
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target actionability \
  --shots-per-class 5 \
  --dry-run
```

SciPy fallback：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  MoDE/optimize_factor_mix.py \
  --target positioning_check \
  --shots-per-class 3 \
  --optimizer scipy_differential_evolution
```

每个 run 记录源 adapter/hash、calibration/test 文件和 hash、全部候选权重、CE、L1、
优化器 recommendation、实测最优权重、软件版本及最终 adapter hash。最终保存的是
所有已评估候选中 objective 最低的权重。

## 6. 独立评测

以 coherence 为例：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/evaluate.py \
  --exp_name mode_factor_mix_coherence_k5pc \
  --model_name model/Qwen3-4B \
  --adapter 'MoDE/outputs/<run-id>/adapter' \
  --dataset_file MoDE/data/coherence/clean_test1046.json \
  --output_path MoDE/eval_outputs \
  --max_model_len 8192 \
  --max_tokens 512 \
  --temp 0 \
  --seed 42
```

优化和 vLLM 评测分进程执行。三个 shot 设置必须传同一个 clean test 文件。

评测脚本会从数据集的 `score_sets` 动态解析合法输出，并据此计算 accuracy、macro-F1、
各类别 precision/recall/F1 和格式有效率，因此同时支持旧任务的 `[0,1]` 与 RevUtil 的
`[1,2,3,4,5]`。RevUtil 必须使用 `MoDE/data` 下按 aspect 拆分的 clean test；原始
4788 条合并文件同时包含两套标签空间，不应作为单次分类评测输入。例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data01/public/yangxin/.conda/envs/small-paper/bin/python \
  training/evaluate.py \
  --exp_name mode_factor_mix_actionability_k5pc \
  --model_name model/Qwen3-4B \
  --adapter 'MoDE/outputs/<run-id>/adapter' \
  --dataset_file MoDE/data/actionability/clean_test1000.json \
  --output_path MoDE/eval_outputs \
  --max_model_len 8192 \
  --max_tokens 512 \
  --temp 0 \
  --seed 42
```
