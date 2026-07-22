# Distillation Data

本目录存放由 [`train_data/cleaned_data`](../cleaned_data/) 生成的教师 CoT 轨迹。脚本使用原始 cleaned prompt，不额外添加 Zero-shot-CoT 提示；只有教师输出严格符合 `<reasoning>...</reasoning><score>整数</score>`、分数属于样本的 `score_sets` 且与金标签一致时，轨迹才标记为 `accepted=true`。因此脚本同时支持 RW 的二分类评分和 RevUtil 的 1–5 分评分。

## 单文件格式

[`generate_mode_cot.py`](generate_mode_cot.py) 对每个输入只维护一个追加式 JSONL 文件：

```text
任务_维度_输入数量_distill.jsonl
```

例如全量 RW coherence 数据输出为：

```text
rw_gen_coherence_4811_distill.jsonl
```

文件中包含三类记录：

- `run_start`：本次运行的教师模型、解码设置、已有模型统计和待处理数量。
- `distillation`：一条样本使用一个教师模型得到的完整轨迹。
- `run_end`：本次运行的完成状态、成功/拒绝数量和剩余数量。

每条 `distillation` 记录包含：

- 样本 `id`、原始位置、任务、维度、金标签和完整训练 messages；
- 本次指定的 `teacher_model`、API 实际返回的模型名和 `run_id`；
- 可见原始输出 `raw_output`、解析后的 `teacher_reasoning` 和标准化 `completion`；
- API 返回的内部推理字段 `internal_reasoning`；若服务商不返回 `reasoning_content`，该字段为空字符串；
- 教师标签、格式是否合法、是否通过金标签过滤及拒绝原因；
- response ID、finish reason、token 用量、解码参数和耗时。

## 可接受轨迹数据集

每个任务保留一份追加式原始蒸馏文件，并由
[`extract_accepted_distill.js`](extract_accepted_distill.js) 生成四份只包含
`accepted=true` 记录的派生文件：

1. `*_distill_glm-5.2.jsonl`：全部可接受的 GLM 轨迹；
2. `*_distill_deepseek-v4-pro.jsonl`：全部可接受的 DeepSeek 轨迹；
3. `*_distill_deepseek-v4-pro_glm-5.2_consensus_glm-5.2.jsonl`：两位教师答案一致，只保留 GLM 轨迹；
4. `*_distill_deepseek-v4-pro_glm-5.2_consensus_deepseek-v4-pro.jsonl`：两位教师答案一致，只保留 DeepSeek 轨迹。

共识数据按 `id` 取两位教师的交集，要求 `teacher_label` 相同，并校验
`gold_label`、`task`、`aspect` 和 prompt 完全一致。由于派生数据只包含标签等于
金标签的可接受记录，两份共识文件的样本集合和顺序相同，仅保留的教师完整轨迹
不同。共识记录额外包含 `consensus_models`、`consensus_teacher_labels` 和
`consensus_trajectory_teacher` 字段。

重新生成全部三个任务的四份派生数据：

```bash
node train_data/distill_data/extract_accepted_distill.js
```

本次抽取结果如下：

| 任务维度 | DeepSeek | GLM | 两位教师一致 |
|---|---:|---:|---:|
| coherence | 3629 | 3625 | 3247 |
| positioning_check | 2666 | 2693 | 2613 |
| positioning_type | 944 | 953 | 943 |

`positioning_check` 的两位教师蒸馏均已完成，表中数据已从完整原始蒸馏文件刷新。

## 按模型续跑

启动时脚本扫描现有 JSONL，以 `(样本 id, teacher_model)` 作为唯一键：

- 同一样本已经使用过本次 `--model` 时直接跳过，无论该轨迹最终通过还是被拒绝；
- 同一样本只使用过其他教师时，仍会调用本次指定的教师并追加一条新轨迹；
- API 请求中断且尚未写入 `distillation` 记录的样本，下次会重新调用；
- 每条响应立即追加并 `fsync`，中断前已经完成的轨迹不会丢失。

`--overwrite` 会清空该输入对应的整个 JSONL，包括其他教师已有轨迹，只应在确认需要从头生成时使用。

## 运行命令

默认输入是 2 条 preview，教师模型应在每次运行时显式指定：

```bash
python train_data/distill_data/generate_mode_cot.py \
  --model deepseek-v4-pro
```

同一 preview 更换教师时使用相同命令并修改模型名，结果会追加到同一个文件：

```bash
python train_data/distill_data/generate_mode_cot.py \
  --model glm-5
```

运行 4811 条全量训练数据：

```bash
python train_data/distill_data/generate_mode_cot.py \
  --input train_data/cleaned_data/rw_gen_coherence_4811.json \
  --output-dir train_data/distill_data \
  --model deepseek-v4-pro
```

API 配置从本目录 `.env` 读取 `OPENBITFUN_API_KEY` 和 `OPENBITFUN_BASE_URL`。也可以配置 `OPENBITFUN_MODEL` 作为默认教师，但显式使用 `--model` 更适合多教师蒸馏。
