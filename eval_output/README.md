# 评测输出（eval_output）

评测配置与结果都放在本目录：按任务分子目录，配置在 `configs/`，结果在对应任务文件夹下。

每个可训练任务有 **8 份**微调相关配置（命名规则 `{train}_on_{test}.yaml`）+ 2 份 base：

```text
configs/<task>/
  base_on_cot.yaml
  base_on_label_only.yaml
  ft_cot_on_cot.yaml
  ft_label_only_on_label_only.yaml
  ft_cot_on_label_only.yaml
  ft_label_only_on_cot.yaml
  ft_align_on_cot.yaml
  ft_align_on_label_only.yaml
```

## 训练 × 测试矩阵

| 训练 \\ 测试 | Label-only 测试 | CoT 测试 |
|---|---|---|
| Base | `base_on_label_only.yaml` | `base_on_cot.yaml` |
| Label-only SFT | `ft_label_only_on_label_only.yaml` | `ft_label_only_on_cot.yaml` |
| CoT SFT | `ft_cot_on_label_only.yaml` | `ft_cot_on_cot.yaml` |
| Align SFT | `ft_align_on_label_only.yaml` | `ft_align_on_cot.yaml` |

解码统一：`temp=0`，`rollout=1`；Label-only 测试 `max_tokens=32`，CoT 测试 `max_tokens=512`。

> 数据文件路径仍使用 `data/<task>/score_only/`（历史目录名），含义为 **Label-only** 测试 prompt。

```text
eval_output/
  configs/<task>/...
  evaluation_analysis.md           # 唯一汇总分析（自动重建）
  <task>/<exp_name>/               # 单次评测结果（不入库）
    resolved_config.json
    metrics.json
    predictions.jsonl
```

## 记号（evaluation_analysis.md）

| 记号 | 训练 | 测试 prompt |
|---|---|---|
| B-L / B-C | Base | label-only / cot |
| L-L / C-C | Label-only / CoT SFT | 同格式 |
| C→L / L→C | 交叉 | cot→label-only / label-only→cot |
| A-C / A→L | Align SFT | cot / label-only（交叉） |

## 命令示例

```bash
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \
  --config eval_output/configs/rw_gen_coherence/base_on_cot.yaml

CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \
  --config eval_output/configs/rev_util_actionability/ft_cot_on_cot.yaml

CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \
  --config eval_output/configs/rev_util_actionability/ft_align_on_cot.yaml
```

仅重建分析：

```bash
python training/evaluate.py --refresh-analysis-only
```
