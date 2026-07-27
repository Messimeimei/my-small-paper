# Multitask cleaned score-only

由 `train_data/cleaned_data` 合并的 gold score-only 训练数据。

## 文件

| 路径 | 说明 |
|---|---|
| `multitask_32587_cleaned_score_only.jsonl` | 8 任务混合 |
| `splits/multitask_32587_cleaned_score_only_seed20260720.json` | `(task, aspect, label)` 分层划分 |

重建：

```bash
python train_data/lora_data/build_multitask_cleaned.py
```
