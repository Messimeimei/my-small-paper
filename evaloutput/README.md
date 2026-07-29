# 评测输出

评测配置与结果都放在本目录：按任务分子目录，配置在 `configs/`，结果在对应任务文件夹下。

每个可训练任务有 4 份配置；仅测试集任务（novelty / revision_*）只有 base：

```text
configs/<task>/
  base_cot.yaml           # 基座 + CoT 测试集
  base_score_only.yaml    # 基座 + Score-only 测试集
  ft_cot.yaml             # CoT 微调 adapter
  ft_score_only.yaml      # Score-only 微调 adapter
```

```text
evaloutput/
  configs/<task>/...
  <task>/<exp_name>/                     # 单次评测结果（不入库）
    resolved_config.json
    metrics.json
    predictions.jsonl
  <task>/comparison_table.md             # 该任务 4 路结果汇总
  comparison_table.md                    # 全局汇总
```

对比表每次评测都会把对应一格写进去，四列配置为：

- `base-score_only`：基座 + Score-only
- `base-cot`：基座 + CoT
- `score_only`：微调 + Score-only（ft）
- `cot`：微调 + CoT（ft）

每种配置额外展示速度指标：

- `samp/s`：每秒完成样本数（更能体现 score-only 相对 CoT 的速度优势）

对比表高亮（纯 Markdown 标记）：

- **base** 列：普通显示
- **🟦 score_only**：微调 Score-only（表头 + 加粗单元格）
- **🟧 cot**：微调 CoT（表头 + 加粗单元格）

基座评测：

```bash
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \
  --config evaloutput/configs/rw_gen_coherence/base_cot.yaml
```

微调评测（`ft_*.yaml` 里把 `adapter` 的 `<run>` 换成实际训练目录；已有 run 的可直接改好）：

```bash
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py \
  --config evaloutput/configs/rev_util_actionability/ft_cot.yaml
```
