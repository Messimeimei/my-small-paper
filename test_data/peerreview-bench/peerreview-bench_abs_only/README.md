# PeerReview-Bench abs-only

阉割版 PeerReview-Bench：只保留能解析出 **标题 + 摘要 + 介绍 + 结论** 的论文，并把 `paper_content` 截断为这四部分，便于小模型 zero-shot 评测。

源数据：`data/peerreview-bench/`（[prometheus-eval/peerreview-bench](https://huggingface.co/datasets/prometheus-eval/peerreview-bench)）

## 筛选与截断规则

1. **筛选**：论文正文中必须能检测到 Abstract/Summary、Introduction/Background、Conclusion(s)（若无则用 Discussion）三节。
2. **截断**：`paper_content` 仅保留：

```
<title>

<abstract section>

<introduction section>

<conclusion section>
```

不含 Results / Methods / References 等其余章节。未拷贝 `submitted_papers`。

## 规模

| config | 原始 | abs-only |
|--------|-----:|---------:|
| reviewer | 78 | 47 |
| meta_reviewer | 908 | 516 |
| expert_annotation | 3881 | 2288 |
| similarity_check | 164 | 105 |

论文正文平均长度约 58.8k → 15.0k 字符（压缩比 ~0.27）。详见 `manifest.json`。

## 加载

```python
from datasets import load_dataset
base = "data/peerreview-bench/peerreview-bench_abs_only"
meta = load_dataset(base, "meta_reviewer", split="eval")  # 516
papers = load_dataset(base, "reviewer", split="eval")     # 47
```

## 推理数据（Task 2 / meta-reviewer 10-class）

由 `build_prompted_meta_reviewer_data.py` 生成，格式对齐
`data/Reward Modeling for Scientific Writing Evaluation/inference.py`：

- `prompted_meta_reviewer_abs_only.json`：`{"train": [], "test": [...]}`，共 516 条 zero-shot 样本
- `task=meta_reviewer_eval`，`aspect=cascade_10way`，`labels=label_id∈{1..10}`

指标：Appendix F Table 49 secondary 10-class Acc / F1 → `peerreview_task2_metrics`。

## 推理数据（Primary / 三轴级联 Acc）

由 `build_prompted_meta_reviewer_cascade_data.py` 生成：

- `prompted_meta_reviewer_cascade_abs_only.json`：516 条 zero-shot
- `task=meta_reviewer_cascade_eval`，`aspect=three_axis`
- 模型输出：`<correctness>` / `<significance>` / `<evidence>`（级联跳过写 `null`）
- GT：`*_primary`；评测子集：`eval_correctness/significance/evidence`（两位专家该轴一致，对齐 Table 47 分母）
- abs-only 子集规模：Corr **464** / Sig **288** / Evid **232**

```bash
cd "/data01/public/yangxin/small-paper/data/Reward Modeling for Scientific Writing Evaluation"
CUDA_VISIBLE_DEVICES=0 /data01/public/yangxin/.conda/envs/small-paper/bin/python inference.py \
  --exp_name peerreview_cascade_qwen3_4b \
  --model_name /data01/public/yangxin/small-paper/model/Qwen3-4B \
  --dataset_file /data01/public/yangxin/small-paper/data/peerreview-bench/peerreview-bench_abs_only/prompted_meta_reviewer_cascade_abs_only.json \
  --output_path /data01/public/yangxin/small-paper/data/eval_outputs \
  --max_model_len 16384 \
  --max_tokens 1024 \
  --temp 0 \
  --top_p 1.0 \
  --rollout 1 \
  --batch_size 2
```

指标：`peerreview_cascade_metrics`（correctness / significance / evidence Acc）。
