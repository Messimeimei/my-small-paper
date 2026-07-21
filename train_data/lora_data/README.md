# LoRA Training Data

本目录包含从 `train_data/distill_data` 自动导出的 TRL conversational
prompt-completion 数据。每行格式如下：

```json
{
  "id": "train_0002",
  "task": "rw_gen",
  "aspect": "coherence",
  "label": 0,
  "teacher_models": ["deepseek-v4-pro"],
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "content": "<reasoning>...</reasoning>\n<score>0</score>"}
  ]
}
```

`trl>=0.23` 的 `SFTTrainer` 可直接读取该 conversational prompt-completion
格式，训练时应设置 `completion_only_loss=True`。Qwen3 推理时使用
`apply_chat_template(..., enable_thinking=False)`，使生成协议与本数据中的显式
`<reasoning>` 保持一致。数据划分应在加载后按 `label` 分层完成，测试集不得参与划分。

重新生成：

```bash
node train_data/lora_data/prepare_lora_data.js
```

输出文件：

- `rw_gen_coherence_4811_distill_deepseek-v4-pro.jsonl`
- `rw_gen_coherence_4811_distill_glm-5.2.jsonl`
- `rw_gen_coherence_4811_distill_deepseek-v4-pro_glm-5.2_consensus.jsonl`
