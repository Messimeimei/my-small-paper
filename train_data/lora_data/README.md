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

脚本直接读取 `train_data/distill_data` 中的可接受轨迹数据集，逐一转换为同名
LoRA 文件，并以临时文件 + rename 的方式原子更新输出。已完成双教师蒸馏的 RW
任务包含以下四个版本：

1. DeepSeek 全部可接受轨迹；
2. GLM 全部可接受轨迹；
3. 两位教师一致，保留 DeepSeek completion；
4. 两位教师一致，保留 GLM completion。

单教师版本的 `teacher_models` 只记录对应教师，共识版本记录两位教师。实际
completion 来源由文件名最后的教师名以及 `export_manifest.json` 中的
`trajectory_teacher` 标明。manifest 同时记录源蒸馏文件、读取字节数、样本数和
标签分布。

## 当前输出

| aspect | 版本（轨迹来源） | 样本数 | 文件 |
|---|---|---:|---|
| coherence | DeepSeek | 3629 | `rw_gen_coherence_3629_distill_deepseek-v4-pro.jsonl` |
| coherence | GLM | 3625 | `rw_gen_coherence_3625_distill_glm-5.2.jsonl` |
| coherence | 双教师共识（DeepSeek） | 3247 | `rw_gen_coherence_3247_distill_deepseek-v4-pro_glm-5.2_consensus_deepseek-v4-pro.jsonl` |
| coherence | 双教师共识（GLM） | 3247 | `rw_gen_coherence_3247_distill_deepseek-v4-pro_glm-5.2_consensus_glm-5.2.jsonl` |
| positioning_check | DeepSeek | 2666 | `rw_gen_positioning_check_2666_distill_deepseek-v4-pro.jsonl` |
| positioning_check | GLM | 2693 | `rw_gen_positioning_check_2693_distill_glm-5.2.jsonl` |
| positioning_check | 双教师共识（DeepSeek） | 2613 | `rw_gen_positioning_check_2613_distill_deepseek-v4-pro_glm-5.2_consensus_deepseek-v4-pro.jsonl` |
| positioning_check | 双教师共识（GLM） | 2613 | `rw_gen_positioning_check_2613_distill_deepseek-v4-pro_glm-5.2_consensus_glm-5.2.jsonl` |
| positioning_type | DeepSeek | 944 | `rw_gen_positioning_type_944_distill_deepseek-v4-pro.jsonl` |
| positioning_type | GLM | 953 | `rw_gen_positioning_type_953_distill_glm-5.2.jsonl` |
| positioning_type | 双教师共识（DeepSeek） | 943 | `rw_gen_positioning_type_943_distill_deepseek-v4-pro_glm-5.2_consensus_deepseek-v4-pro.jsonl` |
| positioning_type | 双教师共识（GLM） | 943 | `rw_gen_positioning_type_943_distill_deepseek-v4-pro_glm-5.2_consensus_glm-5.2.jsonl` |
| actionability | DeepSeek | 1788 | `rev_util_actionability_1788_distill_deepseek-v4-pro.jsonl` |
| grounding_specificity | DeepSeek | 2652 | `rev_util_grounding_specificity_2652_distill_deepseek-v4-pro.jsonl` |

`positioning_check` 的两位教师蒸馏均已完成，表中 LoRA 数据已从完整的派生蒸馏
数据刷新。

`actionability` 与 `grounding_specificity` 都是 1–5 分五分类任务，当前只导出已完成的
DeepSeek-v4-pro 单教师版本；固定划分分别为
`splits/rev_util_actionability_deepseek-v4-pro_seed20260720.json` 和
`splits/rev_util_grounding_specificity_deepseek-v4-pro_seed20260720.json`。

coherence 两个共识版本使用相同的训练集/验证集 ID 划分，但数据哈希不同：

- DeepSeek 轨迹：`splits/rw_gen_coherence_consensus_deepseek-v4-pro_seed20260720.json`；
- GLM 轨迹：`splits/rw_gen_coherence_consensus_glm-5.2_seed20260720.json`。

原 `splits/rw_gen_coherence_consensus_seed20260720.json` 保留为 DeepSeek 轨迹版本
的兼容别名。
