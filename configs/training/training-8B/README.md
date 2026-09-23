# Qwen3-8B training configs

Seven tasks and five methods mirror the Qwen3-4B and Qwen3-0.6B experiments.
Core hyperparameters are unchanged. Standard methods use effective batch 16;
paired Align methods use effective batch 8.

Self-correct configs consume data/cot_self_generated_data-8B and
data/align_self_generated_data-8B, generated from completed seed42 adapters.
