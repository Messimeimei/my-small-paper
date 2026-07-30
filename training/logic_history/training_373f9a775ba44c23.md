# Training Logic Snapshot

- Logic ID: `373f9a775ba44c23`
- Variant: `standard:label_only`
- Generated at: `2026-07-30T08:33:57.281929+00:00`

## Logic

```json
{
  "objective": {
    "name": "standard_completion_sft",
    "views_per_original_sample": 1,
    "prompt_mode": "label_only",
    "target": "<score> only",
    "loss": "Mean next-token cross-entropy over completion tokens; prompt and padding tokens are ignored.",
    "micro_batch_backward": "Each micro-batch runs one forward pass, computes its completion loss, and immediately accumulates gradients with backward()."
  },
  "parameter_update": {
    "adapter": "LoRA",
    "target_modules": "all-linear",
    "micro_batch_size_per_device": 2,
    "gradient_accumulation_steps": 8,
    "views_per_optimizer_step_per_device": 16,
    "order": "forward -> loss -> backward for each micro-batch; after the configured accumulation count: gradient clipping -> optimizer step -> scheduler step -> zero gradients"
  },
  "validation_and_selection": {
    "frequency": "end of every epoch",
    "teacher_forced": "completion loss on the fixed validation split",
    "generation": "Greedy model.generate on each validation prompt; parse the last valid <score> value.",
    "best_checkpoint_metric": "eval_generation_accuracy",
    "generation_batch_size": 16,
    "max_new_tokens": 32
  }
}
```

## Effective Run

```json
{
  "note": "See run-local snapshot for effective settings."
}
```

## Source Fingerprints

- `training/logic_snapshot.py`: `49d1780903c5ecbfbd5b27d3712599995ed727ab8585e2e3949144e15db49318`
- `training/pipeline.py`: `5630537e2f04aa9a70bf1d78ab46cf30386b8b475f25a9a49b3f3001733ae56f`
- `training/data_utils.py`: `0bde55bc5f04875c5a1d1de7a73548f0a8a131de4f8262f7510e260e0ef004c2`
- `training/generative_trainer.py`: `39c71e69df069975d97bdcb0c9581656a2df1ae6b9683e8044cc669b2e27340f`
- `training/supervision/__init__.py`: `474d5b2562c937fc32b471beb5a7dc27171a3c147683dc66b5a6d163919d9873`
- `training/supervision/standard.py`: `39000047141023ee0e9ed337a51ec5a48b846081ae738d2c3332aaf14771cf91`
- `training/supervision/align.py`: `50374ef88f2338fd6bbc96353836b188e98580fb3eb695fa75d94b9aa968d951`
- `training/trainers/align_trainer.py`: `5e4af766caf2e8479f6e9481d4f61ebe292dd4c776d3b307f2339ef4ddb6fd95`
