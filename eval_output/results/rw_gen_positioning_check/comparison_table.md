## Positioning Check

| Id | Train | Prompt | Accuracy (%) | Macro-F1 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |
| B-L | Base | Label-only | 43.0 | 42.3 | <u>72.2</u> |
| B-C | Base | CoT | 90.9 | 91.1 | 18.4 |
| **Fine-tuned** |  |  |  |  |  |
| FT-L | Fine-tuned | Label-only | <u>98.2</u> | <u>98.2</u> | **74.0** |
| FT-C | Fine-tuned | CoT | **99.5** | **99.5** | 22.3 |
