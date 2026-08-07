## Helpfulness

| Id | Train | Prompt | Accuracy (%) | Macro-F1 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |
| B-L | Base | Label-only | 2.4 | 4.4 | **173.8** |
| B-C | Base | CoT | 34.9 | 28.6 | 25.1 |
| **Fine-tuned** |  |  |  |  |  |
| FT-L | Fine-tuned | Label-only | <u>57.0</u> | <u>44.9</u> | <u>166.9</u> |
| FT-C | Fine-tuned | CoT | **60.2** | **50.5** | 25.5 |
