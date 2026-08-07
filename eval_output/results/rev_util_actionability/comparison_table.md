## Actionability

| Id | Train | Prompt | Accuracy (%) | Macro-F1 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |
| B-L | Base | Label-only | 5.6 | 8.2 | **166.6** |
| B-C | Base | CoT | 36.0 | 33.3 | 19.6 |
| **Fine-tuned** |  |  |  |  |  |
| FT-L | Fine-tuned | Label-only | **52.6** | **47.4** | <u>164.6</u> |
| FT-C | Fine-tuned | CoT | <u>50.5</u> | <u>42.7</u> | 20.7 |
