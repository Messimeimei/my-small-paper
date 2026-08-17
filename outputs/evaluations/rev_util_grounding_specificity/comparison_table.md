## Grounding Specificity

| Id | Train | Prompt | Accuracy (%) | Macro-F1 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |
| B-L | Base | Label-only | 3.2 | 4.8 | **162.8** |
| B-C | Base | CoT | 39.1 | 31.7 | 19.1 |
| **Fine-tuned** |  |  |  |  |  |
| FT-L | Fine-tuned | Label-only | <u>56.1</u> | <u>36.6</u> | <u>160.5</u> |
| FT-C | Fine-tuned | CoT | **62.9** | **48.5** | 25.6 |
