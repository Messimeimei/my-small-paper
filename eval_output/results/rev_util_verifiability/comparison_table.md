## Verifiability

| Id | Train | Prompt | Accuracy (%) | Macro-F1 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |
| B-L | Base | Label-only | 6.6 | 9.0 | **154.2** |
| B-C | Base | CoT | <u>47.8</u> | <u>42.6</u> | 18.2 |
| **Fine-tuned** |  |  |  |  |  |
| FT-L | Fine-tuned | Label-only | 42.3 | 38.4 | <u>153.4</u> |
| FT-C | Fine-tuned | CoT | **55.5** | **48.6** | 21.5 |
