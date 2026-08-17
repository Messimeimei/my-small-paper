## Coherence

| Id | Train | Prompt | Accuracy (%) | Macro-F1 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: |
| **Baselines** |  |  |  |  |  |
| B-L | Base | Label-only | 23.2 | 33.9 | <u>17.8</u> |
| B-C | Base | CoT | 63.5 | 62.9 | 5.9 |
| **Fine-tuned** |  |  |  |  |  |
| FT-L | Fine-tuned | Label-only | <u>71.5</u> | <u>72.6</u> | **18.0** |
| FT-C | Fine-tuned | CoT | **77.4** | **77.4** | 7.1 |
