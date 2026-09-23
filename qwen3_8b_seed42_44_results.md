# Qwen3-8B Seed 42-44 Results

Updated: 2026-09-22T19:49:53.053972+00:00

Completed evaluations: 18 / 210

| Seed | Task | Train | Interface | Condition | N | Accuracy | Macro-F1 | QWK | MAE | RAIL MAE | RAIL MSE | RAIL RMSE | Format valid | Score-prefix valid | Reasoning valid | Score mass | Train-val accuracy | Avg output tok | Avg reasoning tok | GPU sec | Wall sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | rev_util_actionability | cot | cot | CC | 1000 | 0.5050 | 0.4640 | 0.7396 | 0.6350 | - | - | - | 1.0000 | - | - | - | 0.7542 | 133.6390 | 118.6390 | 82.7130 | 199.4480 |
| 42 | rev_util_actionability | cot | label_only | CL | 1000 | 0.5180 | 0.4494 | 0.7736 | 0.6200 | - | - | - | 1.0000 | - | - | - | 0.7542 | 7.0000 | 0.0000 | 10.3210 | 98.1250 |
| 42 | rev_util_actionability | label_only | cot | LC | 1000 | 0.5750 | 0.4705 | 0.7760 | 0.5900 | - | - | - | 1.0000 | - | - | - | 0.8156 | 110.9670 | 95.9670 | 62.8060 | 169.3840 |
| 42 | rev_util_actionability | label_only | label_only | LL | 1000 | 0.5420 | 0.5043 | 0.7881 | 0.5620 | - | - | - | 1.0000 | - | - | - | 0.8156 | 7.0000 | 0.0000 | 10.2240 | 82.5240 |
| 42 | rev_util_actionability | paper_align | cot | PAC | 1000 | 0.4930 | 0.4520 | 0.7365 | 0.6490 | - | - | - | 1.0000 | - | - | - | 0.8268 | 131.5340 | 116.5340 | 82.1250 | 191.2500 |
| 42 | rev_util_actionability | paper_align | label_only | PAL | 1000 | 0.5550 | 0.5224 | 0.7843 | 0.5570 | - | - | - | 1.0000 | - | - | - | 0.8268 | 7.0000 | 0.0000 | 10.2410 | 81.7360 |
| 42 | rev_util_grounding_specificity | cot | cot | CC | 1000 | 0.6930 | 0.5028 | 0.6871 | 0.5420 | - | - | - | 1.0000 | - | - | - | 0.8830 | 119.3260 | 104.3260 | 67.2940 | 174.6280 |
| 42 | rev_util_grounding_specificity | cot | label_only | CL | 1000 | 0.6100 | 0.4089 | 0.7152 | 0.5560 | - | - | - | 1.0000 | - | - | - | 0.8830 | 7.0000 | 0.0000 | 10.4510 | 81.0160 |
| 42 | rev_util_grounding_specificity | label_only | cot | LC | 1000 | 0.6610 | 0.4844 | 0.6990 | 0.5950 | - | - | - | 1.0000 | - | - | - | 0.9132 | 128.7700 | 113.7700 | 69.7510 | 174.5020 |
| 42 | rev_util_grounding_specificity | label_only | label_only | LL | 1000 | 0.7080 | 0.5245 | 0.7341 | 0.4840 | - | - | - | 1.0000 | - | - | - | 0.9132 | 7.0000 | 0.0000 | 10.5540 | 80.9960 |
| 42 | rev_util_grounding_specificity | paper_align | cot | PAC | 1000 | 0.7090 | 0.5082 | 0.6982 | 0.5170 | - | - | - | 1.0000 | - | - | - | 0.9094 | 117.9190 | 102.9190 | 70.0080 | 172.2180 |
| 42 | rev_util_grounding_specificity | paper_align | label_only | PAL | 1000 | 0.7060 | 0.5349 | 0.7345 | 0.4860 | - | - | - | 1.0000 | - | - | - | 0.9094 | 7.0000 | 0.0000 | 10.5200 | 76.1150 |
| 42 | rev_util_helpfulness | cot | cot | CC | 1000 | 0.5700 | 0.4985 | 0.6863 | 0.4540 | - | - | - | 1.0000 | - | - | - | 0.6886 | 142.3880 | 127.3880 | 72.6200 | 170.4450 |
| 42 | rev_util_helpfulness | cot | label_only | CL | 1000 | 0.5410 | 0.4515 | 0.6729 | 0.4890 | - | - | - | 1.0000 | - | - | - | 0.6886 | 7.0000 | 0.0000 | 10.0200 | 69.8950 |
| 42 | rev_util_helpfulness | label_only | cot | LC | 1000 | 0.5910 | 0.4783 | 0.7002 | 0.4370 | - | - | - | 1.0000 | - | - | - | 0.7544 | 129.1510 | 114.1510 | 65.6410 | 164.2980 |
| 42 | rev_util_helpfulness | label_only | label_only | LL | 1000 | 0.6170 | 0.5275 | 0.7323 | 0.3950 | - | - | - | 1.0000 | - | - | - | 0.7544 | 7.0000 | 0.0000 | 10.0100 | 73.9910 |
| 42 | rev_util_helpfulness | paper_align | cot | PAC | 1000 | 0.5860 | 0.4842 | 0.6964 | 0.4320 | - | - | - | 1.0000 | - | - | - | 0.7588 | 141.3000 | 126.3000 | 72.7150 | 168.5610 |
| 42 | rev_util_helpfulness | paper_align | label_only | PAL | 1000 | 0.6190 | 0.5356 | 0.7292 | 0.3960 | - | - | - | 1.0000 | - | - | - | 0.7588 | 7.0000 | 0.0000 | 10.0170 | 70.8370 |
