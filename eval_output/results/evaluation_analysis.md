# Qwen3-4B 评测结果分析

> 自动生成于 2026-08-04T05:50:01.972784+00:00；当前纳入 105 条按任务、条件和训练 seed 去重后的有效记录。
> 本文件由 `training/evaluate.py` 在每次评测后重建。

## 1. 统计口径

当前覆盖条件：B-L、B-C、LL、LC、CL、CC、PAL、PAC、LL-R、RAFT-G、RAFT-R、COT-RAFT-G、COT-RAFT-R。

有序任务以 QWK 为主指标，并同时展示 Accuracy、Pearson、Macro-F1 和 MAE；二分类任务以 Macro-F1 为主指标，并展示 Accuracy 与 Pearson。Pearson 使用离散预测与真实标签计算：先分别计算每个 rollout 的 Pearson 系数，再按照现有评测口径对 rollout 取均值。

同一任务和方法存在多个训练 seed 时，指标在对应方法行内展示为 `均值 ± 样本标准差 (var=样本方差)`，标准差和方差均使用 `n-1`；单 seed 只展示该次结果。Accuracy 使用百分数，方差单位为百分点平方。

每个任务表内的最优值使用下划线标出；QWK、Accuracy、Pearson、Macro-F1 取最高值，MAE 取最低值。RAIL 结果仅纳入 `probability_normalization=full_vocab_raw` 的官方口径。

## 2. Actionability

主指标为 QWK；本任务展示 QWK、Accuracy、Pearson、Macro-F1 和 MAE。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 1000 | 0.413 | 5.6 | 0.532 | 0.082 | 1.069 |
| B-C | Base | Greedy | CoT | base | 1000 | 0.492 | 36.0 | 0.546 | 0.333 | 0.905 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 1000 | 0.781 ± 0.000 (var=0.000000) | <u>54.2 ± 0.2 (var=0.0450)</u> | 0.788 ± 0.001 (var=0.000000) | 0.486 ± 0.016 (var=0.000245) | <u>0.576 ± 0.000 (var=0.000000)</u> |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 1000 | 0.735 ± 0.004 (var=0.000015) | 50.7 ± 0.3 (var=0.0800) | 0.753 ± 0.001 (var=0.000001) | 0.423 ± 0.007 (var=0.000044) | 0.688 ± 0.002 (var=0.000004) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 1000 | 0.764 ± 0.005 (var=0.000022) | 51.8 ± 0.6 (var=0.3200) | 0.767 ± 0.009 (var=0.000077) | 0.462 ± 0.008 (var=0.000062) | 0.613 ± 0.007 (var=0.000050) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 1000 | 0.722 ± 0.003 (var=0.000008) | 49.6 ± 0.5 (var=0.2450) | 0.728 ± 0.000 (var=0.000000) | 0.455 ± 0.008 (var=0.000065) | 0.661 ± 0.010 (var=0.000098) |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 1000 | <u>0.781 ± 0.004 (var=0.000017)</u> | 53.4 ± 0.2 (var=0.0450) | <u>0.790 ± 0.002 (var=0.000002)</u> | <u>0.501 ± 0.005 (var=0.000021)</u> | 0.577 ± 0.015 (var=0.000221) |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 1000 | 0.732 ± 0.009 (var=0.000074) | 49.4 ± 1.9 (var=3.6450) | 0.738 ± 0.008 (var=0.000065) | 0.443 ± 0.013 (var=0.000175) | 0.653 ± 0.037 (var=0.001405) |

## 3. Grounding Specificity

主指标为 QWK；本任务展示 QWK、Accuracy、Pearson、Macro-F1 和 MAE。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 1000 | 0.282 | 3.2 | 0.382 | 0.048 | 1.043 |
| B-C | Base | Greedy | CoT | base | 1000 | 0.518 | 39.1 | 0.571 | 0.317 | 0.997 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 1000 | <u>0.751 ± 0.026 (var=0.000662)</u> | <u>71.4 ± 1.7 (var=2.8800)</u> | 0.770 ± 0.017 (var=0.000293) | <u>0.522 ± 0.013 (var=0.000163)</u> | <u>0.462 ± 0.037 (var=0.001404)</u> |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 1000 | 0.726 ± 0.001 (var=0.000000) | 63.5 ± 0.8 (var=0.7200) | 0.732 ± 0.000 (var=0.000000) | 0.468 ± 0.024 (var=0.000592) | 0.603 ± 0.002 (var=0.000003) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 1000 | 0.686 ± 0.049 (var=0.002362) | 56.0 ± 0.1 (var=0.0200) | 0.705 ± 0.029 (var=0.000828) | 0.360 ± 0.008 (var=0.000062) | 0.623 ± 0.031 (var=0.000968) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 1000 | 0.680 ± 0.017 (var=0.000303) | 68.8 ± 0.8 (var=0.6050) | 0.718 ± 0.018 (var=0.000332) | 0.468 ± 0.002 (var=0.000005) | 0.548 ± 0.022 (var=0.000480) |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 1000 | 0.748 ± 0.011 (var=0.000128) | 71.2 ± 0.6 (var=0.4050) | <u>0.773 ± 0.005 (var=0.000022)</u> | 0.518 ± 0.024 (var=0.000573) | 0.469 ± 0.012 (var=0.000144) |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 1000 | 0.701 ± 0.015 (var=0.000213) | 69.3 ± 1.5 (var=2.2050) | 0.736 ± 0.019 (var=0.000353) | 0.466 ± 0.014 (var=0.000200) | 0.528 ± 0.025 (var=0.000613) |
| LL-R | Label-only CE | RAIL | Label-only | 42 | 1000 | 0.739 | 66.5 | 0.769 | 0.472 | 0.503 |
| RAFT-G | RAFT without CoT | Greedy | Label-only | 43 | 1000 | 0.697 | 69.3 | 0.730 | 0.464 | 0.527 |
| RAFT-R | RAFT without CoT | RAIL | Label-only | 43 | 1000 | 0.704 | 67.7 | 0.740 | 0.445 | 0.526 |
| COT-RAFT-G | CoT-RAFT | Greedy | CoT | 42 | 1000 | 0.662 | 67.2 | 0.701 | 0.447 | 0.576 |
| COT-RAFT-R | CoT-RAFT | CoT-RAIL | CoT | 42 | 1000 | 0.672 | 67.6 | 0.709 | 0.450 | 0.565 |

## 4. Helpfulness

主指标为 QWK；本任务展示 QWK、Accuracy、Pearson、Macro-F1 和 MAE。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 1000 | 0.289 | 2.4 | 0.429 | 0.044 | 1.000 |
| B-C | Base | Greedy | CoT | base | 1000 | 0.388 | 34.9 | 0.451 | 0.286 | 0.830 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 1000 | 0.723 ± 0.011 (var=0.000122) | 60.3 ± 1.6 (var=2.6450) | 0.737 ± 0.006 (var=0.000040) | <u>0.521 ± 0.027 (var=0.000718)</u> | 0.412 ± 0.014 (var=0.000200) |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 1000 | 0.701 ± 0.008 (var=0.000065) | 60.3 ± 0.1 (var=0.0200) | 0.710 ± 0.006 (var=0.000037) | 0.500 ± 0.006 (var=0.000040) | 0.423 ± 0.002 (var=0.000003) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 1000 | 0.669 ± 0.002 (var=0.000003) | 55.8 ± 1.7 (var=2.8800) | 0.681 ± 0.001 (var=0.000000) | 0.459 ± 0.014 (var=0.000208) | 0.464 ± 0.019 (var=0.000364) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 1000 | 0.670 ± 0.010 (var=0.000108) | 56.4 ± 0.5 (var=0.2450) | 0.677 ± 0.007 (var=0.000044) | 0.456 ± 0.003 (var=0.000011) | 0.461 ± 0.010 (var=0.000098) |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 1000 | <u>0.727 ± 0.019 (var=0.000361)</u> | <u>61.1 ± 0.7 (var=0.5000)</u> | <u>0.737 ± 0.017 (var=0.000296)</u> | 0.503 ± 0.044 (var=0.001978) | <u>0.400 ± 0.008 (var=0.000072)</u> |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 1000 | 0.682 ± 0.047 (var=0.002184) | 58.2 ± 3.1 (var=9.6800) | 0.689 ± 0.047 (var=0.002217) | 0.469 ± 0.035 (var=0.001225) | 0.440 ± 0.044 (var=0.001922) |

## 5. Verifiability

主指标为 QWK；本任务展示 QWK、Accuracy、Pearson、Macro-F1 和 MAE。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | QWK | Accuracy (%) | Pearson | Macro-F1 | MAE |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 788 | 0.474 | 6.6 | 0.556 | 0.090 | 0.767 |
| B-C | Base | Greedy | CoT | base | 788 | 0.577 | 47.8 | 0.582 | 0.426 | 0.675 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 788 | 0.739 ± 0.018 (var=0.000333) | 60.3 ± 0.9 (var=0.8052) | 0.763 ± 0.002 (var=0.000005) | 0.536 ± 0.015 (var=0.000214) | 0.484 ± 0.012 (var=0.000136) |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 788 | 0.672 ± 0.019 (var=0.000377) | 56.1 ± 0.9 (var=0.8052) | 0.695 ± 0.009 (var=0.000082) | 0.487 ± 0.000 (var=0.000000) | 0.588 ± 0.005 (var=0.000029) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 788 | 0.659 ± 0.046 (var=0.002094) | 44.5 ± 3.2 (var=10.4357) | 0.691 ± 0.028 (var=0.000784) | 0.404 ± 0.029 (var=0.000830) | 0.604 ± 0.029 (var=0.000825) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 788 | 0.652 ± 0.005 (var=0.000021) | 50.8 ± 0.1 (var=0.0081) | 0.672 ± 0.001 (var=0.000001) | 0.441 ± 0.003 (var=0.000007) | 0.641 ± 0.000 (var=0.000000) |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 788 | <u>0.770 ± 0.002 (var=0.000006)</u> | <u>61.0 ± 1.3 (var=1.8118)</u> | <u>0.781 ± 0.003 (var=0.000009)</u> | <u>0.556 ± 0.015 (var=0.000232)</u> | <u>0.461 ± 0.013 (var=0.000158)</u> |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 788 | 0.662 ± 0.013 (var=0.000175) | 52.3 ± 0.1 (var=0.0081) | 0.684 ± 0.012 (var=0.000151) | 0.442 ± 0.007 (var=0.000043) | 0.617 ± 0.003 (var=0.000007) |

## 6. Coherence

主指标为 Macro-F1；本任务展示 Accuracy、Macro-F1 和 Pearson。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | Accuracy (%) | Macro-F1 | Pearson |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 1046 | 23.2 | 0.339 | 0.266 |
| B-C | Base | Greedy | CoT | base | 1046 | 63.5 | 0.629 | 0.284 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 1046 | 78.9 ± 0.5 (var=0.2239) | 0.789 ± 0.005 (var=0.000022) | 0.579 ± 0.010 (var=0.000092) |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 1046 | 77.8 ± 0.5 (var=0.2239) | 0.778 ± 0.005 (var=0.000025) | 0.556 ± 0.010 (var=0.000107) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 1046 | 71.8 ± 0.5 (var=0.2239) | 0.726 ± 0.001 (var=0.000000) | 0.491 ± 0.011 (var=0.000112) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 1046 | 77.3 ± 0.2 (var=0.0411) | 0.773 ± 0.002 (var=0.000004) | 0.546 ± 0.004 (var=0.000018) |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 1046 | <u>80.6 ± 2.5 (var=6.2562)</u> | <u>0.806 ± 0.025 (var=0.000646)</u> | <u>0.615 ± 0.048 (var=0.002291)</u> |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 1046 | 79.5 ± 0.5 (var=0.2239) | 0.795 ± 0.005 (var=0.000022) | 0.590 ± 0.009 (var=0.000089) |
| RAFT-G | RAFT without CoT | Greedy | Label-only | 43 | 1046 | 0.0 | 0.000 | — |
| RAFT-R | RAFT without CoT | RAIL | Label-only | 43 | 1046 | 77.2 | 0.772 | 0.546 |

## 7. Positioning Check

主指标为 Macro-F1；本任务展示 Accuracy、Macro-F1 和 Pearson。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | Accuracy (%) | Macro-F1 | Pearson |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 603 | 43.0 | 0.423 | — |
| B-C | Base | Greedy | CoT | base | 603 | 90.9 | 0.911 | 0.841 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 603 | 99.7 ± 0.2 (var=0.0550) | 0.997 ± 0.002 (var=0.000006) | 0.993 ± 0.005 (var=0.000023) |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 603 | 99.8 ± 0.4 (var=0.1238) | 0.997 ± 0.004 (var=0.000013) | 0.995 ± 0.007 (var=0.000051) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 603 | 87.0 ± 15.8 (var=250.6126) | 0.920 ± 0.087 (var=0.007496) | 0.971 ± 0.010 (var=0.000104) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 603 | 99.3 ± 0.0 (var=0.0000) | 0.993 ± 0.000 (var=0.000000) | 0.987 ± 0.000 (var=0.000000) |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 603 | <u>99.8 ± 0.0 (var=0.0000)</u> | <u>0.998 ± 0.000 (var=0.000000)</u> | <u>0.997 ± 0.000 (var=0.000000)</u> |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 603 | 99.7 ± 0.0 (var=0.0000) | 0.997 ± 0.000 (var=0.000000) | 0.993 ± 0.000 (var=0.000000) |

## 8. Positioning Type

主指标为 Macro-F1；本任务展示 Accuracy、Macro-F1 和 Pearson。

| 条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | N | Accuracy (%) | Macro-F1 | Pearson |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| B-L | Base | Greedy | Label-only | base | 204 | 35.3 | 0.351 | 0.158 |
| B-C | Base | Greedy | CoT | base | 204 | 58.3 | 0.589 | 0.425 |
| LL | Label-only SFT | Greedy | Label-only | 42, 43 | 204 | <u>100.0 ± 0.0 (var=0.0000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> |
| LC | Label-only SFT | Greedy | CoT | 42, 43 | 204 | 99.3 ± 0.3 (var=0.1201) | 0.995 ± 0.001 (var=0.000002) | 0.994 ± 0.008 (var=0.000061) |
| CL | CoT SFT | Greedy | Label-only | 42, 43 | 204 | 84.8 ± 4.2 (var=17.3010) | 0.843 ± 0.041 (var=0.001648) | 0.730 ± 0.061 (var=0.003690) |
| CC | CoT SFT | Greedy | CoT | 42, 43 | 204 | <u>100.0 ± 0.0 (var=0.0000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> |
| PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | 204 | <u>100.0 ± 0.0 (var=0.0000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> |
| PAC | Paper Align SFT | Greedy | CoT | 42, 43 | 204 | <u>100.0 ± 0.0 (var=0.0000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> | <u>1.000 ± 0.000 (var=0.000000)</u> |

## 9. 七任务汇总

每个任务按其主指标选择最优条件；下表汇总该条件的主指标、Accuracy 和 Pearson。

| 任务 | 主指标 | 最优条件 | 训练方式 | 推理方式 | 测试数据 | 训练 seed | 主指标结果 | Accuracy (%) | Pearson |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Actionability | QWK | PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | <u>0.781 ± 0.004 (var=0.000017)</u> | 53.4 ± 0.2 (var=0.0450) | 0.790 ± 0.002 (var=0.000002) |
| Grounding Specificity | QWK | LL | Label-only SFT | Greedy | Label-only | 42, 43 | <u>0.751 ± 0.026 (var=0.000662)</u> | 71.4 ± 1.7 (var=2.8800) | 0.770 ± 0.017 (var=0.000293) |
| Helpfulness | QWK | PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | <u>0.727 ± 0.019 (var=0.000361)</u> | 61.1 ± 0.7 (var=0.5000) | 0.737 ± 0.017 (var=0.000296) |
| Verifiability | QWK | PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | <u>0.770 ± 0.002 (var=0.000006)</u> | 61.0 ± 1.3 (var=1.8118) | 0.781 ± 0.003 (var=0.000009) |
| Coherence | Macro-F1 | PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | <u>0.806 ± 0.025 (var=0.000646)</u> | 80.6 ± 2.5 (var=6.2562) | 0.615 ± 0.048 (var=0.002291) |
| Positioning Check | Macro-F1 | PAL | Paper Align SFT | Greedy | Label-only | 42, 43 | <u>0.998 ± 0.000 (var=0.000000)</u> | 99.8 ± 0.0 (var=0.0000) | 0.997 ± 0.000 (var=0.000000) |
| Positioning Type | Macro-F1 | LL | Label-only SFT | Greedy | Label-only | 42, 43 | <u>1.000 ± 0.000 (var=0.000000)</u> | 100.0 ± 0.0 (var=0.0000) | 1.000 ± 0.000 (var=0.000000) |

## 10. 重建报告

```bash
python training/evaluate.py --refresh-analysis-only --output_path eval_output/results
```
