# CoT Supervision for Scientific-Writing Judges: Mechanism-Analysis Literature Map

## Abstract

This report asks three questions: whether a mechanism analysis is necessary for the current paper, what counts as mechanism evidence in neighboring LLM-as-a-Judge and rationale-supervision work, and which additional analyses are minimally sufficient. The literature does not require neural-level interpretability for an empirical or method paper. Instead, the common pattern is to state a concrete failure hypothesis, hold other factors fixed, intervene on one component, and examine paired behavioral changes. For the current project, the existing training-by-inference matrix already supports a behavioral explanation based on mode specialization, output-format drift, and label-boundary drift. Paper Align appears to restore the direct-scoring path, but the current experiments do not yet separate the effect of adding Direct examples from the effect of balancing Direct and Reason losses. A matched Mix baseline is therefore the highest-value new experiment; gradient analysis is optional strengthening evidence.

## 1. Research Questions

- RQ1: Is mechanism analysis required for a paper centered on the observed failure of CoT supervision in scientific-writing evaluation?
- RQ2: How do neighboring papers substantiate claims about why reasoning, rationale supervision, or judge training succeeds or fails?
- RQ3: What is the minimum evidence needed to support the current paper's explanation without overclaiming?

## 2. Search Method

The search covered four branches: LLM-as-a-Judge methods, CoT effectiveness and faithfulness, rationale-supervised classification, and multi-objective optimization. Candidate papers were checked against arXiv or official conference metadata, then inspected for the experiments used to support mechanism claims. The synthesis prioritizes papers closest to scientific-writing evaluation and the current Direct/Reason paired training setup.

## 3. What Neighboring Papers Count as Mechanism Evidence

### 3.1 Controlled behavioral comparisons are the field norm

TRACT motivates three distinct components: regression-aware score learning, CoT generation, and self-generated CoT. It supports the explanation through component ablations, loss-weight sensitivity, and a controlled comparison of score RMSE under annotation CoT versus self-generated CoT. It does not measure gradients or internal representations. Its mechanism claim is therefore based on changing one training or inference component and observing the expected behavioral consequence.

Reasoning Is Not Free similarly compares matched reasoning and non-reasoning modes, stratifies results by task domain, and uses case-level analysis to explain why reasoning helps structured verification but can hurt simpler judgments. The paper's "when and why" section is behavioral rather than neural: model family and size are held fixed while the inference mode changes.

JudgeLM decomposes judge failure into position, knowledge, and format biases, then applies targeted interventions such as swap augmentation and reference manipulation. Prometheus removes rubric, feedback, and reference-answer components one at a time. Together, these papers establish that controlled interface and component interventions are accepted as explanatory evidence in LLM-as-a-Judge research.

### 3.2 Rationale papers use error taxonomy and counterfactual intervention

Investigating the Impact of Rationales for LLMs on Natural Language Understanding is the closest source for the current Paper Align setup. It compares Direct and CoT inference, manually categorizes Direct-correct/CoT-wrong cases, sweeps the label-versus-rationale loss coefficient, analyzes rationale length, and evaluates held-out tasks. Its interpretation that CoT can over-analyze NLU inputs is based on paired errors and ablation, not direct gradient evidence.

Supervised Fine-tuning with Synthetic Rationale Data Hurts Real-World Disease Prediction first demonstrates a stable negative result across many configurations, then rules out weak-rationale quality by expert review and shows that similar rationales can help as few-shot demonstrations. Its proposed explanation is a mismatch between narrative plausibility and discriminative prediction. This is strong behavioral triangulation, although it does not directly establish gradient conflict.

Faithfulness work raises the evidence bar only when the claim concerns whether a rationale causally drives a decision. Language Models Don't Always Say What They Think introduces controlled biasing features and observes whether the model's answer and explanation track the bias. Measuring Faithfulness in Chain-of-Thought Reasoning intervenes directly on the CoT through mistakes, paraphrases, and related perturbations, then measures answer changes. These papers show that rationale quality or plausibility alone cannot establish causal use of a rationale.

### 3.3 Optimization literature constrains stronger causal claims

PCGrad operationalizes conflicting task gradients through negative gradient interaction, while GradNorm studies imbalance through task-gradient magnitudes. These works provide suitable diagnostics if the paper claims gradient conflict or optimization imbalance. However, large-scale reassessment of multi-task optimization methods also shows that specialized gradient methods do not necessarily outperform carefully tuned weighted losses. Therefore, a performance recovery under Paper Align cannot by itself be attributed to gradient conflict resolution.

## 4. Interpretation of the Current Results

The current evidence supports a three-part behavioral mechanism.

First, CoT-only supervision causes output-mode specialization. The same CoT-trained checkpoint often recovers when evaluated through the Reason path rather than the Direct path. This means that at least part of the Direct-path degradation is mode mismatch rather than complete loss of judgment capability.

Second, the observed degradation contains two separable failure types. Some tasks exhibit output-schema drift, where the model fails to produce a parseable direct score. Other tasks retain valid outputs but show systematic label transitions, indicating movement of the direct-path decision boundary. These effects must not be averaged into one accuracy number.

Third, Paper Align acts as a direct-view anchor. Across the existing paired predictions, Paper Align agrees with the Label-only model on roughly 88--89% of test items and restores roughly 75% of the predictions changed by CoT-only training. This is consistent with preserving the Direct path while also learning a Reason path. Because the objective contains no explicit representation-consistency term, the current evidence does not establish internal representation alignment.

The RAFT results form a separate mechanism. Pure RAFT should primarily be evaluated through RAIL because its score regression objective does not teach reliable free-form score generation. CoT-RAFT produces an almost one-hot score distribution in the current results, making its expected score nearly identical to discrete decoding. Thus, the current negative result supports probability saturation and unresolved view mismatch more directly than it supports a general failure of regression-aware training.

## 5. Minimum Sufficient Analysis Section

The paper can use the following analysis sequence.

1. Present `LL/LC/CL/CC/PAL/PAC` as a training-view by inference-view factorial design rather than as unrelated baselines.
2. Decompose every error into parse/format failure and valid-but-wrong judgment; for ordinal tasks further report adjacent and severe errors.
3. Report paired `LL -> CL -> PAL` harmed, helped, rescued, and lost transitions with paired bootstrap confidence intervals; use McNemar tests for binary outcomes.
4. Plot label-transition matrices for the tasks with the clearest boundary shifts and show representative examples from predeclared transition categories.
5. Quantify Direct-versus-Reason prediction flips for the same checkpoint, separating changes that correct an error from changes that introduce one.
6. Analyze RAIL expected scores, entropy, margin, continuous Pearson/RMSE, and the difference between expected and discretized scores.
7. Keep the claim at the behavioral level: CoT supervision induces mode specialization and direct-scoring drift; paired Direct/Reason supervision restores the direct path.

## 6. Highest-Value Additional Experiments

### Required for the Paper Align mechanism claim

Add a matched `Mix` condition that uses the same Direct and Reason examples as Paper Align but applies ordinary token-averaged SFT. This single control distinguishes two explanations:

- `Mix` approximately equals Paper Align: recovery mainly comes from including Direct examples.
- `Mix` approximately equals CoT-only and remains below Paper Align: separate view normalization and balanced losses are important.

Without this condition, the paper can claim that paired Direct/Reason supervision works, but cannot confidently attribute the recovery to view-balanced loss.

### Recommended for generalization

Evaluate one additional model size or model family on a small representative task set. Grounding or Verifiability can represent ordinal semantic drift, while Coherence can represent a non-saturated binary task. This is more valuable than a third seed because the closest rationale literature reports model-size dependence.

### Optional strengthening evidence

- Rationale intervention: truncate, paraphrase, shuffle, or make the rationale counterfactual, then probe the score distribution.
- Gradient diagnostics: compare Direct, Reason, and Score gradient cosine similarity and norm on a fixed sample without optimizer updates.
- Coefficient sweep: vary the Direct/Reason loss weight on one representative task.

These are necessary only for stronger claims about causal rationale use, gradient conflict, or token-level optimization imbalance.

## 7. Recommended Paper Positioning

The strongest current positioning is an empirical study with a companion mitigation method:

> CoT supervision is not uniformly beneficial for scientific-writing judges. It creates task-dependent specialization to the reasoning interface, expressed through both output-schema failures and valid-score boundary drift. Balanced Direct/Reason supervision restores the direct-scoring path, while regression-aware scoring alone does not resolve this interface mismatch.

This positioning is stronger than presenting another generic judge-training method because it contributes a controlled diagnosis specific to scientific-writing evaluation. It also complements Reward Modeling for Scientific Writing Evaluation, which studies scientific-writing reward models and reasoning ablations but does not provide the same matched training-view by inference-view design.

## 8. Answers to the Research Questions

RQ1: Mechanism analysis is not a formal requirement, but it is important for this paper because the central contribution is an empirical failure mode rather than a wholly new architecture. Aggregate benchmark scores alone would leave obvious alternative explanations unresolved.

RQ2: Neighboring papers mainly use matched behavioral comparisons, targeted perturbations, component ablations, error taxonomies, and task stratification. Neural-level analysis is uncommon and is not required unless the paper makes neural or optimization-causal claims.

RQ3: The existing paired outputs are sufficient for a substantial Analysis section. One matched Mix baseline is the key missing experiment for attributing Paper Align's recovery to balanced losses. Model-family replication is the next priority; gradient analysis is optional.

## References

[1] Cheng-Han Chiang, Hung-yi Lee, and Michal Lukasik, "TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge," ACL, 2025.

[2] Wenhang Shi et al., "Investigating the Impact of Rationales for LLMs on Natural Language Understanding," arXiv:2510.16686, 2025.

[3] Buxin Su et al., "Supervised Fine-tuning with Synthetic Rationale Data Hurts Real-World Disease Prediction," arXiv:2606.10279, 2026.

[4] Furkan Sahinuc, Subhabrata Dutta, and Iryna Gurevych, "Reward Modeling for Scientific Writing Evaluation," ACL, 2026.

[5] Wenbo Zhang et al., "Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge," ICML, 2026.

[6] Lianghui Zhu, Xinggang Wang, and Xinlong Wang, "JudgeLM: Fine-tuned Large Language Models are Scalable Judges," ICLR, 2025.

[7] Seungone Kim et al., "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models," ICLR, 2024.

[8] Zayne Sprague et al., "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning," ICLR, 2025.

[9] Miles Turpin et al., "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting," NeurIPS, 2023.

[10] Tamera Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning," arXiv:2307.13702, 2023.

[11] Melanie Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting," ICLR, 2024.

[12] Tianhe Yu et al., "Gradient Surgery for Multi-Task Learning," NeurIPS, 2020.

[13] Zhao Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks," ICML, 2018.

[14] Derrick Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" NeurIPS, 2022.
