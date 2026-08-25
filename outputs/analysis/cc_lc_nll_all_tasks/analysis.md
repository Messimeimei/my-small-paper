# CC vs LC NLL analysis

All conditions use the same CoT training-validation examples. NLL is teacher-forced; QWK/Macro-F1 comes from free greedy generation.

| Task | Condition | Seeds | Rationale NLL | Completion NLL | Rationale share | Primary metric | Valid rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| rev_util_actionability | Base | base | 3.3984 | 3.0749 | 0.9723 | 0.4743 (qwk) | 1.0000 |
| rev_util_actionability | CC | 42,43,44 | 0.7894 | 0.6945 | 1.0000 | 0.8291 (qwk) | 1.0000 |
| rev_util_actionability | LC | 42,43,44 | 1.5891 | 1.4309 | 0.9770 | 0.8382 (qwk) | 0.9963 |
| rev_util_grounding_specificity | Base | base | 3.6439 | 3.2816 | 0.9704 | 0.5098 (qwk) | 1.0000 |
| rev_util_grounding_specificity | CC | 42,43,44 | 0.6761 | 0.5908 | 1.0000 | 0.8911 (qwk) | 1.0000 |
| rev_util_grounding_specificity | LC | 42,43,44 | 1.6757 | 1.4925 | 0.9812 | 0.8354 (qwk) | 1.0000 |
| rev_util_helpfulness | Base | base | 3.5397 | 3.2347 | 0.9749 | 0.5091 (qwk) | 1.0000 |
| rev_util_helpfulness | CC | 42,43,44 | 0.8856 | 0.7890 | 1.0000 | 0.7581 (qwk) | 1.0000 |
| rev_util_helpfulness | LC | 42,43,44 | 1.7811 | 1.6128 | 0.9840 | 0.8300 (qwk) | 0.9971 |
| rev_util_verifiability | Base | base | 3.5677 | 3.2419 | 0.9674 | 0.7487 (qwk) | 1.0000 |
| rev_util_verifiability | CC | 42,43,44 | 0.8401 | 0.7385 | 1.0000 | 0.8646 (qwk) | 1.0000 |
| rev_util_verifiability | LC | 42,43,44 | 1.9243 | 1.7352 | 0.9749 | 0.8265 (qwk) | 1.0000 |
| rw_gen_coherence | Base | base | 3.4272 | 3.1192 | 0.9716 | 0.6553 (macro_f1) | 0.9804 |
| rw_gen_coherence | CC | 42,43,44 | 0.7852 | 0.6944 | 1.0000 | 0.8649 (macro_f1) | 1.0000 |
| rw_gen_coherence | LC | 42,43,44 | 1.4634 | 1.3198 | 0.9805 | 0.8517 (macro_f1) | 1.0000 |
| rw_gen_positioning_check | Base | base | 3.1925 | 2.8931 | 0.9633 | 0.8961 (macro_f1) | 0.9888 |
| rw_gen_positioning_check | CC | 42,43,44 | 0.7112 | 0.6209 | 1.0000 | 0.9949 (macro_f1) | 1.0000 |
| rw_gen_positioning_check | LC | 42,43,44 | 1.5362 | 1.3579 | 0.9877 | 1.0000 (macro_f1) | 1.0000 |
| rw_gen_positioning_type | Base | base | 3.3664 | 3.0812 | 0.9693 | 0.6151 (macro_f1) | 0.9681 |
| rw_gen_positioning_type | CC | 42,43,44 | 0.7676 | 0.6810 | 1.0000 | 1.0000 (macro_f1) | 1.0000 |
| rw_gen_positioning_type | LC | 42,43,44 | 1.9051 | 1.7128 | 0.9867 | 0.9906 (macro_f1) | 0.9965 |

## Interpretation

Rationale modeling is supported when CC has lower rationale NLL than LC. CC lower than Base additionally shows learning of the teacher rationale distribution. Scoring ability is judged only by free-generation QWK (ordinal) or Macro-F1 (classification). Score NLL is audit-only. NLL does not measure rationale quality or causal gradients.

Paired CC-LC rows: 21. See paired_differences.csv.
