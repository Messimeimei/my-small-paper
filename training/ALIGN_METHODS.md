# Align Training Method Versions

This repository intentionally keeps two distinct Align implementations. Do not use
the unqualified name "align" when reporting a new run.

## legacy_align_unpaired_split_view_v1

This is the implementation used by the 2026-07-30 Align runs.

For each CoT row with prompt P and completion `reasoning + score`, it creates two
independent dataset rows:

```text
P -> <score>...</score> + EOS
P -> <reasoning>...</reasoning> + EOS
```

Both rows reuse the same CoT prompt. They are independently shuffled and are not
guaranteed to share a micro-batch. Within a micro-batch, completion-token CE is
averaged separately for score-only and reasoning-only rows, then combined as:

```text
label_coeff * label_loss + rationale_coeff * rationale_loss
```

This behavior is preserved under `supervision.method: legacy_align`. The immutable
snapshot from the original run remains in
`training/logic_history/training_29b0891030e44e6e.md`.

## paper_align_paired_direct_reason_v1

This implementation follows Section 4 and Appendix D.1/D.4 of
"Investigating the Impact of Rationales for LLMs on Natural Language Understanding."

CoT and label-only rows are matched by source ID. The loader requires identical ID
sets, labels, and user/context messages. It also validates the output contracts:

```text
Direct prompt -> <score>...</score> + EOS
Reason prompt -> <reasoning>...</reasoning><score>...</score> + EOS
```

One dataset item contains both views. The collator expands each source item into an
adjacent Direct/Reason sequence pair, so shuffling cannot separate the pair across
micro-batches.

The loss is:

```text
L = label_coeff * mean_ce(Direct completion tokens)
  + rationale_coeff * mean_ce(Reason completion tokens)
```

The Reason loss covers the full rationale-plus-label completion. With the default
coefficients, each view contributes 0.5 regardless of its token count.

Teacher-forced validation, generation accuracy, and best-checkpoint selection use
the matched label-only Direct validation rows, consistent with the paper's RQ2
training evaluation.

Every run records its method name, data hashes, pair counts, objective version, and
source-code hashes in `logic_snapshots/`, `logic_history/`, `data_summary.json`,
and `resolved_config.json`.
