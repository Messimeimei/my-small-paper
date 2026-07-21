---
license: cc-by-4.0
language:
- en
pretty_name: PeerReview Bench
task_categories:
- text-classification
- text-generation
tags:
- peer-review
- scientific-papers
- expert-annotation
- evaluation
- multimodal
configs:
- config_name: expert_annotation
  data_files:
  - split: eval
    path: expert_annotation/eval-*
- config_name: meta_reviewer
  data_files:
  - split: eval
    path: meta_reviewer/eval-*
- config_name: reviewer
  data_files:
  - split: eval
    path: reviewer/eval-*
- config_name: similarity_check
  data_files:
  - split: eval
    path: similarity_check/eval-*
- config_name: submitted_papers
  data_files:
  - split: eval
    path: submitted_papers/eval-*
dataset_info:
- config_name: expert_annotation
  features:
  - name: paper_id
    dtype: int64
  - name: paper_title
    dtype: string
  - name: paper_content
    dtype: string
  - name: file_refs
    list:
    - name: content_hash
      dtype: string
    - name: is_text
      dtype: bool
    - name: path
      dtype: string
    - name: size_bytes
      dtype: int64
  - name: annotator_source
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: reviewer_type
    dtype: string
  - name: review_item_number
    dtype: int64
  - name: review_item
    dtype: string
  - name: correctness
    dtype: string
  - name: significance
    dtype: string
  - name: evidence
    dtype: string
  - name: annotator_comments
    dtype: string
  splits:
  - name: eval
    num_bytes: 358234433.90870005
    num_examples: 3881
  download_size: 56164786
  dataset_size: 358234433.90870005
- config_name: meta_reviewer
  features:
  - name: paper_id
    dtype: int64
  - name: paper_title
    dtype: string
  - name: paper_content
    dtype: string
  - name: file_refs
    list:
    - name: content_hash
      dtype: string
    - name: is_text
      dtype: bool
    - name: path
      dtype: string
    - name: size_bytes
      dtype: int64
  - name: reviewer_id
    dtype: string
  - name: reviewer_type
    dtype: string
  - name: review_item_number
    dtype: int64
  - name: review_item
    dtype: string
  - name: correctness_primary
    dtype: string
  - name: correctness_secondary
    dtype: string
  - name: significance_primary
    dtype: string
  - name: significance_secondary
    dtype: string
  - name: evidence_primary
    dtype: string
  - name: evidence_secondary
    dtype: string
  - name: label_id
    dtype: int64
  - name: label
    dtype: string
  splits:
  - name: eval
    num_bytes: 74407740
    num_examples: 908
  download_size: 2262537
  dataset_size: 74407740
- config_name: reviewer
  features:
  - name: paper_id
    dtype: int64
  - name: paper_title
    dtype: string
  - name: paper_content
    dtype: string
  - name: file_refs
    list:
    - name: content_hash
      dtype: string
    - name: is_text
      dtype: bool
    - name: path
      dtype: string
    - name: size_bytes
      dtype: int64
  - name: rubric
    sequence: string
  splits:
  - name: eval
    num_bytes: 8171594.925
    num_examples: 78
  download_size: 4579964
  dataset_size: 8171594.925
---

# PeerReview Bench

- **CMU Paper Reviewer:https://prometheus-eval.github.io/cmu-paper-reviewer/** 
- **Repository:https://github.com/prometheus-eval/cmu-paper-reviewer** 
- **Paper:https://arxiv.org/abs/2605.20668** 
- **Point of Contact:seungone@kaist.ac.kr** 

Expert-annotated review items from scientific papers, organized for three
complementary **evaluation** tasks. All data in this dataset is intended
for evaluation, not training. All configs reference a shared, deduplicated
file store (`submitted_papers`) via SHA256 content hashes.

Every config exposes a single `eval` split.

## Configs

### `reviewer`
For evaluating **AI reviewers** (models that *generate* reviews from a paper).
- One row per paper.
- Minimal fields: `paper_id`, `paper_title`, `paper_content` (preprint.md text),
  `file_refs` (pointers to `submitted_papers`).
- Use this by loading one paper, reconstructing its files via `file_refs` +
  `submitted_papers`, feeding the content to your AI reviewer, and comparing
  the generated review to the ground-truth reviews in `expert_annotation`.

### `meta_reviewer`
For evaluating **AI meta-reviewers** (LLMs or agents that *label* an existing
review item with correctness / significance / evidence).
- One row per (paper, reviewer, review_item), **only for the papers where
  both primary and secondary annotators contributed**.
- Each row includes per-annotator labels (`correctness_primary`,
  `correctness_secondary`, etc.) plus a single collapsed `label` of one of 10
  classes that encodes both the cascade outcome and the per-metric agreement:

  | ID | Label | Meaning |
  |---:|:---|:---|
  | 1 | `correct_significant_sufficient` | Both annotators: Correct + Significant + Sufficient |
  | 2 | `correct_significant_insufficient` | Both: Correct + Significant + Requires More |
  | 3 | `correct_significant_disagree_on_evidence` | Both: Correct + Significant, but disagree on evidence |
  | 4 | `correct_marginal_sufficient` | Both: Correct + Marginally Significant + Sufficient |
  | 5 | `correct_marginal_insufficient` | Both: Correct + Marginally Sig. + Requires More |
  | 6 | `correct_marginal_disagree_on_evidence` | Both: Correct + Marginally Sig., disagree on evidence |
  | 7 | `correct_not_significant` | Both: Correct + Not Significant |
  | 8 | `correct_disagree_on_significance` | Both: Correct, disagree on significance |
  | 9 | `incorrect` | Both: Not Correct |
  | 10 | `disagree_on_correctness` | Annotators disagree on whether the item is correct |

  A well-designed meta-reviewer should predict **both** the labels (cascade)
  and whether experts would agree on each metric — the collapsed label captures
  both pieces in one class.
- `file_refs` included so agent-based meta-reviewers can browse the paper's
  preprint files.

Schema columns: `paper_id`, `paper_title`, `paper_content`, `file_refs`,
`reviewer_id`, `reviewer_type`, `review_item_number`, `review_item`,
`correctness_primary`, `correctness_secondary`,
`significance_primary`, `significance_secondary`,
`evidence_primary`, `evidence_secondary`, `label_id`, `label`.

### `expert_annotation`
For **statistical analysis** and **human-vs-AI review similarity** measurement.
- One row per (paper, reviewer, review_item, annotator_source); items annotated
  by both primary and secondary annotators appear as two rows.
- `annotator_source` ∈ {`primary`, `secondary`}.
- Per-row validity stripping is applied: items with incomplete cascades are
  dropped, and labels beyond the cascade break are nulled (see validity rules
  below).
- `file_refs` included for LLM-agent similarity evaluation.

Schema columns: `paper_id`, `paper_title`, `paper_content`, `file_refs`,
`annotator_source`, `reviewer_id`, `reviewer_type`, `review_item_number`,
`review_item`, `correctness`, `significance`, `evidence`, `annotator_comments`.

### `similarity_check`
For benchmarking **automated similarity metrics** (embedding-based or
LLM-based) against expert judgments of when two peer-review items are
about the same underlying concern.

- **164** (paper, review item A, review item B) tuples after a post-hoc
  label-quality review (see *Label-quality filter* below).
- Each pair has a binary `binary_label` (`similar` or `not_similar`)
  matching the annotator's implicit judgment, plus a four-category
  diagnostic `finegrained_label` from a manual audit:
  - `"same subject, same argument, same evidence"` — near-paraphrase
  - `"same subject, same argument, different evidence"` — convergent conclusion
  - `"same subject, different argument"` — topical neighbor
  - `"different subject"` — unrelated
- 70 similar (48 convergent + 22 near-paraphrase) + 94 not-similar
  (27 topical neighbors + 67 unrelated).
- 85 AI-AI / 79 AI-Human pairs.
- `paper_content` is inlined for self-contained baselines. `file_refs`
  is **not** inlined here — join on `paper_id` against the `reviewer`
  config if you need the supplementary code/data files.

**Label-quality filter**. The initial annotation yielded 238 pairs, but
a post-hoc review identified 74 pairs where both gpt-5.4 and
gemini-3.1-pro (each running a carefully tuned 4-way classification
prompt) disagreed with the ground truth label, and independent reviewer
agents reading the full item texts judged the ground truth label to be
wrong under a strict reading of the taxonomy or genuinely ambiguous at
the category boundary. The three dominant label-error patterns were:
(1) `c` (near-paraphrase) over-applied to pairs where one item had
substantively additional independent observations beyond elaboration of
a shared core; (2) `b` (convergent) over-applied to pairs whose two
items actually make different flaw-types about the same subject;
(3) `d` (different subject) over-applied to pairs that share a broad
subject but attack different aspects of it. The 74 dropped pairs are
listed in `upload_to_hf.py::DROP_PAIR_IDS_SIMILARITY_CHECK`.

Schema columns: `eval_pair_id`, `source_pair_id`, `paper_id`, `paper_title`,
`paper_content`, `item_a_reviewer_id`, `item_a_reviewer_type`,
`item_a_item_number`, `item_a_text`, `item_b_reviewer_id`,
`item_b_reviewer_type`, `item_b_item_number`, `item_b_text`,
`binary_label`, `finegrained_label`, `pair_type`, `rationale`,
`source_bucket`.

```python
# Loading example — paper_content is already inlined
from datasets import load_dataset
sim = load_dataset('prometheus-eval/peerreview-bench', 'similarity_check', split='eval')
for pair in sim:
    paper_content = pair['paper_content']
    text_a = pair['item_a_text']
    text_b = pair['item_b_text']
    # ... feed paper_content + the two item texts to your similarity metric
    # ground truth: pair['binary_label'] and pair['finegrained_label']

# If you also need code/supplementary files, join with the reviewer config:
papers = load_dataset('prometheus-eval/peerreview-bench', 'reviewer', split='eval')
paper_by_id = {r['paper_id']: r for r in papers}
file_refs = paper_by_id[some_pair['paper_id']]['file_refs']
```

### `submitted_papers`
Deduplicated blob storage for every file under each paper's `preprint/`
directory. One row per unique SHA256 hash.
- `content_hash`, `content_bytes` (binary), `size_bytes`, `is_text`.
- No per-file size cap — every file under `preprint/` is included
  (excluding `.DS_Store` and common metadata dirs).
- To look up a file: build a hash → bytes dict once, then index by the hashes
  in other configs' `file_refs` columns.

## Usage

```python
from datasets import load_dataset

# 1. AI reviewer evaluation
papers = load_dataset('prometheus-eval/peerreview-bench', 'reviewer', split='eval')
files  = load_dataset('prometheus-eval/peerreview-bench', 'submitted_papers', split='eval')
hash_to_bytes = {r['content_hash']: r['content_bytes'] for r in files}
for paper in papers:
    content_files = {ref['path']: hash_to_bytes[ref['content_hash']] for ref in paper['file_refs']}
    # Feed paper['paper_content'] and content_files to your AI reviewer...

# 2. AI meta-reviewer evaluation
items = load_dataset('prometheus-eval/peerreview-bench', 'meta_reviewer', split='eval')
# Each row's `label` is one of the 10 classes; `label_id` is 1..10.
# `review_item` is the free-form review text being meta-reviewed.

# 3. Analysis (human-vs-AI similarity, paired paper-level statistics)
rows = load_dataset('prometheus-eval/peerreview-bench', 'expert_annotation', split='eval')
# Rows for both primary and secondary annotators. Filter with
# rows.filter(lambda r: r['annotator_source'] == 'primary') if you
# want the primary set only.
```

## The `review_item` column

For both `expert_annotation` and `meta_reviewer`, each review item is a single
free-form `review_item` string that reads like a natural reviewer comment.

- For human reviewers, `review_item` is the reviewer's own prose as written.
- For AI reviewers, `review_item` is a merged version of the underlying
  structured markdown: the main point of criticism, followed by the
  evidence quotes and comments (with the `* Main point of criticism:`,
  `* Quote:`, `* Comment:`, and `* Evaluation criteria:` markup stripped),
  followed by any cited references.

## Validity rules (applied in expert_annotation and meta_reviewer)

Annotations follow a cascade: mark correctness first; mark significance only
if Correct; mark evidence only if at least Marginally Significant.

- **Rule 2**: Correct with no significance label → dropped entirely.
- **Rule 3**: Correct + (Marginally) Significant with no evidence label → dropped.
- **Rule 5**: Not Correct *but* a significance label was entered → sig and
  evidence are stripped (nulled).
- **Rule 6**: Correct + Not Significant *but* an evidence label was entered →
  evidence is stripped.

Significance is always 3-class: the original 4-option "Very Significant"
choice is merged into "Significant".

## License

CC-BY-4.0

### Citation Information
If you find the following model helpful, please consider citing our paper!

```bibtex
@article{kim2026limits,
  title={On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists},
  author={Kim, Seungone and Yoon, Dongkeun and Gashteovski, Kiril and Suk, Juyoung and Baek, Jinheon and Aggarwal, Pranjal and Wu, Ian and Zaverkin, Viktor and Petkoski, Spase and Schrider, Daniel R and others},
  journal={arXiv preprint arXiv:2605.20668},
  year={2026}
}
```
