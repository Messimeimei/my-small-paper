# Reward Modeling for Scientific Writing Evaluation

Name: Scientific Writing Evaluation Datasets

Version: 0.1

Authors: Furkan Şahinuç, Subhabrata Dutta Iryna Gurevych (UKP Lab, Technical University of Darmstadt)


The components of this dataset are used in the experiments of the paper "[Reward Modeling for Scientific Writing Evaluation](https://arxiv.org/abs/2601.11374)".

If you utilize this repository and our work, please cite:

```bibtex
@misc{sahinuc2026reward,
    title       = {Reward Modeling for Scientific Writing Evaluation}, 
    author      = {Furkan \c{S}ahinu\c{c} and Subhabrata Dutta and Iryna Gurevych},
    year        = {2026},
    eprint      = {2601.11374},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url         = {https://arxiv.org/abs/2601.11374}, 
}
```

✉️ Contact person: [Furkan Şahinuç](mailto:furkan.sahinuc@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Introduction

Datasets proposed in this repository belongs to the manuscript named "Reward Modeling for Scientific Writing Evaluation". The datasets cover several distinct scientific writing tasks, including related work generation, review generation, novelty summarization, and paper revision. Each dataset instance is structured as a chat-style prompt consisting of a system prompt, a user query, evaluation criteria, and illustrative examples. Only related work generation and review generation included in the training data.

## Files

`final_reward_data.json`: Training data of reward models consisting of related work and review evaluation. Those datasets are combined from [1] and [2]. (Due to submission limitations, we include 1/2 subset of the dataset).

`prompted_novelty_data.json`: Test data including comparisons of novelty summaries. Data instances are based on [3].

`prompted_revision_data.json`: Test data including paper revision instances and instructions. Data instances are based on [4].

## Dataset Structure

Structure of the datasets are as follows:

```json
{
  "train": [
    {
      "task": "rev_util | rw_gen ",
      "aspect": "task specific evaluation aspect",
      "labels": "gold evaluation score",
      "score_sets": "task and aspect specific possible scoring set",
      "prompt": [{"role":  "system"}, {"content":  "system prompt"},
                 {"role":  "user"}, {"content":  "prompt including query, evaluation criteria, task examples and answer to be evaluated"}]
    }
  ],
  "test": [
    {
      "task": "rev_util | rw_gen | novelty_eval | revision_eval",
      "aspect": "task specific evaluation aspect",
      "labels": "gold evaluation score",
      "score_sets": "task and aspect specific possible scoring set",
      "prompt": [{"role":  "system"}, {"content":  "system prompt"},
                 {"role":  "user"}, {"content":  "prompt including query, evaluation criteria, task examples and answer to be evaluated"}]
    }
  ]
}
```

## Dataset Statistics

For each task, statistics of each evaluation aspect are given below.

| Related Work            | Train     | Test      | Scoring    |
|-------------------------|-----------|-----------|------------|
| Positioning Type        | 954       | 204       | 0-1        |
| Positioning Consistency | 2,822     | 605       | 0-1        |
| Coherence               | 4,890     | 1,048     | 0-1        |
| **Total**               | **8,666** | **1,857** | **10,523** |

| Review                   | Train      | Test      | Scoring    |
|--------------------------|------------|-----------|------------|
| Actionability            | 10,432     | 1,000     | 1-5        |
| Grounding Specificity    | 10,431     | 1,000     | 1-5        |
| Verifiability Extraction | 10,430     | 1,000     | 0-1        |
| Verifiability            | 8,323      | 788       | 1-5        |
| Helpfulness              | 10,430     | 1,000     | 1-5        |
| **Total**                | **50,046** | **4,788** | **54,834** |

| Novelty                  | Test   | Scoring |
|--------------------------|--------|---------|
| Coherence/Alignment      | 76     | 0-1     |
| **Total**                | **76** | **76**  |

| Revision    | Test      | Scoring   |
|-------------|-----------|-----------|
| Relatedness | 3,092     | 0-1       |
| Correctness | 3,092     | 0-1       |
| **Total**   | **6,184** | **6,184** |


## References

[1] Furkan Sahinuc et al. 2025. Expert Preference-based Evaluation of Automated Related Work Generation. Preprint, arXiv:2508.07955.

[2] Abdelrahman Sadallah et al. 2025. The Good, The Bad and The Constructive: Automatically Measuring Peer Review’s Utility for Authors. In EMNLP, pp. 28979–29009.

[3] Osama Mohammed Afzal et al. 2025. Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM  Assisted Feedback. Preprint, arXiv:2508.10795.

[4] Leane Jourdan et al. 2025. Identifying Reliable Evaluation Metrics for Scientific Text Revision. In ACL, pp. 6731–6756.
