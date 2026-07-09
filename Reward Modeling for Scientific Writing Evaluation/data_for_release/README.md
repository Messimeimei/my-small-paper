# ICLR Novelty Assessment Dataset

**Project Page:** [https://ukplab.github.io/arxiv2025-assessing-paper-novelty/](https://ukplab.github.io/arxiv2025-assessing-paper-novelty/)

## Overview
This dataset contains 182 ICLR paper submissions with human reviews and automated novelty assessments for research on automated novelty evaluation in academic papers.

## Dataset Structure
Each submission is organized in its own folder named by its OpenReview forum ID. The folder structure is as follows:

```
data_for_release/
├── {forum_id}/
│   ├── annotation.json              # Original reviews and metadata
│   ├── human_novelty_assessments/   # Novelty assessments derived from original reviews
│   │   └── review_{id}.txt          # Individual novelty assessment files
│   └── ours/                        # Our system's novelty analysis outputs
│       ├── novelty_delta_analysis.txt
│       ├── research_landscape.txt
│       ├── structured_representation.json
│       └── summary.txt
└── README.md                        # This file
```

## File Descriptions

### annotation.json
Contains the original review data and metadata for each submission with the following fields:
- `paper_id`: Unique identifier for the paper
- `input`: Input data for the paper
- `output`: List of review objects, each containing:
  - `review_id`: Unique identifier for the review
  - `review`: The full original human review text
  - `novelty_statements`: Extracted novelty-related statements from the original review. Labelled at sentence level where sentence segmentation is done using nltk sentence tokenizer.
  - Additional review metadata
- `publicationDate`: Date of publication
- `year`: Year of submission

### human_novelty_assessments/
This folder contains novelty assessments derived from the original human reviews by prompting an LLM. For submissions where human annotations are available (`annotation.json`), the annotations are provided alongside the review to help identify the novelty-relevant portions. The files are named `review_{id}.txt` where `{id}` corresponds to the `review_id` in the annotation.json file. Refer to the paper for prompt details and methodology.

### ours/
This folder contains our system's automated novelty analysis outputs:
- `novelty_delta_analysis.txt`: Analysis of the novelty delta (difference from prior work)
- `research_landscape.txt`: Overview of the research landscape and related work
- `structured_representation.json`: Structured JSON representation of the paper's contributions
- `summary.txt`: Summary of the novelty delta analysis

Refer to the paper for exact details about prompts and methodology.

### PDF Files
The pdf file can be fetched from OpenReview using the forum_id.

## Data Statistics
- **Total Submissions**: 182

## Access Links
Each submission can be accessed on OpenReview using the forum ID:
- Forum page: `https://openreview.net/forum?id={forum_id}`
- Direct PDF: `https://openreview.net/pdf?id={forum_id}`


## Data Fields Reference
When working with `annotation.json`:
- Original full review text: `output[i].review`
- Extracted novelty statements: `output[i].novelty_statements`
- Review ID for matching with assessment files: `output[i].review_id`
- Generated novelty assessment: `human_novelty_assessments/review_{review_id}.txt`

## Citation
```bibtex
@inproceedings{
afzal2026beyond,
title={Beyond {\textquotedblleft}Not Novel Enough{\textquotedblright}: Enriching Scholarly Critique with {LLM}-Assisted Feedback},
author={Osama Mohammed Afzal and Preslav Nakov and Tom Hope and Iryna Gurevych},
booktitle={19th Conference of the European Chapter of the Association for Computational Linguistics},
year={2026},
url={https://openreview.net/forum?id=VNB3bhnHhL}
}
```

## License
This dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). You are free to share and adapt this dataset for non-commercial purposes, provided you give appropriate credit.