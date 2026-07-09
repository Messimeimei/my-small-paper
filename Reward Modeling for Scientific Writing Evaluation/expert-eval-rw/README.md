# Expert Preference-based Evaluation of Automated Related Work Generation

Name: Related Work Generation Dataset with Main Text of Citing and Cited Papers

Version: 0.1

Authors: Furkan Şahinuç, Subhabrata Dutta Iryna Gurevych (UKP Lab, Technical University of Darmstadt)


The components of this dataset are used in the experiments of the paper "[Expert Preference-based Evaluation of Automated Related Work Generation](https://arxiv.org/abs/2508.07955)".

If you utilize this repository and our work, please cite:

```bibtex
@misc{sahinuc2025expertEval,
    title       = {Expert Preference-based Evaluation of Automated Related Work Generation}, 
    author      = {Furkan \c{S}ahinu\c{c} and Subhabrata Dutta and Iryna Gurevych},
    year        = {2025},
    eprint      = {2508.07955},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url         = {https://arxiv.org/abs/2508.07955}, 
}
```

✉️ Contact person: [Furkan Şahinuç](mailto:furkan.sahinuc@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Introduction

This dataset focuses on automatic related work generation. In particular, it includes contents from main papers of target related work sections along with content of papers that are cited in the corresponding related work section. 

For each main paper, the dataset includes detailed metadata, raw and clean version of the abstract, introduction and related work sections. We use `clean_numbered` version that are using numbers for citation marks in our experiments.

For each cited paper, the dataset includes abstract and introduction sections under the corresponding related work sections.

For further information regarding experiments please refer to the paper's [GitHub repository](https://github.com/UKPLab/arxiv2025-expert-eval-rw).

### Data Loading

```python
import json
data = json.load(open("final_rw_data.json", 'r'))
```

## Dataset Structure

```json
{
    "1607.04423": { //arxiv identifier of main paper
        "metadata": {...},
        "abstract": {
            "raw": "...",
            "clean": "..."
        },
        "introduction": {
            "raw": "...",
            "clean": "..."
        },
        "related work": {
            "raw": "...",
            "clean": "...",
            "clean_numbered": "..."
        },
        "cited_papers_in_rw": {
            "3883a54ad417208e144836940ece05e42a0fb09f": { // unarxive id
                "authors": [...],
                "title": "...",
                "venue": "...",
                "year": ...,
                "semantic_scholar_id": "...",
                "semantic_scholar_corpus_id": ...,
                "abstract": "...",
                "introduction": "..."
            },
            ...
        }
    },
    ...
    ...
}
```

## Dataset Statistics

| Dataset item                  | Counts |
|-------------------------------|--------|
| Main (citing) papers          | 44     |
| Total citation count in RW    | 644    |
| Avg. citation count per paper | 14.63  |

| Dataset item            | Avg. Token Counts |
|-------------------------|-------------------|
| Main paper abstract     | 190.77            |
| Main paper intro.       | 891.84            |
| Main paper related work | 454.20            |
| Cited paper abstract    | 181.07            |
| Cited paper intro.      | 781.75            |

