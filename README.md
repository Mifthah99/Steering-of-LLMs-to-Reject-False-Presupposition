# Original thesis scripts

This branch holds the scripts as they were used during the thesis work on steering
large language models to reject false presuppositions. They are kept for reference.
For the cleaned-up, documented version of the pipeline, use the `main` branch.

| Script | Role |
|---|---|
| `parallel_con_snippets.py` | Generate stance snippets for the contrastive prompts |
| `gen_prompts.py` | Build contrastive multiple-choice prompts from the snippets |
| `store_steering_vectors.py` | Compute and save steering vectors |
| `base_run.py` | Unsteered baseline generation |
| `base_judge.py` | Judge generated responses |
| `full_pipeline_find_layer.py` | Layer and multiplier search |
| `parallel_true_find_layer.py` | Layer and multiplier search, true claims |
| `full_pipeline_chosen.py` | Evaluation of the chosen configurations |
| `compute_perplexity.py` | Perplexity on WikiText-2, with optional steering |
| `mmlu_full_eval.py` | MMLU evaluation, baseline and steered |

Notes

- Settings such as model choice, paths and GPU IDs are set inside each script; edit them before running.
- Set your Hugging Face token in the `HF_TOKEN` environment variable. No tokens are stored here.
- Input data and generated files are not included on this branch.

## Dataset and citation

These scripts use the **UPHILL** benchmark ([code and data](https://github.com/navreeetkaur/UPHILL), [paper](https://aclanthology.org/2024.findings-acl.850/)) by Navreet Kaur, Monojit Choudhury and Danish Pruthi.

UPHILL builds on the [PubHealth](https://arxiv.org/abs/2010.09926) and [Monant Medical Misinformation](https://dl.acm.org/doi/abs/10.1145/3477495.3531726) datasets. If you use this code or data, please cite all three, as the UPHILL authors request:

```bibtex
@inproceedings{kaur-etal-2024-evaluating,
    title = "Evaluating Large Language Models for Health-related Queries with Presuppositions",
    author = "Kaur, Navreet  and
      Choudhury, Monojit  and
      Pruthi, Danish",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.850/",
    doi = "10.18653/v1/2024.findings-acl.850",
    pages = "14308--14331"
}
```

```bibtex
@misc{kotonya2020explainable,
      title={Explainable Automated Fact-Checking for Public Health Claims}, 
      author={Neema Kotonya and Francesca Toni},
      year={2020},
      eprint={2010.09926},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@inproceedings{srba2022monant,
  title={Monant medical misinformation dataset: Mapping articles to fact-checked claims},
  author={Srba, Ivan and Pecher, Branislav and Tomlein, Matus and Moro, Robert and Stefancova, Elena and Simko, Jakub and Bielikova, Maria},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2949--2959},
  year={2022}
}
```
