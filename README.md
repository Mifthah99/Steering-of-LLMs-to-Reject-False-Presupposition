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
