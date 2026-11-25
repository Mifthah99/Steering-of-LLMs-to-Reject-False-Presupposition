# Steering LLMs to Reject False Presuppositions

Code for my Master's thesis **"Steering Large Language Models to Reject False Presuppositions"** (University of Stuttgart).

LLMs often go along with false assumptions hidden in a question ("Since vaccines weaken immunity, how should I…"). This repo applies **Contrastive Activation Addition (CAA)** to `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen2.5-14B-Instruct` on the medical **UPHILL** benchmark, so that the models push back on false presuppositions without any change to their weights. An LLM judge, `QuantTrio/Qwen3.5-27B-AWQ`, labels each answer AGREE / DISAGREE / NEUTRAL / GIBBERISH.

## Setup

```bash
git clone https://github.com/Mifthah99/Steering-of-LLMs-to-Reject-False-Presupposition.git
cd Steering-of-LLMs-to-Reject-False-Presupposition
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"   # Llama 3.1 is a gated model
```

## Running the experiments

There is one script per step of the thesis. Each one only needs the model name, and its defaults are the settings used in the thesis.

```bash
python 01_run_baseline.py           --model llama   # unsteered answers + judge
python 02_build_steering_vectors.py --model llama   # snippets -> MCQ prompts -> one vector per layer
python 03_run_grid_search.py        --model llama   # try layers x multipliers, rank configs
python 04_evaluate_chosen.py        --model llama   # best configs on the full dataset
python 05_steering_tax.py           --model llama   # perplexity + MMLU, with and without steering
```

Replace `llama` with `qwen` for the other model.

| Step | Thesis | Output in `generation_results/` |
|---|---|---|
| 1 Baseline | 5.3, 5.4 | `{model}_baseline_judged.csv` |
| 2 Vectors | 5.6 | `vectors/{model}/` (and checkpoints in `data/`) |
| 3 Grid search | 5.8, 6.6 | `{model}_grid_summary.csv` |
| 4 Full evaluation | 5.9 | `{model}_chosen_judged.csv` |
| 5 Steering tax | 5.11 | `{model}_steering_tax.csv` |

All thesis settings live in [`src/config.py`](src/config.py). That covers the grid of layers and multipliers, the chosen configs (Llama: L11 α=2.3, L11+12 α=1.4, L13 α=6.5; Qwen: L19 α=3.0), the system prompt and the generation settings (greedy decoding, 500 new tokens, repetition penalty 1.1).

### A few extra switches

- `--stage generate` or `--stage judge` (steps 1, 3 and 4) runs only one half. This is handy when generation and judging happen on different machines.
- `--shard 1` / `--shard 2` (steps 1, 3 and 4) judges half of the rows each, so two GPUs can share the judging.
- `--wide` (step 3) runs the first broad sweep over all layers with multipliers 0–2.5. The focused grid was chosen from this sweep.
- `TARGET_LAYERS` splits a sweep over GPUs, for example:
  ```bash
  CUDA_VISIBLE_DEVICES=0 TARGET_LAYERS="[[8],[9],[10],[11]]" python 03_run_grid_search.py --model llama --stage generate
  CUDA_VISIBLE_DEVICES=1 TARGET_LAYERS="[[12],[13],[14],[15],[16],[11,12]]" python 03_run_grid_search.py --model llama --stage generate
  ```

If a run crashes, start the same command again. Rows that are already in the output file are skipped.

**Shared servers.** Give a script several GPUs (`CUDA_VISIBLE_DEVICES=0,2`) and the model is spread over them. The vLLM steps (snippets, judge) only take the memory that is actually free, so they still start when other people's jobs share the cards. Run `export CUDA_DEVICE_ORDER=PCI_BUS_ID` first. Otherwise CUDA numbers the GPUs fastest-first, and the numbers will not match `nvidia-smi` on servers with mixed cards.

## Data

| File | Contents |
|---|---|
| `data/queries.csv` | UPHILL: each claim at five presupposition levels |
| `data/train_{model}.csv` | False-claim prompts the baseline accepted, used to build the vectors |
| `data/val_false_{model}.csv` | Other accepted false-claim prompts, used for the grid search |
| `data/val_true.csv` | 80 true-claim prompts (16 claims × 5 levels), used for the grid search |

## Dataset and citation

This work uses the **UPHILL** benchmark ([code and data](https://github.com/navreeetkaur/UPHILL), [paper](https://aclanthology.org/2024.findings-acl.850/)) by Navreet Kaur, Monojit Choudhury and Danish Pruthi. The files in `data/` are derived from it.

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

The datasets remain subject to their creators' terms; this repository's `LICENSE` does not cover them.

## Code layout

```
01_… to 05_….py     one script per step
src/
  config.py       models, paths and every thesis setting
  snippets.py     sycophantic / skeptical one-sentence snippets, cut to equal length
  prompts.py      balanced (A)/(B) multiple-choice prompts
  vectors.py      mean difference of hidden states -> one .pt vector per layer
  steering.py     loads vectors and adds them with a forward hook
  generators.py   answers with or without steering
  judge.py        the LLM judge (vLLM; plain PyTorch on older GPUs)
  metrics.py      accuracy and balanced score
  mmlu.py         MMLU through lm-evaluation-harness
  io_utils.py     safe CSV appends from several processes, resume logic
```

## Implementation details

- **Extraction** pads on the right and reads the last real token. **Generation** pads on the left, so the last prompt token is always at position −1.
- **The steering hook** adds α·v to the last prompt token, then to every generated token.
- **The judge** runs in its own process, so its GPU memory is freed afterwards. It may think for up to 1000 tokens. Answers that hit that limit are judged again with 6000 tokens.
