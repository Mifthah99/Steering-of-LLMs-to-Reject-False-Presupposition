"""Step 5: steering tax (Section 5.11).

    python 05_steering_tax.py --model llama

For the unsteered model and every config in CHOSEN_CONFIGS this measures:
  - perplexity on WikiText-2 (raw, test split), with the vector added at every token
  - 5-shot MMLU accuracy (generative version, lm-evaluation-harness), with the same hook as generation

The model is loaded once, so baseline and steered runs differ only in the hooks.
MMLU results are saved per config in generation_results/mmlu/ and reused on the next run
(delete a file to redo it). The final table goes to generation_results/{model}_steering_tax.csv.
"""
import argparse
import json
import logging

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import MODELS, CHOSEN_CONFIGS, RESULTS_DIR
from src.generators import load_model
from src.mmlu import get_accuracy, run_mmlu
from src.steering import get_steering_hook, load_vector_bank

# Tokenizing all of WikiText at once gives a harmless "sequence longer than max" warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

MAX_LENGTH = 4096   # tokens per window
STRIDE = 512        # how far the window moves each step


def full_steering_hook(vector, multiplier):
    """Adds the vector at EVERY position. When perplexity is scored in one forward pass,
    this is what steering every generated token looks like (the generation hook would
    only touch the last token of each window)."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden += multiplier * vector.to(hidden.device, hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    return hook


def perplexity(model, tokenizer, text):
    """Sliding-window perplexity: only the new tokens in each window are scored."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for start in tqdm(range(0, ids.size(1), STRIDE), desc="Perplexity"):
        end = min(start + MAX_LENGTH, ids.size(1))
        window = ids[:, start:end]
        targets = window.clone()
        targets[:, :-(end - prev_end)] = -100   # context tokens are not scored
        with torch.no_grad():
            loss = model(window, labels=targets).loss
        n = (targets != -100).sum().item()
        nll_sum += loss.item() * n
        n_tokens += n
        prev_end = end
        if end == ids.size(1):
            break
    return torch.exp(torch.tensor(nll_sum / n_tokens)).item()


def add_hooks(model, model_key, cfg, hook_fn, calls):
    """Attach the steering hook to each layer of `cfg` (nothing for the baseline).

    calls["n"] counts how often a hook actually ran, so we can check that steering
    really happened inside lm-eval / the perplexity loop.
    """
    if cfg is None:
        return []
    bank = load_vector_bank(model_key, cfg["layers"])

    def counted(hook):
        def wrapper(module, inputs, output):
            calls["n"] += 1
            return hook(module, inputs, output)
        return wrapper

    return [model.model.layers[l].register_forward_hook(counted(hook_fn(bank[l], cfg["multiplier"])))
            for l in cfg["layers"]]


def run_steered(model, model_key, cfg, hook_fn, fn):
    """Run fn() with the hooks attached, and fail loudly if a steered run never used them."""
    calls = {"n": 0}
    handles = add_hooks(model, model_key, cfg, hook_fn, calls)
    try:
        result = fn()
    finally:
        for h in handles:
            h.remove()
    if cfg is not None:
        if calls["n"] == 0:
            raise RuntimeError(f"Steering hook for {cfg} was never called - the result would be unsteered.")
        print(f"  steering hook ran {calls['n']} times")
    return result, calls["n"]


def load_cached_mmlu(path, cfg):
    """Reuse a saved MMLU result only if it was made with exactly this steering config."""
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    if results.get("steering", "missing") != cfg:
        print(f"  {path} was not made with steering={cfg} - running MMLU again")
        return None
    print(f"  using saved MMLU result {path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in wikitext["text"] if line.strip())

    rows = []
    for cfg in [None] + CHOSEN_CONFIGS[args.model]:
        name = "baseline" if cfg is None else "L" + "+".join(map(str, cfg["layers"])) + f"_M{cfg['multiplier']}"
        print(f"\n=== {name} ===")

        ppl, _ = run_steered(model, args.model, cfg, full_steering_hook,
                             lambda: perplexity(model, tokenizer, text))

        mmlu_file = RESULTS_DIR / "mmlu" / f"{args.model}_{name}.json"
        results = load_cached_mmlu(mmlu_file, cfg)
        if results is None:
            results, n_calls = run_steered(model, args.model, cfg, get_steering_hook,
                                           lambda: run_mmlu(model, tokenizer))
            results["steering"] = cfg          # so a saved file can be checked later
            results["hook_calls"] = n_calls
            mmlu_file.parent.mkdir(parents=True, exist_ok=True)
            mmlu_file.write_text(json.dumps(results, indent=2, default=str))

        rows.append({"config": name, "perplexity": ppl, "mmlu": get_accuracy(results)})
        print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"{args.model}_steering_tax.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
