"""
compute_perplexity.py

Sliding-window perplexity on WikiText-2 (raw) for a causal LM,
with an optional CAA steering hook applied during the forward pass.

Usage:
    python compute_perplexity.py --model meta-llama/Llama-3.1-8B-Instruct
    python compute_perplexity.py --model meta-llama/Llama-3.1-8B-Instruct \
        --steering_vector_path vectors/llama_layer12.pt --layer 12 --alpha 4.0

Notes:
- Uses the standard sliding-window perplexity protocol (same one used by the
  HF "perplexity of fixed-length models" guide): concatenate the whole test
  split into one long token stream, then slide a fixed-size window over it
  with a stride, only scoring the "new" tokens in each window (the rest is
  context). This avoids the bias of just chunking the text into independent
  blocks.
- This intentionally mirrors your existing generation pipeline: same model
  loading pattern, same left-padding-safe tokenizer setup, and a hook
  attachment point that should match how you already register hooks for
  full_pipeline_chosen.py / full_pipeline_find_layer.py. Swap in your own
  register_steering_hook() import if you already have one implemented.
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The initial tokenizer(text, ...) call tokenizes the *entire* WikiText-2
# split (hundreds of thousands of tokens) before the sliding-window loop
# below chunks it into max_length-sized windows. transformers logs a
# "sequence length longer than max" warning at that point, but it's a false
# alarm here -- the full tensor is never passed to the model directly, only
# max_length-sized slices are. Silence just this one warning.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def load_wikitext2_text(split: str = "test") -> str:
    """Load WikiText-2-raw and join into a single text blob.

    Note: the old no-namespace "wikitext" repo on the Hub is legacy and
    raises huggingface_hub.errors.HfUriError on recent `datasets` versions,
    which require a "namespace/name" repo id. Salesforce/wikitext is the
    actively maintained mirror with identical configs and splits.
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    # Join non-empty lines; WikiText has a lot of blank lines / headers.
    text = "\n\n".join([line for line in ds["text"] if line.strip() != ""])
    return text


def make_steering_hook(vector: torch.Tensor, alpha: float):
    """
    Returns a forward hook matching the same prefill/decoding logic
    described in your methodology (Section 4.4): during prefill (seq_len > 1)
    only the last token position is steered; during decoding (seq_len == 1)
    every generated token is steered.

    IMPORTANT CAVEAT FOR PERPLEXITY: teacher-forced scoring feeds the whole
    window in one pass (seq_len == max_length throughout, never 1), so this
    hook's prefill branch fires every time and only the single last token
    of each window gets steered -- roughly 1 in `max_length` scored tokens.
    That's correct behavior for matching your *generation* pipeline, but it
    means this hook alone will barely move perplexity even if steering has
    a real effect on multi-token generation. Use make_full_steering_hook()
    below if you want to test what happens when every position is steered
    (i.e. mimicking what actually happens across a real decode sequence).

    Attach with: layer_module.register_forward_hook(hook)
    where layer_module = model.model.layers[l]
    """
    vector = vector.to(dtype=torch.float16)

    def hook(module, inputs, output):
        # output is usually a tuple; hidden_states is output[0]
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        seq_len = hidden_states.shape[1]
        v = vector.to(hidden_states.device)

        if seq_len > 1:
            # Prefill: steer only the final token slice
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + alpha * v
        else:
            # Decoding: steer every new token
            hidden_states[:, :, :] = hidden_states[:, :, :] + alpha * v

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    return hook


def make_full_steering_hook(vector: torch.Tensor, alpha: float):
    """
    Steers EVERY token position regardless of seq_len. Use this for the
    perplexity check that actually tests the "steering tax" claim in RQ3 --
    it approximates what your intervention does across a real multi-token
    decode sequence (where every generated token gets steered), rather than
    the near-inert last-token-only effect that the standard hook has under
    single-pass teacher forcing.
    """
    vector = vector.to(dtype=torch.float16)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        v = vector.to(hidden_states.device)
        hidden_states[:, :, :] = hidden_states[:, :, :] + alpha * v

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    return hook


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = 4096,
    stride: int = 512,
):
    """
    Standard sliding-window perplexity.
    max_length: the model's context window to use per forward pass.
    stride: how far the window moves each step; smaller stride = more
            overlap = more accurate but slower. stride = max_length gives
            no overlap (faster, slightly less accurate).
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids.to(device)
    seq_len = input_ids_full.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # number of NEW tokens to score this step

        input_ids = input_ids_full[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        # Mask out the context tokens (everything except the last trg_len tokens)
        # so loss is only computed on the new portion of the window.
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # outputs.loss is the mean NLL per (non-masked) token in this window
            neg_log_likelihood = outputs.loss

        # Number of tokens actually scored in this window
        num_valid = (target_ids != -100).sum().item()
        nll_sum += neg_log_likelihood.item() * num_valid
        n_tokens += num_valid

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    return ppl, n_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # Steering args -- omit these three to just get the baseline (unsteered) PPL
    parser.add_argument("--steering_vector_path", default=None,
                         help="Path to a saved steering vector tensor (.pt)")
    parser.add_argument("--layer", type=int, default=None,
                         help="Layer index to attach the steering hook to")
    parser.add_argument("--alpha", type=float, default=None,
                         help="Steering multiplier")
    parser.add_argument("--hook_mode", default="full", choices=["generation_matched", "full"],
                         help="'generation_matched' reuses the exact prefill/decode hook from "
                              "your pipeline (steers ~1 token per window under teacher forcing -- "
                              "near-inert, use only as a diagnostic). 'full' (default) steers every "
                              "token position, which is what actually tests RQ3's steering-tax claim.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device
    )
    model.eval()

    print(f"Loading WikiText-2 ({args.split} split)")
    text = load_wikitext2_text(args.split)

    handle = None
    if args.steering_vector_path is not None:
        assert args.layer is not None and args.alpha is not None, \
            "Must provide --layer and --alpha alongside --steering_vector_path"
        print(f"Attaching steering hook at layer {args.layer}, alpha={args.alpha}")
        vector = torch.load(args.steering_vector_path, map_location=device)
        if args.hook_mode == "full":
            hook_fn = make_full_steering_hook(vector, args.alpha)
        else:
            hook_fn = make_steering_hook(vector, args.alpha)
        target_layer = model.model.layers[args.layer]
        handle = target_layer.register_forward_hook(hook_fn)

    try:
        ppl, n_tokens = compute_perplexity(
            model, tokenizer, text, device,
            max_length=args.max_length, stride=args.stride,
        )
    finally:
        if handle is not None:
            handle.remove()

    label = "STEERED" if args.steering_vector_path else "BASELINE"
    print(f"\n[{label}] Model: {args.model}")
    print(f"[{label}] Tokens scored: {n_tokens}")
    print(f"[{label}] Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
