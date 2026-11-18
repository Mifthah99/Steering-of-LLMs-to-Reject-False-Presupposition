"""Response generation with Hugging Face transformers, with or without steering (Sections 5.3, 6.4).

The same loop is used for:
  - the unsteered baseline   -> configs = [None]
  - steered runs             -> configs = [{"layers": [11], "multiplier": 2.3}, ...]
Every config's results are appended to the output CSV as soon as it finishes, and rows
that are already in the CSV are skipped, so the script can be restarted after a crash.
"""
import gc

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODELS, SYS_PROMPT, MAX_NEW_TOKENS, REPETITION_PENALTY, BATCH_SIZE
from .io_utils import append_results_to_csv, get_missing_rows
from .steering import load_vector_bank, register_steering_hooks

# Metadata copied from the input CSV into every result row (if present)
META_COLS = ["claim_id", "claim", "claim_veracity", "presupposition_level", "query_with_presupposition"]


def free_memory_map(headroom_gb=4):
    """How much of each GPU the weights may fill: what is free right now (other jobs may share the
    cards) minus some room for activations and the KV cache during generation."""
    if not torch.cuda.is_available():
        return None
    return {i: f"{max(0, torch.cuda.mem_get_info(i)[0] / 1024**3 - headroom_gb):.0f}GiB"
            for i in range(torch.cuda.device_count())}


def load_model(model_key):
    model_id = MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Left padding so the last prompt token of every row sits at index -1,
    # which is where the steering hook adds the vector during prefill.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 where supported, float16 on older GPUs (e.g. RTX 2080 Ti)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {model_id} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", max_memory=free_memory_map(), dtype=dtype, attn_implementation="sdpa"
    )
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, query):
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, prompts):
    """Greedy decoding for one batch of chat-formatted prompts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def run_generation(model_key, rows_df, output_csv, configs=(None,), batch_size=BATCH_SIZE):
    """Generate a response for every row in `rows_df`, once per steering config.

    configs: list of None (unsteered) and/or {"layers": [...], "multiplier": float}.
    """
    rows_df = rows_df.reset_index(drop=True)
    rows_df = rows_df[[c for c in META_COLS if c in rows_df.columns]]

    # Work out what is still missing before loading the model
    tasks = []
    for cfg in configs:
        layers = cfg["layers"] if cfg else None
        mult = cfg["multiplier"] if cfg else None
        todo = get_missing_rows(output_csv, rows_df, layers, mult)
        name = f"layers={layers} multiplier={mult}" if cfg else "unsteered"
        if todo.empty:
            print(f"[{name}] all {len(rows_df)} rows already in {output_csv} - skipping.")
        else:
            print(f"[{name}] {len(todo)} of {len(rows_df)} rows still need generation.")
            tasks.append((cfg, todo))

    if not tasks:
        print("Nothing left to generate.")
        return

    model, tokenizer = load_model(model_key)
    all_layers = [l for cfg, _ in tasks if cfg for l in cfg["layers"]]
    vector_bank = load_vector_bank(model_key, all_layers) if all_layers else {}

    try:
        for cfg, todo in tqdm(tasks, desc="Configs"):
            todo = todo.copy()
            todo["prompt"] = [build_prompt(tokenizer, q) for q in todo["query_with_presupposition"]]
            # Sort by length so each batch needs little padding
            todo = todo.loc[todo["prompt"].str.len().sort_values().index]

            handles = register_steering_hooks(model, vector_bank, cfg["layers"], cfg["multiplier"]) if cfg else []
            try:
                for start in tqdm(range(0, len(todo), batch_size), desc="Batches", leave=False):
                    batch = todo.iloc[start:start + batch_size].copy()
                    with torch.inference_mode():
                        batch["model_response"] = generate_batch(model, tokenizer, batch["prompt"].tolist())
                    if cfg:
                        batch["layers"] = str(cfg["layers"])
                        batch["multiplier"] = float(cfg["multiplier"])
                    # Save after every batch so little work is lost on a crash
                    append_results_to_csv(batch, output_csv)
            finally:
                for h in handles:
                    h.remove()
            torch.cuda.empty_cache()
    finally:
        print("Generation done - releasing model from GPU memory...")
        del model, vector_bank
        gc.collect()
        torch.cuda.empty_cache()

    print(f"All generation complete. Results in {output_csv}")

