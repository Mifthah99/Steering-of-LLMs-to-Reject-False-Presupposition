import os
import pandas as pd
import multiprocessing as mp
from pathlib import Path
m='qwen'
# ==========================================
# 1. WORKER FUNCTION (Runs on each GPU)
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    # raise RuntimeError("Set the HF_TOKEN environment variable before running this script.")
    HF_TOKEN = os.environ.get("HF_TOKEN")
def process_shard_on_gpu(shard_df, gpu_id=0):
    from vllm import LLM, SamplingParams

    print(f"[GPU {gpu_id}] Starting... processing {len(shard_df)} rows.")
    if m=="qwen":

        model_id = "Qwen/Qwen2.5-14B-Instruct"  
    else :
        model_id="meta-llama/Llama-3.1-8B-Instruct"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_seqs=1056,
        max_model_len=600,
        enforce_eager=True, 
    )
    tokenizer = llm.get_tokenizer()
    raw_queries = shard_df['query_with_presupposition'].astype(str).tolist()

    # ==========================================
    # SHORT, MATCHED-LENGTH CONTRASTIVE SNIPPETS
    # ==========================================
    # These are used as literal MCQ answer OPTIONS during steering-vector
    # extraction (gen_prompts.py) — NOT as full free-text responses. CAA's
    # signal isolation depends on the two options differing in as little as
    # possible besides the target stance (matched length/topic/structure),
    # so we ask for exactly one short, direct sentence per side instead of
    # a full report, then length-match each pair in post-processing below.
    state_a_sys = (
        "You are a sycophantic assistant. Respond with EXACTLY ONE short sentence "
        "(15-25 words) that directly and confidently agrees with and validates the "
        "premise embedded in the user's question, as if it were an established fact. "
        "Do not hedge, qualify, or explain — state it as a confident fact in one sentence only. "
        "Do not write a report, list, or multiple sentences."
    )
    state_b_sys = (
        "You are a skeptical fact-checking assistant. Respond with EXACTLY ONE short "
        "sentence (15-25 words) that directly and confidently corrects the false premise "
        "embedded in the user's question. Do not hedge or add caveats — state the "
        "correction as a confident fact in one sentence only. "
        "Do not write a report, list, or multiple sentences."
    )

    # Short on purpose: this is a snippet, not a response. max_tokens is a
    # generous ceiling (not a target) — the stop sequence does the real work
    # of enforcing single-sentence output.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=60,
        repetition_penalty=1.1,
        stop=["\n\n", "\n"],
    )

    def get_prompt(sys_msg, user_msg):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True
        )

    prompts_a = [get_prompt(state_a_sys, q) for q in raw_queries]
    prompts_b = [get_prompt(state_b_sys, q) for q in raw_queries]

    # ==========================================
    # GENERATION
    # ==========================================
    print(f"[GPU {gpu_id}] Generating State A (Sycophant) snippets...")
    outputs_a = llm.generate(prompts_a, sampling_params)

    print(f"[GPU {gpu_id}] Generating State B (Skeptic) snippets...")
    outputs_b = llm.generate(prompts_b, sampling_params)

    raw_a = [out.outputs[0].text.strip() for out in outputs_a]
    raw_b = [out.outputs[0].text.strip() for out in outputs_b]

    # ==========================================
    # LENGTH MATCHING
    # ==========================================
    # Truncate each pair to the SHORTER of the two, at a whole-word boundary,
    # so raw length difference doesn't leak into the steering vector as a
    # confound alongside stance. This directly implements the "matched for
    # length, topic, and syntactic structure" requirement from the CAA method.
    def truncate_to_word_count(text, n_words):
        words = text.split()
        return " ".join(words[:n_words])

    matched_a, matched_b = [], []
    needs_review = []

    for idx, (text_a, text_b) in enumerate(zip(raw_a, raw_b)):
        words_a, words_b = len(text_a.split()), len(text_b.split())

        if words_a == 0 or words_b == 0:
            # Model returned nothing before hitting the stop sequence (e.g.
            # started with a blank line). Flag rather than silently produce
            # an empty/degenerate contrastive pair.
            needs_review.append(idx)
            matched_a.append(text_a)
            matched_b.append(text_b)
            continue

        n_words = min(words_a, words_b)
        matched_a.append(truncate_to_word_count(text_a, n_words))
        matched_b.append(truncate_to_word_count(text_b, n_words))

    if needs_review:
        print(f"[GPU {gpu_id}] WARNING: {len(needs_review)} row(s) produced an empty snippet "
              f"on one side and were left unmatched — inspect these rows directly "
              f"(indices within this shard: {needs_review[:20]}{'...' if len(needs_review) > 20 else ''}).")

    shard_df = shard_df.copy()
    # Raw (unmatched) snippets kept for debugging/inspection
    shard_df['snippet_sycophant_raw'] = raw_a
    shard_df['snippet_skeptic_raw'] = raw_b
    # Length-matched snippets — these are what gen_prompts.py should use as MCQ options
    shard_df['snippet_sycophant'] = matched_a
    shard_df['snippet_skeptic'] = matched_b

    return shard_df


# ==========================================
# 2. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(f"data/train_{m}.csv")  # replace with your actual path

    import torch
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    NUM_GPUS = 1  # gpu_count if gpu_count > 0 else 1
    pdf = process_shard_on_gpu(df)
    final_df = pdf

    save_path = f"data/contrastive_snippets_{m}.csv"
    final_df.to_csv(save_path, index=False)

    print(f"Success! Snippets saved to {save_path}")
    print(f"Median snippet length (words): "
          f"sycophant={final_df['snippet_sycophant'].str.split().str.len().median():.0f}, "
          f"skeptic={final_df['snippet_skeptic'].str.split().str.len().median():.0f}")