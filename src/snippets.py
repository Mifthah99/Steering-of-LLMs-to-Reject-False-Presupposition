"""Step 2a: contrastive sycophant / skeptic snippets with vLLM + whole-word length matching (Sections 5.6.1, 5.6.2)."""
import pandas as pd
from .config import MODELS, DATA_DIR, TRAIN_CSV


def generate_snippets(model):
    """Generates sycophantic and skeptical snippets for each query in the input CSV, using vLLM for efficient GPU inference."""
    # Imported here so the rest of the package can be used without vLLM installed
    from vllm import LLM, SamplingParams
    input_df = pd.read_csv(str(TRAIN_CSV).format(model=model))
    print(f" Starting... processing {len(input_df)} rows.")
    model_id = MODELS[model]
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_seqs=1056,
        max_model_len=600,
        enforce_eager=True, 
    )
    tokenizer = llm.get_tokenizer()
    raw_queries = input_df['query_with_presupposition'].astype(str).tolist()

   
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
    print(f"Generating State A (Sycophant) snippets...")
    outputs_a = llm.generate(prompts_a, sampling_params)

    print(f"Generating State B (Skeptic) snippets...")
    outputs_b = llm.generate(prompts_b, sampling_params)

    raw_a = [out.outputs[0].text.strip() for out in outputs_a]
    raw_b = [out.outputs[0].text.strip() for out in outputs_b]

    # Truncate each pair to the SHORTER of the two, at a whole-word boundary, so that the model doesn't learn to associate length with a particular state.
    def truncate_to_word_count(text, n_words):
        words = text.split()
        return " ".join(words[:n_words])

    matched_a, matched_b = [], []
    needs_review = []

    for idx, (text_a, text_b) in enumerate(zip(raw_a, raw_b)):
        words_a, words_b = len(text_a.split()), len(text_b.split())

        if words_a == 0 or words_b == 0:
            # Model returned nothing before hitting the stop sequence
            needs_review.append(idx)
            matched_a.append(text_a)
            matched_b.append(text_b)
            continue

        n_words = min(words_a, words_b)
        matched_a.append(truncate_to_word_count(text_a, n_words))
        matched_b.append(truncate_to_word_count(text_b, n_words))

    if needs_review:
        print(f"WARNING: {len(needs_review)} row produced an empty snippet "
              f"on one side and were left unmatched — inspect these rows directly "
              f"(indices within this shard: {needs_review[:20]}{'...' if len(needs_review) > 20 else ''}).")

    input_df = input_df.copy()
    # Raw (unmatched) snippets kept for debugging/inspection
    input_df['snippet_sycophant_raw'] = raw_a
    input_df['snippet_skeptic_raw'] = raw_b
    # Length-matched snippets — these are what gen_prompts.py should use as MCQ options
    input_df['snippet_sycophant'] = matched_a
    input_df['snippet_skeptic'] = matched_b
    save_path = f"{DATA_DIR}/contrastive_snippets_{model}.csv"
    input_df.to_csv(save_path, index=False)
    print(f"Success! Snippets saved to {save_path}")
    print(f"Median snippet length (words): "
        f"sycophant={input_df['snippet_sycophant'].str.split().str.len().median():.0f}, "
        f"skeptic={input_df['snippet_skeptic'].str.split().str.len().median():.0f}")

    return input_df
