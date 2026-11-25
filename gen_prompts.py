#generate contrastive mcq prompts from short, length-matched stance snippets

import pandas as pd
import os
from pathlib import Path
from transformers import AutoTokenizer
m="qwen"
if m=="qwen":

    model_id = "Qwen/Qwen2.5-14B-Instruct"  
else :
    model_id="meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Read the SNIPPETS file (produced by parallel_contrastive_snippets.py), not
# the raw training queries directly — we need the generated
# snippet_sycophant / snippet_skeptic columns to use as the literal MCQ
# answer options, matching how Anthropic's actual sycophancy datasets are
# structured (options ARE the candidate content, not a description of a
# behavior/strategy).
df_train = pd.read_csv(f"data/contrastive_snippets_{m}.csv")

# NEUTRAL extraction-time system prompt. Do NOT reuse the judge script's
# "You are an expert evaluator. You must evaluate the assistant's stance..."
# prompt here — that was the actual bug found earlier: it puts the model in
# an unrelated evaluator persona at the exact token position being
# extracted from, contaminating the hidden state with "I am judging
# someone else's response" rather than "I am directly answering a
# question," which is the persona active at real generation time.
sys_prompt = "You are a helpful assistant."

print("Generating perfectly balanced MCQ pairs from matched snippets...")

kept_queries = []
prompts_sycophant = []
prompts_skeptic = []
skipped = 0

for idx, row in df_train.iterrows():
    sycophant_snippet = str(row['snippet_sycophant']).strip()
    skeptic_snippet = str(row['snippet_skeptic']).strip()

    # Skip rows where snippet generation produced nothing usable on one
    # side (flagged as needs_review in parallel_contrastive_snippets.py) —
    # a degenerate pair here would inject noise straight into the vector.
    if (not sycophant_snippet or not skeptic_snippet
            or sycophant_snippet.lower() == 'nan' or skeptic_snippet.lower() == 'nan'):
        skipped += 1
        continue

    query = row['query_with_presupposition']

    # Alternating 50/50 split based on row index parity, so the model
    # doesn't learn "the sycophantic answer is always (A)" — only the
    # STANCE alternates sides; what's forced is the letter position, not
    # a fixed mapping between letter and stance.
    if idx % 2 == 0:
        user_mcq_string = f"""{query}

(A) {sycophant_snippet}
(B) {skeptic_snippet}

Answer: ("""
        sycophant_letter, skeptic_letter = "A", "B"
    else:
        user_mcq_string = f"""{query}

(A) {skeptic_snippet}
(B) {sycophant_snippet}

Answer: ("""
        sycophant_letter, skeptic_letter = "B", "A"

    base_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_mcq_string}],
        tokenize=False, add_generation_prompt=True
    )

    kept_queries.append(query)
    prompts_sycophant.append(base_prompt + sycophant_letter)  # Forced sycophant state
    prompts_skeptic.append(base_prompt + skeptic_letter)      # Forced skeptic state

if skipped:
    print(f"Skipped {skipped} row(s) with a missing/empty snippet on one side.")

# Save the dataset
df_pairs = pd.DataFrame({
    'user_query': kept_queries,
    'prompt_sycophant': prompts_sycophant,
    'prompt_skeptic': prompts_skeptic
})

output_path = f"data/prompts_{m}.csv"
df_pairs.to_csv(output_path, index=False)
print(f"Balanced dataset saved successfully to {output_path}! ({len(df_pairs)} pairs)")