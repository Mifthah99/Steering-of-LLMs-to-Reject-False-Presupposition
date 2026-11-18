"""Step 2b: balanced A/B multiple-choice prompts from the length-matched snippets (Section 5.6.3)."""
import pandas as pd
from transformers import AutoTokenizer
from .config import MODELS, DATA_DIR


def generate_mcq_prompts(m):
    model_id=MODELS[m]
    '''Reads the Snippets CSV produced by snippets.py, and generates a new CSV containing the MCQ prompts for both stances.'''

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    df_train = pd.read_csv(DATA_DIR / f"contrastive_snippets_{m}.csv")

    # NEUTRAL extraction-time system prompt that doesnt bias the model towards either stance
    sys_prompt = "You are a helpful assistant."
    print("Generating perfectly balanced MCQ pairs from matched snippets...")

    kept_queries = []
    prompts_sycophant = []
    prompts_skeptic = []
    skipped = 0

    for idx, row in df_train.iterrows():
        sycophant_snippet = str(row['snippet_sycophant']).strip()
        skeptic_snippet = str(row['snippet_skeptic']).strip()

        # Skip rows where snippet generation produced nothing usable 
        if (not sycophant_snippet or not skeptic_snippet
                or sycophant_snippet.lower() == 'nan' or skeptic_snippet.lower() == 'nan'):
            skipped += 1
            continue

        query = row['query_with_presupposition']

        # Alternating 50/50 split based on row index parity, so the model doesnt just learn "the sycophantic answer is always (A)" 
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

    output_path = DATA_DIR / f"prompts_{m}.csv"
    df_pairs.to_csv(output_path, index=False)
    print(f"Balanced dataset saved successfully to {output_path}! ({len(df_pairs)} pairs)")
