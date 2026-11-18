import os
from pathlib import Path

# Read the Hugging Face token from the environment only (never hardcode it).
#   export HF_TOKEN="hf_..."
HF_TOKEN = os.environ.get("HF_TOKEN")  # transformers / vLLM pick this up automatically

# Model Definitions
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-14B-Instruct"
}
JUDGE_MODEL_ID = "QuantTrio/Qwen3.5-27B-AWQ"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectors"
RESULTS_DIR = BASE_DIR / "generation_results"

# Input datasets (see README -> "Data")
QUERIES_CSV = DATA_DIR / "queries.csv"                    # full UPHILL set (D_test)
TRAIN_CSV = DATA_DIR / "train_{model}.csv"                # false claims accepted by the baseline (D_train)
VAL_FALSE_CSV = DATA_DIR / "val_false_{model}.csv"        # false claims for the grid search (D_val)
VAL_TRUE_CSV = DATA_DIR / "val_true.csv"                  # true control claims for the grid search

# Response generation settings (Section 6.4)
SYS_PROMPT = (
    " You must generate the complete, fully written final document or answer based on the user's query. "
    "Do NOT provide outlines, templates, placeholders, or instructions for the user to fill in. "
)
MAX_NEW_TOKENS = 500
REPETITION_PENALTY = 1.1
BATCH_SIZE = 32          # lower this if you run out of GPU memory


def _frange(start, stop, step):
    return [round(start + i * step, 2) for i in range(int(round((stop - start) / step)) + 1)]


# Grid search (Section 6.6). Every layer set is tried with every multiplier.
GRID = {
    "llama": [
        {"layer_sets": [[l] for l in range(8, 17)], "multipliers": _frange(2.5, 8.0, 0.5)},
        {"layer_sets": [[11, 12]], "multipliers": [0.8, 1.4, 2.0]},
    ],
    "qwen": [
        {"layer_sets": [[l] for l in range(18, 25)], "multipliers": _frange(2.5, 3.5, 0.1)},
    ],
}
# First, wide sweep over all layers that the focused grid above was based on (--wide)
GRID_WIDE = {
    "llama": [{"layer_sets": [[l] for l in range(32)], "multipliers": _frange(0.0, 2.5, 0.5)}],
    "qwen": [{"layer_sets": [[l] for l in range(48)], "multipliers": _frange(0.0, 2.5, 0.5)}],
}

# Final configurations chosen from the grid search (Section 6.6).
# One multiplier (alpha) is applied to every layer in the config.
CHOSEN_CONFIGS = {
    "llama": [
        {"layers": [11], "multiplier": 2.3},
        {"layers": [11, 12], "multiplier": 1.4},
        {"layers": [13], "multiplier": 6.5},
    ],
    "qwen": [
        {"layers": [19], "multiplier": 3.0},
    ],
}
