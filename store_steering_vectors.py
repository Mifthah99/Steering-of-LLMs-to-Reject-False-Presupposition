import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ============================================================
# Configuration
# ============================================================

models = ["qwen", "llama"]
model_choice = models[1]  # "qwen" or "llama"

DATA_PATH = f"data/prompts_{model_choice}.csv"
VECTOR_DIR = f"vectors/{model_choice}"

os.makedirs(VECTOR_DIR, exist_ok=True)

BATCH_SIZE = 64         # Try 16 first on an L4. Increase to 32 if memory allows.
DTYPE = torch.bfloat16

# ============================================================
# GPU Information
# ============================================================

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ============================================================
# HuggingFace Login
# ============================================================

hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

# ============================================================
# Model Selection
# ============================================================

if model_choice == "qwen":
    MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
    MODEL_PATH = MODEL_ID
else:
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = MODEL_ID

print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Llama has no pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer padding_side: {tokenizer.padding_side}")
assert tokenizer.padding_side == "right", "Extraction indexing assumes right-padding — fix or update the indexing logic"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
)

model.eval()

# ============================================================
# Dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

vector_sums = {
    layer: torch.zeros(hidden_size, device=model.device)
    for layer in range(num_layers)
}

# ============================================================
# Hidden State Hooks
# ============================================================

hidden_states_cache = {}
current_attention_mask = None


def get_hook(layer_idx):
    def hook(module, inputs, output):

        hidden = output[0] if isinstance(output, tuple) else output

        # Index of final non-padding token
        lengths = current_attention_mask.sum(dim=1) - 1

        batch_indices = torch.arange(
            hidden.size(0),
            device=hidden.device
        )

        hidden_states_cache[layer_idx] = hidden[
            batch_indices,
            lengths,
            :
        ].detach()

    return hook


handles = []

for layer in range(num_layers):
    handles.append(
        model.model.layers[layer].register_forward_hook(
            get_hook(layer)
        )
    )

# ============================================================
# Extraction
# ============================================================

print("Extracting hidden states...")

with torch.inference_mode():

    for start in range(0, len(df), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(df))

        batch = df.iloc[start:end]

        prompts_A = batch["prompt_sycophant"].tolist()
        prompts_B = batch["prompt_skeptic"].tolist()

        inputs_A = tokenizer(
            prompts_A,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        inputs_B = tokenizer(
            prompts_B,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # -----------------------------
        # Forward pass A
        # -----------------------------
        hidden_states_cache.clear()

        current_attention_mask = inputs_A["attention_mask"]

        model(**inputs_A)

        states_A = {
            k: v.clone()
            for k, v in hidden_states_cache.items()
        }

        # -----------------------------
        # Forward pass B
        # -----------------------------
        hidden_states_cache.clear()

        current_attention_mask = inputs_B["attention_mask"]

        model(**inputs_B)

        states_B = {
            k: v.clone()
            for k, v in hidden_states_cache.items()
        }

        # -----------------------------
        # Accumulate batch differences
        # -----------------------------
        for layer in range(num_layers):

            # (batch, hidden_size)
            diff = states_B[layer] - states_A[layer]

            # Sum over batch
            vector_sums[layer] += diff.sum(dim=0)

        print(f"Processed {end}/{len(df)}")

# ============================================================
# Remove Hooks
# ============================================================

for h in handles:
    h.remove()

# ============================================================
# Save Mean Vectors
# ============================================================

print("Saving vectors...")

model_name_safe = MODEL_ID.split("/")[-1]

for layer in range(num_layers):

    mean_vector = vector_sums[layer] / len(df)

    save_path = os.path.join(
        VECTOR_DIR,
        f"{model_name_safe}_layer_{layer}_caa.pt"
    )

    torch.save(mean_vector.cpu(), save_path)

print("Done!")