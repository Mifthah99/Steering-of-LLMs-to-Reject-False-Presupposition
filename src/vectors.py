"""Step 2c: layer-wise CAA steering vectors = mean(skeptic) - mean(sycophant) hidden state (Sections 5.6.3, 5.6.4)."""
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MODELS, DATA_DIR, VECTOR_DIR as VECTOR_ROOT


def generate_steering_vector(model):
    """Generates and stores a steering vector for each layer for the given model, based on the prompts generated in prompts.py."""
    model_id = MODELS[model]

    prompts = f"{DATA_DIR}/prompts_{model}.csv"
    VECTOR_DIR = str(VECTOR_ROOT / model)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    BATCH_SIZE = 64        
    DTYPE = torch.bfloat16
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding: the last real token is found with attention_mask.sum(dim=1) - 1 below
    tokenizer.padding_side = "right"
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")
    assert tokenizer.padding_side == "right", "Extraction indexing assumes right-padding — fix or update the indexing logic"

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    model.eval()

    df = pd.read_csv(prompts)


    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    vector_sums = {
        layer: torch.zeros(hidden_size, device=model.device)
        for layer in range(num_layers)
    }

    hidden_states_cache = {}
    current_attention_mask = None
    # extract hidden states at the final non-padding token for each layer, for both stances, and accumulate the differences

    def get_hook(layer_idx):
        def hook(module, inputs, output):

            hidden = output[0] if isinstance(output, tuple) else output

            # Index of final non-padding token
            # (.to(...) because with several GPUs each layer can live on a different card)
            lengths = (current_attention_mask.sum(dim=1) - 1).to(hidden.device)

            batch_indices = torch.arange(
                hidden.size(0),
                device=hidden.device
            )

            hidden_states_cache[layer_idx] = hidden[
                batch_indices,
                lengths,
                :
            ].detach().to(model.device)

        return hook


    handles = []

    for layer in range(num_layers):
        handles.append(
            model.model.layers[layer].register_forward_hook(
                get_hook(layer)
            )
        )
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

            # Forward pass A
            
            hidden_states_cache.clear()
            current_attention_mask = inputs_A["attention_mask"]
            model(**inputs_A)
            states_A = {
                k: v.clone()
                for k, v in hidden_states_cache.items()
            }
            
            # Forward pass B
            
            hidden_states_cache.clear()
            current_attention_mask = inputs_B["attention_mask"]
            model(**inputs_B)
            states_B = {
                k: v.clone()
                for k, v in hidden_states_cache.items()
            }

            # Accumulate batch differences
            
            for layer in range(num_layers):
                # (batch, hidden_size)
                diff = states_B[layer] - states_A[layer]

                # Sum over batch
                vector_sums[layer] += diff.sum(dim=0)

            print(f"Processed {end}/{len(df)}")

    
    # Remove Hooks
   
    for h in handles:
        h.remove()
 
    # Save Steering Vectors

    print("Saving steering vectors...")
    model_name_safe = model_id.split("/")[-1]
    for layer in range(num_layers):
        mean_vector = vector_sums[layer] / len(df)
        save_path = os.path.join(
            VECTOR_DIR,
            f"{model_name_safe}_layer_{layer}_caa.pt"
        )
        torch.save(mean_vector.cpu(), save_path)
    print("vectors saved successfully at "+VECTOR_DIR)
