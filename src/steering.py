"""Inference-time steering: load the saved .pt vectors and add them with forward hooks (Section 5.7)."""
import torch

from .config import MODELS, VECTOR_DIR


def vector_path(model_key, layer):
    """Path written by src/vectors.py for one layer."""
    model_name_safe = MODELS[model_key].split("/")[-1]
    return VECTOR_DIR / model_key / f"{model_name_safe}_layer_{layer}_caa.pt"


def load_vector_bank(model_key, layers):
    """Load the steering vector of every requested layer onto the CPU."""
    bank = {}
    for layer in sorted(set(layers)):
        path = vector_path(model_key, layer)
        if not path.exists():
            raise FileNotFoundError(f"Missing steering vector for layer {layer}: {path} "
                                    f"(run 02_build_steering_vectors.py first)")
        bank[layer] = torch.load(path, map_location="cpu", weights_only=True)
    return bank


def get_steering_hook(vector, multiplier):
    """Forward hook that adds `multiplier * vector` to a decoder layer's output.

    Prefill (seq_len > 1): only the last prompt token is steered. With left padding
    this is always position -1.
    Decoding (seq_len == 1): every newly generated token is steered.
    """
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        v = vector.view(1, 1, -1) if vector.dim() == 1 else vector
        v = v.to(device=hidden_states.device, dtype=hidden_states.dtype)

        seq_len = hidden_states.shape[1]
        if seq_len > 1:
            hidden_states[:, -1:, :] += (v * multiplier)
        else:
            hidden_states += (v * multiplier)

        if is_tuple:
            return (hidden_states,) + output[1:]
        return hidden_states
    return hook


def register_steering_hooks(model, vector_bank, layers, multiplier):
    """Attach one hook per layer. Returns the handles - call h.remove() on each when done."""
    handles = []
    for layer in layers:
        layer_module = model.model.layers[layer]
        handles.append(layer_module.register_forward_hook(get_steering_hook(vector_bank[layer], multiplier)))
    return handles
