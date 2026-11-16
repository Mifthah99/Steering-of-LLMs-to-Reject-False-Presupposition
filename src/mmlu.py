"""MMLU (5-shot, generative) with lm-evaluation-harness, with or without steering (Section 5.11).

The generative task (`mmlu_generative`) is used on purpose: the steering hook acts on the
last prompt token and on every generated token, so it has an effect while the model
writes its answer. The log-likelihood `mmlu` task would not be comparable.
"""
TASK = "mmlu_generative"
NUM_FEWSHOT = 5


def run_mmlu(model, tokenizer, limit=None):
    """Evaluate an already-loaded model (with any steering hooks already attached)."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=[TASK],
        num_fewshot=NUM_FEWSHOT,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        limit=limit,  # docs per subtask, for a quick test; None = full eval
    )


def get_accuracy(results):
    metrics = results.get("results", {}).get(TASK, {})
    return metrics.get("exact_match,get_response", metrics.get("exact_match"))
