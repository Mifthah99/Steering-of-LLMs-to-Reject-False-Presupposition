"""Step 2: steering vectors (Section 5.6).

    python 02_build_steering_vectors.py --model llama

Three sub-steps, each saving its output so it can be checked:
    1. snippets  data/train_{model}.csv -> data/contrastive_snippets_{model}.csv
    2. prompts   -> data/prompts_{model}.csv (balanced A/B multiple-choice prompts)
    3. vectors   -> vectors/{model}/*_layer_{l}_caa.pt (one vector per layer)

The snippet file is reused if it already exists; delete it to generate new snippets.
"""
import argparse
import multiprocessing as mp

from src.config import MODELS, DATA_DIR
from src.prompts import generate_mcq_prompts
from src.snippets import generate_snippets
from src.vectors import generate_steering_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()

    snippets = DATA_DIR / f"contrastive_snippets_{args.model}.csv"
    if snippets.exists():
        print(f"Using existing {snippets}")
    else:
        # vLLM runs in its own process so its GPU memory is freed before the extraction below
        p = mp.get_context("spawn").Process(target=generate_snippets, args=(args.model,))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Snippet generation failed - see the log above.")

    generate_mcq_prompts(args.model)
    generate_steering_vector(args.model)


if __name__ == "__main__":
    main()
