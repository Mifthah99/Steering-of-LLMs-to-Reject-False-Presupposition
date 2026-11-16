"""Step 4: full evaluation of the chosen configs (Section 5.9).

    python 04_evaluate_chosen.py --model llama

Runs every config in CHOSEN_CONFIGS (src/config.py) on all false- and true-claim prompts:
    llama: layer 11 (alpha 2.3), layers 11+12 (alpha 1.4), layer 13 (alpha 6.5)
    qwen:  layer 19 (alpha 3.0)

Writes generation_results/{model}_chosen.csv and {model}_chosen_judged.csv.
"""
import argparse

import pandas as pd

from src.config import MODELS, QUERIES_CSV, RESULTS_DIR, CHOSEN_CONFIGS
from src.generators import run_generation
from src.io_utils import filter_by_target_layers, select_rows
from src.judge import run_judging
from src.metrics import accuracy_table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=["generate", "judge", "both"], default="both")
    parser.add_argument("--shard", choices=["1", "2"], help="judge only half of the rows (run 1 and 2 on two GPUs)")
    args = parser.parse_args()

    responses = RESULTS_DIR / f"{args.model}_chosen.csv"
    judged = RESULTS_DIR / f"{args.model}_chosen_judged.csv"

    if args.stage in ("generate", "both"):
        rows = select_rows(pd.read_csv(QUERIES_CSV), ["false", "true"])
        run_generation(args.model, rows, responses, filter_by_target_layers(CHOSEN_CONFIGS[args.model]))

    if args.stage in ("judge", "both"):
        run_judging(responses, judged, shard=args.shard)
        print(accuracy_table(pd.read_csv(judged)).to_string(index=False))


if __name__ == "__main__":
    main()
