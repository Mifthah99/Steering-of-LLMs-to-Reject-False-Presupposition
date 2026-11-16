"""Step 1: baseline (Sections 5.3, 5.4).

Generates unsteered answers to every false- and true-claim prompt in UPHILL and labels them with the judge.

    python 01_run_baseline.py --model llama

Writes generation_results/{model}_baseline.csv and {model}_baseline_judged.csv.
The false-claim prompts labelled AGREE here are the ones used to build the
steering vectors (data/train_{model}.csv) and the grid-search set (data/val_false_{model}.csv).
"""
import argparse

import pandas as pd

from src.config import MODELS, QUERIES_CSV, RESULTS_DIR
from src.generators import run_generation
from src.io_utils import select_rows
from src.judge import run_judging
from src.metrics import accuracy_table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=["generate", "judge", "both"], default="both")
    parser.add_argument("--shard", choices=["1", "2"], help="judge only half of the rows (run 1 and 2 on two GPUs)")
    args = parser.parse_args()

    responses = RESULTS_DIR / f"{args.model}_baseline.csv"
    judged = RESULTS_DIR / f"{args.model}_baseline_judged.csv"

    if args.stage in ("generate", "both"):
        rows = select_rows(pd.read_csv(QUERIES_CSV), ["false", "true"])
        run_generation(args.model, rows, responses)

    if args.stage in ("judge", "both"):
        run_judging(responses, judged, shard=args.shard)
        print(accuracy_table(pd.read_csv(judged), ["presupposition_level"]).to_string(index=False))


if __name__ == "__main__":
    main()
