"""Step 3: grid search for the best layer and multiplier (Sections 5.8, 6.6).

    python 03_run_grid_search.py --model llama            # focused grid used in the thesis
    python 03_run_grid_search.py --model llama --wide     # first sweep: all layers, multipliers 0-2.5

Every config in GRID (src/config.py) steers the false-claim validation prompts and the
80 true-claim control prompts. After judging, each config gets a correction rate,
a preservation rate and their harmonic mean, the balanced score (eqs. 9-11).

Writes generation_results/{model}_grid.csv, {model}_grid_judged.csv and {model}_grid_summary.csv.

To split the generation over GPUs, give each process its own layers:
    CUDA_VISIBLE_DEVICES=0 TARGET_LAYERS="[[8],[9],[10],[11]]" python 03_run_grid_search.py --model llama --stage generate
"""
import argparse

import pandas as pd

from src.config import MODELS, RESULTS_DIR, VAL_FALSE_CSV, VAL_TRUE_CSV, GRID, GRID_WIDE
from src.generators import run_generation
from src.io_utils import filter_by_target_layers, file_has_data
from src.judge import run_judging
from src.metrics import balanced_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=["generate", "judge", "both"], default="both")
    parser.add_argument("--shard", choices=["1", "2"], help="judge only half of the rows (run 1 and 2 on two GPUs)")
    parser.add_argument("--wide", action="store_true", help="run the first wide sweep instead of the focused grid")
    args = parser.parse_args()

    name = f"{args.model}_grid_wide" if args.wide else f"{args.model}_grid"
    responses = RESULTS_DIR / f"{name}.csv"
    judged = RESULTS_DIR / f"{name}_judged.csv"

    if args.stage in ("generate", "both"):
        grid = (GRID_WIDE if args.wide else GRID)[args.model]
        configs = [{"layers": layers, "multiplier": m}
                   for block in grid for layers in block["layer_sets"] for m in block["multipliers"]]
        configs = filter_by_target_layers(configs)
        print(f"{len(configs)} configs to run")

        false_rows = pd.read_csv(str(VAL_FALSE_CSV).format(model=args.model)).assign(claim_veracity="false")
        true_rows = pd.read_csv(VAL_TRUE_CSV).assign(claim_veracity="true")
        run_generation(args.model, pd.concat([false_rows, true_rows], ignore_index=True), responses, configs)

    if args.stage in ("judge", "both"):
        run_judging(responses, judged, shard=args.shard)

    if file_has_data(judged):
        summary = balanced_scores(pd.read_csv(judged))
        summary.to_csv(RESULTS_DIR / f"{name}_summary.csv", index=False)
        print("Best configs (avoid any with n_gibberish > 0):")
        print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
