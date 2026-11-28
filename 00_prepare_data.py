"""Step 0: rebuild the data files from your own copy of UPHILL.

The dataset is not shipped with this repository. Download `queries.csv` from the
UPHILL authors (see README -> "Data"), then run:

    python 00_prepare_data.py --source path/to/queries.csv

data/ids/ lists, in order, which UPHILL rows each file used. A row is identified by
(claim_id, presupposition_level); query_sha1 is a checksum of the query text, so a
changed upstream file is noticed instead of silently used.

Writes data/queries.csv, data/train_{model}.csv, data/val_false_{model}.csv and
data/val_true.csv. data/prompts_{model}.csv is produced later by step 2.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
IDS_DIR = DATA_DIR / "ids"
FILES = ["queries", "train_llama", "train_qwen", "val_false_llama", "val_false_qwen", "val_true"]
KEY = ("claim_id", "presupposition_level")
REQUIRED = ("claim_id", "claim", "claim_veracity", "presupposition_level", "query_with_presupposition")


def checksum(text):
    normalised = text.replace("\r\n", "\n")
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="queries.csv downloaded from the UPHILL authors")
    parser.add_argument("--out", default=str(DATA_DIR), help="output folder (default: data/)")
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = parser.parse_args()

    csv.field_size_limit(10 ** 9)
    columns, rows = read_csv(args.source)
    missing_columns = [c for c in REQUIRED if c not in columns]
    if missing_columns:
        sys.exit(f"{args.source} lacks the column(s) {missing_columns}. Use UPHILL's queries.csv.")
    index = {tuple(r[k] for k in KEY): r for r in rows}
    # the first column of the download is an unnamed row number
    header = ["Unnamed: 0" if c == "" else c for c in columns]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [f"{n}.csv" for n in FILES if (out_dir / f"{n}.csv").exists()]
    if existing and not args.force:
        sys.exit(f"Already present in {out_dir}: {', '.join(existing)}. Use --force to overwrite.")

    # check everything first, so that nothing is written if any file cannot be rebuilt
    selected = {}
    for name in FILES:
        _, wanted = read_csv(IDS_DIR / f"{name}.csv")
        picked, problems = [], []
        for item in wanted:
            key = tuple(item[k] for k in KEY)
            row = index.get(key)
            if row is None:
                problems.append(f"{key} not found")
            elif checksum(row["query_with_presupposition"]) != item["query_sha1"]:
                problems.append(f"{key} has a different query text")
            else:
                picked.append(row)
        if problems:
            sys.exit(f"{name}: {len(problems)} of {len(wanted)} rows could not be rebuilt, e.g. {problems[:3]}. "
                     "The downloaded file differs from the version used in the thesis.")
        selected[name] = picked

    for name, picked in selected.items():
        with open(out_dir / f"{name}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows([row[c] for c in columns] for row in picked)
        print(f"{name}.csv: {len(picked)} rows")
    print(f"Done. Files written to {out_dir}")


if __name__ == "__main__":
    main()
