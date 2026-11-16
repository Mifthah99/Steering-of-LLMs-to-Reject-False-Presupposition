"""Shared CSV helpers: locked appends, header alignment, resume support and data selection.

Several processes (one per GPU) may append to the same results CSV at once, so every
read and write of a results file goes through a FileLock on "<file>.lock".
"""
import ast
import csv
import os
from pathlib import Path

import pandas as pd
from filelock import FileLock


def get_lock(path):
    return FileLock(str(path) + ".lock", timeout=120)


def file_has_data(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_csv_locked(path, **kwargs):
    """Read a CSV under its lock so we never see a half-written file."""
    with get_lock(path):
        return pd.read_csv(path, **kwargs)


def append_results_to_csv(rows, path):
    """Append rows (list of dicts or DataFrame) to a CSV, safe across concurrent processes.

    If the file already exists, the new rows are re-ordered to match its header exactly
    (missing columns are filled with NaN, extra columns are dropped), so columns never shift.
    """
    if rows is None:
        return
    df_new = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df_new.empty:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with get_lock(path):
        file_exists = file_has_data(path)
        if file_exists:
            with open(path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
            dropped = [c for c in df_new.columns if c not in header]
            if dropped:
                print(f"WARNING: columns {dropped} are not in the header of {path} and were dropped.")
            df_new = df_new.reindex(columns=header)

        df_new.to_csv(path, mode="a" if file_exists else "w", header=not file_exists,
                      index=False, quoting=csv.QUOTE_MINIMAL)


def dedupe_csv(path, key_cols):
    """Keep only the LAST row per key (used after the judge retry pass)."""
    with get_lock(path):
        full_df = pd.read_csv(path)
        keys = [c for c in key_cols if c in full_df.columns]
        before = len(full_df)
        full_df.drop_duplicates(subset=keys, keep="last", inplace=True)
        full_df.to_csv(path, index=False)
    print(f"De-duplicated {path}: {before} -> {len(full_df)} rows.")


def get_missing_rows(path, rows_df, layers=None, multiplier=None,
                     key_cols=("query_with_presupposition", "claim_id")):
    """Return the rows of `rows_df` that are not yet in the results file `path`.

    For steered runs only rows with the same (layers, multiplier) count as done, so a
    crashed run resumes exactly where it stopped.
    """
    if not file_has_data(path):
        return rows_df

    existing = read_csv_locked(path)
    if layers is not None:
        mask = (existing["layers"].astype(str) == str(list(layers))) & \
               (existing["multiplier"].astype(float).round(6) == round(float(multiplier), 6))
        existing = existing[mask]

    key_cols = [c for c in key_cols if c in rows_df.columns and c in existing.columns]
    if not key_cols or rows_df.empty or existing.empty:
        return rows_df
    done = set(map(tuple, existing[key_cols].astype(str).values))
    keys = rows_df[key_cols].astype(str).apply(tuple, axis=1)
    return rows_df[~keys.isin(done)]


def filter_by_target_layers(configs):
    """Keep only configs whose layer list appears in the TARGET_LAYERS env var.

    Lets you split a sweep across GPUs, e.g.
        CUDA_VISIBLE_DEVICES=0 TARGET_LAYERS="[[8],[9],[10]]" python 03_run_grid_search.py ...
        CUDA_VISIBLE_DEVICES=1 TARGET_LAYERS="[[11],[12],[13]]" python 03_run_grid_search.py ...
    """
    env_layers = os.environ.get("TARGET_LAYERS")
    if not env_layers:
        return configs
    wanted = ast.literal_eval(env_layers)
    configs = [cfg for cfg in configs if cfg["layers"] in wanted]
    print(f"TARGET_LAYERS={env_layers} -> {len(configs)} config(s) for this process.")
    return configs


def select_rows(df, veracities, max_claims_other=20):
    """Pick the UPHILL rows to evaluate.

    'false' and 'true' rows are always kept in full. For 'fabricated' and 'mixture'
    only the first `max_claims_other` claims (all their presupposition levels) are kept.
    """
    parts = []
    for v in veracities:
        part = df[df["claim_veracity"] == v]
        if v in ("fabricated", "mixture"):
            keep_ids = part["claim_id"].drop_duplicates().head(max_claims_other)
            part = part[part["claim_id"].isin(keep_ids)]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
