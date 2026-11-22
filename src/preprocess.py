"""Clean the raw Kaggle `hotel_bookings.csv` into the file we host in the cloud.

Usage:
    python -m src.preprocess                      # uses paths from config.yaml
    python -m src.preprocess --raw path/to/raw.csv --out path/to/clean.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import feature_lists, get, load_config

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def clean_raw(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Deterministic cleaning. Feature engineering happens later, inside the
    model Pipeline, so that the API applies exactly the same transforms."""
    df = df.copy()
    n0 = len(df)

    # 1. Drop leakage / ID columns
    drop_cols = [c for c in get(cfg, "preprocess.drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 2. Missing values
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")
    if "meal" in df.columns:
        # "Undefined" and "SC" both mean no meal package in this dataset
        df["meal"] = df["meal"].replace("Undefined", "SC")

    # 3. Impossible / outlier rows
    guests = df[["adults", "children", "babies"]].sum(axis=1)
    df = df[guests > 0]
    adr_max = get(cfg, "preprocess.adr_max", None)
    df = df[df["adr"] >= 0]
    if adr_max is not None:
        df = df[df["adr"] <= adr_max]

    # 4. Duplicates (the raw file has ~30k exact duplicate rows)
    if get(cfg, "preprocess.drop_duplicates", True):
        df = df.drop_duplicates()

    # 5. Types
    int_cols = ["children", "adults", "babies"]
    for c in int_cols:
        df[c] = df[c].astype(int)

    # 6. Keep only the columns the pipeline can use (+ target, + country so the
    #    include_country flag can be flipped without re-uploading data)
    numeric, categorical, _ = feature_lists(cfg)
    keep = [get(cfg, "data.target")] + numeric + categorical
    if "country" in df.columns and "country" not in keep:
        keep.append("country")
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Raw data is missing expected columns: {missing}")
    df = df[keep].reset_index(drop=True)

    print(f"[preprocess] {n0:,} raw rows -> {len(df):,} clean rows, {df.shape[1]} columns")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw", default=None, help="raw CSV path (default: data.raw_path)")
    parser.add_argument("--out", default=None, help="clean CSV path (default: data.clean_path)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = Path(args.raw or get(cfg, "data.raw_path"))
    out_path = Path(args.out or get(cfg, "data.clean_path"))

    if not raw_path.exists():
        raise SystemExit(
            f"Raw file not found: {raw_path}\n"
            "Download hotel_bookings.csv from "
            "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand "
            f"and save it to {raw_path}"
        )
    df = clean_raw(pd.read_csv(raw_path), cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[preprocess] wrote {out_path}")


if __name__ == "__main__":
    main()
