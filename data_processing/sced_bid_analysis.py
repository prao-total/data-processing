#!/usr/bin/env python3
"""
Ingest SCED bidding inputs from an aggregated directory of time-series CSVs.

Directory structure:
  aggregation_HSL.csv
  aggregation_SCED1_Curve-MW1.csv
  aggregation_SCED1_Curve-Price1.csv
  aggregation_SCED1_Curve-MW2.csv
  aggregation_SCED1_Curve-Price2.csv
  ...
Each CSV:
  - First columns: "Resource Name", "Resource Type"
  - Remaining columns: timestamps (5-min resolution), each cell is the metric value at that timestamp.

This script:
  - Loads & melts each CSV to long form
  - Detects step number and whether MW or Price
  - Merges MW_k, Price_k with HSL by (Resource Name, Resource Type, timestamp, Step)
  - Computes MW_frac = MW / HSL (pairwise, before any aggregation)
  - Outputs a single normalized long CSV for downstream analysis

Edit INPUT_DIR and OUTPUT_DIR in the CONFIG block.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
INPUT_DIR = Path("/path/to/your/aggregated_timeseries_dir")  # <-- set this
OUTPUT_DIR = Path("/path/to/output/sced_bidding_from_aggregates")  # <-- set this
MAX_STEPS = 35  # enforce up to 35 (G..BX equivalent)
TIMESTAMP_TZ = None  # e.g., "America/Chicago" or None to keep naive
# ---------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------- Filename parsing helpers ---------
RE_HSL = re.compile(r"^aggregation[_-]HSL\.csv$", re.IGNORECASE)
RE_MW  = re.compile(r"^aggregation[_-]SCED1[_ ]?Curve[-_]?MW(\d+)\.csv$", re.IGNORECASE)
RE_PRC = re.compile(r"^aggregation[_-]SCED1[_ ]?Curve[-_]?Price(\d+)\.csv$", re.IGNORECASE)

def classify_file(fname: str) -> Tuple[str, Optional[int]]:
    """
    Classify a file:
      - returns ("HSL", None) for HSL file
      - returns ("MW", step) for curve MW step file
      - returns ("PRICE", step) for curve Price step file
      - returns ("OTHER", None) if we don't recognize it
    """
    if RE_HSL.match(fname):
        return "HSL", None
    m = RE_MW.match(fname)
    if m:
        return "MW", int(m.group(1))
    p = RE_PRC.match(fname)
    if p:
        return "PRICE", int(p.group(1))
    return "OTHER", None

# --------- IO / melt helpers ---------
def melt_metric_csv(path: Path, value_name: str) -> pd.DataFrame:
    """
    Read a single 'aggregation_*.csv' file and melt:
      id_vars = ["Resource Name", "Resource Type"]
      value_vars = all timestamp columns
    Return columns: Resource Name, Resource Type, timestamp, <value_name>
    """
    df = pd.read_csv(path)
    # Normalize expected id columns
    rename_map = {}
    if "Resource Node" in df.columns:
        rename_map["Resource Node"] = "Resource Name"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["Resource Name", "Resource Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")

    id_vars = ["Resource Name", "Resource Type"]
    ts_cols = [c for c in df.columns if c not in id_vars]
    if not ts_cols:
        # no data columns
        return pd.DataFrame(columns=["Resource Name","Resource Type","timestamp", value_name])

    long_df = df.melt(id_vars=id_vars, value_vars=ts_cols,
                      var_name="timestamp", value_name=value_name)
    # Parse timestamps
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], errors="coerce")
    if TIMESTAMP_TZ:
        # Localize (if naive) or convert
        try:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_localize(TIMESTAMP_TZ, nonexistent="NaT", ambiguous="NaT")
        except TypeError:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_convert(TIMESTAMP_TZ)
    return long_df

# --------- Build long tables for HSL, MW, Price ---------
def ingest_directory(input_dir: Path, max_steps: int = 35) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Scan input_dir for aggregation files, melt them, and return:
      - hsl_long: long dataframe with column 'HSL'
      - mw_long_map: dict step -> long dataframe with column 'MW'
      - price_long_map: dict step -> long dataframe with column 'Price'
    """
    hsl_long = None
    mw_long_map: Dict[int, pd.DataFrame] = {}
    price_long_map: Dict[int, pd.DataFrame] = {}

    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        kind, step = classify_file(f.name)
        if kind == "HSL":
            hsl_long = melt_metric_csv(f, "HSL")
        elif kind == "MW" and step is not None and 1 <= step <= max_steps:
            mw_long_map[step] = melt_metric_csv(f, "MW")
        elif kind == "PRICE" and step is not None and 1 <= step <= max_steps:
            price_long_map[step] = melt_metric_csv(f, "Price")
        else:
            # Ignore OTHER or steps beyond max_steps
            continue

    if hsl_long is None or hsl_long.empty:
        raise ValueError("No valid HSL file found (aggregation_HSL.csv) or it is empty.")

    return hsl_long, mw_long_map, price_long_map

def merge_step(hsl_long: pd.DataFrame, mw_long: pd.DataFrame, price_long: pd.DataFrame, step: int) -> pd.DataFrame:
    """
    Merge HSL, MW_step, Price_step on (Resource Name, Resource Type, timestamp).
    Add 'Step' column and compute MW_frac safely.
    """
    keys = ["Resource Name", "Resource Type", "timestamp"]
    # Start from HSL to ensure we only keep timestamps/units present in HSL (adjust if desired)
    m = hsl_long.merge(mw_long, on=keys, how="inner")
    m = m.merge(price_long, on=keys, how="inner")
    m["Step"] = step

    # Normalize pair-wise
    m["MW"] = pd.to_numeric(m["MW"], errors="coerce")
    m["Price"] = pd.to_numeric(m["Price"], errors="coerce")
    m["HSL"] = pd.to_numeric(m["HSL"], errors="coerce").fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        m["MW_frac"] = np.where(m["HSL"].to_numpy() > 0, m["MW"].to_numpy() / m["HSL"].to_numpy(), np.nan)

    m["hour"] = m["timestamp"].dt.hour
    # Keep canonical columns first
    cols = ["Resource Name","Resource Type","timestamp","hour","Step","MW","Price","HSL","MW_frac"]
    return m[cols]

def build_bids_long(input_dir: Path, max_steps: int = 35) -> pd.DataFrame:
    """
    Produce a unified long dataframe combining steps 1..max_steps where
    each row is (Resource Name, Resource Type, timestamp, hour, Step, MW, Price, HSL, MW_frac).
    """
    hsl_long, mw_map, price_map = ingest_directory(input_dir, max_steps=max_steps)

    # Intersect steps present in both MW and Price
    steps = sorted(set(mw_map.keys()).intersection(set(price_map.keys())))
    if not steps:
        raise ValueError("No overlapping steps found between MW and Price files.")
    if max_steps is not None:
        steps = [s for s in steps if 1 <= s <= max_steps]

    merged_list: List[pd.DataFrame] = []
    for s in steps:
        mw_long = mw_map[s]
        price_long = price_map[s]
        # Basic sanity checks on id columns
        for df, label in [(mw_long, f"MW{s}"), (price_long, f"Price{s}")]:
            for c in ["Resource Name", "Resource Type", "timestamp"]:
                if c not in df.columns:
                    raise ValueError(f"{label} long table missing '{c}'")

        merged = merge_step(hsl_long, mw_long, price_long, s)
        merged_list.append(merged)

    bids_long = pd.concat(merged_list, ignore_index=True)
    # Clean invalids
    bids_long = bids_long.dropna(subset=["Price", "MW_frac"])
    bids_long = bids_long[(bids_long["MW_frac"] >= 0) & np.isfinite(bids_long["MW_frac"])]
    return bids_long

def main():
    print(f"Reading aggregated files from: {INPUT_DIR}")
    bids_long = build_bids_long(INPUT_DIR, max_steps=MAX_STEPS)

    out_csv = OUTPUT_DIR / "bids_long_steps_from_aggregates.csv"
    bids_long.to_csv(out_csv, index=False)
    print(f"✅ Wrote normalized long table: {out_csv}")
    print("Sample:")
    print(bids_long.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
