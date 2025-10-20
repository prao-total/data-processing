#!/usr/bin/env python3
"""
SCED bidding from aggregated time-series directory + Price heatmaps (months x hours)

Input directory contains files:
  aggregation_HSL.csv
  aggregation_SCED1_Curve-MW1.csv
  aggregation_SCED1_Curve-Price1.csv
  ...
Each file:
  - Columns: ["Resource Name", "Resource Type", <timestamp_1>, <timestamp_2>, ...]
  - Timestamps are 5-min resolution column headers.

This script:
  1) Loads & melts all needed files (HSL, MWk, Pricek), pairs MWk/Pricek, merges with HSL
  2) Computes MW_frac = MW/HSL (per pair, before any aggregation)
  3) Builds 12 x 24 heatmaps of Price by (month, hour) for every Resource Type × Step
     - Saves each heatmap PNG under OUTPUT_DIR/price_heatmaps/<ResourceType>/
"""

import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
INPUT_DIR  = Path("/path/to/your/aggregated_timeseries_dir")     # <-- set this
OUTPUT_DIR = Path("/path/to/output/sced_bidding_from_aggregates")# <-- set this
MAX_STEPS  = 35  # cap number of (MW, Price) pairs to consider
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
      - ("HSL", None)     for HSL file
      - ("MW", step)      for curve MW step file
      - ("PRICE", step)   for curve Price step file
      - ("OTHER", None)   otherwise
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
        return pd.DataFrame(columns=["Resource Name","Resource Type","timestamp", value_name])

    long_df = df.melt(id_vars=id_vars, value_vars=ts_cols,
                      var_name="timestamp", value_name=value_name)
    # Parse timestamps
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], errors="coerce")
    if TIMESTAMP_TZ:
        try:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_localize(
                TIMESTAMP_TZ, nonexistent="NaT", ambiguous="NaT"
            )
        except TypeError:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_convert(TIMESTAMP_TZ)
    return long_df

def ingest_directory(input_dir: Path, max_steps: int = 35) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Scan input_dir, melt files, and return:
      - hsl_long: DataFrame with column 'HSL'
      - mw_long_map: dict[step] -> DataFrame with column 'MW'
      - price_long_map: dict[step] -> DataFrame with column 'Price'
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
            continue

    if hsl_long is None or hsl_long.empty:
        raise ValueError("No valid HSL file found (aggregation_HSL.csv) or it is empty.")

    return hsl_long, mw_long_map, price_long_map

def merge_step(hsl_long: pd.DataFrame, mw_long: pd.DataFrame, price_long: pd.DataFrame, step: int) -> pd.DataFrame:
    """
    Merge HSL, MW_step, Price_step on (Resource Name, Resource Type, timestamp).
    Add 'Step', compute MW_frac, and derive hour/month.
    """
    keys = ["Resource Name", "Resource Type", "timestamp"]
    m = hsl_long.merge(mw_long, on=keys, how="inner")
    m = m.merge(price_long, on=keys, how="inner")
    m["Step"] = step

    # Normalize pair-wise (before any aggregation)
    m["MW"] = pd.to_numeric(m["MW"], errors="coerce")
    m["Price"] = pd.to_numeric(m["Price"], errors="coerce")
    m["HSL"] = pd.to_numeric(m["HSL"], errors="coerce").fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        m["MW_frac"] = np.where(m["HSL"].to_numpy() > 0, m["MW"].to_numpy() / m["HSL"].to_numpy(), np.nan)

    m["hour"] = m["timestamp"].dt.hour
    m["month"] = m["timestamp"].dt.month
    cols = ["Resource Name","Resource Type","timestamp","month","hour","Step","MW","Price","HSL","MW_frac"]
    return m[cols]

def build_bids_long(input_dir: Path, max_steps: int = 35) -> pd.DataFrame:
    """
    Unified long dataframe combining steps 1..max_steps.
    Columns: Resource Name, Resource Type, timestamp, month, hour, Step, MW, Price, HSL, MW_frac
    """
    hsl_long, mw_map, price_map = ingest_directory(input_dir, max_steps=max_steps)

    steps = sorted(set(mw_map.keys()).intersection(set(price_map.keys())))
    if not steps:
        raise ValueError("No overlapping steps found between MW and Price files.")
    if max_steps is not None:
        steps = [s for s in steps if 1 <= s <= max_steps]

    merged_list: List[pd.DataFrame] = []
    for s in steps:
        merged_list.append(merge_step(hsl_long, mw_map[s], price_map[s], s))

    bids_long = pd.concat(merged_list, ignore_index=True)
    bids_long = bids_long.dropna(subset=["Price"])  # keep price rows
    return bids_long

# ---------------------- Heatmap generation ---------------------- #
def plot_price_heatmaps_month_hour(bids_long: pd.DataFrame, out_dir: Path):
    """
    Create 12 x 24 heatmaps of Price for each (Resource Type, Step).
      - x-axis: hour 0..23
      - y-axis: month 1..12
      - value: mean Price for that (month, hour)
    Saves: out_dir/price_heatmaps/<ResourceType>/heatmap_price_<ResourceType>_step<k>.png
    """
    heatmaps_root = out_dir / "price_heatmaps"
    heatmaps_root.mkdir(parents=True, exist_ok=True)

    # Pre-aggregate: mean price for each (rtype, step, month, hour)
    grp = (bids_long.groupby(["Resource Type","Step","month","hour"], observed=True)["Price"]
           .mean().reset_index())

    resource_types = sorted(grp["Resource Type"].dropna().unique().tolist())
    steps = sorted(grp["Step"].dropna().unique().astype(int).tolist())

    # Fixed axes
    hours = np.arange(0, 24)
    months = np.arange(1, 13)

    for rtype in resource_types:
        type_dir = heatmaps_root / re.sub(r"[^A-Za-z0-9._-]+","_", rtype)
        type_dir.mkdir(parents=True, exist_ok=True)

        sub_type = grp[grp["Resource Type"] == rtype]
        for s in steps:
            sub = sub_type[sub_type["Step"] == s]
            if sub.empty:
                # still emit an empty heatmap for consistency
                mat = np.full((len(months), len(hours)), np.nan)
            else:
                # Build matrix: rows=months(1..12), cols=hours(0..23)
                pivot = sub.pivot(index="month", columns="hour", values="Price")
                pivot = pivot.reindex(index=months, columns=hours)
                mat = pivot.to_numpy()

            fig = plt.figure(figsize=(10, 6))
            # imshow with explicit extents to label axes nicely
            # y from 1..12; x from 0..23
            plt.imshow(mat, aspect="auto", origin="lower",
                       extent=[hours.min()-0.5, hours.max()+0.5, months.min()-0.5, months.max()+0.5])
            cbar = plt.colorbar()
            cbar.set_label("Mean bid price ($/MWh)")
            plt.xlabel("Hour of day")
            plt.ylabel("Month (1–12)")
            plt.title(f"Price Heatmap — {rtype} — Step {s}")
            # ticks
            plt.xticks(np.arange(0, 24, 1))
            plt.yticks(np.arange(1, 13, 1))
            out_path = type_dir / f"heatmap_price_{re.sub(r'[^A-Za-z0-9._-]+','_', rtype)}_step{s:02d}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

# --------------------------- Main --------------------------- #
def main():
    print(f"Reading aggregated files from: {INPUT_DIR}")
    bids_long = build_bids_long(INPUT_DIR, max_steps=MAX_STEPS)

    # Save the normalized long table for downstream analysis
    out_csv = OUTPUT_DIR / "bids_long_steps_from_aggregates.csv"
    bids_long.to_csv(out_csv, index=False)
    print(f"✅ Wrote normalized long table: {out_csv}")

    # Generate price heatmaps (months x hours) for each resource type & step
    print("Generating months×hours price heatmaps for each Resource Type × Step...")
    plot_price_heatmaps_month_hour(bids_long, OUTPUT_DIR)
    print(f"✅ Heatmaps saved under: {OUTPUT_DIR / 'price_heatmaps'}")

if __name__ == "__main__":
    main()
