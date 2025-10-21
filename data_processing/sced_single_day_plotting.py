#!/usr/bin/env python3
"""
Plot hourly stacked bars of total Base Point by fuel type from wide 5-min CSVs.

Input per day:
  <ROOT>/<YYYY-MM-DD>/aggregation_Base_Point.csv
  Columns:
    1) Resource Name
    2) Resource Type (fuel)
    3+) 5-min timestamps as column headers (parseable datetimes)

Output:
  <PLOTS_DIR>/<YYYY-MM-DD>_base_point_stacked_<agg>.png
  (optional) hourly pivot CSV used for plotting

Aggregation mode:
  "mean" -> hourly average MW (typical for instantaneous Base Point)
  "mwh"  -> hourly energy (sum of 5-min MW * 5/60)

Usage:
  python plot_base_point_stacked_by_hour.py \
    --root "/path/to/output_root" \
    --plots-dir "/path/to/plots" \
    --agg-mode mean \
    --save-hourly-csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Defaults (override with CLI) -----------------
ROOT_DIR = r"/path/to/output_root"   # has YYYY-MM-DD subfolders
PLOTS_DIR = r"/path/to/plots_out"
AGG_MODE = "mean"                    # "mean" or "mwh"
SAVE_HOURLY_CSV = False
# ---------------------------------------------------------------

def normalize_key_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Find 'Resource Name' and 'Resource Type' (case/space tolerant)."""
    def norm(s: str) -> str:
        return s.strip().lower()
    norm_map = {norm(c): c for c in df.columns}
    name_col = norm_map.get("resource name")
    type_col = norm_map.get("resource type")
    if not name_col or not type_col:
        raise ValueError(f"Expected columns 'Resource Name' and 'Resource Type'; found: {list(df.columns)}")
    return name_col, type_col

def detect_timestamp_columns(df: pd.DataFrame, keys: Tuple[str, str]) -> Dict[pd.Timestamp, str]:
    """Return map {Timestamp -> original column name} for all 3+ columns that parse as datetimes."""
    name_col, type_col = keys
    ts_map: Dict[pd.Timestamp, str] = {}
    for col in df.columns:
        if col in (name_col, type_col):
            continue
        ts = pd.to_datetime(col, errors="coerce")
        if pd.isna(ts):
            continue
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        ts_map[pd.Timestamp(ts)] = col
    if not ts_map:
        raise ValueError("No timestamp columns detected (headers must be parseable datetimes).")
    return ts_map

def hourly_by_fuel_from_wide(df: pd.DataFrame, agg_mode: str) -> pd.DataFrame:
    """
    Wide 5-min -> hourly by fuel:
      1) Sum across resources within each fuel for each 5-min timestamp
      2) Resample to hourly:
           - 'mean' -> average MW
           - 'mwh'  -> sum * (5/60)
    Returns DataFrame indexed by hour (Timestamp), columns = fuel types.
    """
    name_col, type_col = normalize_key_columns(df)
    ts_map = detect_timestamp_columns(df, (name_col, type_col))

    # Ensure numeric, keep only timestamp columns; coerce errors to 0
    time_cols = [ts_map[t] for t in sorted(ts_map.keys())]
    numeric = df[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Attach fuel, group-sum across resources -> fuel totals per 5-min stamp
    numeric.insert(0, "Resource Type", df[type_col].astype(str).fillna("Unknown"))
    fuel_5min = numeric.groupby("Resource Type", dropna=False).sum(numeric_only=True)

    # Label the columns with actual datetimes
    col_ts = pd.to_datetime(fuel_5min.columns, errors="coerce")
    fuel_5min.columns = col_ts
    fuel_5min = fuel_5min.T.sort_index()  # index: 5-min datetime, columns: fuel

    # Handle potential duplicate timestamp headers by summing
    fuel_5min = fuel_5min.groupby(level=0).sum()

    # Resample to hourly
    if agg_mode.lower() == "mwh":
        hourly = fuel_5min.resample("H").sum() * (5.0 / 60.0)
    else:
        hourly = fuel_5min.resample("H").mean()

    # Clean and stable ordering
    hourly.columns = [str(c) if c is not None else "Unknown" for c in hourly.columns]
    hourly = hourly.reindex(sorted(hourly.columns), axis=1)

    return hourly

def plot_day_stacked(day: str, hourly_df: pd.DataFrame, out_png: Path, agg_mode: str) -> None:
    if hourly_df.empty:
        print(f"[WARN] {day}: no hourly data; skipping.")
        return

    # Clip exactly to that calendar day (robust to stray times)
    day_start = pd.to_datetime(day)
    day_end = day_start + pd.Timedelta(days=1)
    hourly_df = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]

    # Reindex to full 24 hours for nice x-axis, fill missing with 0
    full_hours = pd.date_range(day_start, day_end, freq="H", inclusive="left")
    hourly_df = hourly_df.reindex(full_hours, fill_value=0.0)

    # X-axis as 0..23
    plot_df = hourly_df.copy()
    plot_df.index = plot_df.index.hour

    ax = plot_df.plot(kind="bar", stacked=True, figsize=(14, 7))
    ylabel = "Average MW" if agg_mode.lower() == "mean" else "MWh"
    ax.set_title(f"{day} – Base Point by Fuel Type (Hourly {ylabel})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved plot: {out_png}")

def process_root(root: Path, plots_dir: Path, agg_mode: str, save_hourly_csv: bool) -> None:
    day_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not day_dirs:
        print(f"[WARN] No day subfolders found in {root}")
        return

    for day_dir in day_dirs:
        day = day_dir.name
        csv_path = day_dir / "aggregation_Base_Point.csv"
        if not csv_path.exists():
            matches = list(day_dir.glob("aggregation_Base_Point*.csv")) or list(day_dir.glob("*Base*Point*.csv"))
            if not matches:
                print(f"[INFO] {day}: no aggregation_Base_Point.csv; skipping.")
                continue
            csv_path = matches[0]

        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"[ERROR] {day}: failed to read {csv_path.name}: {e}")
            continue

        try:
            hourly = hourly_by_fuel_from_wide(df, agg_mode=agg_mode)
        except Exception as e:
            print(f"[ERROR] {day}: aggregation failed: {e}")
            continue

        # Create per-day output folder (mirrors input structure)
        day_out_dir = plots_dir / day
        day_out_dir.mkdir(parents=True, exist_ok=True)

        if save_hourly_csv:
            out_csv = day_out_dir / f"base_point_hourly_{agg_mode}.csv"
            try:
                to_save = hourly.copy()
                to_save.insert(0, "hour", to_save.index.strftime("%Y-%m-%d %H:%M"))
                to_save.to_csv(out_csv, index=False)
                print(f"[OK] Saved hourly table: {out_csv}")
            except Exception as e:
                print(f"[ERROR] {day}: failed to write hourly CSV: {e}")

        out_png = day_out_dir / f"base_point_stacked_{agg_mode}.png"
        plot_day_stacked(day, hourly, out_png, agg_mode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stacked bar of hourly Base Point by fuel type (wide 5-min headers).")
    p.add_argument("--root", type=str, default=ROOT_DIR, help="Root dir containing YYYY-MM-DD subfolders")
    p.add_argument("--plots-dir", type=str, default=PLOTS_DIR, help="Output folder for plots and CSVs")
    p.add_argument("--agg-mode", type=str, choices=["mean", "mwh"], default=AGG_MODE,
                   help="Hourly aggregation: 'mean' (MW) or 'mwh' (energy)")
    p.add_argument("--save-hourly-csv", action="store_true", default=SAVE_HOURLY_CSV,
                   help="Save the hourly pivot CSV used for plotting")
    return p.parse_args()

def main():
    args = parse_args()
    process_root(
        root=Path(args.root),
        plots_dir=Path(args.plots_dir),
        agg_mode=args.agg_mode,
        save_hourly_csv=args.save_hourly_csv,
    )

if __name__ == "__main__":
    main()
