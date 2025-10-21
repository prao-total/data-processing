#!/usr/bin/env python3
"""
Plot hourly stacked bars of total MW by fuel type from daily Base Point files.

Directory layout (from your extractor step):
ROOT/
  YYYY-MM-DD/
    aggregation_Base_Point.csv
    (and other aggregation_*.csv)

This script:
- Finds every date folder containing aggregation_Base_Point.csv
- Aggregates 5-minute Base Points to HOURLY totals by fuel type (default: hourly MEAN MW)
- Produces a stacked bar chart for each day and saves it to PLOTS_DIR/<YYYY-MM-DD>_base_point_stacked.png
- Also writes a CSV of the hourly pivot table used for the plot (optional toggle)

To switch to hourly MWh instead of average MW:
- Change AGG_MODE = "mwh" (see config below)

CLI usage:
python plot_base_point_stacked_by_hour.py \
  --root "/path/to/output_root" \
  --plots-dir "/path/to/plots" \
  --save-hourly-csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Config (defaults; can override via CLI) -------------------------
ROOT_DIR = r"/path/to/output_root"    # root with YYYY-MM-DD subfolders
PLOTS_DIR = r"/path/to/plots_out"     # where to save plots

# Choose how to aggregate 5-min MW to hourly:
#   "mean" -> hourly average MW (typical for instantaneous MW signals)
#   "mwh"  -> hourly energy by summing (MW * 5/60) across 12 intervals
AGG_MODE = "mean"   # "mean" or "mwh"

SAVE_HOURLY_CSV = False
# ------------------------------------------------------------------------------------------


def normalize_key_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Return actual column names for Resource Name and Resource Type (case/space tolerant)."""
    def norm(s: str) -> str:
        return s.strip().lower()

    cols_norm = {norm(c): c for c in df.columns}
    name_col = cols_norm.get("resource name")
    type_col = cols_norm.get("resource type")
    if not name_col or not type_col:
        raise ValueError(
            f"Expected 'Resource Name' and 'Resource Type' columns. Found: {list(df.columns)}"
        )
    return name_col, type_col


def detect_timestamp_columns(df: pd.DataFrame, key_cols: Tuple[str, str]) -> Dict[pd.Timestamp, str]:
    """Map parsed timestamps -> original column names for all 5-min columns."""
    name_col, type_col = key_cols
    ts_map = {}
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
        raise ValueError("No timestamp columns detected.")
    return ts_map


def hourly_by_fuel(df: pd.DataFrame, agg_mode: str) -> pd.DataFrame:
    """
    From wide 5-min base points:
      - group by Resource Type
      - sum across resources to get total MW per 5-min per fuel
      - resample to hourly:
          mean (MW) if agg_mode == "mean"
          sum * (5/60) if agg_mode == "mwh" (energy in MWh)
    Returns a DataFrame indexed by hour (DatetimeIndex), columns=fuel types.
    """
    name_col, type_col = normalize_key_columns(df)
    ts_map = detect_timestamp_columns(df, (name_col, type_col))

    # Keep only numeric time columns; coerce non-numeric to NaN then fill with 0
    time_cols = [ts_map[ts] for ts in sorted(ts_map.keys())]
    numeric = df[time_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Attach Resource Type
    numeric.insert(0, "Resource Type", df[type_col].astype(str).fillna("Unknown"))

    # Sum across resources per fuel type for each 5-min timestamp
    grouped = numeric.groupby("Resource Type", dropna=False).sum(numeric_only=True)

    # Now columns are the time headers; convert them to a DatetimeIndex
    ts_sorted = pd.to_datetime(grouped.columns, errors="coerce")
    grouped.columns = ts_sorted

    # Transpose to have time on index, fuel on columns
    fuel_time = grouped.T.sort_index()  # index = 5-min timestamps, columns = fuel types

    # Resample to hourly
    if agg_mode.lower() == "mwh":
        # sum(MW_per_5min) * (5/60) => MWh per hour
        hourly = fuel_time.resample("H").sum() * (5.0 / 60.0)
    else:
        # hourly average MW
        hourly = fuel_time.resample("H").mean()

    # Clean column names and sort columns alphabetically for a stable legend order
    hourly.columns = [str(c) if c is not None else "Unknown" for c in hourly.columns]
    hourly = hourly.reindex(sorted(hourly.columns), axis=1)

    # Keep only the hours present in data (e.g., 00:00 … 23:00 for a single day)
    return hourly


def plot_stacked(day: str, hourly_df: pd.DataFrame, out_png: Path, agg_mode: str) -> None:
    """Create and save a stacked bar chart for a single day."""
    if hourly_df.empty:
        print(f"[WARN] {day}: hourly dataframe is empty; skipping plot.")
        return

    # Ensure only that day's hours (robust if extra bleed)
    try:
        day_start = pd.to_datetime(day)
        day_end = day_start + pd.Timedelta(days=1)
        hourly_df = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
    except Exception:
        pass

    # Make an hour-of-day index (0..23)
    hourly_df = hourly_df.copy()
    hourly_df.index = hourly_df.index.hour

    # Plot
    ax = hourly_df.plot(kind="bar", stacked=True, figsize=(14, 7))
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
            # Try case-insensitive fallback
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
            hourly = hourly_by_fuel(df, agg_mode=agg_mode)
        except Exception as e:
            print(f"[ERROR] {day}: aggregation failed: {e}")
            continue

        # Optional: save the hourly pivot used for plotting
        if save_hourly_csv:
            out_csv = plots_dir / f"{day}_base_point_hourly_{agg_mode}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            try:
                hourly.to_csv(out_csv, index_label="hour")
                print(f"[OK] Saved hourly table: {out_csv}")
            except Exception as e:
                print(f"[ERROR] {day}: failed to write hourly CSV: {e}")

        # Plot
        out_png = plots_dir / f"{day}_base_point_stacked_{agg_mode}.png"
        plot_stacked(day, hourly, out_png, agg_mode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stacked bar of hourly Base Point by fuel type.")
    p.add_argument("--root", type=str, default=ROOT_DIR, help="Root dir containing YYYY-MM-DD subfolders")
    p.add_argument("--plots-dir", type=str, default=PLOTS_DIR, help="Output folder for plots/CSVs")
    p.add_argument("--agg-mode", type=str, choices=["mean", "mwh"], default=AGG_MODE,
                   help="Hourly aggregation: 'mean' (MW) or 'mwh' (energy)")
    p.add_argument("--save-hourly-csv", action="store_true", default=SAVE_HOURLY_CSV,
                   help="Also save the hourly pivot table used for plotting")
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
