#!/usr/bin/env python3
# sced_single_day_plotting.py
"""
Generates:
  1) Base Point stacked bar plots (hourly by fuel)
  2) SCED step price violin plots (by fuel)

Input (from your extractor step):
ROOT_DIR/
  YYYY-MM-DD/
    aggregation_Base_Point.csv
    aggregation_SCED1_Curve-Price1.csv
    aggregation_SCED1_Curve-Price2.csv
    ...

Outputs:
PLOTS_DIR/
  YYYY-MM-DD/
    base_point_stacked_<agg>.png
    base_point_hourly_<agg>.csv           # optional
    sced_violin/
      step_01_violin.png
      step_01_values.csv                  # optional
      step_02_violin.png
      ...

Notes:
- Wide format expected:
    col0 = "Resource Name"
    col1 = "Resource Type"
    col2+ = 5-minute timestamps as column headers
- Base Point hourly agg:
    "mean" -> hourly average MW (typical for instantaneous base point)
    "mwh"  -> sum(MW) * (5/60) => MWh per hour
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== EDIT THESE =====================
ROOT_DIR  = r"/path/to/output_root"   # contains YYYY-MM-DD subfolders
PLOTS_DIR = r"/path/to/plots_out"     # results written here, mirrored per-day
# Base Point controls
BP_AGG_MODE = "mean"                  # "mean" or "mwh"
BP_SAVE_HOURLY_CSV = True             # write the hourly pivot per day
# SCED violin controls
SCED_SAVE_VALUES_CSV = False          # write flattened per-fuel values per step
# ======================================================

STEP_PATTERN = re.compile(r"aggregation_SCED1_Curve-Price(\d+)\.csv$", re.IGNORECASE)

# --------- Shared helpers ---------
def normalize_key_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Find actual 'Resource Name' and 'Resource Type' columns (case/space tolerant)."""
    def norm(s: str) -> str:
        return s.strip().lower()
    m = {norm(c): c for c in df.columns}
    name_col = m.get("resource name")
    type_col = m.get("resource type")
    if not name_col or not type_col:
        raise ValueError(f"Expected columns 'Resource Name' and 'Resource Type'; found: {list(df.columns)}")
    return name_col, type_col

def detect_timestamp_columns(df: pd.DataFrame, keys: Tuple[str, str]) -> List[str]:
    """Return original column names that parse as datetimes (excluding the two key columns)."""
    name_col, type_col = keys
    ts_cols: List[str] = []
    for col in df.columns:
        if col in (name_col, type_col): 
            continue
        ts = pd.to_datetime(col, errors="coerce")
        if pd.isna(ts):
            continue
        ts_cols.append(col)
    if not ts_cols:
        raise ValueError("No timestamp columns detected (headers must be parseable datetimes).")
    ts_cols = sorted(ts_cols, key=lambda c: pd.to_datetime(c))
    return ts_cols

# --------- Base Point (stacked bar) ---------
def hourly_by_fuel_from_wide(df: pd.DataFrame, agg_mode: str) -> pd.DataFrame:
    """
    Wide 5-min -> hourly by fuel:
      - sum across resources within each fuel for each 5-min timestamp
      - resample to hourly (mean MW or MWh)
    Returns DataFrame indexed by hour (Timestamp), columns = fuels (sorted).
    """
    name_col, type_col = normalize_key_columns(df)
    ts_cols = detect_timestamp_columns(df, (name_col, type_col))

    # numeric 5-min values, coerce bad to 0
    numeric = df[ts_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Sum across resources per fuel for each 5-min timestamp
    numeric.insert(0, "Resource Type", df[type_col].astype(str).fillna("Unknown"))
    fuel_5min = numeric.groupby("Resource Type", dropna=False).sum(numeric_only=True)

    # index by actual timestamps (columns are the times)
    col_ts = pd.to_datetime(fuel_5min.columns, errors="coerce")
    fuel_5min.columns = col_ts
    fuel_5min = fuel_5min.T.sort_index()           # index: 5-min timestamps, cols: fuel
    fuel_5min = fuel_5min.groupby(level=0).sum()   # collapse duplicate headers if any

    if agg_mode.lower() == "mwh":
        hourly = fuel_5min.resample("H").sum() * (5.0/60.0)
    else:
        hourly = fuel_5min.resample("H").mean()

    hourly.columns = [str(c) if c is not None else "Unknown" for c in hourly.columns]
    hourly = hourly.reindex(sorted(hourly.columns), axis=1)
    return hourly

def plot_base_point_day(day: str, hourly_df: pd.DataFrame, out_png: Path, agg_mode: str) -> None:
    if hourly_df.empty:
        print(f"[WARN] {day}: no hourly data; skipping Base Point plot.")
        return
    # clamp to that day & ensure a full 24-hour axis
    day_start = pd.to_datetime(day)
    day_end   = day_start + pd.Timedelta(days=1)
    hourly_df = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
    hourly_df = hourly_df.reindex(pd.date_range(day_start, day_end, freq="H", inclusive="left"), fill_value=0.0)

    plot_df = hourly_df.copy()
    plot_df.index = plot_df.index.hour  # 0..23

    ax = plot_df.plot(kind="bar", stacked=True, figsize=(14, 7))
    ylabel = "Average MW" if agg_mode.lower() == "mean" else "MWh"
    ax.set_title(f"{day} – Base Point by Fuel Type (Hourly {ylabel})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out_png}")

def process_base_point_day(day_dir: Path, plots_root: Path, agg_mode: str, save_hourly_csv: bool) -> None:
    day = day_dir.name
    # expected name; allow some flexibility
    csv_path = day_dir / "aggregation_Base_Point.csv"
    if not csv_path.exists():
        matches = list(day_dir.glob("aggregation_Base_Point*.csv")) or list(day_dir.glob("*Base*Point*.csv"))
        if not matches:
            print(f"[INFO] {day}: no Base Point file; skipping.")
            return
        csv_path = matches[0]

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: read failed for {csv_path.name}: {e}")
        return

    try:
        hourly = hourly_by_fuel_from_wide(df, agg_mode)
    except Exception as e:
        print(f"[ERROR] {day}: Base Point hourly aggregation failed: {e}")
        return

    day_out = plots_root / day
    day_out.mkdir(parents=True, exist_ok=True)

    if save_hourly_csv:
        out_csv = day_out / f"base_point_hourly_{agg_mode}.csv"
        to_save = hourly.copy()
        to_save.insert(0, "hour", to_save.index.strftime("%Y-%m-%d %H:%M"))
        try:
            to_save.to_csv(out_csv, index=False)
            print(f"[OK] Saved: {out_csv}")
        except Exception as e:
            print(f"[ERROR] {day}: failed to write hourly CSV: {e}")

    out_png = day_out / f"base_point_stacked_{agg_mode}.png"
    plot_base_point_day(day, hourly, out_png, agg_mode)

# --------- SCED (violin plots) ---------
def load_violin_data_by_fuel(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build dict fuel -> flattened vector of all 5-min prices that day
    (across resources & timestamps), dropping NaN/inf.
    """
    name_col, type_col = normalize_key_columns(df)
    ts_cols = detect_timestamp_columns(df, (name_col, type_col))
    prices = df[ts_cols].apply(pd.to_numeric, errors="coerce")
    fuels = df[type_col].astype(str).fillna("Unknown")
    prices = prices.assign(**{"__fuel__": fuels})

    out: Dict[str, np.ndarray] = {}
    for fuel, block in prices.groupby("__fuel__"):
        vals = block[ts_cols].to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        out[str(fuel)] = vals
    return out

def plot_violin_step(day: str, step_csv: Path, day_out_dir: Path, save_values_csv: bool) -> None:
    m = STEP_PATTERN.search(step_csv.name)
    step_num = int(m.group(1)) if m else None

    try:
        df = pd.read_csv(step_csv, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: read failed for {step_csv.name}: {e}")
        return

    try:
        data_by_fuel = load_violin_data_by_fuel(df)
    except Exception as e:
        print(f"[ERROR] {day}: processing failed for {step_csv.name}: {e}")
        return

    if not data_by_fuel:
        print(f"[INFO] {day}: no data in {step_csv.name}; skipping.")
        return

    fuels = sorted(data_by_fuel.keys())
    datasets = [data_by_fuel[f] for f in fuels]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.violinplot(datasets, showmeans=False, showmedians=True, showextrema=True)
    title_step = f"SCED Step {step_num}" if step_num is not None else "SCED Step"
    ax.set_title(f"{day} – {title_step} Price Distribution by Fuel")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Price")
    ax.set_xticks(range(1, len(fuels) + 1))
    ax.set_xticklabels(fuels, rotation=30, ha="right")
    plt.tight_layout()

    out_dir = day_out_dir / "sced_violin"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{step_num:02d}" if step_num is not None else "X"
    out_png = out_dir / f"step_{suffix}_violin.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out_png}")

    if save_values_csv:
        vals_list = []
        for fuel in fuels:
            vals = data_by_fuel[fuel]
            vals_list.append(pd.DataFrame({"fuel": fuel, "price": vals}))
        long_df = pd.concat(vals_list, ignore_index=True)
        out_csv = out_dir / f"step_{suffix}_values.csv"
        long_df.to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")

def process_sced_violins_day(day_dir: Path, plots_root: Path, save_values_csv: bool) -> None:
    day = day_dir.name
    day_out = plots_root / day
    day_out.mkdir(parents=True, exist_ok=True)

    step_files = list(day_dir.glob("aggregation_SCED1_Curve-Price*.csv"))
    if not step_files:
        # broader fallback
        step_files = list(day_dir.glob("*SCED*Curve*Price*.csv"))
    if not step_files:
        print(f"[INFO] {day}: no SCED price files; skipping violins.")
        return

    def step_key(p: Path):
        m = STEP_PATTERN.search(p.name)
        return int(m.group(1)) if m else 1_000_000
    step_files = sorted(step_files, key=step_key)

    for step_csv in step_files:
        plot_violin_step(day, step_csv, day_out, save_values_csv)

# --------- Driver ---------
def main():
    root = Path(ROOT_DIR).resolve()
    out  = Path(PLOTS_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)

    day_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not day_dirs:
        print(f"[WARN] No day subfolders found in {root}")
        return

    for day_dir in day_dirs:
        # Base Point (stacked bar)
        process_base_point_day(day_dir, out, BP_AGG_MODE, BP_SAVE_HOURLY_CSV)
        # SCED violins
        process_sced_violins_day(day_dir, out, SCED_SAVE_VALUES_CSV)

if __name__ == "__main__":
    main()
