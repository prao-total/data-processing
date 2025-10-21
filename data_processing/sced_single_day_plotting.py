#!/usr/bin/env python3
# sced_single_day_plotting.py
"""
Generates per day:
  1) Base Point stacked bar plots (hourly by fuel)
  2) SCED step price violin plots (by fuel)
  3) SCED step normalized-bid line plot (by fuel):
       - Normalize each aggregation_SCED1_Curve-MW<k>.csv by aggregation_HSL.csv
       - Save normalized CSVs alongside inputs
       - Plot average normalized bid vs step, one line per fuel

Input (from your extractor step):
ROOT_DIR/
  YYYY-MM-DD/
    aggregation_Base_Point.csv
    aggregation_HSL.csv
    aggregation_SCED1_Curve-Price1.csv, ...
    aggregation_SCED1_Curve-MW1.csv, ...

Outputs:
PLOTS_DIR/
  YYYY-MM-DD/
    base_point_stacked_<agg>.png
    base_point_hourly_<agg>.csv         # optional
    sced_violin/
      step_01_violin.png
      step_01_values.csv                # optional
    sced_normalized/
      normalized_bids_by_stage.png
      normalized_bids_by_stage.csv

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
from typing import Dict, Tuple, List, Optional
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
# SCED price violins
SCED_SAVE_VALUES_CSV = False          # write flattened per-fuel values per step
VIOLIN_MIN_SAMPLES = 1                # min samples per fuel to draw a violin
# SCED normalized-bids (MW/HSL) summary output
SAVE_NORMALIZED_SUMMARY_CSV = True
# ======================================================

PRICE_STEP_PATTERN = re.compile(r"aggregation_SCED1_Curve-Price(\d+)\.csv$", re.IGNORECASE)
MW_STEP_PATTERN    = re.compile(r"aggregation_SCED1_Curve-MW(\d+)\.csv$", re.IGNORECASE)

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

# =========================
# 1) Base Point (stacked)
# =========================
def hourly_by_fuel_from_wide(df: pd.DataFrame, agg_mode: str) -> pd.DataFrame:
    name_col, type_col = normalize_key_columns(df)
    ts_cols = detect_timestamp_columns(df, (name_col, type_col))

    numeric = df[ts_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric.insert(0, "Resource Type", df[type_col].astype(str).fillna("Unknown"))
    fuel_5min = numeric.groupby("Resource Type", dropna=False).sum(numeric_only=True)

    col_ts = pd.to_datetime(fuel_5min.columns, errors="coerce")
    fuel_5min.columns = col_ts
    fuel_5min = fuel_5min.T.sort_index()
    fuel_5min = fuel_5min.groupby(level=0).sum()

    if agg_mode.lower() == "mwh":
        hourly = fuel_5min.resample("h").sum() * (5.0 / 60.0)
    else:
        hourly = fuel_5min.resample("h").mean()

    hourly.columns = [str(c) if c is not None else "Unknown" for c in hourly.columns]
    hourly = hourly.reindex(sorted(hourly.columns), axis=1)
    return hourly

def plot_base_point_day(day: str, hourly_df: pd.DataFrame, out_png: Path, agg_mode: str) -> None:
    if hourly_df.empty:
        print(f"[WARN] {day}: no hourly data; skipping Base Point plot.")
        return
    day_start = pd.to_datetime(day)
    day_end   = day_start + pd.Timedelta(days=1)
    hourly_df = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
    hourly_df = hourly_df.reindex(pd.date_range(day_start, day_end, freq="h", inclusive="left"), fill_value=0.0)

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

# =========================
# 2) SCED price violins
# =========================
def load_violin_data_by_fuel(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    name_col, type_col = normalize_key_columns(df)
    ts_cols = detect_timestamp_columns(df, (name_col, type_col))
    prices = df[ts_cols].apply(pd.to_numeric, errors="coerce")
    fuels = df[type_col].astype(str).fillna("Unknown")
    prices = prices.assign(**{"__fuel__": fuels})

    out: Dict[str, np.ndarray] = {}
    for fuel, block in prices.groupby("__fuel__"):
        vals = block[ts_cols].to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        out[str(fuel)] = vals.astype(float)
    return out

def plot_violin_step(day: str, step_csv: Path, day_out_dir: Path, save_values_csv: bool) -> None:
    m = PRICE_STEP_PATTERN.search(step_csv.name)
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

    fuels_all = sorted(data_by_fuel.keys())
    fuels, datasets, skipped = [], [], []
    for f in fuels_all:
        arr = data_by_fuel[f]
        if arr is None or arr.size < VIOLIN_MIN_SAMPLES:
            skipped.append(f); continue
        fuels.append(f); datasets.append(arr)
    if skipped:
        print(f"[INFO] {day}: {step_csv.name} – skipped empty/short fuels: {', '.join(skipped)}")
    if not datasets:
        title_step = f"SCED Step {step_num}" if step_num is not None else "SCED Step"
        print(f"[INFO] {day}: {title_step} – no fuels with data; skipping plot.")
        return

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
        vals_list = [pd.DataFrame({"fuel": f, "price": arr}) for f, arr in zip(fuels, datasets)]
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
        step_files = list(day_dir.glob("*SCED*Curve*Price*.csv"))
    if not step_files:
        print(f"[INFO] {day}: no SCED price files; skipping violins.")
        return

    def step_key(p: Path):
        m = PRICE_STEP_PATTERN.search(p.name)
        return int(m.group(1)) if m else 1_000_000
    step_files = sorted(step_files, key=step_key)

    for step_csv in step_files:
        plot_violin_step(day, step_csv, day_out, save_values_csv)

# =========================================
# 3) SCED normalized-bids (MW/HSL) lines
# =========================================
def normalize_mw_by_hsl(stage_df: pd.DataFrame, hsl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a normalized wide dataframe with:
      - col0: Resource Name (matching intersection)
      - col1: Resource Type (from stage_df for those resources)
      - col2+: timestamps (intersection of parseable headers in both), values = stage / hsl
    Any division where HSL <= 0 or NaN => NaN result.
    """
    s_name, s_type = normalize_key_columns(stage_df)
    h_name, h_type = normalize_key_columns(hsl_df)

    # Intersection of resources by Resource Name
    stage_df = stage_df.copy()
    hsl_df   = hsl_df.copy()
    stage_df[s_name] = stage_df[s_name].astype(str).str.strip()
    hsl_df[h_name]   = hsl_df[h_name].astype(str).str.strip()

    stage_df = stage_df.set_index(s_name)
    hsl_df   = hsl_df.set_index(h_name)

    common_units = stage_df.index.intersection(hsl_df.index)
    if common_units.empty:
        raise ValueError("No overlapping resources (Resource Name) between MW stage and HSL.")

    stage_df = stage_df.loc[common_units]
    hsl_df   = hsl_df.loc[common_units]

    # Timestamp intersections (parseable headers only)
    s_ts = detect_timestamp_columns(stage_df.reset_index(), (s_name, s_type))
    h_ts = detect_timestamp_columns(hsl_df.reset_index(), (h_name, h_type))
    common_ts = sorted(set(s_ts).intersection(h_ts), key=lambda c: pd.to_datetime(c))
    if not common_ts:
        raise ValueError("No overlapping timestamp columns between MW stage and HSL.")

    # Build numeric arrays
    s_vals = stage_df[common_ts].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    h_vals = hsl_df[common_ts].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Safe division: HSL<=0 or NaN -> NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_vals = np.where(np.isfinite(h_vals) & (h_vals > 0.0), s_vals / h_vals, np.nan)

    # Compose result frame
    out = pd.DataFrame(norm_vals, index=common_units, columns=common_ts)
    # Resource Type from stage_df (aligned on index)
    type_series = stage_df[s_type].astype(str).fillna("Unknown")
    out.insert(0, "Resource Type", type_series)
    out.insert(0, "Resource Name", out.index)
    out.reset_index(drop=True, inplace=True)
    return out

def average_normalized_by_fuel(norm_df: pd.DataFrame) -> Dict[str, float]:
    """Average across all resources & timestamps for each fuel type -> scalar per fuel."""
    name_col, type_col = normalize_key_columns(norm_df)
    ts_cols = detect_timestamp_columns(norm_df, (name_col, type_col))
    vals = norm_df[ts_cols].apply(pd.to_numeric, errors="coerce")
    vals.insert(0, "Resource Type", norm_df[type_col].astype(str).fillna("Unknown"))
    # mean across rows and timestamps (ignore NaN)
    by_fuel = vals.groupby("Resource Type", dropna=False).mean(numeric_only=True).mean(axis=1, numeric_only=True)
    # Return as dict
    return {str(k): float(v) for k, v in by_fuel.sort_index().items()}

def process_sced_normalized_lines_day(day_dir: Path, plots_root: Path, save_summary_csv: bool) -> None:
    """
    For a given day:
      - Load aggregation_HSL.csv
      - For each aggregation_SCED1_Curve-MW<k>.csv:
          normalize by HSL, save normalized CSV back into day_dir
      - Build a line chart: x = step (k), y = avg normalized bid; one line per fuel
    """
    day = day_dir.name
    hsl_path = day_dir / "aggregation_HSL.csv"
    if not hsl_path.exists():
        # fallback
        matches = list(day_dir.glob("*HSL*.csv"))
        if not matches:
            print(f"[INFO] {day}: no HSL file; skipping normalized MW lines.")
            return
        hsl_path = matches[0]

    try:
        hsl_df = pd.read_csv(hsl_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: failed to read HSL {hsl_path.name}: {e}")
        return

    # Find SCED MW step files
    mw_files = list(day_dir.glob("aggregation_SCED1_Curve-MW*.csv"))
    if not mw_files:
        mw_files = list(day_dir.glob("*SCED*Curve*MW*.csv"))
    if not mw_files:
        print(f"[INFO] {day}: no SCED MW step files; skipping normalized MW lines.")
        return

    def step_key(p: Path):
        m = MW_STEP_PATTERN.search(p.name)
        return int(m.group(1)) if m else 1_000_000
    mw_files = sorted(mw_files, key=step_key)

    # Per-step averages per fuel
    summary: Dict[int, Dict[str, float]] = {}

    for stage_csv in mw_files:
        m = MW_STEP_PATTERN.search(stage_csv.name)
        step_num = int(m.group(1)) if m else None
        try:
            stage_df = pd.read_csv(stage_csv, dtype=str)
        except Exception as e:
            print(f"[ERROR] {day}: read failed for {stage_csv.name}: {e}")
            continue

        try:
            norm_df = normalize_mw_by_hsl(stage_df, hsl_df)
        except Exception as e:
            print(f"[INFO] {day}: {stage_csv.name} – normalization skipped: {e}")
            continue

        # Save normalized CSV back into inputs folder
        suffix = f"{step_num}" if step_num is not None else "X"
        norm_path = day_dir / f"{stage_csv.stem}_normalized.csv"
        try:
            norm_df.to_csv(norm_path, index=False)
            print(f"[OK] Saved normalized: {norm_path}")
        except Exception as e:
            print(f"[ERROR] {day}: failed to write normalized CSV {norm_path.name}: {e}")

        # Compute scalar per fuel for this step
        avg_by_fuel = average_normalized_by_fuel(norm_df)
        if step_num is not None:
            summary[step_num] = avg_by_fuel

    if not summary:
        print(f"[INFO] {day}: no normalized data to summarize; skipping line plot.")
        return

    # Assemble summary table: rows=step, cols=fuels (sorted)
    all_fuels = sorted({fuel for d in summary.values() for fuel in d.keys()})
    steps_sorted = sorted(summary.keys())
    data = []
    for s in steps_sorted:
        row = [summary[s].get(fuel, np.nan) for fuel in all_fuels]
        data.append(row)
    summary_df = pd.DataFrame(data, index=steps_sorted, columns=all_fuels)

    # Plot: lines per fuel across steps
    day_out = plots_root / day / "sced_normalized"
    day_out.mkdir(parents=True, exist_ok=True)

    ax = summary_df.plot(figsize=(14, 7), marker="o")
    ax.set_title(f"{day} – Average Normalized Bid by SCED Step")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("Average Normalized Bid (MW / HSL)")
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    out_png = day_out / "normalized_bids_by_stage.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out_png}")

    if save_summary_csv:
        out_csv = day_out / "normalized_bids_by_stage.csv"
        summary_df.to_csv(out_csv, index_label="step")
        print(f"[OK] Saved: {out_csv}")

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
        # 1) Base Point (stacked bar)
        process_base_point_day(day_dir, out, BP_AGG_MODE, BP_SAVE_HOURLY_CSV)
        # 2) SCED price violins
        process_sced_violins_day(day_dir, out, SCED_SAVE_VALUES_CSV)
        # 3) SCED normalized-bid lines
        process_sced_normalized_lines_day(day_dir, out, SAVE_NORMALIZED_SUMMARY_CSV)

if __name__ == "__main__":
    main()
