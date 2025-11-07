#!/usr/bin/env python3
# sced_single_day_plotting.py
"""
Generates per day:
  1) Base Point stacked bar plots (hourly by fuel)
  2) SCED step price violin plots (by fuel)
  3) SCED step normalized-bid line plot (by fuel):
       - Normalize each aggregation_SCED1_Curve-MW<k>.csv by aggregation_HSL.csv
       - Save normalized CSVs back into the day folder
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

Assumptions:
- Wide format:
    col0 = "Resource Name"
    col1 = "Resource Type"
    col2+ = 5-minute timestamps as column headers
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
# Optional toggles (leave as-is unless you want different behavior)
BP_AGG_MODE = "mean"                  # "mean" (MW) or "mwh" (energy)
BP_SAVE_HOURLY_CSV = True             # write hourly pivot per day
SCED_SAVE_VALUES_CSV = False          # write per-step flattened price values
VIOLIN_MIN_SAMPLES = 1                # min samples per fuel to draw a violin
SAVE_NORMALIZED_SUMMARY_CSV = True    # write the per-day summary used for line plot
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
    Elementwise normalization with strict alignment:
      - Match rows by Resource Name (case/whitespace tolerant).
      - Match columns by timestamp headers (intersection only, chronological).
      - Return wide frame: Resource Name, Resource Type, then common timestamps as MW/HSL.
      - HSL<=0 or NaN -> NaN (avoid divide-by-zero).
    """
    s_name, s_type = normalize_key_columns(stage_df)
    h_name, h_type = normalize_key_columns(hsl_df)

    # Normalize resource-name keys (casefolded, trimmed)
    def norm_key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.casefold()

    stage = stage_df.copy()
    hsl   = hsl_df.copy()
    stage["_key"] = norm_key(stage[s_name])
    hsl["_key"]   = norm_key(hsl[h_name])

    # If duplicates exist, keep first occurrence (customize if needed)
    stage = stage.drop_duplicates(subset=["_key"], keep="first")
    hsl   = hsl.drop_duplicates(subset=["_key"], keep="first")

    stage = stage.set_index("_key")
    hsl   = hsl.set_index("_key")

    # Timestamp intersections
    s_ts = detect_timestamp_columns(stage.reset_index(), (s_name, s_type))
    h_ts = detect_timestamp_columns(hsl.reset_index(),   (h_name, h_type))
    common_ts = sorted(set(s_ts).intersection(h_ts), key=lambda c: pd.to_datetime(c))
    if not common_ts:
        raise ValueError("No overlapping timestamp columns between MW stage and HSL.")

    # Common rows (resources)
    idx_common = stage.index.intersection(hsl.index)
    if idx_common.empty:
        raise ValueError("No overlapping Resource Name rows between MW stage and HSL (after normalization).")

    # Aligned numeric blocks
    stage_block = stage.loc[idx_common, common_ts].apply(pd.to_numeric, errors="coerce")
    hsl_block   = hsl.loc[idx_common,   common_ts].apply(pd.to_numeric, errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_vals = np.where(np.isfinite(hsl_block.values) & (hsl_block.values > 0.0),
                             stage_block.values / hsl_block.values,
                             np.nan)

    # Compose output with display names and fuel from the stage file
    disp_names = stage.loc[idx_common, s_name].astype(str)
    fuel_types = stage.loc[idx_common, s_type].astype(str).fillna("Unknown")

    out = pd.DataFrame(norm_vals, index=idx_common, columns=common_ts)
    out.insert(0, "Resource Type", fuel_types.values)
    out.insert(0, "Resource Name", disp_names.values)
    out.reset_index(drop=True, inplace=True)
    return out

def average_normalized_by_fuel(norm_df: pd.DataFrame) -> Dict[str, float]:
    """Average across all resources & timestamps for each fuel type -> scalar per fuel."""
    name_col, type_col = normalize_key_columns(norm_df)
    ts_cols = detect_timestamp_columns(norm_df, (name_col, type_col))
    vals = norm_df[ts_cols].apply(pd.to_numeric, errors="coerce")
    vals.insert(0, "Resource Type", norm_df[type_col].astype(str).fillna("Unknown"))
    by_fuel = vals.groupby("Resource Type", dropna=False).mean(numeric_only=True).mean(axis=1, numeric_only=True)
    return {str(k): float(v) for k, v in by_fuel.sort_index().items()}


def normalize_by_row_max_with_hsl_and_bp(step_dfs: Dict[int, pd.DataFrame],
                                         hsl_df: pd.DataFrame,
                                         bp_df: pd.DataFrame) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Align all inputs on common resources and timestamps.
    For each resource×timestamp cell, compute:
      M_sced = max over all SCED-MW steps
      M_all  = max(M_sced, HSL)

    Return:
      - norm_steps: dict[step]-> normalized (values / M_all)
      - hsl_norm  : normalized HSL (values / M_all)
      - bp_norm   : normalized Base Point (values / M_all)
      - max_sced_per_row: Series indexed by row (resource) with sum_over_timestamps(M_sced)
      - fuel_per_row: Series fuel type per resource row (from the first step that has it, else HSL/BP)
    """
    # Normalize keys
    # Find key columns for each df
    # Use first step to detect columns
    first_df = next(iter(step_dfs.values()))
    s_name, s_type = normalize_key_columns(first_df)
    h_name, h_type = normalize_key_columns(hsl_df)
    b_name, b_type = normalize_key_columns(bp_df)

    def norm_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.casefold()

    # Set index by normalized resource name
    step_aligned = {}
    for k, df in step_dfs.items():
        df2 = df.copy()
        df2["_key"] = norm_series(df2[s_name])
        df2 = df2.set_index("_key")
        step_aligned[k] = df2

    h2 = hsl_df.copy()
    h2["_key"] = norm_series(h2[h_name])
    h2 = h2.set_index("_key")

    b2 = bp_df.copy()
    b2["_key"] = norm_series(b2[b_name])
    b2 = b2.set_index("_key")

    # Common timestamp columns across all steps + HSL + BP
    ts_sets = []
    for df in step_aligned.values():
        name_col, type_col = normalize_key_columns(df.reset_index())
        ts_sets.append(set(detect_timestamp_columns(df.reset_index(), (name_col, type_col))))
    ts_sets.append(set(detect_timestamp_columns(h2.reset_index(), (h_name, h_type))))
    ts_sets.append(set(detect_timestamp_columns(b2.reset_index(), (b_name, b_type))))
    common_ts = sorted(set.intersection(*ts_sets), key=lambda c: pd.to_datetime(c))
    if not common_ts:
        raise ValueError("No overlapping timestamp columns across SCED steps, HSL, and Base Point.")

    # Common resource rows
    common_idx = set(h2.index) & set(b2.index)
    for df in step_aligned.values():
        common_idx &= set(df.index)
    common_idx = pd.Index(sorted(common_idx))
    if common_idx.empty:
        raise ValueError("No overlapping Resource Name rows across SCED steps, HSL, and Base Point.")

    # Extract numeric blocks
    step_vals = {k: df.loc[common_idx, common_ts].apply(pd.to_numeric, errors="coerce").to_numpy()
                 for k, df in step_aligned.items()}
    h_vals = h2.loc[common_idx, common_ts].apply(pd.to_numeric, errors="coerce").to_numpy()
    b_vals = b2.loc[common_idx, common_ts].apply(pd.to_numeric, errors="coerce").to_numpy()

    # Compute M_sced and M_all
    # Stack steps into 3D: (n_steps, n_rows, n_ts)
    steps_sorted = sorted(step_vals.keys())
    stack = np.stack([step_vals[k] for k in steps_sorted], axis=0)  # shape (S, R, T)
    M_sced = np.nanmax(stack, axis=0)  # (R, T)
    M_all = np.nanmax(np.stack([M_sced, h_vals], axis=0), axis=0)   # include HSL

    # Avoid zeros and non-finite
    M_all_safe = np.where(np.isfinite(M_all) & (M_all > 0.0), M_all, np.nan)

    # Normalize each step, HSL, BP
    norm_steps = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, k in enumerate(steps_sorted):
            norm = np.where(np.isfinite(M_all_safe), stack[i] / M_all_safe, np.nan)
            out = pd.DataFrame(norm, index=common_idx, columns=common_ts)
            # attach display columns from the step df
            src = step_aligned[k].loc[common_idx]
            out.insert(0, "Resource Type", src[s_type].astype(str).fillna("Unknown").values)
            out.insert(0, "Resource Name", src[s_name].astype(str).values)
            norm_steps[k] = out.reset_index(drop=True)

        h_norm = np.where(np.isfinite(M_all_safe), h_vals / M_all_safe, np.nan)
        hsl_norm = pd.DataFrame(h_norm, index=common_idx, columns=common_ts)
        hsl_norm.insert(0, "Resource Type", step_aligned[steps_sorted[0]].loc[common_idx][s_type].astype(str).fillna("Unknown").values)
        hsl_norm.insert(0, "Resource Name", step_aligned[steps_sorted[0]].loc[common_idx][s_name].astype(str).values)
        hsl_norm = hsl_norm.reset_index(drop=True)

        b_norm = np.where(np.isfinite(M_all_safe), b_vals / M_all_safe, np.nan)
        bp_norm = pd.DataFrame(b_norm, index=common_idx, columns=common_ts)
        bp_norm.insert(0, "Resource Type", step_aligned[steps_sorted[0]].loc[common_idx][s_type].astype(str).fillna("Unknown").values)
        bp_norm.insert(0, "Resource Name", step_aligned[steps_sorted[0]].loc[common_idx][s_name].astype(str).values)
        bp_norm = bp_norm.reset_index(drop=True)

    # Sum of largest SCED values per row (over timestamps)
    max_sced_per_row = pd.Series(np.nansum(M_sced, axis=1), index=range(len(common_idx)))  # temp index aligns with reset_index()

    # Fuel per row (from first step)
    fuel_per_row = step_aligned[steps_sorted[0]].loc[common_idx][s_type].astype(str).fillna("Unknown").reset_index(drop=True)

    return norm_steps, hsl_norm, bp_norm, max_sced_per_row, fuel_per_row




def process_sced_normalized_lines_day(day_dir: Path, plots_root: Path, save_summary_csv: bool) -> None:
    """
    UPDATED:
      - Build row-wise maxima across ALL SCED MW steps (M_sced).
      - Define M_all = max(M_sced, HSL).
      - Normalize each SCED step, HSL, and Base Point by M_all.
      - Aggregate by fuel, average over resources×timestamps.
      - Reapply magnitude by multiplying the averaged normalized values by SUM(M_sced) per fuel.
      - Plot one line per fuel across steps; add dashed horizontal lines for HSL and Base Point (rescaled magnitudes).
    """
    day = day_dir.name

    # Load HSL
    hsl_path = day_dir / "aggregation_HSL.csv"
    if not hsl_path.exists():
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

    # Load Base Point
    bp_path = day_dir / "aggregation_Base_Point.csv"
    if not bp_path.exists():
        matches = list(day_dir.glob("*Base*Point*.csv"))
        if not matches:
            print(f"[INFO] {day}: no Base Point file; skipping normalized MW lines.")
            return
        bp_path = matches[0]
    try:
        bp_df = pd.read_csv(bp_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: failed to read Base Point {bp_path.name}: {e}")
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

    # Load step frames
    step_dfs: Dict[int, pd.DataFrame] = {}
    for p in mw_files:
        m = MW_STEP_PATTERN.search(p.name)
        step_num = int(m.group(1)) if m else None
        if step_num is None:
            continue
        try:
            step_dfs[step_num] = pd.read_csv(p, dtype=str)
        except Exception as e:
            print(f"[WARN] {day}: failed to read {p.name}: {e}")

    if not step_dfs:
        print(f"[INFO] {day}: no readable SCED MW step files; skipping.")
        return

    try:
        norm_steps, hsl_norm, bp_norm, max_sced_per_row, fuel_per_row = normalize_by_row_max_with_hsl_and_bp(step_dfs, hsl_df, bp_df)
    except Exception as e:
        print(f"[ERROR] {day}: normalization by row-max failed: {e}")
        return

    # Compute rescaling magnitude per fuel: sum of largest SCED values (M_sced) per row, then sum within fuel
    # max_sced_per_row index aligns with fuel_per_row (both reset)
    fuel_series = fuel_per_row.astype(str).fillna("Unknown")
    rescale_by_fuel = max_sced_per_row.groupby(fuel_series).sum(min_count=1)

    # Build per-step averaged normalized (over resources×timestamps) by fuel, then rescale
    steps_sorted = sorted(norm_steps.keys())
    fuels_sorted = sorted(rescale_by_fuel.index.tolist())

    def avg_over_rows_and_time(df: pd.DataFrame) -> pd.Series:
        name_col, type_col = normalize_key_columns(df)
        ts_cols = detect_timestamp_columns(df, (name_col, type_col))
        vals = df[ts_cols].apply(pd.to_numeric, errors="coerce")
        vals.insert(0, "Resource Type", df[type_col].astype(str).fillna("Unknown"))
        by_fuel = vals.groupby("Resource Type", dropna=False).mean(numeric_only=True).mean(axis=1, numeric_only=True)
        return by_fuel

    scaled_by_step = {}
    for s in steps_sorted:
        avg_norm = avg_over_rows_and_time(norm_steps[s])
        # align with fuels_sorted and rescale
        scaled = []
        for f in fuels_sorted:
            base = avg_norm.get(f, np.nan)
            scale = rescale_by_fuel.get(f, np.nan)
            scaled.append(float(base) * float(scale) if np.isfinite(base) and np.isfinite(scale) else np.nan)
        scaled_by_step[s] = scaled

    summary_scaled = pd.DataFrame.from_dict(scaled_by_step, orient="index", columns=fuels_sorted)
    summary_scaled.index.name = "SCED Step"

    # Compute HSL and Base Point dashed lines (same normalization and rescaling)
    hsl_avg = avg_over_rows_and_time(hsl_norm)
    bp_avg  = avg_over_rows_and_time(bp_norm)

    hsl_scaled = pd.Series({f: (hsl_avg.get(f, np.nan) * rescale_by_fuel.get(f, np.nan)) for f in fuels_sorted})
    bp_scaled  = pd.Series({f: (bp_avg.get(f, np.nan)  * rescale_by_fuel.get(f, np.nan)) for f in fuels_sorted})

    # Plot
    day_out = plots_root / day / "sced_normalized"
    day_out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    for f in fuels_sorted:
        ax.plot(summary_scaled.index.values, summary_scaled[f].values, marker="o", label=f)

    # dashed horizontals
    for f in fuels_sorted:
        y_hsl = hsl_scaled.get(f, np.nan)
        y_bp  = bp_scaled.get(f, np.nan)
        if np.isfinite(y_hsl):
            ax.hlines(y=y_hsl, xmin=summary_scaled.index.min(), xmax=summary_scaled.index.max(),
                      linestyles="dashed", linewidth=1.5, label=f"{f} – HSL")
        if np.isfinite(y_bp):
            ax.hlines(y=y_bp, xmin=summary_scaled.index.min(), xmax=summary_scaled.index.max(),
                      linestyles="dashed", linewidth=1.5, label=f"{f} – Base Point")

    ax.set_title(f"{day} – SCED Steps normalized by row-wise max (rescaled by sum of row maxima)")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("Scaled Value (sum of row max SCED MW × average normalized)")
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    out_png = day_out / "normalized_bids_by_stage.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out_png}")

    if save_summary_csv:
        out_csv = day_out / "normalized_bids_by_stage.csv"
        try:
            summary_scaled.to_csv(out_csv)
            print(f"[OK] Saved: {out_csv}")
        except Exception as e:
            print(f"[ERROR] {day}: failed to write summary CSV: {e}")
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
