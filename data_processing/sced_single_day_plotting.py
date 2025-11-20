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
from typing import Dict, Tuple, List, Optional, Iterable
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== EDIT THESE =====================
ROOT_DIR  = r"C:/Users/L1165683/GitHub_Repos/data-processing/output/sced_single_day_data"   # contains YYYY-MM-DD subfolders
PLOTS_DIR = r"C:/Users/L1165683/GitHub_Repos/data-processing/output/sced_single_day_plotting_sced1"     # results written here, mirrored per-day
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

    # Filter to the exact day
    hourly_df = hourly_df[(hourly_df.index >= day_start) & (hourly_df.index < day_end)]
    hourly_df = hourly_df.reindex(
        pd.date_range(day_start, day_end, freq="h", inclusive="left"),
        fill_value=0.0,
    )

    # Reindex so x-axis is 0..23
    plot_df = hourly_df.copy()
    plot_df.index = plot_df.index.hour

    # Extract fuel names (columns)
    fuels = list(plot_df.columns)

    # ----- Assign unique consistent colors -----
    # Use matplotlib’s largest categorical palette (20 colors)
    import matplotlib.pyplot as plt
    import numpy as np

    base_colors = plt.get_cmap("tab20").colors  # length 20
    # Expand if there are >20 fuels
    n = len(fuels)
    if n <= 20:
        colors = base_colors[:n]
    else:
        # Interpolate more colors if needed
        expanded = plt.get_cmap("tab20")
        colors = expanded(np.linspace(0, 1, n))

    fuel_to_color = {fuel: colors[i] for i, fuel in enumerate(fuels)}

    # Plot with explicit colors
    fig, ax = plt.subplots(figsize=(14, 7))

    bottom = np.zeros(len(plot_df))
    x = plot_df.index.values

    for fuel in fuels:
        ax.bar(
            x,
            plot_df[fuel].values,
            bottom=bottom,
            color=fuel_to_color[fuel],
            label=fuel,
            width=0.9,
        )
        bottom += plot_df[fuel].values

    # ----- Labels & formatting -----
    ylabel = "Average MW" if agg_mode.lower() == "mean" else "MWh"
    ax.set_title(f"{day} – Base Point by Fuel Type (Hourly {ylabel})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.legend(
        title="Fuel Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.,
    )

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

def process_base_point_telemetered_output(day_dir: Path, plots_root: Path) -> None:
    """
    Compare Telemetered Net Output vs Base Point and plot error (tele - dispatch) by fuel.
    Generates a box-and-whisker plot per day and saves the flattened values used to plot.
    """
    day = day_dir.name

    bp_path = day_dir / "aggregation_Base_Point.csv"
    if not bp_path.exists():
        matches = list(day_dir.glob("aggregation_Base_Point*.csv")) or list(day_dir.glob("*Base*Point*.csv"))
        if not matches:
            print(f"[INFO] {day}: no Base Point file; skipping telemetered comparison.")
            return
        bp_path = matches[0]

    tele_path = day_dir / "aggregation_Telemetered_Net_Output.csv"
    if not tele_path.exists():
        matches = list(day_dir.glob("*Telemetered*Net*Output*.csv"))
        if not matches:
            print(f"[INFO] {day}: no Telemetered Net Output file; skipping telemetered comparison.")
            return
        tele_path = matches[0]

    try:
        bp_df = pd.read_csv(bp_path, dtype=str)
        tele_df = pd.read_csv(tele_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: failed to read Base Point/Telemetered files: {e}")
        return

    try:
        bp_name, bp_type = normalize_key_columns(bp_df)
        tele_name, tele_type = normalize_key_columns(tele_df)
        ts_bp   = detect_timestamp_columns(bp_df,   (bp_name,   bp_type))
        ts_tele = detect_timestamp_columns(tele_df, (tele_name, tele_type))
    except Exception as e:
        print(f"[ERROR] {day}: column detection failed: {e}")
        return

    ts_common = sorted(set(ts_bp).intersection(ts_tele), key=lambda c: pd.to_datetime(c))
    if not ts_common:
        print(f"[INFO] {day}: no overlapping timestamp columns between Base Point and Telemetered; skipping.")
        return

    def _key_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.casefold()

    def _prep(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
        d = df.copy()
        d["_key"] = _key_series(d[name_col])
        d = d.drop_duplicates(subset=["_key"], keep="first").set_index("_key")
        return d

    bp_prep = _prep(bp_df, bp_name)
    tele_prep = _prep(tele_df, tele_name)
    master_idx = sorted(set(bp_prep.index).intersection(set(tele_prep.index)))
    if not master_idx:
        print(f"[INFO] {day}: no overlapping resources between Base Point and Telemetered; skipping.")
        return

    def _numeric_block(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return df.reindex(master_idx).reindex(columns=cols).apply(pd.to_numeric, errors="coerce")

    bp_block = _numeric_block(bp_prep, ts_common)
    tele_block = _numeric_block(tele_prep, ts_common)
    error_vals = tele_block - bp_block  # telemetered minus dispatched

    # fuel/resource names for reporting
    fuel_series = tele_prep.reindex(master_idx)[tele_type]
    fuel_series = fuel_series.fillna(bp_prep.reindex(master_idx)[bp_type]).astype(str).replace("", "Unknown")
    name_series = tele_prep.reindex(master_idx)[tele_name]
    name_series = name_series.fillna(bp_prep.reindex(master_idx)[bp_name]).astype(str)

    error_df = pd.DataFrame(error_vals, index=master_idx, columns=ts_common)
    error_df.insert(0, "Resource Type", fuel_series.values)
    error_df.insert(0, "Resource Name", name_series.values)
    error_df.reset_index(drop=True, inplace=True)

    # Aggregate error values by fuel
    errors_by_fuel: Dict[str, np.ndarray] = {}
    for fuel, block in error_df.groupby("Resource Type"):
        vals = block[ts_common].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            errors_by_fuel[str(fuel)] = vals.astype(float)

    if not errors_by_fuel:
        print(f"[INFO] {day}: no finite error values to plot; skipping telemetered comparison.")
        return

    fuels_sorted = sorted(errors_by_fuel.keys())
    datasets = [errors_by_fuel[f] for f in fuels_sorted]

    day_out = (plots_root / day / "base_point_telemetered")
    day_out.mkdir(parents=True, exist_ok=True)

    # Plot box/whisker
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(datasets, labels=fuels_sorted, showfliers=True)
    ax.set_title(f"{day} – Telemetered Output minus Base Point by Fuel")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Error (Telemetered - Base Point)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = day_out / "bp_telemetered_error_boxplot.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    # Save flattened values used for the plot
    long_rows = [{"fuel": f, "error": float(v)} for f, arr in errors_by_fuel.items() for v in arr]
    out_csv = day_out / "bp_telemetered_error_values.csv"
    try:
        pd.DataFrame(long_rows).to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")
    except Exception as e:
        print(f"[WARN] {day}: failed to write telemetered error CSV: {e}")


def process_base_point_monthlies(day_dir: Path, plots_root: Path, resource_type: str, save_values_csv: bool) -> None:
    """
    Aggregate Base Point for a specific Resource Type:
      - Filter rows matching `resource_type`.
      - Sum across units per timestamp, then aggregate by hour-of-day.
      - Plot MW vs hour-of-day and optionally save the hourly values.
    """
    day = day_dir.name
    csv_path = day_dir / "aggregation_Base_Point.csv"
    if not csv_path.exists():
        matches = list(day_dir.glob("aggregation_Base_Point*.csv")) or list(day_dir.glob("*Base*Point*.csv"))
        if not matches:
            print(f"[INFO] {day}: no Base Point file; skipping monthlies for {resource_type}.")
            return
        csv_path = matches[0]

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: failed to read Base Point {csv_path.name}: {e}")
        return

    try:
        name_col, type_col = normalize_key_columns(df)
        ts_cols = detect_timestamp_columns(df, (name_col, type_col))
    except Exception as e:
        print(f"[ERROR] {day}: column detection failed for Base Point: {e}")
        return

    resource_norm = str(resource_type).strip().casefold()
    fuels = df[type_col].astype(str).fillna("Unknown").str.strip()
    mask = fuels.str.casefold() == resource_norm
    if not mask.any():
        print(f"[INFO] {day}: no rows with Resource Type '{resource_type}' in Base Point.")
        return

    filtered = df.loc[mask, ts_cols]
    numeric = filtered.apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        print(f"[INFO] {day}: filtered Base Point data empty for {resource_type}.")
        return

    totals = numeric.sum(axis=0, skipna=True)
    ts_index = pd.to_datetime(totals.index, errors="coerce")
    valid_mask = ts_index.notna()
    totals = totals[valid_mask]
    ts_index = ts_index[valid_mask]
    if totals.empty:
        print(f"[INFO] {day}: no valid timestamps for Base Point monthlies ({resource_type}).")
        return

    series = pd.Series(totals.values, index=ts_index).sort_index()
    hourly = series.groupby(series.index.hour).mean()
    hours = list(range(24))
    hourly = hourly.reindex(hours, fill_value=np.nan)

    day_out = Path(plots_root) / day / "sced_base_point_monthlies"
    day_out.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^0-9A-Za-z]+", "_", resource_type).strip("_") or "fuel"

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(hours)
    y = hourly.to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_title(f"{day} – Base Point (Fuel: {resource_type}) by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Aggregate MW")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = day_out / f"{slug}_base_point_hourly.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    if save_values_csv:
        out_csv = day_out / f"{slug}_base_point_hourly.csv"
        try:
            pd.DataFrame({"hour": hours, "mw": y}).to_csv(out_csv, index=False)
            print(f"[OK] Saved: {out_csv}")
        except Exception as e:
            print(f"[WARN] {day}: failed to write Base Point monthlies CSV: {e}")


def process_representative_quantity_curves(day_dir: Path, plots_root: Path, resource_type: str, save_summary: bool) -> Optional[pd.DataFrame]:
    """
    Build representative SCED quantity curves for a specific fuel:
      - Filter Base Point, HSL, and SCED MW steps to the requested resource_type (case-insensitive).
      - For each resource, assemble its SCED curve across steps per timestamp, normalize by its max, and
        keep the max for scaling.
      - Average normalized curves across resources per timestamp → multiply by averaged max → representative curve.
      - Plot MW vs SCED Step; optionally save the per-timestamp curves and representative curve to CSV.
    """
    day = day_dir.name

    # Base Point + HSL (only to validate presence)
    bp_path = day_dir / "aggregation_Base_Point.csv"
    if not bp_path.exists():
        matches = list(day_dir.glob("aggregation_Base_Point*.csv")) or list(day_dir.glob("*Base*Point*.csv"))
        if not matches:
            print(f"[INFO] {day}: no Base Point file; skipping representative curve for {resource_type}.")
            return
        bp_path = matches[0]
    hsl_path = day_dir / "aggregation_HSL.csv"
    if not hsl_path.exists():
        matches = list(day_dir.glob("*HSL*.csv"))
        if not matches:
            print(f"[INFO] {day}: no HSL file; skipping representative curve for {resource_type}.")
            return
        hsl_path = matches[0]
    try:
        bp_df = pd.read_csv(bp_path, dtype=str)
        hsl_df = pd.read_csv(hsl_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] {day}: failed to read Base Point/HSL: {e}")
        return

    resource_norm = str(resource_type).strip().casefold()

    def _prep_filtered_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        name_col, type_col = normalize_key_columns(df)
        ts_cols = detect_timestamp_columns(df, (name_col, type_col))
        mask = df[type_col].astype(str).fillna("Unknown").str.strip().str.casefold() == resource_norm
        filt = df.loc[mask].copy()
        if filt.empty:
            return pd.DataFrame(), []
        filt["_key"] = filt[name_col].astype(str).str.strip().str.casefold()
        filt = filt.drop_duplicates(subset=["_key"], keep="first").set_index("_key")
        ts_cols_sorted = sorted(ts_cols, key=lambda c: pd.to_datetime(c))
        return filt.reindex(columns=ts_cols_sorted), ts_cols_sorted

    bp_filtered, bp_ts_cols = _prep_filtered_matrix(bp_df)
    hsl_filtered, hsl_ts_cols = _prep_filtered_matrix(hsl_df)
    if bp_filtered.empty or hsl_filtered.empty:
        print(f"[INFO] {day}: No Base Point/HSL rows for Resource Type '{resource_type}'.")
        return

    mw_files = list(day_dir.glob("aggregation_SCED1_Curve-MW*.csv"))
    if not mw_files:
        mw_files = list(day_dir.glob("*SCED*Curve*MW*.csv"))
    if not mw_files:
        print(f"[INFO] {day}: no SCED MW step files; skipping representative curve for {resource_type}.")
        return

    def step_key(p: Path):
        m = MW_STEP_PATTERN.search(p.name)
        return int(m.group(1)) if m else 1_000_000
    mw_files = sorted(mw_files, key=step_key)

    filtered_steps: Dict[int, pd.DataFrame] = {}
    ts_by_step: Dict[int, List[str]] = {}
    for p in mw_files:
        try:
            df = pd.read_csv(p, dtype=str)
        except Exception as e:
            print(f"[WARN] {day}: failed to read {p.name}: {e}")
            continue
        filt, ts_cols_sorted = _prep_filtered_matrix(df)
        if filt.empty:
            continue
        step = step_key(p)
        filtered_steps[step] = filt
        ts_by_step[step] = ts_cols_sorted

    if not filtered_steps:
        print(f"[INFO] {day}: no SCED rows for Resource Type '{resource_type}'.")
        return

    steps_sorted = sorted(filtered_steps.keys())
    bp_keys = set(bp_filtered.index)
    hsl_keys = set(hsl_filtered.index)
    step_key_sets = [set(df.index) for df in filtered_steps.values()]
    common_keys = sorted(set.intersection(*(step_key_sets + [bp_keys, hsl_keys])))
    if not common_keys:
        print(f"[INFO] {day}: no overlapping resources across SCED steps for '{resource_type}'.")
        return
    common_ts = sorted(
        set.intersection(
            *( [set(ts_by_step[s]) for s in steps_sorted] + [set(bp_ts_cols), set(hsl_ts_cols)] )
        ),
        key=lambda c: pd.to_datetime(c)
    )
    if not common_ts:
        print(f"[INFO] {day}: no common timestamps across SCED steps for '{resource_type}'.")
        return

    # Build 3D array: steps x keys x times
    n_steps = len(steps_sorted)
    n_keys = len(common_keys)
    n_times = len(common_ts)
    curves = np.full((n_steps, n_keys, n_times), np.nan, dtype=float)
    for i, step in enumerate(steps_sorted):
        df = filtered_steps[step].reindex(index=common_keys, columns=common_ts)
        curves[i, :, :] = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Max per resource/timestamp
    with np.errstate(all="ignore"):
        max_vals = np.nanmax(curves, axis=0)  # (keys, times)
    max_vals[~np.isfinite(max_vals)] = np.nan
    if np.isnan(max_vals).all():
        print(f"[INFO] {day}: SCED curves all NaN for '{resource_type}'.")
        return

    # Normalize curves per resource per timestamp
    with np.errstate(all="ignore"):
        norm_curves = curves / max_vals[np.newaxis, :, :]
        bp_norm = bp_filtered.reindex(index=common_keys, columns=common_ts).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) / max_vals
        hsl_norm = hsl_filtered.reindex(index=common_keys, columns=common_ts).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) / max_vals

    # Average normalized curves across resources per timestamp (times x steps)
    mean_norm = np.nanmean(norm_curves, axis=1).transpose(1, 0)  # (times, steps)
    avg_max = np.nanmean(max_vals, axis=0)  # (times,)
    scaled = mean_norm * avg_max[:, None]

    bp_scaled = np.nanmean(bp_norm, axis=0) * avg_max
    hsl_scaled = np.nanmean(hsl_norm, axis=0) * avg_max

    ts_index = pd.to_datetime(common_ts)
    step_index = pd.Index(steps_sorted, name="SCED Step")
    curves_only = pd.DataFrame(scaled, index=ts_index, columns=step_index)
    representative = curves_only.mean(axis=0, skipna=True)
    bp_level = float(np.nanmean(bp_scaled)) if np.isfinite(bp_scaled).any() else np.nan
    hsl_level = float(np.nanmean(hsl_scaled)) if np.isfinite(hsl_scaled).any() else np.nan

    out_dir = Path(plots_root) / day / "sced_quantity_curve_representative"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^0-9A-Za-z]+", "_", resource_type).strip("_") or "fuel"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_index.values, representative.to_numpy(dtype=float), marker="o", linewidth=2)
    if np.isfinite(bp_level):
        ax.axhline(bp_level, linestyle="--", linewidth=1.5, color="#1f77b4", label="Base Point")
    if np.isfinite(hsl_level):
        ax.axhline(hsl_level, linestyle="--", linewidth=1.5, color="#ff7f0e", label="HSL")
    ax.set_title(f"{day} – Representative Quantity Curve (Fuel: {resource_type})")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("MW")
    ax.set_xticks(step_index.values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = out_dir / f"{slug}_representative_curve.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    rep_df = pd.DataFrame({
        "SCED Step": step_index.values,
        "Representative MW": representative.values,
        "Base Point MW": [bp_level] * len(representative),
        "HSL MW": [hsl_level] * len(representative),
    })

    if save_summary:
        rep_csv = out_dir / f"{slug}_representative_curve.csv"
        try:
            rep_df.to_csv(rep_csv, index=False)
            print(f"[OK] Saved: {rep_csv}")
        except Exception as e:
            print(f"[WARN] {day}: failed to write representative curve CSV: {e}")

    return rep_df


def process_representative_price_curves(day_dir: Path, plots_root: Path, resource_type: str, save_summary: bool) -> Optional[pd.DataFrame]:
    """
    Representative SCED price curve for a given fuel:
      - Filter all SCED price steps to the specified Resource Type.
      - Construct per-resource SCED price curves across steps and average them per timestamp.
      - Average across timestamps to obtain a single SCED-step price curve.
      - Plot Bid Price vs SCED Step and optionally save the curve to CSV.
    """
    day = day_dir.name

    # find price step files (same discovery as process_sced_price_lines_day)
    step_files = list(day_dir.glob("aggregation_SCED1_Curve-Price*.csv"))
    if not step_files:
        step_files = list(day_dir.glob("*SCED*Curve*Price*.csv"))
    if not step_files:
        print(f"[INFO] {day}: no SCED price files; skipping representative price curves.")
        return

    def step_key(p: Path):
        m = PRICE_STEP_PATTERN.search(p.name)
        return int(m.group(1)) if m else 1_000_000
    step_files = sorted(step_files, key=step_key)

    resource_norm = str(resource_type).strip().casefold()

    def _prep_filtered_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        name_col, type_col = normalize_key_columns(df)
        ts_cols = detect_timestamp_columns(df, (name_col, type_col))
        mask = df[type_col].astype(str).fillna("Unknown").str.strip().str.casefold() == resource_norm
        filt = df.loc[mask].copy()
        if filt.empty:
            return pd.DataFrame(), []
        filt["_key"] = filt[name_col].astype(str).str.strip().str.casefold()
        filt = filt.drop_duplicates(subset=["_key"], keep="first").set_index("_key")
        ts_cols_sorted = sorted(ts_cols, key=lambda c: pd.to_datetime(c))
        return filt.reindex(columns=ts_cols_sorted), ts_cols_sorted

    filtered_steps: Dict[int, pd.DataFrame] = {}
    ts_by_step: Dict[int, List[str]] = {}
    for p in step_files:
        try:
            df = pd.read_csv(p, dtype=str)
        except Exception as e:
            print(f"[WARN] {day}: failed to read {p.name}: {e}")
            continue
        filt, ts_cols_sorted = _prep_filtered_matrix(df)
        if filt.empty:
            continue
        step = step_key(p)
        filtered_steps[step] = filt
        ts_by_step[step] = ts_cols_sorted

    if not filtered_steps:
        print(f"[INFO] {day}: no SCED price rows for Resource Type '{resource_type}'.")
        return

    steps_sorted = sorted(filtered_steps.keys())
    common_keys = sorted(set.intersection(*(set(df.index) for df in filtered_steps.values())))
    if not common_keys:
        print(f"[INFO] {day}: no overlapping resources across SCED price steps for '{resource_type}'.")
        return

    common_ts = sorted(
        set.intersection(*(set(ts_by_step[s]) for s in steps_sorted)),
        key=lambda c: pd.to_datetime(c)
    )
    if not common_ts:
        print(f"[INFO] {day}: no common timestamps across SCED price steps for '{resource_type}'.")
        return

    n_steps = len(steps_sorted)
    n_keys = len(common_keys)
    n_times = len(common_ts)
    curves = np.full((n_steps, n_keys, n_times), np.nan, dtype=float)
    for i, step in enumerate(steps_sorted):
        df = filtered_steps[step].reindex(index=common_keys, columns=common_ts)
        curves[i, :, :] = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    with np.errstate(all="ignore"):
        mean_curves = np.nanmean(curves, axis=1).transpose(1, 0)  # (times, steps)
    if np.isnan(mean_curves).all():
        print(f"[INFO] {day}: SCED price curves all NaN for '{resource_type}'.")
        return

    ts_index = pd.to_datetime(common_ts)
    step_index = pd.Index(steps_sorted, name="SCED Step")
    curves_df = pd.DataFrame(mean_curves, index=ts_index, columns=step_index)
    representative = curves_df.mean(axis=0, skipna=True)

    out_dir = Path(plots_root) / day / "sced_price_curve_representative"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^0-9A-Za-z]+", "_", resource_type).strip("_") or "fuel"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_index.values, representative.to_numpy(dtype=float), marker="o", linewidth=2)
    ax.set_title(f"{day} – Representative Price Curve (Fuel: {resource_type})")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("Bid Price")
    ax.set_xticks(step_index.values)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / f"{slug}_representative_price_curve.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    rep_df = pd.DataFrame({
        "SCED Step": step_index.values,
        "Representative Price": representative.values,
    })

    if save_summary:
        rep_csv = out_dir / f"{slug}_representative_price_curve.csv"
        try:
            rep_df.to_csv(rep_csv, index=False)
            print(f"[OK] Saved: {rep_csv}")
        except Exception as e:
            print(f"[WARN] {day}: failed to write representative price curve CSV: {e}")

    return rep_df

def process_average_bid_quantity(day_dir: Path, plots_root: Path) -> None:
    """
    Build a box/whisker plot of all SCED price values across steps, grouped by fuel.
    Uses any aggregation_SCED1_Curve_Price*.csv (underscore variant) in the day folder.
    Saves both the plot and the flattened values by fuel.
    """
    day = day_dir.name
    price_files = list(day_dir.glob("aggregation_SCED1_Curve_Price*.csv"))
    if not price_files:
        price_files = list(day_dir.glob("*SCED*Curve*Price*.csv"))
    if not price_files:
        print(f"[INFO] {day}: no SCED price files found for average bid quantity plot.")
        return

    prices_by_fuel: Dict[str, List[float]] = {}

    for p in sorted(price_files):
        try:
            df = pd.read_csv(p, dtype=str)
        except Exception as e:
            print(f"[WARN] {day}: failed to read {p.name}: {e}")
            continue
        try:
            data_by_fuel = load_violin_data_by_fuel(df)
        except Exception as e:
            print(f"[WARN] {day}: failed processing {p.name}: {e}")
            continue
        for fuel, arr in data_by_fuel.items():
            if arr is None or arr.size == 0:
                continue
            prices_by_fuel.setdefault(str(fuel), []).extend(arr.astype(float).tolist())

    if not prices_by_fuel:
        print(f"[INFO] {day}: no finite SCED prices to plot for average bid quantity.")
        return

    fuels_sorted = sorted(prices_by_fuel.keys())
    datasets = [np.asarray(prices_by_fuel[f], dtype=float) for f in fuels_sorted]

    day_out = plots_root / day / "average_bid_quantity"
    day_out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(datasets, labels=fuels_sorted, showfliers=True)
    ax.set_title(f"{day} – SCED Price distribution by Fuel")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Price")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = day_out / "average_bid_price_boxplot.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    rows = [{"fuel": f, "price": float(v)} for f, data in prices_by_fuel.items() for v in data]
    out_csv = day_out / "average_bid_price_values.csv"
    try:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")
    except Exception as e:
        print(f"[WARN] {day}: failed to write average bid price CSV: {e}")


def process_average_mw_bid(summary_scaled: pd.DataFrame, day_dir: Path, plots_root: Path) -> None:
    """
    Given summary_scaled (index: SCED Step, columns: fuels), plot average MW per fuel (skip first row).
    Saves both the bar chart and the numeric averages.
    """
    day = day_dir.name
    if summary_scaled is None or summary_scaled.shape[0] < 2:
        print(f"[INFO] {day}: not enough rows in summary_scaled to compute average MW bid (need >=2).")
        return

    df = summary_scaled.copy()
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    mean_by_fuel = df_numeric.iloc[1:].mean(axis=0, skipna=True)  # ignore first row
    mean_by_fuel = mean_by_fuel.dropna()
    if mean_by_fuel.empty:
        print(f"[INFO] {day}: no finite averages for MW bid; skipping bar chart.")
        return

    fuels = mean_by_fuel.index.tolist()
    vals = mean_by_fuel.to_numpy(dtype=float)

    day_out = plots_root / day / "average_mw_bid"
    day_out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(fuels, vals)
    ax.set_title(f"{day} – Average MW bid by Fuel (excluding first SCED step)")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Average MW bid")
    ax.set_xticklabels(fuels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = day_out / "average_mw_bid_bar.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    out_csv = day_out / "average_mw_bid_values.csv"
    try:
        mean_by_fuel.to_csv(out_csv, header=["average_mw_bid"])
        print(f"[OK] Saved: {out_csv}")
    except Exception as e:
        print(f"[WARN] {day}: failed to write average MW bid CSV: {e}")

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

def process_sced_price_lines_day(day_dir: Path, plots_root: Path, save_values_csv: bool, per_fuel: bool = True,) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Build SCED price line plots across steps, by fuel.
      - For each price step CSV, flatten all resource×time prices per fuel and compute Q1, median, Q3.
      - Combined chart (all fuels): median lines ONLY (no IQR shading).
      - Per-fuel charts: median line with Q1–Q3 shaded band.
      - Save wide CSVs for medians, Q1s, and Q3s.

    Assumes helpers exist:
      - normalize_key_columns(df) -> (name_col, type_col)
      - detect_timestamp_columns(df, (name_col, type_col)) -> list[str] sorted
      - PRICE_STEP_PATTERN: compiled regex extracting step number from filename.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    MIN_SAMPLES = 5  # per-step minimum samples per fuel to compute quartiles

    day = day_dir.name
    day_out = plots_root / day
    day_out.mkdir(parents=True, exist_ok=True)

    # find price step files
    step_files = list(day_dir.glob("aggregation_SCED1_Curve-Price*.csv"))
    if not step_files:
        step_files = list(day_dir.glob("*SCED*Curve*Price*.csv"))
    if not step_files:
        print(f"[INFO] {day}: no SCED price files; skipping price line plots.")
        return

    def step_key(p: Path):
        m = PRICE_STEP_PATTERN.search(p.name)
        return int(m.group(1)) if m else 1_000_000

    step_files = sorted(step_files, key=step_key)

    # Hold per-step quartiles (Series indexed by fuel)
    q1_by_step: dict[int, pd.Series] = {}
    med_by_step: dict[int, pd.Series] = {}
    q3_by_step: dict[int, pd.Series] = {}
    fuels_seen: set[str] = set()

    for p in step_files:
        m = PRICE_STEP_PATTERN.search(p.name)
        step_num = int(m.group(1)) if m else None
        step_lbl = step_num if step_num is not None else 1_000_000

        try:
            df = pd.read_csv(p, dtype=str)
        except Exception as e:
            print(f"[ERROR] {day}: read failed for {p.name}: {e}")
            continue

        # Extract numeric prices and fuel
        try:
            name_col, type_col = normalize_key_columns(df)
            ts_cols = detect_timestamp_columns(df, (name_col, type_col))
            if not ts_cols or type_col is None:
                print(f"[INFO] {day}: {p.name} – no timestamps or fuel column; skipping.")
                continue

            prices = df[ts_cols].apply(pd.to_numeric, errors="coerce")
            fuels = df[type_col].astype(str).fillna("Unknown")
            # long-ish aggregation by fuel: flatten resource×time to 1D per fuel
            prices = prices.assign(__fuel__=fuels)

            q1_dict, med_dict, q3_dict = {}, {}, {}
            skipped_fuels = []
            for f, block in prices.groupby("__fuel__"):
                vals = block[ts_cols].to_numpy().ravel()
                vals = vals[np.isfinite(vals)]
                if vals.size < MIN_SAMPLES:
                    skipped_fuels.append(str(f))
                    continue
                q1_dict[str(f)]  = float(np.quantile(vals, 0.25))
                med_dict[str(f)] = float(np.quantile(vals, 0.50))
                q3_dict[str(f)]  = float(np.quantile(vals, 0.75))

            if skipped_fuels:
                print(f"[INFO] {day}: {p.name} – skipped empty/short fuels: {', '.join(skipped_fuels)}")

            if not med_dict:
                print(f"[INFO] {day}: {p.name} – no fuels with enough samples; skipping step.")
                continue

            q1_by_step[step_lbl]  = pd.Series(q1_dict,  dtype="float64")
            med_by_step[step_lbl] = pd.Series(med_dict, dtype="float64")
            q3_by_step[step_lbl]  = pd.Series(q3_dict,  dtype="float64")
            fuels_seen.update(med_dict.keys())

        except Exception as e:
            print(f"[ERROR] {day}: processing failed for {p.name}: {e}")
            continue

    if not med_by_step:
        print(f"[INFO] {day}: no usable steps for price lines; done.")
        return

    # Build wide DataFrames (index = step, columns = fuel)
    fuels_sorted = sorted(fuels_seen)
    steps_sorted = sorted(med_by_step.keys())

    def _to_wide(dstep: dict[int, pd.Series]) -> pd.DataFrame:
        wide = pd.DataFrame(index=steps_sorted, columns=fuels_sorted, dtype="float64")
        for s in steps_sorted:
            if s in dstep:
                for f, v in dstep[s].items():
                    wide.loc[s, f] = v
        wide.index.name = "SCED Step"
        return wide

    med_wide = _to_wide(med_by_step)
    q1_wide  = _to_wide(q1_by_step)
    q3_wide  = _to_wide(q3_by_step)

    # ---------------- Combined chart (all fuels): MEDIAN LINES ONLY ----------------
    out_lines_dir = day_out / "sced_price_lines"
    out_lines_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    for f in fuels_sorted:
        series = pd.to_numeric(med_wide[f], errors="coerce").to_numpy()
        if np.isfinite(series).any():
            ax.plot(med_wide.index.values, series, marker="o", linewidth=2, label=f)

    ax.set_title(f"{day} – SCED Price (Median by Fuel) across Steps")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("Price (Median)")
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = out_lines_dir / "price_lines_all_fuels.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")

    # ---------------- Per-fuel charts: MEDIAN LINE + IQR (Q1–Q3) SHADED ----------------
    if per_fuel:
        per_fuel_dir = out_lines_dir / "per_fuel"
        per_fuel_dir.mkdir(parents=True, exist_ok=True)

        x = med_wide.index.values
        for f in fuels_sorted:
            y_med = pd.to_numeric(med_wide[f], errors="coerce").to_numpy()
            y_q1  = pd.to_numeric(q1_wide[f],  errors="coerce").to_numpy()
            y_q3  = pd.to_numeric(q3_wide[f],  errors="coerce").to_numpy()

            if not (np.isfinite(y_med).any() or np.isfinite(y_q1).any() or np.isfinite(y_q3).any()):
                continue  # nothing to plot

            fig, ax = plt.subplots(figsize=(10, 6))
            # shaded IQR where both bounds are finite
            mask = np.isfinite(y_q1) & np.isfinite(y_q3)
            if mask.any():
                ax.fill_between(x[mask], y_q1[mask], y_q3[mask], alpha=0.25, linewidth=0)

            # median line
            if np.isfinite(y_med).any():
                ax.plot(x, y_med, marker="o", linewidth=2, label=f)

            ax.set_title(f"{day} – SCED Price (Fuel: {f}) — Median with IQR")
            ax.set_xlabel("SCED Step")
            ax.set_ylabel("Price")
            ax.set_xticks(x)
            ax.grid(True, alpha=0.3)
            ax.legend()

            out_pf = per_fuel_dir / f"{f}_price_line_iqr.png"
            fig.tight_layout()
            fig.savefig(out_pf, dpi=150)
            plt.close(fig)
            print(f"[OK] Saved: {out_pf}")

    # ---------------- Save CSVs ----------------
    if save_values_csv:
        med_csv = out_lines_dir / "price_median_wide.csv"
        q1_csv  = out_lines_dir / "price_q1_wide.csv"
        q3_csv  = out_lines_dir / "price_q3_wide.csv"
        try:
            med_wide.to_csv(med_csv)
            q1_wide.to_csv(q1_csv)
            q3_wide.to_csv(q3_csv)
            print(f"[OK] Saved: {med_csv}")
            print(f"[OK] Saved: {q1_csv}")
            print(f"[OK] Saved: {q3_csv}")
        except Exception as e:
            print(f"[ERROR] {day}: failed writing quartile CSVs: {e}")
    
    return med_wide, q1_wide, q3_wide

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


def normalize_by_row_max_with_hsl_and_bp(
    step_dfs: Dict[int, pd.DataFrame],
    hsl_df: pd.DataFrame,
    bp_df: pd.DataFrame,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Robust normalization for SCED steps vs HSL (row-wise baseline), tolerant to partial data.

    Goals implemented:
      1) Drop units (resources) that are empty across ALL steps (so a single bad unit does not nuke its fuel).
      2) Keep units that only have SOME steps; missing steps simply contribute NaN and are ignored in later means.

    Alignment:
      - Resources: (Union of step resources) ∩ (HSL resources).
      - Timestamps: (HSL timestamps) ∩ (Union of step timestamps).
      - Base Point is aligned to the same resources and timestamps; missing BP timestamps are allowed (become NaN).

    Returns:
      norm_steps, hsl_norm, bp_norm, max_sced_per_row (row-mean of M_sced), fuel_per_row.
    """
    # 1) Key columns
    h_name, h_type = normalize_key_columns(hsl_df)
    any_step_df = next(iter(step_dfs.values()))
    s_name, s_type = normalize_key_columns(any_step_df)
    bp_name, bp_type = normalize_key_columns(bp_df)

    # 2) Prepare indices (casefold/trim) and dedup
    def _key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.casefold()

    def _prep(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
        d = df.copy()
        d["_key"] = _key(d[name_col])
        d = d.drop_duplicates(subset=["_key"], keep="first")
        d = d.set_index("_key")
        return d

    hsl = _prep(hsl_df, h_name)
    bp  = _prep(bp_df,  bp_name)
    steps_prepped: Dict[int, pd.DataFrame] = {k: _prep(df, s_name) for k, df in step_dfs.items()}

    # 3) Timestamps: HSL ∩ (Union of step timestamps)
    def _ts(df: pd.DataFrame, name_col: str, type_col: Optional[str]) -> List[str]:
        return detect_timestamp_columns(df.reset_index(), (name_col, type_col))

    union_step_ts: set = set()
    for df in steps_prepped.values():
        union_step_ts |= set(_ts(df, s_name, s_type))

    h_ts_set  = set(_ts(hsl, h_name, h_type))
    bp_ts_set = set(_ts(bp,  bp_name, bp_type))

    common_ts = sorted(h_ts_set & union_step_ts, key=lambda c: pd.to_datetime(c))
    if not common_ts:
        raise ValueError("No overlapping timestamp columns between HSL and the union of SCED step timestamps.")

    # 4) Resources: (Union of step resources) ∩ (HSL resources)
    union_step_idx: set = set().union(*[set(df.index) for df in steps_prepped.values()])
    master_idx = sorted(union_step_idx & set(hsl.index))
    if not master_idx:
        raise ValueError("No overlapping Resource Names between HSL and SCED steps (after normalization).")

    # 5) Numeric blocks (allow partial NaN)
    def _num_block(df: pd.DataFrame, idx: List[str], cols: List[str]) -> pd.DataFrame:
        sub = df.reindex(master_idx)                       # rows aligned; missing -> NaN
        block = sub.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")  # cols aligned; missing -> NaN
        return block

    step_blocks: Dict[int, pd.DataFrame] = {k: _num_block(df, master_idx, common_ts) for k, df in steps_prepped.items()}
    hsl_block = _num_block(hsl, master_idx, common_ts)

    # FIX: intersect as sets, then sort (bp may have fewer timestamps)
    bp_cols = sorted(set(common_ts) & bp_ts_set, key=lambda c: pd.to_datetime(c))
    bp_block = _num_block(bp, master_idx, bp_cols).reindex(columns=common_ts)  # fill missing bp ts with NaN

    # 6) M_sced = rowwise max over steps at each timestamp (K,R,T)
    K, R, T = len(step_blocks), len(master_idx), len(common_ts)
    stack = np.empty((K, R, T), dtype=float)
    stack[:] = np.nan
    for i, k in enumerate(sorted(step_blocks.keys())):
        stack[i, :, :] = step_blocks[k].to_numpy(dtype=float)

    with np.errstate(all="ignore"):
        M_sced = np.nanmax(stack, axis=0)  # (R,T)

    # 7) Drop units empty across ALL steps (your rule #1)
    empty_unit_mask = np.isnan(M_sced).all(axis=1)  # (R,)
    if empty_unit_mask.any():
        keep_mask = ~empty_unit_mask
        M_sced     = M_sced[keep_mask, :]
        hsl_block  = hsl_block.iloc[keep_mask, :]
        bp_block   = bp_block.iloc[keep_mask, :]
        master_idx = [master_idx[i] for i, k in enumerate(keep_mask) if k]
        for k in list(step_blocks.keys()):
            step_blocks[k] = step_blocks[k].iloc[keep_mask, :]

    # 8) M_all and denominator
    with np.errstate(all="ignore"):
        M_all = np.nanmax(np.stack([M_sced, hsl_block.to_numpy(dtype=float)], axis=0), axis=0)
    denom = M_all.copy()
    denom[~np.isfinite(denom) | (denom <= 0.0)] = np.nan

    # 9) Normalize steps/HSL/BP by M_all (rule #2 allows partial NaNs)
    norm_steps: Dict[int, pd.DataFrame] = {}
    for k in sorted(step_blocks.keys()):
        num = step_blocks[k].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            vals = num / denom
        norm_steps[k] = pd.DataFrame(vals, index=master_idx, columns=common_ts)

    with np.errstate(all="ignore"):
        hsl_vals = hsl_block.to_numpy(dtype=float) / denom
        bp_vals  = bp_block.to_numpy(dtype=float)  / denom

    # 10) Display columns
    def _pick_display_cols() -> Tuple[pd.Series, pd.Series]:
        name_series = hsl.reindex(master_idx)[h_name]
        if name_series.isna().any():
            for df in steps_prepped.values():
                name_series = name_series.fillna(df.reindex(master_idx)[s_name])
        if h_type and (h_type in hsl.columns):
            fuel_series = hsl.reindex(master_idx)[h_type]
        else:
            fuel_series = pd.Series(index=master_idx, dtype=object)
        if fuel_series.isna().any() or fuel_series.empty:
            for df in steps_prepped.values():
                if s_type and (s_type in df.columns):
                    fuel_series = fuel_series.fillna(df.reindex(master_idx)[s_type])
        fuel_series = fuel_series.astype(str).str.strip().replace("", "Unknown")
        name_series = name_series.astype(str)
        return name_series, fuel_series

    disp_names, fuels = _pick_display_cols()

    for k in norm_steps.keys():
        df = norm_steps[k]
        df.insert(0, "Resource Type", fuels.values)
        df.insert(0, "Resource Name", disp_names.values)
        norm_steps[k] = df.reset_index(drop=True)

    hsl_norm = pd.DataFrame(hsl_vals, index=master_idx, columns=common_ts)
    hsl_norm.insert(0, "Resource Type", fuels.values)
    hsl_norm.insert(0, "Resource Name", disp_names.values)
    hsl_norm.reset_index(drop=True, inplace=True)

    bp_norm = pd.DataFrame(bp_vals, index=master_idx, columns=common_ts)
    bp_norm.insert(0, "Resource Type", fuels.values)
    bp_norm.insert(0, "Resource Name", disp_names.values)
    bp_norm.reset_index(drop=True, inplace=True)

    # 11) Row-mean of M_sced for rescaling + fuel_per_row for grouping
    with np.errstate(all="ignore"):
        row_mean_Msced = np.nanmean(M_sced, axis=1)
    max_sced_per_row = pd.Series(row_mean_Msced, index=master_idx)
    fuel_per_row     = pd.Series(fuels.values, index=master_idx)

    return norm_steps, hsl_norm, bp_norm, max_sced_per_row, fuel_per_row


def process_sced_normalized_lines_day(day_dir: Path, plots_root: Path, save_summary_csv: bool) -> None:
    """
    UPDATED:
      - Row-wise maxima across all SCED MW steps (M_sced); M_all = max(M_sced, HSL).
      - Normalize SCED steps, HSL, Base Point by M_all.
      - Aggregate by fuel over resources×timestamps, then reapply magnitude via SUM(M_sced) per fuel.
      - Combined plot: one line per fuel (NO HSL/BP lines) + NEW shaded IQR (25–75%) per fuel.
      - Per-fuel plots: that fuel’s line + HSL/BP dashed lines + NEW shaded IQR (25–75%).
      - Diagnostics: raw fuels list (CSV), plotted vs skipped fuels (reasons).
    """
    day = day_dir.name

    # ----- Load inputs (same as before) -----
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

    # ----- Raw fuels quick check (same as before) -----
    print(f"\n[CHECK] {day}: Inspecting fuels present in raw input data...")

    def _extract_fuels(df: pd.DataFrame) -> pd.Series:
        name_col, type_col = normalize_key_columns(df)
        if type_col is None:
            return pd.Series([], dtype=str)
        return (
            df[type_col].astype(str)
            .str.strip()
            .replace({"": "Unknown", "None": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
        )

    fuels_raw_list: List[str] = []
    fuels_raw_list.extend(_extract_fuels(hsl_df).tolist())
    fuels_raw_list.extend(_extract_fuels(bp_df).tolist())
    for _, df in step_dfs.items():
        fuels_raw_list.extend(_extract_fuels(df).tolist())

    fuels_raw_series = pd.Series(fuels_raw_list, dtype=str).replace(
        {"": "Unknown", "None": "Unknown", "nan": "Unknown", "NaN": "Unknown"}
    )
    unique_fuels_raw = fuels_raw_series.value_counts(dropna=False).sort_index()
    raw_fuels_set = set(unique_fuels_raw.index.astype(str))
    print(f"[CHECK] {day}: FOUND FUEL TYPES IN INPUT DATA:")
    for fuel, count in unique_fuels_raw.items():
        print(f"    - {fuel}: {count} rows")
    check_path = day_dir / "fuel_check_raw_input.csv"
    try:
        unique_fuels_raw.to_csv(check_path, header=["count"])
        print(f"[CHECK] Saved raw fuel diagnostics → {check_path}\n")
    except Exception as e:
        print(f"[WARN] {day}: failed to write fuel_check_raw_input.csv: {e}")

    # ----- Normalize (using your robust union/partial-steps normalizer) -----
    try:
        norm_steps, hsl_norm, bp_norm, max_sced_per_row, fuel_per_row = normalize_by_row_max_with_hsl_and_bp(
            step_dfs, hsl_df, bp_df
        )
    except Exception as e:
        print(f"[ERROR] {day}: normalization by row-max failed: {e}")
        return

    # ----- Rescale factor by fuel: SUM of per-row max SCED (same) -----
    fuel_series = fuel_per_row.astype(str).fillna("Unknown")
    rescale_by_fuel = max_sced_per_row.groupby(fuel_series).sum(min_count=1)
    candidate_fuels_set = set(rescale_by_fuel.index.astype(str))

    # Helpers to aggregate stats over rows×timestamps for each fuel
    def _numeric_vals_for_fuel(df: pd.DataFrame, fuel: str) -> np.ndarray:
        name_col, type_col = normalize_key_columns(df)
        ts_cols = detect_timestamp_columns(df, (name_col, type_col))
        if type_col is None or not ts_cols:
            return np.asarray([], dtype=float)
        sub = df[df[type_col].astype(str).fillna("Unknown") == fuel]
        if sub.empty:
            return np.asarray([], dtype=float)
        vals = sub[ts_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return vals.reshape(-1)  # flatten rows×time

    def stats_over_rows_and_time(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return mean, Q25, Q75 per fuel for a given step-normalized frame."""
        # fuels present in df:
        _, type_col = normalize_key_columns(df)
        fuels_here = [] if type_col is None else sorted(df[type_col].astype(str).fillna("Unknown").unique())
        means, q25s, q75s = {}, {}, {}
        for f in fuels_here:
            arr = _numeric_vals_for_fuel(df, f)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                means[f] = np.nan; q25s[f] = np.nan; q75s[f] = np.nan
            else:
                means[f] = float(np.nanmean(arr))
                q25s[f]  = float(np.nanpercentile(arr, 25))
                q75s[f]  = float(np.nanpercentile(arr, 75))
        # return as Series so .get() works for fuels not present in this step
        return pd.Series(means, dtype=float), pd.Series(q25s, dtype=float), pd.Series(q75s, dtype=float)

    # Track which fuels appear in each step (for diagnostics)
    steps_sorted = sorted(norm_steps.keys())
    fuels_present_by_step: Dict[int, set] = {}
    for s in steps_sorted:
        _, type_col = normalize_key_columns(norm_steps[s])
        if type_col is None:
            fuels_present_by_step[s] = set()
        else:
            fuels_present_by_step[s] = set(
                norm_steps[s][type_col].astype(str).fillna("Unknown").str.strip().replace("", "Unknown").unique()
            )

    # ----- Build per-step mean + IQR (25–75%) and then RESCALE all three -----
    fuels_sorted = sorted(candidate_fuels_set)
    scaled_mean_by_step: Dict[int, List[float]] = {}
    scaled_low_by_step:  Dict[int, List[float]] = {}
    scaled_high_by_step: Dict[int, List[float]] = {}

    for s in steps_sorted:
        mean_s, q25_s, q75_s = stats_over_rows_and_time(norm_steps[s])
        row_mean, row_low, row_high = [], [], []
        for f in fuels_sorted:
            scale = rescale_by_fuel.get(f, np.nan)
            mu  = mean_s.get(f,  np.nan)
            q25 = q25_s.get(f,   np.nan)
            q75 = q75_s.get(f,   np.nan)
            if np.isfinite(scale) and np.isfinite(mu):
                row_mean.append(float(mu) * float(scale))
            else:
                row_mean.append(np.nan)
            if np.isfinite(scale) and np.isfinite(q25) and np.isfinite(q75):
                row_low.append(float(q25) * float(scale))
                row_high.append(float(q75) * float(scale))
            else:
                row_low.append(np.nan); row_high.append(np.nan)
        scaled_mean_by_step[s] = row_mean
        scaled_low_by_step[s]  = row_low
        scaled_high_by_step[s] = row_high

    summary_scaled = pd.DataFrame.from_dict(scaled_mean_by_step, orient="index", columns=fuels_sorted)
    summary_low    = pd.DataFrame.from_dict(scaled_low_by_step,  orient="index", columns=fuels_sorted)
    summary_high   = pd.DataFrame.from_dict(scaled_high_by_step, orient="index", columns=fuels_sorted)
    summary_scaled.index.name = "SCED Step"
    summary_low.index    = summary_scaled.index
    summary_high.index   = summary_scaled.index

    # ----- HSL/BP (used only for per-fuel dashed lines) -----
    def avg_over_rows_and_time(df: pd.DataFrame) -> pd.Series:
        name_col, type_col = normalize_key_columns(df)
        ts_cols = detect_timestamp_columns(df, (name_col, type_col))
        vals = df[ts_cols].apply(pd.to_numeric, errors="coerce")
        vals.insert(0, "Resource Type", df[type_col].astype(str).fillna("Unknown"))
        return (
            vals.groupby("Resource Type", dropna=False)
                .mean(numeric_only=True)
                .mean(axis=1, numeric_only=True)
        )
    hsl_scaled = pd.Series({f: (avg_over_rows_and_time(hsl_norm).get(f, np.nan) *
                                rescale_by_fuel.get(f, np.nan)) for f in fuels_sorted})
    bp_scaled  = pd.Series({f: (avg_over_rows_and_time(bp_norm).get(f,  np.nan) *
                                rescale_by_fuel.get(f, np.nan)) for f in fuels_sorted})

    # ----- Diagnostics: who plotted vs skipped (same logic) -----
    plotted_fuels, skipped_fuels = [], {}
    for f in fuels_sorted:
        col = pd.to_numeric(summary_scaled[f], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(col).sum() > 0:
            plotted_fuels.append(f)
        else:
            reasons = []
            appears_any_step = any(f in fuels_present_by_step[s] for s in steps_sorted)
            if not appears_any_step:
                reasons.append("no SCED step rows for this fuel")
            rscale = rescale_by_fuel.get(f, np.nan)
            if not np.isfinite(float(rscale)) or float(rscale) == 0.0:
                reasons.append("invalid/zero rescale (sum of row maxima)")
            if appears_any_step and (np.isnan(col).all()):
                reasons.append("normalized averages all NaN after coercion/alignment")
            if not reasons:
                reasons.append("no finite values to plot")
            skipped_fuels[f] = reasons

    raw_not_candidate = sorted(raw_fuels_set - candidate_fuels_set)
    for f in raw_not_candidate:
        skipped_fuels.setdefault(f, []).append("present in raw input but absent after row-max/rescale grouping")

    if plotted_fuels:
        print(f"[OK] {day}: Fuels plotted in combined/per-fuel charts: {', '.join(sorted(plotted_fuels))}")
    if skipped_fuels:
        print(f"[INFO] {day}: Skipped fuels and reasons:")
        for f, reasons in sorted(skipped_fuels.items()):
            print("   -", f, "→", "; ".join(sorted(set(reasons))))

    # ----- Plot: Combined (NO HSL/BP) + shaded IQR per fuel -----
    day_out = plots_root / day / "sced_normalized"
    day_out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = summary_scaled.index.values
    for f in fuels_sorted:
        y  = pd.to_numeric(summary_scaled[f], errors="coerce").to_numpy(dtype=float)
        yl = pd.to_numeric(summary_low[f],    errors="coerce").to_numpy(dtype=float)
        yh = pd.to_numeric(summary_high[f],   errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).any():
            line, = ax.plot(x, y, marker="o", linewidth=2, label=f)
            # shading only where both bounds are finite
            mask = np.isfinite(yl) & np.isfinite(yh)
            if mask.any():
                ax.fill_between(x[mask], yl[mask], yh[mask], alpha=0.2)

    ax.set_title(f"{day} – Aggregate bidding behavior per fuel over SCED Steps")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("Aggregate MW bid (MW)")
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    out_png = day_out / "normalized_bids_by_stage.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out_png}")

    # ----- Plot: Per-fuel + shaded IQR + HSL/BP dashed lines -----
    per_fuel_dir = day_out / "per_fuel"
    per_fuel_dir.mkdir(parents=True, exist_ok=True)

    for f in fuels_sorted:
        y  = pd.to_numeric(summary_scaled[f], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue
        yl = pd.to_numeric(summary_low[f],  errors="coerce").to_numpy(dtype=float)
        yh = pd.to_numeric(summary_high[f], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, marker="o", linewidth=2, label=f)
        mask = np.isfinite(yl) & np.isfinite(yh)
        if mask.any():
            ax.fill_between(x[mask], yl[mask], yh[mask], alpha=0.2)

        # dashed HSL / BP
        y_hsl = hsl_scaled.get(f, np.nan)
        y_bp  = bp_scaled.get(f, np.nan)
        if np.isfinite(y_hsl):
            ax.axhline(y_hsl, linestyle="--", linewidth=1.75, color="#1f77b4", label="HSL (rescaled)")
        if np.isfinite(y_bp):
            ax.axhline(y_bp,  linestyle="--", linewidth=1.75, color="#d62728", label="Base Point (rescaled)")

        ax.set_title(f"{day} – Aggregate bidding behavior per SCED Step for Fuel: {f}")
        ax.set_xlabel("SCED Step")
        ax.set_ylabel("Aggregate MW bid (MW)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_pf = per_fuel_dir / f"{f}_normalized.png"
        fig.tight_layout()
        plt.savefig(out_pf, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved: {out_pf}")

    # ----- Optional CSV (combined wide) -----
    if save_summary_csv:
        out_csv = day_out / "normalized_bids_by_stage.csv"
        try:
            summary_scaled.to_csv(out_csv)
            print(f"[OK] Saved: {out_csv}")
        except Exception as e:
            print(f"[ERROR] {day}: failed to write summary CSV: {e}")
    
    return summary_scaled, summary_low, summary_high, hsl_scaled, bp_scaled


def plot_price_vs_mw_by_fuel(
    day: str,
    mw_summary: pd.DataFrame,
    price_summary: pd.DataFrame,
    out_dir: Path,
    fuels: Optional[Iterable[str]] = None,
    hsl_scaled: Optional[pd.Series] = None,
    bp_scaled: Optional[pd.Series] = None,
) -> None:
    """
    Plot price (y-axis) vs aggregate MW (x-axis) for each fuel, using
    per-step summaries:

      mw_summary: per-step MW by fuel (e.g. summary_scaled from normalized lines)
      price_summary: per-step price by fuel (e.g. med_wide from price lines)

    Both should have index = SCED Step (or a 'SCED Step' column that can
    be set as the index) and fuel columns matching.

    If hsl_scaled / bp_scaled are provided (Series indexed by fuel),
    vertical lines will be drawn at those MW values.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def _ensure_step_index(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.name == "SCED Step":
            return df
        if "SCED Step" in df.columns:
            df = df.set_index("SCED Step")
        df.index.name = "SCED Step"
        return df

    mw = _ensure_step_index(mw_summary.copy())
    price = _ensure_step_index(price_summary.copy())

    # Align on common steps
    common_steps = mw.index.intersection(price.index)
    mw = mw.loc[common_steps]
    price = price.loc[common_steps]

    if fuels is None:
        fuels = sorted(set(mw.columns).intersection(price.columns))

    out_dir = out_dir / "sced_price_vs_mw"
    out_dir.mkdir(parents=True, exist_ok=True)

    step_vals = common_steps.to_numpy()

    for f in fuels:
        mw_vals = pd.to_numeric(mw[f], errors="coerce").to_numpy(dtype=float)
        pr_vals = pd.to_numeric(price[f], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(mw_vals) & np.isfinite(pr_vals)
        if not mask.any():
            continue

        x = mw_vals[mask]
        y = pr_vals[mask]
        steps_used = step_vals[mask]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, marker="o", linewidth=2, label="SCED steps")

        # annotate with SCED step number so you know which kink is which
        for xx, yy, s in zip(x, y, steps_used):
            ax.annotate(str(int(s)), xy=(xx, yy),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)

        # Optional vertical lines for HSL / Base Point if provided
        # (they are MW quantities, so they live on the x-axis)
        if hsl_scaled is not None:
            x_hsl = hsl_scaled.get(f, np.nan)
            if np.isfinite(x_hsl):
                ax.axvline(
                    float(x_hsl),
                    linestyle="--",
                    linewidth=1.75,
                    color="#1f77b4",
                    label="HSL (rescaled)",
                )
        if bp_scaled is not None:
            x_bp = bp_scaled.get(f, np.nan)
            if np.isfinite(x_bp):
                ax.axvline(
                    float(x_bp),
                    linestyle="--",
                    linewidth=1.75,
                    color="#d62728",
                    label="Base Point (rescaled)",
                )

        ax.set_title(f"{day} – Price vs MW (Fuel: {f})")
        ax.set_xlabel("Aggregate MW bid (MW)")
        ax.set_ylabel("Price (median)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        out_png = out_dir / f"{f}_price_vs_mw.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved: {out_png}")


# ---------------------------------------------------------------------------
# Fuel counts per SCED MW step
# ---------------------------------------------------------------------------

def process_counts_fuel_per_step(root_dir: Path, out_dir: Path) -> None:
    """
    For each aggregation_SCED1_Curve_MW*.csv in root_dir, plot the number of plants per fuel.
    Produces one bar chart per file plus a counts CSV.
    """
    root = Path(root_dir)
    out_base = Path(out_dir)
    day_out = out_base / root.name / "sced_fuel_count"
    files = sorted(root.glob("aggregation_SCED1_Curve-MW*.csv"))
    if not files:
        files = sorted(root.glob("*SCED*Curve*MW*.csv"))
    if not files:
        print(f"[INFO] No SCED MW step files found in {root}")
        return

    def _key_cols(df: pd.DataFrame) -> tuple[str, str]:
        col_map = {str(c).strip().lower(): c for c in df.columns}
        name_col = col_map.get("resource name")
        type_col = col_map.get("resource type")
        if not name_col or not type_col:
            raise ValueError("Expected columns 'Resource Name' and 'Resource Type'.")
        return name_col, type_col

    def _timestamp_cols(df: pd.DataFrame, name_col: str, type_col: str) -> list[str]:
        ts_cols: list[str] = []
        for c in df.columns:
            if c in (name_col, type_col):
                continue
            ts = pd.to_datetime(c, errors="coerce")
            if ts is not pd.NaT:
                ts_cols.append(c)
        return ts_cols

    day_out.mkdir(parents=True, exist_ok=True)
    step_pattern = re.compile(r"_MW(\d+)", re.IGNORECASE)

    for path in files:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            print(f"[WARN] Failed to read {path.name}: {e}")
            continue

        try:
            name_col, type_col = _key_cols(df)
            ts_cols = _timestamp_cols(df, name_col, type_col)
            if not ts_cols:
                print(f"[INFO] {path.name}: no timestamp columns detected; skipping.")
                continue
        except Exception as e:
            print(f"[WARN] {path.name}: column detection failed: {e}")
            continue

        numeric = df[ts_cols].apply(pd.to_numeric, errors="coerce")
        active_mask = np.isfinite(numeric.to_numpy()).any(axis=1)
        fuels = (
            df.loc[active_mask, type_col]
            .astype(str)
            .fillna("Unknown")
            .str.strip()
            .replace("", "Unknown")
        )
        if fuels.empty:
            print(f"[INFO] {path.name}: no active rows to count; skipping.")
            continue

        counts = fuels.value_counts().sort_index()

        match = step_pattern.search(path.name)
        step_label = match.group(1) if match else path.stem
        title = f"SCED Step {step_label}"

        labels = counts.index.tolist()
        xpos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(xpos, counts.to_numpy(dtype=float))
        ax.set_title(title)
        ax.set_xlabel("ERCOT Fuel Type")
        ax.set_ylabel("Number of Plants")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out_png = day_out / f"{path.stem}_fuel_counts.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved: {out_png}")

        out_csv = day_out / f"{path.stem}_fuel_counts.csv"
        try:
            counts.to_csv(out_csv, header=["count"])
            print(f"[OK] Saved: {out_csv}")
        except Exception as e:
            print(f"[WARN] Failed to write counts CSV for {path.name}: {e}")



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
        # 1b) Base Point vs Telemetered error distribution by fuel
        process_base_point_telemetered_output(day_dir, out)
        # 1c) SCED price distribution by fuel (boxplot)
        process_average_bid_quantity(day_dir, out)
        # 2) SCED price violins
        process_sced_violins_day(day_dir, out, SCED_SAVE_VALUES_CSV)

        # Fuel counts per SCED MW step
        process_counts_fuel_per_step(day_dir, out)

        # 3) SCED price lines
        price_result = process_sced_price_lines_day(day_dir, out, save_values_csv=True, per_fuel=True)

        # 4) SCED normalized-bid lines
        norm_result = process_sced_normalized_lines_day(day_dir, out, SAVE_NORMALIZED_SUMMARY_CSV)

        # If either step failed / skipped, don't try to do price vs MW
        if price_result is None or norm_result is None:
            continue

        med_wide, q1_wide, q3_wide = price_result
        summary_scaled, summary_low, summary_high, hsl_scaled, bp_scaled = norm_result

        # 4b) Average MW bid per fuel (skip first SCED step)
        process_average_mw_bid(summary_scaled, day_dir, out)

        # 5) Price vs MW per fuel, with HSL & Base Point verticals
        plot_price_vs_mw_by_fuel(
            day=day_dir.name,
            mw_summary=summary_scaled,
            price_summary=med_wide,
            out_dir=out / day_dir.name,
            hsl_scaled=hsl_scaled,
            bp_scaled=bp_scaled,
        )


if __name__ == "__main__":
    main()
