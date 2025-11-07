
#!/usr/bin/env python3
"""
SCED single-day plotting (per-fuel normalized curves with magnitude reapplication).

Changes in this version:
- Creates one normalized plot per FUEL (instead of one multi-fuel plot).
- Normalization baseline per (resource, timestamp): row-wise max across all SCED MW steps and HSL
  (i.e., M_all = max(max_sced_step, HSL)). Base Point uses the same M_all per row.
- After averaging normalized curves within a fuel, reapply magnitude using the sum of per-row SCED maxima
  (Σ M_sced) for that fuel (as requested).
- Plot dashed horizontal lines for HSL and Base Point with distinct colors.
"""

from __future__ import annotations
import os
import re
import glob
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- Config -----------------------
# Root directory that contains YYYY-MM-DD subfolders with aggregated CSVs
ROOT_DIR = os.environ.get("ROOT_DIR", "./inputs/sced_day")
# Where plots will be written
PLOTS_DIR = os.environ.get("PLOTS_DIR", "./plots")

# Filenames (per-day) expected (case-insensitive tolerant search is used as fallback)
BASE_POINT_NAME_HINT = "aggregation_Base_Point.csv"
HSL_NAME_HINT        = "aggregation_HSL.csv"
SCED_MW_GLOB         = "aggregation_SCED1_Curve-MW*.csv"  # stage files (MW1, MW2, ...)

# Toggle saving intermediate CSVs
SAVE_NORMALIZED_VALUES_CSV = os.environ.get("SAVE_NORMALIZED_VALUES_CSV", "0") == "1"

# Dashed line colors (explicit per user request for distinct colors)
HSL_LINE_COLOR = "#1f77b4"   # blue-ish
BP_LINE_COLOR  = "#d62728"   # red-ish

# ------------------------------------------------------


def _find_case_insensitive(path: str, name_hint: str) -> Optional[str]:
    """Try direct join; otherwise scan for case-insensitive match."""
    p = os.path.join(path, name_hint)
    if os.path.exists(p):
        return p
    # fallback: case-insensitive search in folder
    lower_hint = name_hint.lower()
    for fn in os.listdir(path):
        if fn.lower() == lower_hint:
            return os.path.join(path, fn)
    return None


def _find_glob(path: str, pattern: str) -> List[str]:
    cand = sorted(glob.glob(os.path.join(path, pattern)))
    if cand:
        return cand
    # broader fallback
    cand = sorted(glob.glob(os.path.join(path, "*SCED*Curve*MW*.csv")))
    return cand


def _detect_key_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Return canonical 'Resource Name' column and optional 'Resource Type' column names as present in df."""
    cols = {c.strip().lower(): c for c in df.columns}
    rn = cols.get("resource name") or cols.get("resourcename")
    rt = cols.get("resource type") or cols.get("resourcetype")
    if rn is None:
        # Try fuzzy-ish search
        for c in df.columns:
            if re.sub(r"\s+", "", c.strip().lower()) in ("resourcename",):
                rn = c
                break
    if rn is None:
        raise ValueError("Could not find 'Resource Name' column.")
    return rn, rt


def _timestamp_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Columns that look like timestamps (excluding key cols)."""
    cand = [c for c in df.columns if c not in exclude]
    out = []
    for c in cand:
        try:
            pd.to_datetime(c)
            out.append(c)
        except Exception:
            # not a timestamp header
            pass
    return out


def _coerce_numeric(df: pd.DataFrame, ts_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in ts_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _align_on_resources_and_timestamps(dfs: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str], str, Optional[str]]:
    """
    Given [SCED_step_1, ..., SCED_step_K, HSL, BasePoint], align on common resources and timestamp headers.
    Returns (aligned_dfs, common_ts, resource_name_col, resource_type_col)
    """
    # Detect keys from the first frame that has them
    rn, rt = None, None
    for df in dfs:
        try:
            rn, rt = _detect_key_cols(df)
            break
        except Exception:
            continue
    if rn is None:
        raise ValueError("Could not detect key columns from inputs.")

    # Intersect resource sets (case-tolerant on values)
    def _norm_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()

    resource_sets = []
    for df in dfs:
        if rn not in df.columns:
            raise ValueError(f"Missing key col '{rn}' in an input.")
        resource_sets.append(set(_norm_series(df[rn]).tolist()))
    common_resources_norm = set.intersection(*resource_sets) if resource_sets else set()

    # Intersect timestamp headers
    ts_common = None
    for df in dfs:
        ts = _timestamp_columns(df, exclude=[rn] + ([rt] if rt else []))
        ts_common = set(ts) if ts_common is None else (ts_common & set(ts))
    if not ts_common:
        raise ValueError("No common timestamp columns across inputs.")
    ts_common = sorted(ts_common, key=lambda x: pd.to_datetime(x))

    # Filter and coerce numeric
    aligned = []
    for df in dfs:
        d = df.copy()
        d["_rn_norm"] = _norm_series(d[rn])
        d = d[d["_rn_norm"].isin(common_resources_norm)].drop(columns=["_rn_norm"])
        keep = [rn] + ([rt] if rt else []) + ts_common
        d = d[keep]
        d = _coerce_numeric(d, ts_common)
        aligned.append(d.reset_index(drop=True))

    return aligned, ts_common, rn, rt


def _build_rowwise_max_and_normalized(
    sced_steps: List[pd.DataFrame],
    hsl_df: pd.DataFrame,
    bp_df: pd.DataFrame
) -> Tuple[List[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given aligned frames (same resources, same timestamps):
      - sced_steps: list of SCED MW step dataframes
      - hsl_df: HSL dataframe
      - bp_df: Base Point dataframe
    Returns:
      - sced_norm_steps: list of SCED steps normalized by row-wise M_all (max(max_sced, HSL))
      - hsl_norm: HSL normalized by same M_all
      - bp_norm: Base Point normalized by same M_all
      - sced_row_max: DataFrame of per-row M_sced (max across SCED steps), for re-scaling
    """
    # Keys and timestamps
    rn, rt = _detect_key_cols(hsl_df)
    ts_cols = _timestamp_columns(hsl_df, exclude=[rn] + ([rt] if rt else []))

    # Compute M_sced (max across steps per cell) and M_all (max of M_sced and HSL) per (row, ts)
    # Stack SCED steps into 3D array for vectorized max
    sced_vals = np.stack([df[ts_cols].to_numpy(dtype=float) for df in sced_steps], axis=0)  # (K, R, T)
    M_sced = np.nanmax(sced_vals, axis=0)  # (R, T)
    hsl_vals = hsl_df[ts_cols].to_numpy(dtype=float)
    M_all = np.fmax(M_sced, hsl_vals)  # row-wise baseline

    # Avoid divide-by-zero: where M_all <= 0 -> NaN mask
    denom = M_all.copy()
    denom[~np.isfinite(denom) | (denom <= 0)] = np.nan

    # Normalize SCED steps
    sced_norm_steps: List[pd.DataFrame] = []
    for k, step_df in enumerate(sced_steps):
        num = step_df[ts_cols].to_numpy(dtype=float)
        norm_vals = num / denom
        out = step_df[[rn] + ([rt] if rt else [])].copy()
        for j, c in enumerate(ts_cols):
            out[c] = norm_vals[:, j]
        sced_norm_steps.append(out)

    # Normalize HSL and BP
    def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
        num = df[ts_cols].to_numpy(dtype=float)
        norm_vals = num / denom
        out = df[[rn] + ([rt] if rt else [])].copy()
        for j, c in enumerate(ts_cols):
            out[c] = norm_vals[:, j]
        return out

    hsl_norm = _norm_df(hsl_df)
    bp_norm  = _norm_df(bp_df)

    # Return M_sced as DataFrame for later grouping / re-sum (magnitude reapplication)
    sced_row_max = hsl_df[[rn] + ([rt] if rt else [])].copy()
    for j, c in enumerate(ts_cols):
        sced_row_max[c] = M_sced[:, j]

    return sced_norm_steps, hsl_norm, bp_norm, sced_row_max


def _per_fuel_average_and_rescale(
    sced_norm_steps: List[pd.DataFrame],
    hsl_norm: pd.DataFrame,
    bp_norm: pd.DataFrame,
    sced_row_max: pd.DataFrame
) -> Tuple[Dict[str, pd.Series], Dict[str, float], Dict[str, float], List[str]]:
    """
    For each fuel, compute:
      - step_curve[fuel]: 1D array (len = K steps). Each value is the mean over rows×timestamps of the normalized step,
                          then rescaled by Σ(M_sced) within that fuel.
      - hsl_line[fuel]: scalar, mean of normalized HSL over rows×timestamps, rescaled by same Σ(M_sced)
      - bp_line[fuel]: scalar, mean of normalized Base Point over rows×timestamps, rescaled by same Σ(M_sced)
    Returns:
      step_curve_by_fuel: dict fuel -> pd.Series(index=step_labels, values=rescaled means)
      hsl_by_fuel: dict fuel -> float (rescaled)
      bp_by_fuel: dict fuel -> float (rescaled)
      step_labels: list of labels for x-axis (e.g., ["MW1","MW2",...])
    """
    rn, rt = _detect_key_cols(hsl_norm)
    ts_cols = _timestamp_columns(hsl_norm, exclude=[rn] + ([rt] if rt else []))

    # Determine fuel list
    if rt is None:
        raise ValueError("Resource Type column is required to aggregate by fuel.")
    fuels = sorted(hsl_norm[rt].dropna().astype(str).unique())

    # Prepare Σ(M_sced) per fuel
    # Use mean across timestamps per row, then sum over rows within fuel -> Σ of per-row maxima (time-averaged)
    sced_row_max_mean = sced_row_max.copy()
    sced_row_max_mean["_row_mean"] = sced_row_max[ts_cols].mean(axis=1, skipna=True)

    fuel_max_sum = sced_row_max_mean.groupby(rt)["_row_mean"].sum(min_count=1).to_dict()

    # Prepare step labels
    step_labels = []
    for df in sced_norm_steps:
        # expect columns like ... MW1.csv, MW2.csv; we derive index by position
        step_labels.append(None)  # placeholder; we will label as 1..K
    step_labels = [f"Step {i+1}" for i in range(len(sced_norm_steps))]

    # Compute per-step per-fuel means (over rows×timestamps)
    step_curve_by_fuel: Dict[str, pd.Series] = {}
    for fuel in fuels:
        vals = []
        denom = fuel_max_sum.get(fuel, np.nan)
        for df in sced_norm_steps:
            sub = df[df[rt].astype(str) == fuel]
            if sub.empty:
                vals.append(np.nan)
                continue
            m = sub[ts_cols].to_numpy(dtype=float)
            mean_norm = np.nanmean(m)  # mean over all rows×timestamps for this step, this fuel
            # reapply magnitude using Σ(M_sced) for this fuel
            vals.append(mean_norm * denom if np.isfinite(denom) else np.nan)
        step_curve_by_fuel[fuel] = pd.Series(vals, index=step_labels, dtype="float64")

    # HSL/BP scalars per fuel
    hsl_by_fuel: Dict[str, float] = {}
    bp_by_fuel: Dict[str, float] = {}
    for fuel in fuels:
        denom = fuel_max_sum.get(fuel, np.nan)
        sub_h = hsl_norm[hsl_norm[rt].astype(str) == fuel]
        sub_b = bp_norm [bp_norm [rt].astype(str) == fuel]

        h_mean = np.nanmean(sub_h[ts_cols].to_numpy(dtype=float)) if not sub_h.empty else np.nan
        b_mean = np.nanmean(sub_b[ts_cols].to_numpy(dtype=float)) if not sub_b.empty else np.nan

        hsl_by_fuel[fuel] = h_mean * denom if np.isfinite(denom) else np.nan
        bp_by_fuel[fuel]  = b_mean * denom if np.isfinite(denom) else np.nan

    return step_curve_by_fuel, hsl_by_fuel, bp_by_fuel, step_labels


def process_sced_normalized_lines_day_per_fuel(day_dir: str, plots_root: str, save_csv: bool = False) -> None:
    """
    For a single YYYY-MM-DD folder:
      - Load SCED MW step CSVs, HSL, Base Point
      - Align on resources and timestamps
      - Compute row-wise max normalization + rescale
      - Produce ONE plot per fuel with:
          * Line across steps (rescaled mean normalized step)
          * Dashed horizontal HSL and Base Point lines (distinct colors)
      - Save plots under <plots_root>/<YYYY-MM-DD>/sced_normalized_per_fuel/
    """
    os.makedirs(plots_root, exist_ok=True)
    day = os.path.basename(os.path.normpath(day_dir))
    out_dir = os.path.join(plots_root, day, "sced_normalized_per_fuel")
    os.makedirs(out_dir, exist_ok=True)

    # Find files
    hsl_path = _find_case_insensitive(day_dir, HSL_NAME_HINT)
    bp_path  = _find_case_insensitive(day_dir, BASE_POINT_NAME_HINT)
    step_paths = _find_glob(day_dir, SCED_MW_GLOB)

    if not hsl_path:
        raise FileNotFoundError(f"HSL file not found in {day_dir}")
    if not bp_path:
        raise FileNotFoundError(f"Base Point file not found in {day_dir}")
    if not step_paths:
        raise FileNotFoundError(f"No SCED MW step files found in {day_dir}")

    # Load
    hsl_df = pd.read_csv(hsl_path)
    bp_df  = pd.read_csv(bp_path)
    sced_steps = [pd.read_csv(p) for p in step_paths]

    # Align
    aligned, ts_cols, rn, rt = _align_on_resources_and_timestamps(sced_steps + [hsl_df, bp_df])
    K = len(sced_steps)
    aligned_steps = aligned[:K]
    hsl_aligned  = aligned[K]
    bp_aligned   = aligned[K+1]

    # Normalize & build row-max
    sced_norm_steps, hsl_norm, bp_norm, sced_row_max = _build_rowwise_max_and_normalized(
        aligned_steps, hsl_aligned, bp_aligned
    )

    # Aggregate by fuel & rescale
    step_curve_by_fuel, hsl_by_fuel, bp_by_fuel, step_labels = _per_fuel_average_and_rescale(
        sced_norm_steps, hsl_norm, bp_norm, sced_row_max
    )

    # Optional CSV dump per fuel
    if save_csv:
        for fuel, series in step_curve_by_fuel.items():
            csv_path = os.path.join(out_dir, f"{fuel}_normalized_rescaled_curve.csv")
            series.to_csv(csv_path, header=["value"])

        # HSL/BP summaries
        summ = []
        for fuel in sorted(set(list(hsl_by_fuel.keys()) + list(bp_by_fuel.keys()))):
            summ.append({"fuel": fuel, "hsl_rescaled": hsl_by_fuel.get(fuel, np.nan),
                         "base_point_rescaled": bp_by_fuel.get(fuel, np.nan)})
        pd.DataFrame(summ).to_csv(os.path.join(out_dir, "hsl_bp_rescaled_summary.csv"), index=False)

    # Plot one figure per fuel
    for fuel, series in step_curve_by_fuel.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(series)+1), series.values, marker="o", linewidth=2, label=f"{fuel}")

        # Dashed horizontal lines for HSL and Base Point (distinct colors)
        hval = hsl_by_fuel.get(fuel, np.nan)
        bval = bp_by_fuel.get(fuel, np.nan)
        if np.isfinite(hval):
            ax.axhline(hval, linestyle="--", linewidth=1.75, color=HSL_LINE_COLOR, label="HSL (rescaled)")
        if np.isfinite(bval):
            ax.axhline(bval, linestyle="--", linewidth=1.75, color=BP_LINE_COLOR,  label="Base Point (rescaled)")

        ax.set_title(f"{day} — Normalized & Rescaled SCED Curve (Per Fuel: {fuel})")
        ax.set_xlabel("SCED Step")
        ax.set_ylabel("Rescaled magnitude (Σ row SCED maxima weighted)")
        ax.set_xticks(range(1, len(series)+1))
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_png = os.path.join(out_dir, f"{fuel}_normalized_rescaled.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


def main():
    # Process all subfolders YYYY-MM-DD in ROOT_DIR
    days = [os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
            if os.path.isdir(os.path.join(ROOT_DIR, d))]
    days.sort()

    for day_dir in days:
        try:
            process_sced_normalized_lines_day_per_fuel(
                day_dir,
                PLOTS_DIR,
                save_csv=SAVE_NORMALIZED_VALUES_CSV
            )
            print(f"Processed: {os.path.basename(day_dir)}")
        except Exception as e:
            print(f"! Error processing {day_dir}: {e}")


if __name__ == "__main__":
    main()
