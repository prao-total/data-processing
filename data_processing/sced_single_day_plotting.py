
#!/usr/bin/env python3
"""
SCED single-day plotting — combined multi-fuel normalized curves AND per-fuel splits
(keeping the exact same series values for both).

Normalization & rescaling logic:
- For each (resource, timestamp), compute M_sced = max across SCED MW steps.
- Let M_all = max(M_sced, HSL). Normalize each SCED step, HSL, and Base Point by M_all.
- Aggregate by Resource Type (fuel) across rows × timestamps to get one mean per step (and scalars for HSL/BP).
- Reapply magnitude by multiplying those means with Σ(M_sced) for that fuel (sum of per-row SCED maxima,
  time-averaged per row).
- Plot:
  1) A combined chart with all fuels (legacy "all normalized bid by stage" view).
  2) One chart per fuel using the EXACT SAME series values from (1).

HSL and Base Point are dashed lines with distinct colors.
"""

from __future__ import annotations
import os
import re
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- Config -----------------------
ROOT_DIR = os.environ.get("ROOT_DIR", "./inputs/sced_day")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "./plots")

BASE_POINT_NAME_HINT = "aggregation_Base_Point.csv"
HSL_NAME_HINT        = "aggregation_HSL.csv"
SCED_MW_GLOB         = "aggregation_SCED1_Curve-MW*.csv"

SAVE_NORMALIZED_VALUES_CSV = os.environ.get("SAVE_NORMALIZED_VALUES_CSV", "0") == "1"

# Distinct dashed colors for HSL & BP as requested
HSL_LINE_COLOR = "#1f77b4"   # blue-ish
BP_LINE_COLOR  = "#d62728"   # red-ish
# ------------------------------------------------------


def _find_case_insensitive(path: str, name_hint: str) -> Optional[str]:
    p = os.path.join(path, name_hint)
    if os.path.exists(p):
        return p
    lower_hint = name_hint.lower()
    for fn in os.listdir(path):
        if fn.lower() == lower_hint:
            return os.path.join(path, fn)
    return None


def _find_glob(path: str, pattern: str) -> List[str]:
    cand = sorted(glob.glob(os.path.join(path, pattern)))
    if cand:
        return cand
    cand = sorted(glob.glob(os.path.join(path, "*SCED*Curve*MW*.csv")))
    return cand


def _detect_key_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    cols = {c.strip().lower(): c for c in df.columns}
    rn = cols.get("resource name") or cols.get("resourcename")
    rt = cols.get("resource type") or cols.get("resourcetype")
    if rn is None:
        for c in df.columns:
            if re.sub(r"\s+", "", c.strip().lower()) in ("resourcename",):
                rn = c
                break
    if rn is None:
        raise ValueError("Could not find 'Resource Name' column.")
    return rn, rt


def _timestamp_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        try:
            pd.to_datetime(c)
            out.append(c)
        except Exception:
            pass
    return out


def _coerce_numeric(df: pd.DataFrame, ts_cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in ts_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _align_on_resources_and_timestamps(dfs: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str], str, Optional[str]]:
    rn, rt = None, None
    for df in dfs:
        try:
            rn, rt = _detect_key_cols(df)
            break
        except Exception:
            continue
    if rn is None:
        raise ValueError("Could not detect key columns from inputs.")

    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()

    # common resources
    res_sets = []
    for df in dfs:
        if rn not in df.columns:
            raise ValueError(f"Missing key col '{rn}' in an input.")
        res_sets.append(set(_norm(df[rn]).tolist()))
    common_res = set.intersection(*res_sets) if res_sets else set()

    # common timestamps
    ts_common = None
    for df in dfs:
        ts = _timestamp_columns(df, exclude=[rn] + ([rt] if rt else []))
        ts_common = set(ts) if ts_common is None else (ts_common & set(ts))
    if not ts_common:
        raise ValueError("No common timestamp columns across inputs.")
    ts_common = sorted(ts_common, key=lambda x: pd.to_datetime(x))

    aligned = []
    for df in dfs:
        d = df.copy()
        d["_rn_norm"] = _norm(d[rn])
        d = d[d["_rn_norm"].isin(common_res)].drop(columns=["_rn_norm"])
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
    rn, rt = _detect_key_cols(hsl_df)
    ts_cols = _timestamp_columns(hsl_df, exclude=[rn] + ([rt] if rt else []))

    sced_vals = np.stack([df[ts_cols].to_numpy(dtype=float) for df in sced_steps], axis=0)  # (K,R,T)
    M_sced = np.nanmax(sced_vals, axis=0)  # (R,T)
    hsl_vals = hsl_df[ts_cols].to_numpy(dtype=float)
    M_all = np.fmax(M_sced, hsl_vals)

    denom = M_all.copy()
    denom[~np.isfinite(denom) | (denom <= 0)] = np.nan

    sced_norm_steps: List[pd.DataFrame] = []
    for step_df in sced_steps:
        num = step_df[ts_cols].to_numpy(dtype=float)
        norm_vals = num / denom
        out = step_df[[rn] + ([rt] if rt else [])].copy()
        for j, c in enumerate(ts_cols):
            out[c] = norm_vals[:, j]
        sced_norm_steps.append(out)

    def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
        num = df[ts_cols].to_numpy(dtype=float)
        norm_vals = num / denom
        out = df[[rn] + ([rt] if rt else [])].copy()
        for j, c in enumerate(ts_cols):
            out[c] = norm_vals[:, j]
        return out

    hsl_norm = _norm_df(hsl_df)
    bp_norm  = _norm_df(bp_df)

    sced_row_max = hsl_df[[rn] + ([rt] if rt else [])].copy()
    for j, c in enumerate(ts_cols):
        sced_row_max[c] = M_sced[:, j]

    return sced_norm_steps, hsl_norm, bp_norm, sced_row_max


def _aggregate_all_fuels_same_series(
    sced_norm_steps: List[pd.DataFrame],
    hsl_norm: pd.DataFrame,
    bp_norm: pd.DataFrame,
    sced_row_max: pd.DataFrame
) -> Tuple[Dict[str, pd.Series], Dict[str, float], Dict[str, float], List[str]]:
    rn, rt = _detect_key_cols(hsl_norm)
    ts_cols = _timestamp_columns(hsl_norm, exclude=[rn] + ([rt] if rt else []))
    if rt is None:
        raise ValueError("Resource Type column is required to aggregate by fuel.")
    fuels = sorted(hsl_norm[rt].dropna().astype(str).unique())

    sced_row_max_mean = sced_row_max.copy()
    sced_row_max_mean["_row_mean"] = sced_row_max[ts_cols].mean(axis=1, skipna=True)
    fuel_max_sum = sced_row_max_mean.groupby(rt)["_row_mean"].sum(min_count=1).to_dict()

    step_labels = [f"Step {i+1}" for i in range(len(sced_norm_steps))]

    step_curve_by_fuel: Dict[str, pd.Series] = {}
    for fuel in fuels:
        denom = fuel_max_sum.get(fuel, np.nan)
        vals = []
        for df in sced_norm_steps:
            sub = df[df[rt].astype(str) == fuel]
            m = np.nanmean(sub[ts_cols].to_numpy(dtype=float)) if not sub.empty else np.nan
            vals.append(m * denom if np.isfinite(denom) else np.nan)
        step_curve_by_fuel[fuel] = pd.Series(vals, index=step_labels, dtype="float64")

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


def _plot_combined_all_fuels(day: str,
                             out_dir: str,
                             step_curve_by_fuel: Dict[str, pd.Series],
                             hsl_by_fuel: Dict[str, float],
                             bp_by_fuel: Dict[str, float]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    for fuel, series in step_curve_by_fuel.items():
        ax.plot(range(1, len(series) + 1), series.values, marker="o", linewidth=2, label=fuel)

    ax.set_title(f"{day} — Normalized & Rescaled SCED Curve (All Fuels)")
    ax.set_xlabel("SCED Step")
    ax.set_ylabel("Rescaled magnitude (Σ row SCED maxima weighted)")
    ax.set_xticks(range(1, len(next(iter(step_curve_by_fuel.values()))) + 1))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Fuel", ncols=2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "combined_all_fuels.png"), dpi=150)
    plt.close(fig)


def _plot_per_fuel_from_same_series(day: str,
                                    out_dir: str,
                                    step_curve_by_fuel: Dict[str, pd.Series],
                                    hsl_by_fuel: Dict[str, float],
                                    bp_by_fuel: Dict[str, float]) -> None:
    out_dir = os.path.join(out_dir, "per_fuel")
    os.makedirs(out_dir, exist_ok=True)

    for fuel, series in step_curve_by_fuel.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(series) + 1), series.values, marker="o", linewidth=2, label=fuel)

        hval = hsl_by_fuel.get(fuel, np.nan)
        bval = bp_by_fuel.get(fuel, np.nan)
        if np.isfinite(hval):
            ax.axhline(hval, linestyle="--", linewidth=1.75, color=HSL_LINE_COLOR, label="HSL (rescaled)")
        if np.isfinite(bval):
            ax.axhline(bval, linestyle="--", linewidth=1.75, color=BP_LINE_COLOR,  label="Base Point (rescaled)")

        ax.set_title(f"{day} — Normalized & Rescaled SCED Curve (Fuel: {fuel})")
        ax.set_xlabel("SCED Step")
        ax.set_ylabel("Rescaled magnitude (Σ row SCED maxima weighted)")
        ax.set_xticks(range(1, len(series) + 1))
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{fuel}_normalized_rescaled.png"), dpi=150)
        plt.close(fig)


def process_sced_normalized_lines_day_split(day_dir: str, plots_root: str, save_csv: bool = False) -> None:
    os.makedirs(plots_root, exist_ok=True)
    day = os.path.basename(os.path.normpath(day_dir))
    base_out_dir = os.path.join(plots_root, day, "sced_normalized_all")
    os.makedirs(base_out_dir, exist_ok=True)

    hsl_path = _find_case_insensitive(day_dir, HSL_NAME_HINT)
    bp_path  = _find_case_insensitive(day_dir, BASE_POINT_NAME_HINT)
    step_paths = _find_glob(day_dir, SCED_MW_GLOB)

    if not hsl_path:
        raise FileNotFoundError(f"HSL file not found in {day_dir}")
    if not bp_path:
        raise FileNotFoundError(f"Base Point file not found in {day_dir}")
    if not step_paths:
        raise FileNotFoundError(f"No SCED MW step files found in {day_dir}")

    hsl_df = pd.read_csv(hsl_path)
    bp_df  = pd.read_csv(bp_path)
    sced_steps = [pd.read_csv(p) for p in step_paths]

    aligned, ts_cols, rn, rt = _align_on_resources_and_timestamps(sced_steps + [hsl_df, bp_df])
    K = len(sced_steps)
    aligned_steps = aligned[:K]
    hsl_aligned  = aligned[K]
    bp_aligned   = aligned[K+1]

    sced_norm_steps, hsl_norm, bp_norm, sced_row_max = _build_rowwise_max_and_normalized(
        aligned_steps, hsl_aligned, bp_aligned
    )

    step_curve_by_fuel, hsl_by_fuel, bp_by_fuel, step_labels = _aggregate_all_fuels_same_series(
        sced_norm_steps, hsl_norm, bp_norm, sced_row_max
    )

    if save_csv:
        out_csv_dir = os.path.join(base_out_dir, "csv")
        os.makedirs(out_csv_dir, exist_ok=True)
        for fuel, series in step_curve_by_fuel.items():
            series.to_csv(os.path.join(out_csv_dir, f"{fuel}_normalized_rescaled_curve.csv"), header=["value"])
        pd.DataFrame(
            [{"fuel": f, "hsl_rescaled": hsl_by_fuel.get(f, np.nan), "base_point_rescaled": bp_by_fuel.get(f, np.nan)}
             for f in step_curve_by_fuel.keys()]
        ).to_csv(os.path.join(out_csv_dir, "hsl_bp_rescaled_summary.csv"), index=False)

    _plot_combined_all_fuels(day, base_out_dir, step_curve_by_fuel, hsl_by_fuel, bp_by_fuel)
    _plot_per_fuel_from_same_series(day, base_out_dir, step_curve_by_fuel, hsl_by_fuel, bp_by_fuel)


def main():
    days = [os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
            if os.path.isdir(os.path.join(ROOT_DIR, d))]
    days.sort()
    for day_dir in days:
        try:
            process_sced_normalized_lines_day_split(
                day_dir,
                PLOTS_DIR,
                save_csv=SAVE_NORMALIZED_VALUES_CSV
            )
            print(f"Processed: {os.path.basename(day_dir)}")
        except Exception as e:
            print(f"! Error processing {day_dir}: {e}")


if __name__ == "__main__":
    main()
