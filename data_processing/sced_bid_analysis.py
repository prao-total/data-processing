#!/usr/bin/env python3
"""
SCED bidding analysis from aggregated time-series directory

Inputs (in INPUT_DIR):
  aggregation_HSL.csv
  aggregation_SCED1_Curve-MW1.csv
  aggregation_SCED1_Curve-Price1.csv
  ...
Each file columns:
  ["Resource Name", "Resource Type", <timestamp1>, <timestamp2>, ...] (5-min resolution)

Pipeline:
  1) Load & melt HSL, MW_k, Price_k into long format
  2) Pair (MW_k, Price_k) with HSL, compute MW_frac = MW/HSL (pairwise, pre-aggregation)
  3) Save normalized long table
  4) Produce stacked step bars for every (month, hour)
  5) Produce 12×24 price heatmaps for each (Resource Type × Step)
  6) Produce violin plots of Price-by-Step (one per Resource Type)

Notes:
  - Uses matplotlib only (no seaborn).
  - MAX_STEPS controls how many (MW, Price) pairs we consider (default 35).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================ CONFIG ============================ #
INPUT_DIR  = Path("/path/to/your/aggregated_timeseries_dir")      # <-- set this
OUTPUT_DIR = Path("/path/to/output/sced_bidding_from_aggregates") # <-- set this
MAX_STEPS  = 35
TIMESTAMP_TZ = None  # e.g., "America/Chicago" or None to keep naive

# Toggle outputs on/off if needed
MAKE_STEP_BARS_BY_MONTH_HOUR = True
MAKE_PRICE_HEATMAPS          = True
MAKE_PRICE_VIOLINS           = True
MAKE_AVG_MWFRAC_LINE         = True
# ================================================================ #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Filename parsing ---------- #
RE_HSL = re.compile(r"^aggregation[_-]HSL\.csv$", re.IGNORECASE)
RE_MW  = re.compile(r"^aggregation[_-]SCED1[_ ]?Curve[-_]?MW(\d+)\.csv$", re.IGNORECASE)
RE_PRC = re.compile(r"^aggregation[_-]SCED1[_ ]?Curve[-_]?Price(\d+)\.csv$", re.IGNORECASE)

def classify_file(fname: str) -> Tuple[str, Optional[int]]:
    if RE_HSL.match(fname): return "HSL", None
    m = RE_MW.match(fname)
    if m: return "MW", int(m.group(1))
    p = RE_PRC.match(fname)
    if p: return "PRICE", int(p.group(1))
    return "OTHER", None

# ---------- IO / melt helpers ---------- #
def melt_metric_csv(path: Path, value_name: str) -> pd.DataFrame:
    """
    Read one aggregation_*.csv and melt timestamps wide->long.
    Returns: columns ["Resource Name","Resource Type","timestamp", value_name]
    """
    df = pd.read_csv(path)

    # Normalize id headers as needed
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
        # Localize naive or convert if tz-aware
        try:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_localize(
                TIMESTAMP_TZ, nonexistent="NaT", ambiguous="NaT"
            )
        except TypeError:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_convert(TIMESTAMP_TZ)
    return long_df

def ingest_directory(input_dir: Path, max_steps: int = 35) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Scan directory and return:
      hsl_long: DataFrame with 'HSL'
      mw_long_map: dict[step] -> DataFrame with 'MW'
      price_long_map: dict[step] -> DataFrame with 'Price'
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
        elif kind == "MW" and step and 1 <= step <= max_steps:
            mw_long_map[step] = melt_metric_csv(f, "MW")
        elif kind == "PRICE" and step and 1 <= step <= max_steps:
            price_long_map[step] = melt_metric_csv(f, "Price")

    if hsl_long is None or hsl_long.empty:
        raise ValueError("No valid aggregation_HSL.csv found or it is empty.")

    return hsl_long, mw_long_map, price_long_map

def merge_step(hsl_long: pd.DataFrame, mw_long: pd.DataFrame, price_long: pd.DataFrame, step: int) -> pd.DataFrame:
    """
    Merge HSL, MW_step, Price_step on (Resource Name, Resource Type, timestamp).
    Compute MW_frac pair-wise and derive month/hour.
    """
    keys = ["Resource Name", "Resource Type", "timestamp"]
    m = hsl_long.merge(mw_long, on=keys, how="inner").merge(price_long, on=keys, how="inner")
    m["Step"] = step

    # Normalize pair-wise (before any aggregation)
    m["MW"]    = pd.to_numeric(m["MW"],    errors="coerce")
    m["Price"] = pd.to_numeric(m["Price"], errors="coerce")
    m["HSL"]   = pd.to_numeric(m["HSL"],   errors="coerce").fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        m["MW_frac"] = np.where(m["HSL"].to_numpy() > 0, m["MW"].to_numpy() / m["HSL"].to_numpy(), np.nan)

    m["hour"]  = m["timestamp"].dt.hour
    m["month"] = m["timestamp"].dt.month
    cols = ["Resource Name","Resource Type","timestamp","month","hour","Step","MW","Price","HSL","MW_frac"]
    return m[cols]

def build_bids_long(input_dir: Path, max_steps: int = 35) -> pd.DataFrame:
    """
    Unified long table:
      ["Resource Name","Resource Type","timestamp","month","hour","Step","MW","Price","HSL","MW_frac"]
    """
    hsl_long, mw_map, price_map = ingest_directory(input_dir, max_steps=max_steps)
    steps = sorted(set(mw_map.keys()).intersection(price_map.keys()))
    if not steps:
        raise ValueError("No overlapping steps found between MW and Price files.")
    steps = [s for s in steps if 1 <= s <= max_steps]

    merged = [merge_step(hsl_long, mw_map[s], price_map[s], s) for s in steps]
    bids_long = pd.concat(merged, ignore_index=True)

    # Keep valid normalized rows; price can be NaN in some views, but normalize rows need finite MW_frac
    bids_long = bids_long.dropna(subset=["MW_frac"])
    bids_long = bids_long[(bids_long["MW_frac"] >= 0) & np.isfinite(bids_long["MW_frac"])]
    return bids_long


# ---------- Charts: Step bars by (month, hour) ---------- #
def plot_step_bars_by_month_hour(bids_long: pd.DataFrame, out_dir: Path):
    """
    For each (month, hour), produce a stacked bar chart:
      x = Step (1..N)
      y = mean(MW_frac) for that (month, hour)
      stacks = Resource Type
    Saves: out_dir/step_bars_by_month_hour/<MM>/bars_month_MM_hour_HH.png
    """
    root = out_dir / "step_bars_by_month_hour"
    root.mkdir(parents=True, exist_ok=True)

    grp = (bids_long
           .groupby(["month","hour","Step","Resource Type"], observed=True)["MW_frac"]
           .mean().reset_index())

    all_types = sorted(grp["Resource Type"].dropna().unique().tolist())
    all_steps = sorted(grp["Step"].dropna().unique().astype(int).tolist())
    months = list(range(1, 13))
    hours = list(range(0, 24))

    for mm in months:
        mdir = root / f"{mm:02d}"
        mdir.mkdir(parents=True, exist_ok=True)

        for hh in hours:
            sub = grp[(grp["month"] == mm) & (grp["hour"] == hh)]
            fig = plt.figure(figsize=(14, 6))
            if sub.empty:
                plt.title(f"Normalized Bids by Step — Month {mm}, Hour {hh} (no data)")
                plt.xlabel("Step (1..N)")
                plt.ylabel("Mean normalized MW (MW/HSL)")
            else:
                # Matrix: step x type
                mat = np.zeros((len(all_steps), len(all_types)))
                for i, s in enumerate(all_steps):
                    row = sub[sub["Step"] == s]
                    vals = row.set_index("Resource Type")["MW_frac"].reindex(all_types).fillna(0).to_numpy()
                    mat[i, :] = vals

                x = np.arange(len(all_steps))
                bottom = np.zeros(len(all_steps))
                for j, rtype in enumerate(all_types):
                    heights = mat[:, j]
                    plt.bar(x, heights, bottom=bottom, label=rtype)
                    bottom += heights

                plt.xticks(x, all_steps)
                plt.xlabel("Step (1..N)")
                plt.ylabel("Mean normalized MW (MW/HSL)")
                plt.title(f"How Resource Types Bid by Step — Month {mm}, Hour {hh}")
                plt.legend(ncol=3, fontsize=8)

            out_path = mdir / f"bars_month_{mm:02d}_hour_{hh:02d}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


# ---------- Charts: 12×24 price heatmaps per (Resource Type × Step) ---------- #
def plot_price_heatmaps_month_hour(bids_long: pd.DataFrame, out_dir: Path):
    """
    One heatmap per (Resource Type, Step):
      x = hour (0..23), y = month (1..12), value = mean Price
    Saves: out_dir/price_heatmaps/<ResourceType>/heatmap_price_<rtype>_stepXX.png
    """
    heatmaps_root = out_dir / "price_heatmaps"
    heatmaps_root.mkdir(parents=True, exist_ok=True)

    grp = (bids_long.groupby(["Resource Type","Step","month","hour"], observed=True)["Price"]
           .mean().reset_index())

    resource_types = sorted(grp["Resource Type"].dropna().unique().tolist())
    steps = sorted(grp["Step"].dropna().unique().astype(int).tolist())

    hours = np.arange(0, 24)
    months = np.arange(1, 13)

    for rtype in resource_types:
        type_dir = heatmaps_root / re.sub(r"[^A-Za-z0-9._-]+","_", rtype)
        type_dir.mkdir(parents=True, exist_ok=True)
        sub_type = grp[grp["Resource Type"] == rtype]

        for s in steps:
            sub = sub_type[sub_type["Step"] == s]
            if sub.empty:
                mat = np.full((len(months), len(hours)), np.nan)
            else:
                pivot = sub.pivot(index="month", columns="hour", values="Price")
                pivot = pivot.reindex(index=months, columns=hours)
                mat = pivot.to_numpy()

            fig = plt.figure(figsize=(10, 6))
            plt.imshow(
                mat, aspect="auto", origin="lower",
                extent=[hours.min()-0.5, hours.max()+0.5, months.min()-0.5, months.max()+0.5]
            )
            cbar = plt.colorbar()
            cbar.set_label("Mean bid price ($/MWh)")
            plt.xlabel("Hour of day")
            plt.ylabel("Month (1–12)")
            plt.title(f"Price Heatmap — {rtype} — Step {s}")
            plt.xticks(np.arange(0, 24, 1))
            plt.yticks(np.arange(1, 13, 1))
            out_path = type_dir / f"heatmap_price_{re.sub(r'[^A-Za-z0-9._-]+','_', rtype)}_step{s:02d}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


# ---------- Charts: Violin plots of Price by Step (per Resource Type) ---------- #
def plot_price_violins_by_step_per_type(
    bids_long: pd.DataFrame,
    out_dir: Path,
    steps: List[int] | None = None,
    clip_quantiles: tuple[float, float] | None = (0.01, 0.99),
    max_types: int | None = None,
):
    """
    One violin plot per Resource Type:
      x = Step (1..N)
      y = Price ($/MWh)
      Each violin is the price distribution for that step.

    Parameters:
      steps          : restrict to subset of steps (default = all)
      clip_quantiles : trim tails to reduce extreme outliers (set None to disable)
      max_types      : limit number of resource types to plot (None = all)
    """
    viol_root = out_dir / "price_violins"
    viol_root.mkdir(parents=True, exist_ok=True)

    df = bids_long.dropna(subset=["Price", "Step", "Resource Type"]).copy()

    all_steps = sorted(df["Step"].dropna().unique().astype(int).tolist())
    if steps is None:
        steps = all_steps
    else:
        steps = [s for s in steps if s in all_steps]
        if not steps:
            raise ValueError("No requested steps found in data.")

    type_counts = df.groupby("Resource Type").size().sort_values(ascending=False)
    resource_types = type_counts.index.tolist()
    if max_types is not None and max_types > 0:
        resource_types = resource_types[:max_types]

    for rtype in resource_types:
        sub = df[df["Resource Type"] == rtype]
        if sub.empty:
            continue

        # Build arrays of prices per step
        data_arrays = []
        for s in steps:
            arr = pd.to_numeric(sub.loc[sub["Step"] == s, "Price"], errors="coerce").dropna().to_numpy()
            if arr.size > 0 and clip_quantiles is not None:
                lo, hi = np.nanquantile(arr, clip_quantiles[0]), np.nanquantile(arr, clip_quantiles[1])
                arr = arr[(arr >= lo) & (arr <= hi)]
            data_arrays.append(arr)

        if not any(len(a) for a in data_arrays):
            continue

        fig = plt.figure(figsize=(14, 6))
        parts = plt.violinplot(
            data_arrays,
            positions=np.arange(1, len(steps) + 1),
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.8,
        )
        for pc in parts['bodies']:
            pc.set_alpha(0.7)

        plt.xlabel("Step")
        plt.ylabel("Bid price ($/MWh)")
        plt.title(f"SCED Bid Price by Step — {rtype}")
        plt.xticks(np.arange(1, len(steps) + 1), steps)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)

        safe_rtype = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rtype))
        out_path = viol_root / f"violin_price_by_step_{safe_rtype}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------- Chart: Average MW_frac by Step (line per Resource Type) ---------- #
def plot_avg_mwfrac_by_step_all_types(
    bids_long: pd.DataFrame,
    out_dir: Path,
    steps: list[int] | None = None,
    min_points_per_step: int = 1,
    legend_ncol: int = 3,
):
    """
    Single plot:
      - x-axis: SCED Step (1..N)
      - y-axis: Average MW_frac (normalized MW/HSL)
      - One line per Resource Type (averaged across the whole time range)

    Parameters
    ----------
    bids_long : DataFrame with columns ["Resource Type","Step","MW_frac"]
    out_dir   : base output directory (PNG saved here)
    steps     : optional subset of steps to plot; default = all present (sorted)
    min_points_per_step : require at least this many observations to include a (type, step) mean
    legend_ncol : number of columns for legend layout
    """
    df = bids_long.dropna(subset=["Resource Type", "Step", "MW_frac"]).copy()
    df["Step"] = df["Step"].astype(int)

    # Decide steps to show
    all_steps = sorted(df["Step"].unique().tolist())
    if steps is None:
        steps = all_steps
    else:
        steps = [s for s in steps if s in all_steps]
        if not steps:
            raise ValueError("No requested steps found in data.")

    # Compute mean MW_frac per (Resource Type, Step), with a min sample filter
    stats = (df.groupby(["Resource Type", "Step"], observed=True)
               .agg(n=("MW_frac", "size"),
                    avg=("MW_frac", "mean"))
               .reset_index())
    stats = stats[stats["n"] >= min_points_per_step]

    # Pivot to step columns so we can plot lines per resource type
    wide = stats.pivot(index="Resource Type", columns="Step", values="avg")
    # Restrict to selected steps and keep order
    wide = wide.reindex(columns=steps)
    # Sort resource types by overall average (descending) to reduce legend clutter
    wide["__order"] = wide.mean(axis=1, skipna=True)
    wide = wide.sort_values("__order", ascending=False).drop(columns="__order")

    if wide.empty:
        print("No data available for average MW_frac by step plot.")
        return None

    # Plot
    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(steps))
    for rtype, row in wide.iterrows():
        y = row.to_numpy(dtype=float)
        # Mask all-nan lines
        if np.all(np.isnan(y)):
            continue
        plt.plot(x, y, marker="o", label=str(rtype))

    plt.xticks(x, steps)
    plt.xlabel("SCED Step")
    plt.ylabel("Average normalized MW (MW/HSL)")
    plt.title("Average MW_frac by SCED Step (line per Resource Type)")
    plt.grid(True, linestyle="--", alpha=0.3)
    # Put legend outside if many types
    lgd = plt.legend(ncol=legend_ncol, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend if below

    out_path = out_dir / "avg_mwfrac_by_step_all_types.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path



# ============================ Main ============================ #
def main():
    print(f"Reading aggregated files from: {INPUT_DIR}")
    bids_long = build_bids_long(INPUT_DIR, max_steps=MAX_STEPS)

    out_csv = OUTPUT_DIR / "bids_long_steps_from_aggregates.csv"
    bids_long.to_csv(out_csv, index=False)
    print(f"✅ Wrote normalized long table: {out_csv}")

    if MAKE_STEP_BARS_BY_MONTH_HOUR:
        print("Generating stacked step bars for each (month, hour)...")
        plot_step_bars_by_month_hour(bids_long, OUTPUT_DIR)
        print(f"✅ Bar charts saved under: {OUTPUT_DIR / 'step_bars_by_month_hour'}")

    if MAKE_PRICE_HEATMAPS:
        print("Generating 12×24 price heatmaps for each Resource Type × Step...")
        plot_price_heatmaps_month_hour(bids_long, OUTPUT_DIR)
        print(f"✅ Heatmaps saved under: {OUTPUT_DIR / 'price_heatmaps'}")

    if MAKE_PRICE_VIOLINS:
        print("Generating violin plots of Price by Step per Resource Type...")
        plot_price_violins_by_step_per_type(
            bids_long,
            out_dir=OUTPUT_DIR,
            steps=None,                # or list(range(1, 11)) to limit steps
            clip_quantiles=(0.01, 0.99),  # set to None to disable clipping
            max_types=None             # or an int to limit how many types
        )
        print(f"✅ Violins saved under: {OUTPUT_DIR / 'price_violins'}")
    
    if MAKE_AVG_MWFRAC_LINE:
        print("Generating single plot of Average MW_frac vs SCED Step (line per Resource Type)...")
        avg_line_path = plot_avg_mwfrac_by_step_all_types(
            bids_long,
            out_dir=OUTPUT_DIR,
            steps=None,              # or e.g. list(range(1, 16)) to limit to steps 1–15
            min_points_per_step=5,   # raise if you want to suppress sparse averages
            legend_ncol=4            # adjust legend layout
        )
        print("✅ Saved:", avg_line_path)

    print("Done.")

if __name__ == "__main__":
    main()
