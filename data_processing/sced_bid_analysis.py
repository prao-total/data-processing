#!/usr/bin/env python3
"""
Memory-safe SCED bidding analysis from aggregated time-series directory
- Column-batched reading (usecols) to avoid huge melts
- Streaming aggregations for charts (no giant long DF in memory)
- Reservoir sampling for violin plots

Inputs (in INPUT_DIR):
  aggregation_HSL.csv
  aggregation_SCED1_Curve-MW1.csv
  aggregation_SCED1_Curve-Price1.csv
  ...
Each file columns:
  ["Resource Name", "Resource Type", <timestamp1>, <timestamp2>, ...] (5-min resolution)
"""

from __future__ import annotations
import math, random, re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================ CONFIG ============================ #
INPUT_DIR  = Path("/path/to/your/aggregated_timeseries_dir")      # <-- set this
OUTPUT_DIR = Path("/path/to/output/sced_bidding_streaming")       # <-- set this

MAX_STEPS  = 35                 # number of (MW, Price) pairs to consider
TS_BATCH_SIZE = 200             # how many timestamp columns to process at once
TIMESTAMP_TZ = None             # e.g., "America/Chicago" or None for naive

# Violin sampling cap (per Resource Type x Step)
MAX_VIOLIN_SAMPLES_PER_GROUP = 8000

# Toggle outputs
MAKE_STEP_BARS_BY_MONTH_HOUR = True
MAKE_PRICE_HEATMAPS          = True
MAKE_AVG_MWFRAC_LINE         = True
MAKE_AVG_PRICE_LINE          = True
MAKE_PRICE_VIOLINS           = True
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


# ---------- Utility: list timestamp columns & batching ---------- #
def list_timestamp_columns(sample_path: Path) -> List[str]:
    """Read only header of a file to get timestamp columns (besides id vars)."""
    df = pd.read_csv(sample_path, nrows=1)
    rename_map = {}
    if "Resource Node" in df.columns:
        rename_map["Resource Node"] = "Resource Name"
    if rename_map:
        df = df.rename(columns=rename_map)
    id_vars = ["Resource Name", "Resource Type"]
    ts_cols = [c for c in df.columns if c not in id_vars]
    return ts_cols

def batched(lst: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def read_metric_batch(path: Path, ts_cols: List[str], value_name: str) -> pd.DataFrame:
    """Read id vars + given ts columns for one metric file."""
    usecols = ["Resource Name", "Resource Type"] + ts_cols
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    rename_map = {}
    if "Resource Node" in df.columns:
        rename_map["Resource Node"] = "Resource Name"
    if rename_map:
        df = df.rename(columns=rename_map)
    long_df = df.melt(id_vars=["Resource Name","Resource Type"], value_vars=ts_cols,
                      var_name="timestamp", value_name=value_name)
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], errors="coerce")
    if TIMESTAMP_TZ:
        try:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_localize(
                TIMESTAMP_TZ, nonexistent="NaT", ambiguous="NaT")
        except TypeError:
            long_df["timestamp"] = long_df["timestamp"].dt.tz_convert(TIMESTAMP_TZ)
    return long_df


# ---------- Reservoir sampling for violins ---------- #
class Reservoir:
    """Fixed-size reservoir per group key using Vitter's Algorithm R."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.samples: List[float] = []
        self.n_seen = 0

    def add_many(self, values: np.ndarray):
        for v in values:
            self.add(v)

    def add(self, value: float):
        if np.isnan(value): 
            return
        self.n_seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append(float(value))
        else:
            j = random.randint(1, self.n_seen)
            if j <= self.capacity:
                self.samples[j-1] = float(value)

# ---------- Streaming accumulators ---------- #
# For bars: mean MW_frac by (month, hour, step, resource type)
mwfrac_sum_mhst: Dict[Tuple[int,int,int,str], float] = {}
mwfrac_cnt_mhst: Dict[Tuple[int,int,int,str], int]   = {}

# For heatmaps: mean Price by (resource type, step, month, hour)
price_sum_tsmh: Dict[Tuple[str,int,int,int], float] = {}
price_cnt_tsmh: Dict[Tuple[str,int,int,int], int]   = {}

# For single lines:
#   avg MW_frac by (resource type, step)
mwfrac_sum_ts: Dict[Tuple[str,int], float] = {}
mwfrac_cnt_ts: Dict[Tuple[str,int], int]   = {}
#   avg Price by (resource type, step)
price_sum_ts: Dict[Tuple[str,int], float] = {}
price_cnt_ts: Dict[Tuple[str,int], int]   = {}

# Violins: reservoir per (resource type, step)
violin_reservoirs: Dict[Tuple[str,int], Reservoir] = {}

def acc_add(dsum: dict, dcnt: dict, key, values: np.ndarray):
    vals = values[np.isfinite(values)]
    if vals.size == 0: 
        return
    dsum[key] = dsum.get(key, 0.0) + float(np.sum(vals))
    dcnt[key] = dcnt.get(key, 0) + int(vals.size)


# ---------- Directory ingest & streaming process ---------- #
def ingest_and_stream(input_dir: Path, max_steps: int = 35):
    # Find files
    hsl_path = None
    mw_paths: Dict[int, Path] = {}
    pr_paths: Dict[int, Path] = {}
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        kind, step = classify_file(f.name)
        if kind == "HSL":
            hsl_path = f
        elif kind == "MW" and step and 1 <= step <= max_steps:
            mw_paths[step] = f
        elif kind == "PRICE" and step and 1 <= step <= max_steps:
            pr_paths[step] = f
    if not hsl_path:
        raise ValueError("aggregation_HSL.csv not found.")

    # Steps available in both MW & Price
    steps = sorted(set(mw_paths.keys()).intersection(pr_paths.keys()))
    if not steps:
        raise ValueError("No overlapping steps found between MW and Price files.")

    # Timestamp columns (from HSL header)
    ts_cols_all = list_timestamp_columns(hsl_path)

    # Process in column batches
    for ts_cols in batched(ts_cols_all, TS_BATCH_SIZE):
        # Read HSL batch
        hsl_long = read_metric_batch(hsl_path, ts_cols, "HSL")
        hsl_long["HSL"] = pd.to_numeric(hsl_long["HSL"], errors="coerce").fillna(0.0)

        # Process step by step within the batch
        for s in steps:
            mw_long = read_metric_batch(mw_paths[s], ts_cols, "MW")
            pr_long = read_metric_batch(pr_paths[s], ts_cols, "Price")

            # Inner join on (name, type, ts)
            keys = ["Resource Name","Resource Type","timestamp"]
            m = hsl_long.merge(mw_long, on=keys, how="inner").merge(pr_long, on=keys, how="inner")

            # Numeric & MW_frac
            m["MW"]    = pd.to_numeric(m["MW"], errors="coerce")
            m["Price"] = pd.to_numeric(m["Price"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                mw_frac = np.where(m["HSL"].to_numpy() > 0, m["MW"].to_numpy()/m["HSL"].to_numpy(), np.nan)
            m["MW_frac"] = mw_frac

            # month/hour
            m["month"] = m["timestamp"].dt.month
            m["hour"]  = m["timestamp"].dt.hour

            # Group-level arrays
            # 1) Bars: by (month, hour, step, type)
            for (rt, mo, hr), grp in m.groupby(["Resource Type","month","hour"]):
                key_base = (mo, hr, s, str(rt))
                vals = grp["MW_frac"].to_numpy(dtype=float)
                acc_add(mwfrac_sum_mhst, mwfrac_cnt_mhst, key_base, vals)

            # 2) Heatmaps: by (type, step, month, hour)
            for (rt, mo, hr), grp in m.groupby(["Resource Type","month","hour"]):
                key_hm = (str(rt), s, mo, hr)
                acc_add(price_sum_tsmh, price_cnt_tsmh, key_hm, grp["Price"].to_numpy(dtype=float))

            # 3) Single lines: avg MW_frac & avg Price by (type, step)
            for rt, grp in m.groupby("Resource Type"):
                key_ts = (str(rt), s)
                acc_add(mwfrac_sum_ts, mwfrac_cnt_ts, key_ts, grp["MW_frac"].to_numpy(dtype=float))
                acc_add(price_sum_ts,  price_cnt_ts,  key_ts, grp["Price"].to_numpy(dtype=float))

            # 4) Violins: reservoir per (type, step)
            for rt, grp in m.groupby("Resource Type"):
                key_v = (str(rt), s)
                res = violin_reservoirs.get(key_v)
                if res is None:
                    res = Reservoir(MAX_VIOLIN_SAMPLES_PER_GROUP)
                    violin_reservoirs[key_v] = res
                res.add_many(grp["Price"].to_numpy(dtype=float))

            # drop batch frames quickly
            del mw_long, pr_long, m

        del hsl_long


# ---------- Convert accumulators to DataFrames ---------- #
def df_bars_month_hour() -> pd.DataFrame:
    rows = []
    for key, ssum in mwfrac_sum_mhst.items():
        mo, hr, step, rtype = key
        cnt = mwfrac_cnt_mhst.get(key, 0)
        if cnt > 0:
            rows.append((mo, hr, step, rtype, ssum/cnt))
    return pd.DataFrame(rows, columns=["month","hour","Step","Resource Type","avg_mw_frac"])

def df_heatmap_price() -> pd.DataFrame:
    rows = []
    for key, ssum in price_sum_tsmh.items():
        rtype, step, mo, hr = key
        cnt = price_cnt_tsmh.get(key, 0)
        if cnt > 0:
            rows.append((rtype, step, mo, hr, ssum/cnt))
    return pd.DataFrame(rows, columns=["Resource Type","Step","month","hour","avg_price"])

def df_avg_mwfrac_lines() -> pd.DataFrame:
    rows = []
    for key, ssum in mwfrac_sum_ts.items():
        rtype, step = key
        cnt = mwfrac_cnt_ts.get(key, 0)
        if cnt > 0:
            rows.append((rtype, step, ssum/cnt))
    return pd.DataFrame(rows, columns=["Resource Type","Step","avg_mw_frac"])

def df_avg_price_lines() -> pd.DataFrame:
    rows = []
    for key, ssum in price_sum_ts.items():
        rtype, step = key
        cnt = price_cnt_ts.get(key, 0)
        if cnt > 0:
            rows.append((rtype, step, ssum/cnt))
    return pd.DataFrame(rows, columns=["Resource Type","Step","avg_price"])


# ---------- Plotting (uses aggregates & reservoirs) ---------- #
def plot_step_bars_by_month_hour_from_agg(df: pd.DataFrame, out_dir: Path):
    root = out_dir / "step_bars_by_month_hour"
    root.mkdir(parents=True, exist_ok=True)

    all_types = sorted(df["Resource Type"].dropna().unique().tolist())
    all_steps = sorted(df["Step"].dropna().unique().astype(int).tolist())
    months = list(range(1, 13))
    hours = list(range(0, 24))

    for mm in months:
        mdir = root / f"{mm:02d}"
        mdir.mkdir(parents=True, exist_ok=True)
        for hh in hours:
            sub = df[(df["month"]==mm) & (df["hour"]==hh)]
            fig = plt.figure(figsize=(14, 6))
            if sub.empty:
                plt.title(f"Normalized Bids by Step — Month {mm}, Hour {hh} (no data)")
                plt.xlabel("Step (1..N)"); plt.ylabel("Mean normalized MW (MW/HSL)")
            else:
                mat = np.zeros((len(all_steps), len(all_types)))
                for i, s in enumerate(all_steps):
                    row = sub[sub["Step"] == s].set_index("Resource Type")["avg_mw_frac"].reindex(all_types).fillna(0)
                    mat[i,:] = row.to_numpy()
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
            plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_price_heatmaps_from_agg(df: pd.DataFrame, out_dir: Path):
    heatmaps_root = out_dir / "price_heatmaps"
    heatmaps_root.mkdir(parents=True, exist_ok=True)

    rtypes = sorted(df["Resource Type"].dropna().unique().tolist())
    steps  = sorted(df["Step"].dropna().unique().astype(int).tolist())
    hours  = np.arange(0, 24)
    months = np.arange(1, 13)

    for rtype in rtypes:
        type_dir = heatmaps_root / re.sub(r"[^A-Za-z0-9._-]+","_", rtype)
        type_dir.mkdir(parents=True, exist_ok=True)
        sub_type = df[df["Resource Type"]==rtype]
        for s in steps:
            sub = sub_type[sub_type["Step"]==s]
            if sub.empty:
                mat = np.full((len(months), len(hours)), np.nan)
            else:
                pivot = sub.pivot(index="month", columns="hour", values="avg_price").reindex(index=months, columns=hours)
                mat = pivot.to_numpy()
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(mat, aspect="auto", origin="lower",
                       extent=[hours.min()-0.5, hours.max()+0.5, months.min()-0.5, months.max()+0.5])
            cbar = plt.colorbar(); cbar.set_label("Mean bid price ($/MWh)")
            plt.xlabel("Hour of day"); plt.ylabel("Month (1–12)")
            plt.title(f"Price Heatmap — {rtype} — Step {s}")
            plt.xticks(np.arange(0, 24, 1)); plt.yticks(np.arange(1, 13, 1))
            out_path = type_dir / f"heatmap_price_{re.sub(r'[^A-Za-z0-9._-]+','_', rtype)}_step{s:02d}.png"
            plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_avg_mwfrac_by_step_all_types_from_agg(df: pd.DataFrame, out_dir: Path):
    wide = df.pivot(index="Resource Type", columns="Step", values="avg_mw_frac").sort_index()
    steps = sorted(wide.columns.astype(int).tolist())
    if len(steps)==0: 
        return
    fig = plt.figure(figsize=(12,6))
    x = np.arange(len(steps))
    # Order legend by overall mean
    order = wide.mean(axis=1, skipna=True).sort_values(ascending=False).index.tolist()
    for rtype in order:
        y = wide.loc[rtype, steps].to_numpy(dtype=float)
        if np.all(np.isnan(y)): 
            continue
        plt.plot(x, y, marker="o", label=str(rtype))
    plt.xticks(x, steps); plt.xlabel("SCED Step")
    plt.ylabel("Average normalized MW (MW/HSL)")
    plt.title("Average MW_frac by SCED Step (line per Resource Type)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(rect=[0,0.05,1,1])
    out_path = out_dir / "avg_mwfrac_by_step_all_types.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_avg_price_by_step_one_figure_all_types_from_agg(df: pd.DataFrame, out_dir: Path):
    wide = df.pivot(index="Resource Type", columns="Step", values="avg_price").sort_index()
    steps = sorted(wide.columns.astype(int).tolist())
    if len(steps)==0: 
        return
    fig = plt.figure(figsize=(12,6))
    x = np.arange(len(steps))
    order = wide.mean(axis=1, skipna=True).sort_values(ascending=False).index.tolist()
    for rtype in order:
        y = wide.loc[rtype, steps].to_numpy(dtype=float)
        if np.all(np.isnan(y)): 
            continue
        plt.plot(x, y, marker="o", label=str(rtype))
    plt.xticks(x, steps); plt.xlabel("SCED Step")
    plt.ylabel("Average bid Price ($/MWh)")
    plt.title("Average Price by SCED Step (line per Resource Type)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(rect=[0,0.05,1,1])
    out_path = out_dir / "avg_price_by_step_all_types.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_price_violins_from_reservoirs(out_dir: Path):
    """One violin per Resource Type using reservoir samples for each Step."""
    viol_root = out_dir / "price_violins"
    viol_root.mkdir(parents=True, exist_ok=True)

    # Organize reservoirs -> {rtype: {step: np.array(samples)}}
    by_type: Dict[str, Dict[int, np.ndarray]] = {}
    for (rtype, step), res in violin_reservoirs.items():
        if len(res.samples)==0: 
            continue
        by_type.setdefault(rtype, {})[step] = np.asarray(res.samples, dtype=float)

    for rtype, step_map in by_type.items():
        steps = sorted(step_map.keys())
        if not steps:
            continue
        data_arrays = [step_map[s] for s in steps]
        # Skip if all empty
        if not any(len(a) for a in data_arrays):
            continue

        fig = plt.figure(figsize=(14, 6))
        parts = plt.violinplot(
            data_arrays,
            positions=np.arange(1, len(steps)+1),
            showmeans=False, showmedians=True, showextrema=False, widths=0.85,
        )
        for pc in parts['bodies']:
            pc.set_alpha(0.7)

        plt.xlabel("Step"); plt.ylabel("Bid Price ($/MWh)")
        plt.title(f"SCED Bid Price by Step — {rtype} (reservoir sample)")
        plt.xticks(np.arange(1, len(steps)+1), steps)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)

        safe = re.sub(r"[^A-Za-z0-9._-]+","_", rtype)
        out_path = viol_root / f"violin_price_by_step_{safe}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================ Main ============================ #
def main():
    print(f"Streaming from: {INPUT_DIR}")
    ingest_and_stream(INPUT_DIR, max_steps=MAX_STEPS)

    # Build aggregated DataFrames from accumulators
    bars_df     = df_bars_month_hour()       if MAKE_STEP_BARS_BY_MONTH_HOUR else None
    heatmap_df  = df_heatmap_price()         if MAKE_PRICE_HEATMAPS          else None
    avg_mw_df   = df_avg_mwfrac_lines()      if MAKE_AVG_MWFRAC_LINE         else None
    avg_price_df= df_avg_price_lines()       if MAKE_AVG_PRICE_LINE          else None

    # Save aggregates (optional, handy for reuse)
    if bars_df is not None:
        bars_df.to_csv(OUTPUT_DIR / "agg_mwfrac_month_hour_step_type.csv", index=False)
    if heatmap_df is not None:
        heatmap_df.to_csv(OUTPUT_DIR / "agg_price_type_step_month_hour.csv", index=False)
    if avg_mw_df is not None:
        avg_mw_df.to_csv(OUTPUT_DIR / "avg_mwfrac_by_type_step.csv", index=False)
    if avg_price_df is not None:
        avg_price_df.to_csv(OUTPUT_DIR / "avg_price_by_type_step.csv", index=False)

    # Plots
    if MAKE_STEP_BARS_BY_MONTH_HOUR and bars_df is not None:
        print("Plotting step bars by (month, hour)...")
        plot_step_bars_by_month_hour_from_agg(bars_df, OUTPUT_DIR)
    if MAKE_PRICE_HEATMAPS and heatmap_df is not None:
        print("Plotting 12×24 price heatmaps per (type, step)...")
        plot_price_heatmaps_from_agg(heatmap_df, OUTPUT_DIR)
    if MAKE_AVG_MWFRAC_LINE and avg_mw_df is not None:
        print("Plotting single figure: avg MW_frac vs step (lines per type)...")
        plot_avg_mwfrac_by_step_all_types_from_agg(avg_mw_df, OUTPUT_DIR)
    if MAKE_AVG_PRICE_LINE and avg_price_df is not None:
        print("Plotting single figure: avg Price vs step (lines per type)...")
        plot_avg_price_by_step_one_figure_all_types_from_agg(avg_price_df, OUTPUT_DIR)
    if MAKE_PRICE_VIOLINS:
        print("Plotting violin distributions of Price by step (per type, sampled)...")
        plot_price_violins_from_reservoirs(OUTPUT_DIR)

    print("✅ Done. Outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
