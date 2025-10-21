#!/usr/bin/env python3
"""
Incremental SCED bidding analysis (Polars) with live partial plots + ETA.

Input directory (wide CSVs):
  aggregation_HSL.csv
  aggregation_SCED1_Curve-MW1.csv
  aggregation_SCED1_Curve-Price1.csv
  ...
Columns:
  ["Resource Name", "Resource Type", <timestamp1>, <timestamp2>, ...] (5-min)

What this script does:
- Processes one SCED "Step" at a time using Polars (fast, multithreaded).
- Maintains running aggregates to write partial plots ASAP.
- Prints a live ETA after a quick warmup using observed throughput.

Requires: pip install polars pyarrow matplotlib
"""

from __future__ import annotations
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Iterable, List

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# ============================ CONFIG ============================ #
INPUT_DIR  = Path("/path/to/your/aggregated_timeseries_dir")       # <-- set this
OUTPUT_DIR = Path("/path/to/output/sced_bidding_incremental")      # <-- set this

MAX_STEPS  = 35
TIMESTAMP_TZ: Optional[str] = None          # e.g., "America/Chicago" or None

# Flush plots every N steps processed (1 = after each step)
INCREMENTAL_FLUSH_EVERY_STEPS = 1

# Violin sampling cap per (Resource Type × Step). None = keep all (not recommended)
MAX_VIOLIN_SAMPLES_PER_GROUP: Optional[int] = 8000

# Warmup for ETA (after this many steps we print initial ETA)
WARMUP_STEPS_FOR_ETA = 2
# ================================================================ #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Filename parsing ---------------- #
RE_HSL = re.compile(r"^aggregation[_-]HSL\.csv$", re.IGNORECASE)
RE_MW  = re.compile(r"^aggregation[_-]SCED1[_ ]?Curve[-_]?MW(\d+)\.csv$", re.IGNORECASE)
RE_PRC = re.compile(r"^aggregation[_-]SCED1[_ ]?Curve[-_]?Price(\d+)\.csv$", re.IGNORECASE)

def classify_file(fname: str):
    if RE_HSL.match(fname): return "HSL", None
    m = RE_MW.match(fname)
    if m: return "MW", int(m.group(1))
    p = RE_PRC.match(fname)
    if p: return "PRICE", int(p.group(1))
    return "OTHER", None

# ---------------- Polars helpers ---------------- #
def scan_csv_normalized(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(str(path))
    cols = lf.columns
    if "Resource Node" in cols and "Resource Name" not in cols:
        lf = lf.rename({"Resource Node": "Resource Name"})
    return lf

def melt_metric_csv(path: Path, value_name: str) -> pl.LazyFrame:
    lf = scan_csv_normalized(path)
    cols = lf.columns
    id_vars = [c for c in ["Resource Name", "Resource Type"] if c in cols]
    if len(id_vars) != 2:
        raise ValueError(f"{path.name}: missing id columns.")
    ts_cols = [c for c in cols if c not in id_vars]
    if not ts_cols:
        return pl.LazyFrame({"Resource Name": [], "Resource Type": [], "timestamp": [], value_name: []})
    lf = lf.melt(id_vars=id_vars, value_vars=ts_cols,
                 variable_name="timestamp", value_name=value_name)
    lf = lf.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))
    if TIMESTAMP_TZ:
        lf = lf.with_columns(pl.col("timestamp").dt.replace_time_zone(TIMESTAMP_TZ))
    return lf

def join_step_long(hsl_path: Path, mw_path: Path, pr_path: Path, step: int) -> pl.DataFrame:
    keys = ["Resource Name", "Resource Type", "timestamp"]
    hsl = melt_metric_csv(hsl_path, "HSL").with_columns(pl.col("HSL").cast(pl.Float64))
    mw  = melt_metric_csv(mw_path,  "MW").with_columns(pl.col("MW").cast(pl.Float64))
    pr  = melt_metric_csv(pr_path,  "Price").with_columns(pl.col("Price").cast(pl.Float64))
    joined = (
        hsl.join(mw, on=keys, how="inner")
           .join(pr, on=keys, how="inner")
           .with_columns(
               pl.lit(step).alias("Step"),
               (pl.when(pl.col("HSL") > 0)
                .then(pl.col("MW")/pl.col("HSL"))
                .otherwise(None)
               ).alias("MW_frac"),
               pl.col("timestamp").dt.hour().alias("hour"),
               pl.col("timestamp").dt.month().alias("month"),
           )
           .select(["Resource Name","Resource Type","timestamp","month","hour",
                    "Step","MW","Price","HSL","MW_frac"])
    )
    return joined.collect(streaming=True)

# ---------------- Running aggregates (SUM/COUNT) ---------------- #
# Bars: mean(MW_frac) by (month, hour, Step, Resource Type)
bars_sum: Dict[Tuple[int,int,int,str], float] = {}
bars_cnt: Dict[Tuple[int,int,int,str], int]   = {}

# Heatmaps: mean(Price) by (Resource Type, Step, month, hour)
heat_sum: Dict[Tuple[str,int,int,int], float] = {}
heat_cnt: Dict[Tuple[str,int,int,int], int]   = {}

# Avg lines: MW_frac / Price by (Resource Type, Step)
mw_sum: Dict[Tuple[str,int], float] = {}
mw_cnt: Dict[Tuple[str,int], int]   = {}
pr_sum: Dict[Tuple[str,int], float] = {}
pr_cnt: Dict[Tuple[str,int], int]   = {}

# Violins: sampled prices for (Resource Type, Step)
rng = np.random.default_rng(1234)
viol_samples: Dict[Tuple[str,int], np.ndarray] = {}
viol_seen:    Dict[Tuple[str,int], int] = {}

def _accumulate_sums_counts(sum_dict, cnt_dict, key, values: np.ndarray):
    if values.size == 0: return
    finite = values[np.isfinite(values)]
    if finite.size == 0: return
    sum_dict[key] = sum_dict.get(key, 0.0) + float(np.sum(finite))
    cnt_dict[key] = cnt_dict.get(key, 0)   + int(finite.size)

def _reservoir_add(key: Tuple[str,int], values: np.ndarray, cap: Optional[int]):
    if values.size == 0: return
    values = values[np.isfinite(values)]
    if values.size == 0: return
    if cap is None:
        arr = viol_samples.get(key)
        viol_samples[key] = values if arr is None else np.concatenate([arr, values])
        viol_seen[key] = viol_seen.get(key, 0) + len(values)
        return
    n_old = viol_seen.get(key, 0)
    pool  = viol_samples.get(key)
    if pool is None:
        take = values[:cap] if len(values) > cap else values
        viol_samples[key] = take.copy()
        viol_seen[key] = n_old + len(values)
        return
    if pool.shape[0] < cap:
        needed = cap - pool.shape[0]
        add = values[:needed] if len(values) >= needed else values
        viol_samples[key] = np.concatenate([pool, add])
        viol_seen[key] = n_old + len(values)
        return
    # Full reservoir: replace with probability cap/t
    for v in values:
        n_old += 1
        j = rng.integers(1, n_old + 1)
        if j <= cap:
            viol_samples[key][j-1] = v
    viol_seen[key] = n_old

# ---------------- Incremental update ---------------- #
def update_running_aggregates(step_df: pl.DataFrame):
    if step_df.is_empty():
        return
    pdf = step_df.select([
        "Resource Type","Step","month","hour","MW_frac","Price"
    ]).to_pandas()

    # Bars & Heatmaps
    for (rtype, s, mo, hr), grp in pdf.groupby(["Resource Type","Step","month","hour"]):
        key_b = (int(mo), int(hr), int(s), str(rtype))
        _accumulate_sums_counts(bars_sum, bars_cnt, key_b, grp["MW_frac"].to_numpy(dtype=float))
        key_h = (str(rtype), int(s), int(mo), int(hr))
        _accumulate_sums_counts(heat_sum, heat_cnt, key_h, grp["Price"].to_numpy(dtype=float))

    # Avg lines
    for (rtype, s), grp in pdf.groupby(["Resource Type","Step"]):
        key = (str(rtype), int(s))
        _accumulate_sums_counts(mw_sum, mw_cnt, key, grp["MW_frac"].to_numpy(dtype=float))
        _accumulate_sums_counts(pr_sum, pr_cnt, key, grp["Price"].to_numpy(dtype=float))

    # Violins
    if MAX_VIOLIN_SAMPLES_PER_GROUP is not None:
        for (rtype, s), grp in pdf.groupby(["Resource Type","Step"]):
            vals = grp["Price"].to_numpy(dtype=float)
            _reservoir_add((str(rtype), int(s)), vals, MAX_VIOLIN_SAMPLES_PER_GROUP)

# ---- Helpers to convert aggregates to small DataFrames for plotting ---- #
def df_bars_rows():
    rows = []
    for k, sm in bars_sum.items():
        mo, hr, step, rtype = k
        cnt = bars_cnt.get(k, 0)
        if cnt > 0:
            rows.append((mo, hr, step, rtype, sm/cnt))
    return rows

def df_heatmap_rows_for_step(step: int):
    rows = []
    for k, sm in heat_sum.items():
        rtype, s, mo, hr = k
        if s != step: continue
        cnt = heat_cnt.get(k, 0)
        if cnt > 0:
            rows.append((rtype, s, mo, hr, sm/cnt))
    return rows

def df_avg_rows(which: str):
    rows = []
    if which == "mw":
        for k, sm in mw_sum.items():
            rtype, step = k
            cnt = mw_cnt.get(k, 0)
            if cnt > 0:
                rows.append((rtype, step, sm/cnt))
    else:
        for k, sm in pr_sum.items():
            rtype, step = k
            cnt = pr_cnt.get(k, 0)
            if cnt > 0:
                rows.append((rtype, step, sm/cnt))
    return rows

# ---------------- Plotters (overwrite files each flush) ---------------- #
def plot_step_bars_by_month_hour_from_rows(rows, out_dir: Path):
    import pandas as pd
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["month","hour","Step","Resource Type","avg_mw_frac"])
    all_types = sorted(df["Resource Type"].unique().tolist())
    all_steps = sorted(df["Step"].unique().astype(int).tolist())
    months = range(1,13)
    hours = range(0,24)

    root = out_dir / "step_bars_by_month_hour"
    root.mkdir(parents=True, exist_ok=True)

    for mm in months:
        mdir = root / f"{mm:02d}"
        mdir.mkdir(parents=True, exist_ok=True)
        sub_m = df[df["month"] == mm]
        for hh in hours:
            sub = sub_m[sub_m["hour"] == hh]
            fig = plt.figure(figsize=(14,6))
            if sub.empty:
                plt.title(f"Normalized Bids by Step — Month {mm}, Hour {hh} (partial/no data)")
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
                    if np.allclose(heights, 0.0): continue
                    plt.bar(x, heights, bottom=bottom, label=rtype)
                    bottom += heights
                plt.xticks(x, all_steps)
                plt.xlabel("Step (1..N)"); plt.ylabel("Mean normalized MW (MW/HSL)")
                plt.title(f"How Resource Types Bid by Step — Month {mm}, Hour {hh} (partial)")
                plt.legend(ncol=3, fontsize=8)
            out_path = mdir / f"bars_month_{mm:02d}_hour_{hh:02d}.png"
            plt.tight_layout(); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)

def plot_price_heatmap_for_step_from_rows(rows_for_step, out_dir: Path, step: int):
    import pandas as pd
    hours = np.arange(0,24); months = np.arange(1,13)
    if not rows_for_step:
        return
    df = pd.DataFrame(rows_for_step, columns=["Resource Type","Step","month","hour","avg_price"])
    rtypes = sorted(df["Resource Type"].unique().tolist())

    heat_root = out_dir / "price_heatmaps"
    heat_root.mkdir(parents=True, exist_ok=True)

    for rtype in rtypes:
        sub = df[df["Resource Type"] == rtype]
        if sub.empty: continue
        pivot = sub.pivot(index="month", columns="hour", values="avg_price").reindex(index=months, columns=hours)
        mat = pivot.to_numpy()

        fig = plt.figure(figsize=(10,6))
        plt.imshow(mat, aspect="auto", origin="lower",
                   extent=[hours.min()-0.5, hours.max()+0.5, months.min()-0.5, months.max()+0.5])
        cbar = plt.colorbar(); cbar.set_label("Mean bid price ($/MWh)")
        plt.xlabel("Hour of day"); plt.ylabel("Month (1–12)")
        plt.title(f"Price Heatmap — {rtype} — Step {step} (partial)")
        plt.xticks(np.arange(0,24,1)); plt.yticks(np.arange(1,13,1))
        safe = re.sub(r"[^A-Za-z0-9._-]+","_", str(rtype))
        type_dir = heat_root / safe
        type_dir.mkdir(parents=True, exist_ok=True)
        out_path = type_dir / f"heatmap_price_{safe}_step{step:02d}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)

def plot_avg_lines_from_rows(rows, out_path: Path, ylabel: str, title: str):
    import pandas as pd
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["Resource Type","Step","avg"])
    steps = sorted(df["Step"].unique().astype(int).tolist())
    if not steps:
        return
    wide = df.pivot(index="Resource Type", columns="Step", values="avg").reindex(columns=steps)
    order = wide.mean(axis=1, skipna=True).sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=(12,6))
    x = np.arange(len(steps))
    for rtype in order:
        y = wide.loc[rtype, steps].to_numpy(dtype=float)
        if np.all(np.isnan(y)): continue
        plt.plot(x, y, marker="o", label=str(rtype))
    plt.xticks(x, steps); plt.xlabel("SCED Step")
    plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)

def plot_price_violins_incremental(out_dir: Path):
    """One violin per Resource Type using the (partial) reservoirs so far."""
    if not viol_samples:
        return
    viol_root = out_dir / "price_violins"
    viol_root.mkdir(parents=True, exist_ok=True)

    # Collect steps per type
    by_type: Dict[str, Dict[int, np.ndarray]] = {}
    for (rtype, step), arr in viol_samples.items():
        if arr.size == 0: continue
        by_type.setdefault(rtype, {})[step] = arr

    for rtype, step_map in by_type.items():
        steps = sorted(step_map.keys())
        arrays = [step_map[s] for s in steps]
        if not any(len(a) for a in arrays):
            continue
        fig = plt.figure(figsize=(14, 6))
        parts = plt.violinplot(
            arrays,
            positions=np.arange(1, len(steps)+1),
            showmeans=False, showmedians=True, showextrema=False, widths=0.85,
        )
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        plt.xlabel("Step"); plt.ylabel("Bid Price ($/MWh)")
        plt.title(f"SCED Bid Price by Step — {rtype} (partial)")
        plt.xticks(np.arange(1, len(steps)+1), steps)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        safe = re.sub(r"[^A-Za-z0-9._-]+","_", rtype)
        out_path = viol_root / f"violin_price_by_step_{safe}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------------- Main incremental loop w/ ETA ---------------- #
def main():
    # Map files
    hsl_path: Optional[Path] = None
    mw_paths: Dict[int, Path] = {}
    pr_paths: Dict[int, Path] = {}

    for f in INPUT_DIR.iterdir():
        if not f.is_file(): continue
        kind, step = classify_file(f.name)
        if kind == "HSL":
            hsl_path = f
        elif kind == "MW" and step and 1 <= step <= MAX_STEPS:
            mw_paths[step] = f
        elif kind == "PRICE" and step and 1 <= step <= MAX_STEPS:
            pr_paths[step] = f
    if hsl_path is None:
        raise ValueError("aggregation_HSL.csv not found.")
    steps = sorted(set(mw_paths.keys()).intersection(pr_paths.keys()))
    if not steps:
        raise ValueError("No overlapping steps between MW and Price files.")

    # --- ETA init ---
    total_bytes = sum((p.stat().st_size for p in INPUT_DIR.glob("*.csv")), start=0)
    processed_bytes = 0
    t0_all = time.time()
    print(f"Detected input size: {total_bytes/1e9:.2f} GB across {len(list(INPUT_DIR.glob('*.csv')))} files")

    processed_steps = 0
    for s in steps:
        print(f"• Processing Step {s} ...")
        t0_step = time.time()

        # Build this step
        step_df = join_step_long(hsl_path, mw_paths[s], pr_paths[s], s)
        print(f"  rows after join: {step_df.height:,}")
        update_running_aggregates(step_df)

        # --- ETA accounting for this step ---
        # Approx bytes attributed to this step (MW+Price for this step + share of HSL)
        step_size = (mw_paths[s].stat().st_size if s in mw_paths else 0) \
                  + (pr_paths[s].stat().st_size if s in pr_paths else 0)
        step_size += hsl_path.stat().st_size / len(steps)
        processed_bytes += step_size
        processed_steps += 1

        elapsed_step = time.time() - t0_step
        print(f"  step time: {elapsed_step:.2f}s")

        # After warmup, compute ETA based on observed throughput
        if processed_steps >= WARMUP_STEPS_FOR_ETA:
            elapsed_all = time.time() - t0_all
            bps = processed_bytes / elapsed_all if elapsed_all > 0 else 0.0
            if bps > 0:
                remaining_bytes = max(0.0, total_bytes - processed_bytes)
                eta_sec = remaining_bytes / bps
                print(f"  ~Throughput: {bps/1e6:.1f} MB/s; "
                      f"progress {processed_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB; "
                      f"ETA ~{eta_sec/60:.1f} min")

        # Partial flush?
        if INCREMENTAL_FLUSH_EVERY_STEPS and (processed_steps % INCREMENTAL_FLUSH_EVERY_STEPS == 0):
            print("  ↳ Writing partial plots...")
            plot_step_bars_by_month_hour_from_rows(df_bars_rows(), OUTPUT_DIR)
            plot_price_heatmap_for_step_from_rows(df_heatmap_rows_for_step(s), OUTPUT_DIR, s)
            plot_avg_lines_from_rows(
                df_avg_rows("mw"),
                OUTPUT_DIR / "avg_mwfrac_by_step_all_types.png",
                ylabel="Average normalized MW (MW/HSL)",
                title="Average MW_frac by SCED Step (partial)"
            )
            plot_avg_lines_from_rows(
                df_avg_rows("price"),
                OUTPUT_DIR / "avg_price_by_step_all_types.png",
                ylabel="Average bid Price ($/MWh)",
                title="Average Price by SCED Step (partial)"
            )
            plot_price_violins_incremental(OUTPUT_DIR)

    # Final flush (all steps)
    print("Finalizing plots with all steps...")
    plot_step_bars_by_month_hour_from_rows(df_bars_rows(), OUTPUT_DIR)
    for s in steps:
        plot_price_heatmap_for_step_from_rows(df_heatmap_rows_for_step(s), OUTPUT_DIR, s)
    plot_avg_lines_from_rows(
        df_avg_rows("mw"),
        OUTPUT_DIR / "avg_mwfrac_by_step_all_types.png",
        ylabel="Average normalized MW (MW/HSL)",
        title="Average MW_frac by SCED Step"
    )
    plot_avg_lines_from_rows(
        df_avg_rows("price"),
        OUTPUT_DIR / "avg_price_by_step_all_types.png",
        ylabel="Average bid Price ($/MWh)",
        title="Average Price by SCED Step"
    )
    plot_price_violins_incremental(OUTPUT_DIR)

    total_elapsed = time.time() - t0_all
    print(f"✅ Done in {total_elapsed/60:.1f} min. Outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
