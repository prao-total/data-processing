#!/usr/bin/env python3
"""
Incremental SCED bidding analysis (Polars) with:
- Column-BATCHED unpivot per step (stability + low RAM)
- All heavy work in Polars (no pandas groupby on big tables)
- Timestamp kept as string; month/hour via string slicing (fast)
- Live ETA + partial plots

Input folder (wide CSVs):
  aggregation_HSL.csv
  aggregation_SCED1_Curve-MW1.csv
  aggregation_SCED1_Curve-Price1.csv
  ...

Requires: pip install polars pyarrow matplotlib
"""

from __future__ import annotations
import re, time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Iterable

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# ============================ CONFIG ============================ #
INPUT_DIR  = Path("C:/Users/<you>/Documents/aggregated_timeseries")     # <-- set this
OUTPUT_DIR = Path("C:/Users/<you>/Documents/sced_bidding_incremental")  # <-- set this

MAX_STEPS  = 35
# How many timestamp columns to process at once (lower = safer)
TS_BATCH_SIZE = 100

# Write partial plots after every N steps (set 1 for after each step)
INCREMENTAL_FLUSH_EVERY_STEPS = 1

# Limit samples for violin plots per (Resource Type × Step). None = skip violins.
MAX_VIOLIN_SAMPLES_PER_GROUP: Optional[int] = 8000

# How many steps before we print an ETA
WARMUP_STEPS_FOR_ETA = 1
# ================================================================ #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- file name parsing ---------- #
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

# ---------- utility: list timestamp columns ---------- #
def list_ts_columns(path: Path) -> List[str]:
    # Cheap schema read
    names = pl.scan_csv(str(path), infer_schema_length=100).collect_schema().names()
    id_vars = {"Resource Name", "Resource Type", "Resource Node"}  # some files use Resource Node
    ts_cols = [c for c in names if c not in id_vars]
    return ts_cols

def batched(lst: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# ---------- Polars helpers (column-batched unpivot) ---------- #
def scan_csv_normalized(path: Path) -> pl.LazyFrame:
    # Shorter inference for speed; rename Resource Node -> Resource Name if needed
    lf = pl.scan_csv(str(path), infer_schema_length=100)
    names = lf.collect_schema().names()
    if "Resource Node" in names and "Resource Name" not in names:
        lf = lf.rename({"Resource Node": "Resource Name"})
    return lf

def unpivot_batch(path: Path, batch_cols: List[str], value_name: str) -> pl.LazyFrame:
    """
    Return LF with: Resource Name, Resource Type, timestamp (Utf8), value_name
    Only includes the given batch of timestamp columns to keep memory low.
    """
    lf = scan_csv_normalized(path)
    # Some files may be missing some batch columns; intersect safely
    names = lf.collect_schema().names()
    id_vars = [c for c in ("Resource Name", "Resource Type") if c in names]
    if len(id_vars) != 2:
        raise ValueError(f"{path.name}: missing id columns.")
    present = [c for c in batch_cols if c in names]
    if not present:
        # return empty LF with expected columns
        return pl.LazyFrame({"Resource Name": [], "Resource Type": [], "timestamp": [], value_name: []})
    # Select only id_vars + present, then unpivot (keep timestamp as string)
    lf = (
        lf.select(id_vars + present)
          .unpivot(index=id_vars, on=present, variable_name="timestamp", value_name=value_name)
          .with_columns(pl.col(value_name).cast(pl.Float64))
    )
    return lf

def step_lazy_batched(hsl_path: Path, mw_path: Path, pr_path: Path, step: int, ts_cols_all: List[str]) -> Iterable[pl.LazyFrame]:
    """
    Yield a sequence of LazyFrames, each covering a small batch of timestamp columns
    for the given step. Each LF already has month/hour (from string slices) and MW_frac.
    """
    for batch in batched(ts_cols_all, TS_BATCH_SIZE):
        hsl = unpivot_batch(hsl_path, batch, "HSL")
        mw  = unpivot_batch(mw_path,  batch, "MW")
        pr  = unpivot_batch(pr_path,  batch, "Price")

        # Inner join on keys; compute month/hour from the timestamp string directly:
        # 'YYYY-MM-DD HH:MM:SS' -> month at [5:7], hour at [11:13]
        keys = ["Resource Name", "Resource Type", "timestamp"]
        lf = (
            hsl.join(mw, on=keys, how="inner")
               .join(pr, on=keys, how="inner")
               .with_columns(
                   pl.lit(step).alias("Step"),
                   pl.col("timestamp").str.slice(5, 2).cast(pl.Int8).alias("month"),
                   pl.col("timestamp").str.slice(11, 2).cast(pl.Int8).alias("hour"),
                   pl.when(pl.col("HSL") > 0).then(pl.col("MW")/pl.col("HSL")).otherwise(None).alias("MW_frac")
               )
               .select(["Resource Name","Resource Type","timestamp","month","hour","Step","MW","Price","HSL","MW_frac"])
        )
        yield lf

# ---------- running accumulators (tiny dicts) ---------- #
bars_sum: Dict[Tuple[int,int,int,str], float] = {}
bars_cnt: Dict[Tuple[int,int,int,str], int]   = {}

heat_sum: Dict[Tuple[str,int,int,int], float] = {}
heat_cnt: Dict[Tuple[str,int,int,int], int]   = {}

mw_sum: Dict[Tuple[str,int], float] = {}
mw_cnt: Dict[Tuple[str,int], int]   = {}
pr_sum: Dict[Tuple[str,int], float] = {}
pr_cnt: Dict[Tuple[str,int], int]   = {}

rng = np.random.default_rng(1234)
viol_samples: Dict[Tuple[str,int], np.ndarray] = {}
viol_seen:    Dict[Tuple[str,int], int] = {}

def _acc(sumd, cntd, key, s, c):
    if c and c > 0 and s is not None:
        sumd[key] = sumd.get(key, 0.0) + float(s)
        cntd[key] = cntd.get(key, 0)   + int(c)

def _reservoir_add(key: Tuple[str,int], values: np.ndarray, cap: Optional[int]):
    values = values[np.isfinite(values)]
    if values.size == 0: return
    if cap is None:
        arr = viol_samples.get(key)
        viol_samples[key] = values if arr is None else np.concatenate([arr, values])
        viol_seen[key] = viol_seen.get(key, 0) + values.size
        return
    n_old = viol_seen.get(key, 0)
    pool  = viol_samples.get(key)
    if pool is None:
        take = values[:cap] if values.size > cap else values
        viol_samples[key] = take.copy()
        viol_seen[key] = n_old + values.size
        return
    if pool.shape[0] < cap:
        need = cap - pool.shape[0]
        add = values[:need] if values.size >= need else values
        viol_samples[key] = np.concatenate([pool, add])
        viol_seen[key] = n_old + values.size
        return
    for v in values:
        n_old += 1
        j = rng.integers(1, n_old+1)
        if j <= cap:
            pool[j-1] = v
    viol_seen[key] = n_old

# ---------- update from ONE small batch LF (Polars-only) ---------- #
def update_from_batch_lazy(batch_lf: pl.LazyFrame):
    # Bars + heatmaps by (rtype, step, month, hour)
    agg_mh = (
        batch_lf
        .group_by(["Resource Type","Step","month","hour"])
        .agg(
            mwfrac_sum = pl.col("MW_frac").sum(),
            mwfrac_cnt = pl.col("MW_frac").is_finite().sum(),
            price_sum  = pl.col("Price").sum(),
            price_cnt  = pl.col("Price").is_finite().sum(),
        )
        .collect()
    )
    for r in agg_mh.iter_rows(named=True):
        rtype = str(r["Resource Type"]); s = int(r["Step"]); mo = int(r["month"]); hr = int(r["hour"])
        _acc(bars_sum, bars_cnt, (mo, hr, s, rtype), r["mwfrac_sum"], r["mwfrac_cnt"])
        _acc(heat_sum, heat_cnt, (rtype, s, mo, hr), r["price_sum"], r["price_cnt"])

    # Lines by (rtype, step)
    agg_lines = (
        batch_lf
        .group_by(["Resource Type","Step"])
        .agg(
            mwfrac_sum = pl.col("MW_frac").sum(),
            mwfrac_cnt = pl.col("MW_frac").is_finite().sum(),
            price_sum  = pl.col("Price").sum(),
            price_cnt  = pl.col("Price").is_finite().sum(),
        )
        .collect()
    )
    for r in agg_lines.iter_rows(named=True):
        rtype = str(r["Resource Type"]); s = int(r["Step"])
        _acc(mw_sum, mw_cnt, (rtype, s), r["mwfrac_sum"], r["mwfrac_cnt"])
        _acc(pr_sum, pr_cnt, (rtype, s), r["price_sum"],  r["price_cnt"])

    # Violins: sample a few per batch (keeps memory tiny)
    if MAX_VIOLIN_SAMPLES_PER_GROUP is not None:
        viol = (
            batch_lf
            .group_by(["Resource Type","Step"])
            .agg(pl.col("Price").drop_nulls().sample(n=MAX_VIOLIN_SAMPLES_PER_GROUP, with_replacement=False).alias("samples"))
            .collect()
        )
        for r in viol.iter_rows(named=True):
            rtype = str(r["Resource Type"]); s = int(r["Step"])
            arr = np.array(r["samples"], dtype=float) if r["samples"] is not None else np.array([], dtype=float)
            if arr.size:
                _reservoir_add((rtype, s), arr, MAX_VIOLIN_SAMPLES_PER_GROUP)

# ---------- convert aggregates to tiny row-iterators for plotting ---------- #
def rows_bars():
    for (mo,hr,s,rtype), ssum in bars_sum.items():
        cnt = bars_cnt.get((mo,hr,s,rtype), 0)
        if cnt > 0:
            yield (mo, hr, s, rtype, ssum/cnt)

def rows_heat_step(step: int):
    for (rtype,s,mo,hr), ssum in heat_sum.items():
        if s != step: continue
        cnt = heat_cnt.get((rtype,s,mo,hr), 0)
        if cnt > 0:
            yield (rtype, s, mo, hr, ssum/cnt)

def rows_avg(which: str):
    if which == "mw":
        for (rtype,s), ssum in mw_sum.items():
            cnt = mw_cnt.get((rtype,s), 0)
            if cnt > 0:
                yield (rtype, s, ssum/cnt)
    else:
        for (rtype,s), ssum in pr_sum.items():
            cnt = pr_cnt.get((rtype,s), 0)
            if cnt > 0:
                yield (rtype, s, ssum/cnt)

# ---------- plotting (small aggregated slices → pandas) ---------- #
def plot_step_bars_by_month_hour_from_rows(rows_iter, out_dir: Path):
    import pandas as pd
    rows = list(rows_iter)
    if not rows: return
    df = pd.DataFrame(rows, columns=["month","hour","Step","Resource Type","avg_mw_frac"])
    all_types = sorted(df["Resource Type"].unique().tolist())
    all_steps = sorted(df["Step"].unique().astype(int).tolist())
    months = range(1,13); hours = range(0,24)
    root = out_dir / "step_bars_by_month_hour"; root.mkdir(parents=True, exist_ok=True)
    for mm in months:
        mdir = root / f"{mm:02d}"; mdir.mkdir(parents=True, exist_ok=True)
        sub_m = df[df["month"]==mm]
        for hh in hours:
            sub = sub_m[sub_m["hour"]==hh]
            fig = plt.figure(figsize=(14,6))
            if sub.empty:
                plt.title(f"Normalized Bids by Step — Month {mm}, Hour {hh} (partial)")
                plt.xlabel("Step (1..N)"); plt.ylabel("Mean normalized MW (MW/HSL)")
            else:
                mat = np.zeros((len(all_steps), len(all_types)))
                for i, s in enumerate(all_steps):
                    row = sub[sub["Step"]==s].set_index("Resource Type")["avg_mw_frac"].reindex(all_types).fillna(0)
                    mat[i,:] = row.to_numpy()
                x = np.arange(len(all_steps)); bottom = np.zeros(len(all_steps))
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

def plot_price_heatmap_for_step_from_rows(rows_iter, out_dir: Path, step: int):
    import pandas as pd
    rows = list(rows_iter)
    if not rows: return
    df = pd.DataFrame(rows, columns=["Resource Type","Step","month","hour","avg_price"])
    rtypes = sorted(df["Resource Type"].unique().tolist())
    hours = np.arange(0,24); months = np.arange(1,13)
    heat_root = out_dir / "price_heatmaps"; heat_root.mkdir(parents=True, exist_ok=True)
    for rtype in rtypes:
        sub = df[df["Resource Type"]==rtype]
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
        out_path = (heat_root / safe); out_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(out_path / f"heatmap_price_{safe}_step{step:02d}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

def plot_avg_lines_from_rows(rows_iter, out_path: Path, ylabel: str, title: str):
    import pandas as pd
    rows = list(rows_iter)
    if not rows: return
    df = pd.DataFrame(rows, columns=["Resource Type","Step","avg"])
    steps = sorted(df["Step"].unique().astype(int).tolist())
    if not steps: return
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
    if not viol_samples: return
    viol_root = out_dir / "price_violins"; viol_root.mkdir(parents=True, exist_ok=True)
    by_type: Dict[str, Dict[int, np.ndarray]] = {}
    for (rtype, step), arr in viol_samples.items():
        if arr.size == 0: continue
        by_type.setdefault(rtype, {})[step] = arr
    for rtype, s_map in by_type.items():
        steps = sorted(s_map.keys()); arrays = [s_map[s] for s in steps]
        if not any(len(a) for a in arrays): continue
        fig = plt.figure(figsize=(14,6))
        parts = plt.violinplot(arrays, positions=np.arange(1, len(steps)+1),
                               showmeans=False, showmedians=True, showextrema=False, widths=0.85)
        for pc in parts['bodies']: pc.set_alpha(0.7)
        plt.xlabel("Step"); plt.ylabel("Bid Price ($/MWh)")
        plt.title(f"SCED Bid Price by Step — {rtype} (partial)")
        plt.xticks(np.arange(1, len(steps)+1), steps)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        safe = re.sub(r"[^A-Za-z0-9._-]+","_", rtype)
        out_path = viol_root / f"violin_price_by_step_{safe}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------- main loop with ETA & batches ---------- #
def main():
    # Map files
    hsl_path: Optional[Path] = None
    mw_paths: Dict[int, Path] = {}
    pr_paths: Dict[int, Path] = {}
    for f in INPUT_DIR.iterdir():
        if not f.is_file(): continue
        kind, step = classify_file(f.name)
        if kind == "HSL": hsl_path = f
        elif kind == "MW" and step and 1 <= step <= MAX_STEPS: mw_paths[step] = f
        elif kind == "PRICE" and step and 1 <= step <= MAX_STEPS: pr_paths[step] = f
    if hsl_path is None: raise ValueError("aggregation_HSL.csv not found.")
    steps = sorted(set(mw_paths.keys()).intersection(pr_paths.keys()))
    if not steps: raise ValueError("No overlapping MW/Price steps.")

    # Timestamp columns (from HSL schema)
    ts_cols_all = list_ts_columns(hsl_path)
    if not ts_cols_all:
        raise ValueError("No timestamp columns found in HSL.")

    total_bytes = sum((p.stat().st_size for p in INPUT_DIR.glob("*.csv")), start=0)
    processed_bytes = 0.0
    t0_all = time.time()
    print(f"Detected input size: {total_bytes/1e9:.2f} GB across {len(list(INPUT_DIR.glob('*.csv')))} files")
    print(f"~{len(ts_cols_all)} timestamp columns; batching {TS_BATCH_SIZE} at a time")

    processed_steps = 0
    for s in steps:
        print(f"• Processing Step {s} ...")
        t0_step = time.time()
        # For ETA, approximate per-batch share of these three files
        per_step_bytes = (mw_paths[s].stat().st_size + pr_paths[s].stat().st_size + hsl_path.stat().st_size)
        batch_share = TS_BATCH_SIZE / max(1, len(ts_cols_all))

        batch_idx = 0
        for batch_lf in step_lazy_batched(hsl_path, mw_paths[s], pr_paths[s], s, ts_cols_all):
            batch_idx += 1
            t0_batch = time.time()
            update_from_batch_lazy(batch_lf)
            processed_bytes += per_step_bytes * batch_share
            elapsed_batch = time.time() - t0_batch
            if batch_idx % 5 == 0:
                print(f"  - batch {batch_idx} done in {elapsed_batch:.2f}s")

        processed_steps += 1
        elapsed_step = time.time() - t0_step
        print(f"  step time: {elapsed_step:.1f}s")

        # ETA after warmup
        if processed_steps >= WARMUP_STEPS_FOR_ETA:
            elapsed_all = time.time() - t0_all
            bps = processed_bytes / max(elapsed_all, 1e-9)
            remaining = max(0.0, total_bytes - processed_bytes)
            eta_sec = remaining / max(bps, 1e-9)
            print(f"  ~Throughput: {bps/1e6:.1f} MB/s; "
                  f"progress {processed_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB; "
                  f"ETA ~{eta_sec/60:.1f} min")

        # Partial plots
        if INCREMENTAL_FLUSH_EVERY_STEPS and (processed_steps % INCREMENTAL_FLUSH_EVERY_STEPS == 0):
            print("  ↳ Writing partial plots...")
            plot_step_bars_by_month_hour_from_rows(rows_bars(), OUTPUT_DIR)
            plot_price_heatmap_for_step_from_rows(rows_heat_step(s), OUTPUT_DIR, s)
            plot_avg_lines_from_rows(
                rows_avg("mw"),
                OUTPUT_DIR / "avg_mwfrac_by_step_all_types.png",
                ylabel="Average normalized MW (MW/HSL)",
                title="Average MW_frac by SCED Step (partial)"
            )
            plot_avg_lines_from_rows(
                rows_avg("price"),
                OUTPUT_DIR / "avg_price_by_step_all_types.png",
                ylabel="Average bid Price ($/MWh)",
                title="Average Price by SCED Step (partial)"
            )
            if MAX_VIOLIN_SAMPLES_PER_GROUP is not None:
                plot_price_violins_incremental(OUTPUT_DIR)

    # Final plots
    print("Finalizing plots with all steps...")
    plot_step_bars_by_month_hour_from_rows(rows_bars(), OUTPUT_DIR)
    for s in steps:
        plot_price_heatmap_for_step_from_rows(rows_heat_step(s), OUTPUT_DIR, s)
    plot_avg_lines_from_rows(
        rows_avg("mw"),
        OUTPUT_DIR / "avg_mwfrac_by_step_all_types.png",
        ylabel="Average normalized MW (MW/HSL)",
        title="Average MW_frac by SCED Step"
    )
    plot_avg_lines_from_rows(
        rows_avg("price"),
        OUTPUT_DIR / "avg_price_by_step_all_types.png",
        ylabel="Average bid Price ($/MWh)",
        title="Average Price by SCED Step"
    )
    if MAX_VIOLIN_SAMPLES_PER_GROUP is not None:
        plot_price_violins_incremental(OUTPUT_DIR)

    total_elapsed = time.time() - t0_all
    print(f"✅ Done in {total_elapsed/60:.1f} min. Outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
