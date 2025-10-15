"""
Analyze SCED bidding behavior from ERCOT-style CSVs.

What it does
------------
1) Loads the CSV
   - Expects columns akin to:
     "Resource Name" (or "Resource Node"),
     "Resource Type",
     "SCED Time Stamp" (or "SCED Time Step"),
     "HSL" (available capacity)
     and stepwise bid columns like
     "SCED1 Curve-MW1", "SCED1 Curve-Price1", "SCED1 Curve-MW2", "SCED1 Curve-Price2", ...

2) Detects neighboring (MW, Price) column pairs for the SCED curve and melts them long.

3) Normalizes each step MW by HSL: MW_frac = MW / HSL.

4) Aggregates mean(MW_frac) by (Resource Type, hour of day, price bin).

5) Plots:
   - Heatmap: hour (x) × price bins (y) of mean normalized MW for each Resource Type.
   - Average stepwise bid curve per Resource Type (cumulative MW_frac vs. average step price).

Outputs
-------
- bids_long.csv
- bidding_behavior_summary.csv
- average_bid_curves.csv
- heatmap_<ResourceType>.png
- avg_curve_<ResourceType>.png

Usage
-----
python analyze_sced_bidding.py \
  --csv 60d_SCED_Gen_Resource_Data-17-APR-25.csv \
  --out sced_bidding_outputs \
  --price-bin 10 \
  --top-types 6
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Column detection helpers ------------------------- #
def is_mw_col(name: str) -> bool:
    """Heuristically detect MW columns for SCED curve steps."""
    n = str(name).lower()
    # Prefer curve MW columns; fall back to generic MW-not-Price.
    return ("curve-mw" in n) or (("mw" in n) and ("price" not in n))


def is_price_col(name: str) -> bool:
    """Heuristically detect Price columns for SCED curve steps."""
    n = str(name).lower()
    # Prefer curve Price columns; fall back to generic price.
    return ("curve-price" in n) or ("price" in n)


def detect_bid_pairs(columns: list[str]) -> list[tuple[str, str]]:
    """
    Scan neighboring columns to find (MW, Price) pairs in order.
    Returns a list of (mw_col, price_col).
    """
    pairs: list[tuple[str, str]] = []
    cols = list(columns)
    for i in range(len(cols) - 1):
        c0, c1 = cols[i], cols[i + 1]
        if is_mw_col(c0) and is_price_col(c1):
            pairs.append((c0, c1))

    # If available, keep only pairs that clearly belong to the SCED "Curve"
    curve_pairs = [(m, p) for (m, p) in pairs
                   if ("curve" in str(m).lower()) and ("curve" in str(p).lower())]
    return curve_pairs if curve_pairs else pairs


# ---------------------------- Loading & reshaping --------------------------- #
def load_and_prepare(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Map alternative names to canonical
    rename_map = {}
    if "Resource Node" in df.columns:
        rename_map["Resource Node"] = "Resource Name"
    if "SCED Time Step" in df.columns:
        rename_map["SCED Time Step"] = "SCED Time Stamp"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["Resource Name", "Resource Type", "SCED Time Stamp", "HSL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Types
    df["SCED Time Stamp"] = pd.to_datetime(df["SCED Time Stamp"], errors="coerce")
    df["hour"] = df["SCED Time Stamp"].dt.hour
    df["HSL"] = pd.to_numeric(df["HSL"], errors="coerce").fillna(0.0)

    return df


def melt_bids(df: pd.DataFrame, bid_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Convert wide step pairs into long rows:
    columns -> Resource Name, Resource Type, hour, HSL, Step, Price, MW_frac
    """
    melted = []
    for step_idx, (mw_col, price_col) in enumerate(bid_pairs, start=1):
        mw = pd.to_numeric(df[mw_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")

        with np.errstate(divide="ignore", invalid="ignore"):
            mw_frac = np.where(df["HSL"].to_numpy() > 0,
                               mw.to_numpy() / df["HSL"].to_numpy(),
                               np.nan)

        temp = pd.DataFrame({
            "Resource Name": df["Resource Name"],
            "Resource Type": df["Resource Type"],
            "hour": df["hour"],
            "HSL": df["HSL"],
            "Step": step_idx,
            "Price": price,
            "MW_frac": mw_frac
        })
        melted.append(temp)

    bids_long = pd.concat(melted, ignore_index=True)
    bids_long = bids_long.dropna(subset=["Price", "MW_frac"])
    bids_long = bids_long[(bids_long["MW_frac"] >= 0) & np.isfinite(bids_long["MW_frac"])]
    return bids_long


# ------------------------------ Aggregations -------------------------------- #
def aggregate_price_bins(bids_long: pd.DataFrame,
                         bin_width: float = 10.0) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Bin by price, then aggregate mean normalized MW by Resource Type × hour × Price_bin.
    Returns (agg_df, bins).
    """
    if bids_long.empty:
        raise ValueError("No bid rows available after cleaning. Check input columns and detection.")

    pmin = max(0.0, float(np.nanmin(bids_long["Price"].to_numpy())))
    pmax = float(np.nanmax(bids_long["Price"].to_numpy()))
    if not np.isfinite(pmax) or pmax <= pmin:
        pmax = pmin + bin_width

    # Build bins, make sure max fits
    bins = np.arange(np.floor(pmin/bin_width)*bin_width,
                     np.ceil((pmax + bin_width)/bin_width)*bin_width,
                     bin_width)
    bins = np.unique(bins)
    if len(bins) < 3:
        bins = np.array([pmin, pmin + bin_width, pmin + 2*bin_width], dtype=float)

    price_bin = pd.cut(bids_long["Price"], bins=bins, right=False, include_lowest=True)
    bids_long = bids_long.assign(Price_bin=price_bin)

    agg = (bids_long
           .groupby(["Resource Type", "hour", "Price_bin"], observed=True)["MW_frac"]
           .mean()
           .reset_index())

    # Helper for plotting on a numeric Y axis
    bin_mid = agg["Price_bin"].apply(lambda x: (x.left + x.right)/2 if pd.notna(x) else np.nan)
    agg = agg.assign(Price_bin_mid=bin_mid)
    return agg, bins


def compute_average_bid_curves(bids_long: pd.DataFrame) -> pd.DataFrame:
    """
    Average price and MW_frac per step by Resource Type, then cumulative MW_frac to form
    an average stepwise curve per Resource Type.
    """
    step_avg = (bids_long
                .groupby(["Resource Type", "Step"], observed=True)
                .agg(avg_price=("Price", "mean"),
                     avg_mw_frac=("MW_frac", "mean"))
                .reset_index())

    step_avg = step_avg.sort_values(["Resource Type", "Step"])
    step_avg["cum_mw_frac"] = step_avg.groupby("Resource Type")["avg_mw_frac"].cumsum()
    return step_avg


# --------------------------------- Plots ------------------------------------ #
def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def plot_heatmap_for_type(agg_df: pd.DataFrame,
                          rtype: str,
                          bins: np.ndarray,
                          out_dir: Path) -> Path | None:
    """
    Hour × Price heatmap for one resource type.
    """
    sub = agg_df[agg_df["Resource Type"] == rtype]
    if sub.empty:
        return None

    hours = np.arange(0, 24, 1)
    y_edges = bins.astype(float)
    mat = np.full((len(y_edges) - 1, len(hours)), np.nan)

    # Map intervals to indices
    interval_to_idx = {
        pd.Interval(y_edges[i], y_edges[i+1], closed='left'): i
        for i in range(len(y_edges) - 1)
    }

    # Fill matrix
    for _, row in sub.iterrows():
        h = int(row["hour"])
        if 0 <= h < 24 and isinstance(row["Price_bin"], pd.Interval):
            r = interval_to_idx.get(row["Price_bin"])
            if r is not None:
                mat[r, h] = row["MW_frac"] if np.isfinite(row["MW_frac"]) else np.nan

    # Plot
    fig = plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(np.arange(0, 25, 1), y_edges)
    plt.pcolormesh(X, Y, mat, shading="auto")
    plt.colorbar(label="Mean normalized MW (MW/HSL)")
    plt.xlabel("Hour of day")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"SCED Bidding Heatmap — {rtype}")
    out_path = out_dir / f"heatmap_{sanitize_filename(rtype)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_avg_curve_for_type(curves_df: pd.DataFrame,
                            rtype: str,
                            out_dir: Path) -> Path | None:
    """
    Average stepwise bid curve for one resource type:
      x = cumulative MW fraction
      y = average step price
    """
    sub = curves_df[curves_df["Resource Type"] == rtype].sort_values("Step")
    if sub.empty:
        return None

    x = sub["cum_mw_frac"].to_numpy()
    y = sub["avg_price"].to_numpy()

    fig = plt.figure(figsize=(7, 5))
    plt.step(x, y, where="post")
    plt.scatter(x, y, s=12)
    plt.xlabel("Cumulative MW / HSL")
    plt.ylabel("Average step price ($/MWh)")
    plt.title(f"Average Stepwise Bid Curve — {rtype}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    out_path = out_dir / f"avg_curve_{sanitize_filename(rtype)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------- Runner ----------------------------------- #
def run(csv_path: str,
        out_dir: str,
        price_bin_width: float,
        top_types: int) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load & detect
    df = load_and_prepare(csv_path)
    bid_pairs = detect_bid_pairs(df.columns)
    if not bid_pairs:
        raise ValueError(
            "No (MW, Price) step pairs detected. "
            "Ensure SCED curve columns are neighboring '...MWk' and '...Pricek' fields."
        )

    # Melt & normalize
    bids_long = melt_bids(df, bid_pairs)

    # Aggregations
    agg_df, bins = aggregate_price_bins(bids_long, bin_width=price_bin_width)
    curves_df = compute_average_bid_curves(bids_long)

    # Save tables
    bids_long_out = out / "bids_long.csv"
    agg_out = out / "bidding_behavior_summary.csv"
    curves_out = out / "average_bid_curves.csv"

    bids_long.to_csv(bids_long_out, index=False)
    agg_df.to_csv(agg_out, index=False)
    curves_df.to_csv(curves_out, index=False)

    # Choose which resource types to plot
    type_counts = bids_long["Resource Type"].value_counts().sort_values(ascending=False)
    selected_types = type_counts.head(max(1, top_types)).index.tolist()

    heatmap_paths, curve_paths = [], []
    for rtype in selected_types:
        hp = plot_heatmap_for_type(agg_df, rtype, bins, out)
        cp = plot_avg_curve_for_type(curves_df, rtype, out)
        if hp: heatmap_paths.append(hp)
        if cp: curve_paths.append(cp)

    # Console summary
    print("Saved tables:")
    print(f" - Long bids: {bids_long_out}")
    print(f" - Aggregated summary: {agg_out}")
    print(f" - Average bid curves: {curves_out}")
    print("Saved plots:")
    for p in heatmap_paths:
        print(f" - Heatmap: {p}")
    for p in curve_paths:
        print(f" - Avg curve: {p}")
    print("Done.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SCED bidding behavior analysis")
    ap.add_argument("--csv", required=True, help="Path to SCED CSV file")
    ap.add_argument("--out", default="sced_bidding_outputs", help="Output directory")
    ap.add_argument("--price-bin", type=float, default=10.0, help="Price bin width ($/MWh)")
    ap.add_argument("--top-types", type=int, default=6, help="How many resource types to plot")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(csv_path=args.csv,
        out_dir=args.out,
        price_bin_width=args.price_bin,
        top_types=args.top_types)
#!/usr/bin/env python3
"""
Simplified SCED bidding analysis (step-based version)

✅ What’s included:
- Uses direct file paths (no argparse)
- Normalizes each MW/Price pair before any aggregation
- Uses 35 step pairs (columns G–BX)
- Produces:
  • bids_long_steps.csv  (normalized bids per step)
  • noncumulative_step_avgs.csv  (averages per step)
  • 24 hourly bar charts (hour_00 to hour_23)
  • Non-cumulative step curves per resource type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# ---------------- CONFIG ----------------
FILE_PATH = "/path/to/your/60d_SCED_Gen_Resource_Data-17-APR-25.csv"
OUTPUT_DIR = Path("/path/to/output/sced_bidding_outputs_steps")
MAX_STEPS = 35
TOP_N_TYPES_FOR_INLINE = 6  # top resource types for line plots
# -----------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helper functions ----------
def is_mw_col(name: str) -> bool:
    n = str(name).lower()
    return ("curve-mw" in n) or (("mw" in n) and ("price" not in n))

def is_price_col(name: str) -> bool:
    n = str(name).lower()
    return ("curve-price" in n) or ("price" in n)

def detect_bid_pairs(columns):
    pairs = []
    cols = list(columns)
    for i in range(len(cols) - 1):
        c0, c1 = cols[i], cols[i + 1]
        if is_mw_col(c0) and is_price_col(c1):
            pairs.append((c0, c1))
    curve_pairs = [(m,p) for (m,p) in pairs if "curve" in str(m).lower() and "curve" in str(p).lower()]
    if curve_pairs:
        pairs = curve_pairs
    return pairs[:MAX_STEPS]

def load_and_prepare(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    rename_map = {}
    if "Resource Node" in df.columns:
        rename_map["Resource Node"] = "Resource Name"
    if "SCED Time Step" in df.columns:
        rename_map["SCED Time Step"] = "SCED Time Stamp"
    if rename_map:
        df = df.rename(columns=rename_map)
    required = ["Resource Name", "Resource Type", "SCED Time Stamp", "HSL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    df["SCED Time Stamp"] = pd.to_datetime(df["SCED Time Stamp"], errors="coerce")
    df["hour"] = df["SCED Time Stamp"].dt.hour
    df["HSL"] = pd.to_numeric(df["HSL"], errors="coerce").fillna(0.0)
    return df

def melt_bids(df: pd.DataFrame, bid_pairs):
    melted = []
    for step_idx, (mw_col, price_col) in enumerate(bid_pairs, start=1):
        mw = pd.to_numeric(df[mw_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            mw_frac = np.where(df["HSL"].to_numpy() > 0, mw.to_numpy() / df["HSL"].to_numpy(), np.nan)
        temp = pd.DataFrame({
            "Resource Name": df["Resource Name"],
            "Resource Type": df["Resource Type"],
            "hour": df["hour"],
            "HSL": df["HSL"],
            "Step": step_idx,
            "Price": price,
            "MW_frac": mw_frac
        })
        melted.append(temp)
    bids_long = pd.concat(melted, ignore_index=True)
    bids_long = bids_long.dropna(subset=["Price", "MW_frac"])
    bids_long = bids_long[(bids_long["MW_frac"] >= 0) & np.isfinite(bids_long["MW_frac"])]
    return bids_long

def plot_hourly_step_bars(bids_long: pd.DataFrame, out_dir: Path):
    piv = (bids_long.groupby(["hour","Step","Resource Type"], observed=True)["MW_frac"]
           .mean().reset_index())
    all_types = sorted(piv["Resource Type"].dropna().unique().tolist())
    steps = sorted(piv["Step"].dropna().unique().astype(int).tolist())

    for hr in range(24):
        sub = piv[piv["hour"] == hr]
        fig = plt.figure(figsize=(14, 6))
        if sub.empty:
            plt.title(f"Normalized Bids by Step — Hour {hr} (no data)")
            plt.xlabel("Step"); plt.ylabel("Mean MW/HSL")
        else:
            mat = np.zeros((len(steps), len(all_types)))
            for i, s in enumerate(steps):
                row = sub[sub["Step"] == s]
                vals = row.set_index("Resource Type")["MW_frac"].reindex(all_types).fillna(0).to_numpy()
                mat[i,:] = vals
            x = np.arange(len(steps))
            bottom = np.zeros(len(steps))
            for j, rtype in enumerate(all_types):
                heights = mat[:, j]
                plt.bar(x, heights, bottom=bottom, label=rtype)
                bottom += heights
            plt.xticks(x, steps)
            plt.xlabel("Step (1..N)")
            plt.ylabel("Mean normalized MW (MW/HSL)")
            plt.title(f"How Resource Types Bid by Step — Hour {hr}")
            plt.legend(ncol=3, fontsize=8)
        out_path = out_dir / f"hour_{hr:02d}_step_bars.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def compute_non_cumulative_step_avgs(bids_long: pd.DataFrame) -> pd.DataFrame:
    step_avg = (bids_long.groupby(["Resource Type","Step"], observed=True)
                .agg(avg_price=("Price","mean"),
                     avg_mw_frac=("MW_frac","mean"))
                .reset_index().sort_values(["Resource Type","Step"]))
    return step_avg

def plot_non_cumulative_curves(step_avg: pd.DataFrame, out_dir: Path, top_n_types: int = 6):
    counts = step_avg["Resource Type"].value_counts().sort_values(ascending=False)
    selected = counts.head(top_n_types).index.tolist()
    for rtype in selected:
        sub = step_avg[step_avg["Resource Type"] == rtype].sort_values("Step")
        if sub.empty: continue
        x = sub["avg_mw_frac"].to_numpy()
        y = sub["avg_price"].to_numpy()
        fig = plt.figure(figsize=(7.5, 5.5))
        plt.plot(x, y, marker="o")
        plt.xlabel("Average normalized MW per step (MW/HSL)")
        plt.ylabel("Average step price ($/MWh)")
        plt.title(f"Non-cumulative Step Bid Curve — {rtype}")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        out_path = out_dir / f"noncum_curve_{re.sub(r'[^A-Za-z0-9._-]+','_', rtype)}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------- Main ----------------
def main():
    df = load_and_prepare(FILE_PATH)
    bid_pairs = detect_bid_pairs(df.columns)
    if not bid_pairs:
        raise ValueError("No (MW, Price) step pairs found.")
    if len(bid_pairs) < MAX_STEPS:
        print(f"Warning: detected only {len(bid_pairs)} pairs; proceeding.")

    bids_long = melt_bids(df, bid_pairs)
    bids_long_out = OUTPUT_DIR / "bids_long_steps.csv"
    bids_long.to_csv(bids_long_out, index=False)
    print("Saved normalized step data:", bids_long_out)

    plot_hourly_step_bars(bids_long, OUTPUT_DIR)

    step_avg = compute_non_cumulative_step_avgs(bids_long)
    curves_out = OUTPUT_DIR / "noncumulative_step_avgs.csv"
    step_avg.to_csv(curves_out, index=False)
    print("Saved step averages:", curves_out)

    plot_non_cumulative_curves(step_avg, OUTPUT_DIR, TOP_N_TYPES_FOR_INLINE)
    print(f"✅ All charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
