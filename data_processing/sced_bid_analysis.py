#!/usr/bin/env python3
"""
Simplified SCED bidding analysis (step-based)
No argparse version — edit FILE_PATH and OUTPUT_DIR below.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# ---------------- CONFIG ----------------
FILE_PATH = "60d_SCED_Gen_Resource_Data-17-APR-25.csv"  # <-- your CSV path
OUTPUT_DIR = Path("sced_bidding_outputs_steps")          # <-- output folder
MAX_STEPS = 35
TOP_N_TYPES_FOR_INLINE = 6
# -----------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
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
    print(f"Detected {len(bid_pairs)} (MW, Price) pairs; using up to {MAX_STEPS}.")
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
