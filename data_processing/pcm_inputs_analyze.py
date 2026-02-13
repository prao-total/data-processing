from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PcmInputPaths:
    heatrate: Path
    vom: Path
    fom: Path
    capacity: Path


def default_input_paths(base_dir: Path) -> PcmInputPaths:
    inputs_dir = base_dir / "inputs"
    return PcmInputPaths(
        heatrate=inputs_dir / "heatrate.csv",
        vom=inputs_dir / "vom.csv",
        fom=inputs_dir / "fom.csv",
        capacity=inputs_dir / "capacity.csv",
    )


def load_pcm_inputs(paths: PcmInputPaths) -> Dict[str, pd.DataFrame]:
    return {
        "heatrate": pd.read_csv(paths.heatrate),
        "vom": pd.read_csv(paths.vom),
        "fom": pd.read_csv(paths.fom),
        "capacity": pd.read_csv(paths.capacity),
    }


def plot_value_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    values = df["Value"].dropna()
    property_label = str(df["Property"].iloc[0]).strip()
    units_label = str(df["Units"].iloc[0]).strip()
    xlabel = f"{property_label} {units_label}".strip()

    property_dir = output_dir / property_label
    property_dir.mkdir(parents=True, exist_ok=True)
    output_path = property_dir / f"distribution_of_{property_label}.png"

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins="auto", edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"Distribution of {property_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_value_boxplot_by_category(df: pd.DataFrame, output_dir: Path) -> Path | None:
    values = df["Value"]
    categories = df["Category"]
    property_label = str(df["Property"].iloc[0]).strip()
    units_label = str(df["Units"].iloc[0]).strip()
    ylabel = f"{property_label} {units_label}".strip()

    data = pd.DataFrame({"Category": categories, "Value": values}).dropna()
    if data.empty:
        return

    property_dir = output_dir / property_label
    property_dir.mkdir(parents=True, exist_ok=True)
    output_path = property_dir / f"boxplot_by_category_{property_label}.png"

    grouped = data.groupby("Category")["Value"].apply(list)
    labels = grouped.index.tolist()
    box_data = grouped.tolist()

    plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, labels=labels, showfliers=True)
    plt.xlabel("Category")
    plt.ylabel(ylabel)
    plt.title(f"{property_label} by Fuel")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def preprocess_categories(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    category_series = processed["Category"].astype(str).str.strip()
    mask = category_series.isin(
        ["Base Expansion", "Historical Proxies", "Spark Spread Proxies"]
    )
    if not mask.any():
        return processed

    child_object = processed.loc[mask, "Child Object"].astype(str)
    base_idx = child_object.index

    mapping = [
        (r"\bwind\b", "WT_WND"),
        (r"\bnuclear\b", "ST_NUC"),
        (r"\bscct\b", "SCCT_NG"),
        (r"\bsolar\b", "PV_SUN"),
        (r"\bhybrid\s*pv[-\s]*bess\b", "PVGR-BESS"),
        (r"\bccct\b", "CCCT_NG"),
        (r"\bcgt\b|\bgt\b", "CCGT_NG"),
        (r"\bcoal\b", "ST_COAL"),
    ]

    for pattern, replacement in mapping:
        match_mask = child_object.str.contains(pattern, case=False, regex=True)
        if match_mask.any():
            processed.loc[base_idx[match_mask], "Category"] = replacement
    return processed


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_paths = default_input_paths(base_dir)
    dataframes = load_pcm_inputs(input_paths)

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for df in dataframes.values():
        df = preprocess_categories(df)
        plot_value_distribution(df, output_dir)
        plot_value_boxplot_by_category(df, output_dir)


if __name__ == "__main__":
    main()
