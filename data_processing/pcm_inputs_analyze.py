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
    inputs_dir = base_dir / "inputs" / "pcm_inputs"
    return PcmInputPaths(
        heatrate="C:/Users/L1165683/GitHub_Repos/data-processing/inputs/pcm_inputs/heatrate.csv",
        vom="C:/Users/L1165683/GitHub_Repos/data-processing/inputs/pcm_inputs/vom.csv",
        fom="C:/Users/L1165683/GitHub_Repos/data-processing/inputs/pcm_inputs/fom.csv",
        capacity="C:/Users/L1165683/GitHub_Repos/data-processing/inputs/pcm_inputs/capacity.csv",
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
        (r"\bhydro\b", "HY_WAT"),
        (r"\bgas\b", "OtherTech_NG"),
    ]

    for pattern, replacement in mapping:
        match_mask = child_object.str.contains(pattern, case=False, regex=True)
        if match_mask.any():
            processed.loc[base_idx[match_mask], "Category"] = replacement
    return processed


def plot_value_scatter(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Scatter plot of values from two dataframes (x and y)."""
    x_property = str(x_df["Property"].iloc[0]).strip()
    x_units = str(x_df["Units"].iloc[0]).strip()
    y_property = str(y_df["Property"].iloc[0]).strip()
    y_units = str(y_df["Units"].iloc[0]).strip()

    x_label = f"{x_property} {x_units}".strip()
    y_label = f"{y_property} {y_units}".strip()

    merged = (
        x_df[["Child Object", "Value", "Category"]]
        .merge(
            y_df[["Child Object", "Value"]],
            on="Child Object",
            how="inner",
            suffixes=("_x", "_y"),
        )
        .dropna()
    )

    if merged.empty:
        return None

    property_dir = output_dir / f"{x_property}_vs_{y_property}"
    property_dir.mkdir(parents=True, exist_ok=True)
    output_path = property_dir / f"scatter_{x_property}_vs_{y_property}.png"

    categories = merged["Category"].astype(str)
    unique_categories = sorted(categories.unique())
    cmap = plt.get_cmap("tab20")
    color_map = {
        category: cmap(idx % cmap.N) for idx, category in enumerate(unique_categories)
    }

    plt.figure(figsize=(10, 6))
    for category in unique_categories:
        mask = categories == category
        plt.scatter(
            merged.loc[mask, "Value_x"],
            merged.loc[mask, "Value_y"],
            label=category,
            color=color_map[category],
            alpha=0.8,
            edgecolors="none",
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_property} vs {x_property}")
    plt.legend(title="Category", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_paths = default_input_paths(base_dir)
    dataframes = load_pcm_inputs(input_paths)
    processed = {name: preprocess_categories(df) for name, df in dataframes.items()}

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for df in processed.values():
        plot_value_distribution(df, output_dir)
        plot_value_boxplot_by_category(df, output_dir)

    capacity_df = processed["capacity"]
    heatrate_df = processed["heatrate"]
    vom_df = processed["vom"]
    fom_df = processed["fom"]

    plot_value_scatter(capacity_df, heatrate_df, output_dir)
    plot_value_scatter(capacity_df, vom_df, output_dir)
    plot_value_scatter(capacity_df, fom_df, output_dir)
    plot_value_scatter(heatrate_df, vom_df, output_dir)
    plot_value_scatter(heatrate_df, fom_df, output_dir)


if __name__ == "__main__":
    main()
