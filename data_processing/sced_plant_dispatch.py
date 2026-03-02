import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CSV_PATH = "C:/Users/L1165683/Downloads/two_axis_plot.csv"
OUTPUT_DIR = "C:/Users/L1165683/GitHub_Repos/data-processing/data_processing/output/dispatch_count_plot"


def _extract_year(col_name):
    match = re.search(r"(20\d{2})", col_name)
    return int(match.group(1)) if match else None


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    plants_cols = [c for c in df.columns if c.startswith("Number of Plants")]
    dispatch_cols = [c for c in df.columns if c.startswith("Dispatch (GWh)")]

    plants_long = df.melt(
        id_vars=["Resource Type"],
        value_vars=plants_cols,
        var_name="Plants Column",
        value_name="Number of Plants",
    )
    plants_long["Year"] = plants_long["Plants Column"].apply(_extract_year)

    dispatch_long = df.melt(
        id_vars=["Resource Type"],
        value_vars=dispatch_cols,
        var_name="Dispatch Column",
        value_name="Dispatch (GWh)",
    )
    dispatch_long["Year"] = dispatch_long["Dispatch Column"].apply(_extract_year)

    merged = plants_long.merge(
        dispatch_long,
        on=["Resource Type", "Year"],
        how="inner",
    )

    merged = merged.drop(columns=["Plants Column", "Dispatch Column"])
    merged = merged.sort_values(["Resource Type", "Year"]).reset_index(drop=True)
    return merged


def plot_dispatch(df):
    resources = df["Resource Type"].unique().tolist()
    years = sorted(df["Year"].unique().tolist())

    x = np.arange(len(resources))
    width = 0.2
    if len(years) > 1:
        offsets = np.linspace(-width, width, len(years))
    else:
        offsets = [0.0]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for offset, year in zip(offsets, years):
        subset = df[df["Year"] == year]
        subset = subset.set_index("Resource Type").reindex(resources)

        ax1.bar(
            x + offset,
            subset["Dispatch (GWh)"],
            width=width,
            label=str(year),
            alpha=0.8,
        )
        ax2.scatter(
            x + offset,
            subset["Number of Plants"],
            label=None,
            marker="o",
            color="black",
        )

    ax1.set_xlabel("Fuel Type")
    ax1.set_ylabel("Dispatch (GWh)")
    ax2.set_ylabel("# Plants")

    ax1.set_xticks(x)
    ax1.set_xticklabels(resources, rotation=45, ha="right")

    handles1, labels1 = ax1.get_legend_handles_labels()
    dot_proxy = Line2D([0], [0], marker="o", color="black", linestyle="None")
    legend_handles = handles1 + [dot_proxy]
    legend_labels = labels1 + ["Plant Count"]
    ax1.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=min(len(legend_labels), 4),
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    return fig, ax1, ax2


def save_plot(fig, output_dir=OUTPUT_DIR, folder_name="sced_plant_dispatch"):
    output_root = Path(output_dir).expanduser().resolve()
    output_path = output_root / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "sced_plant_dispatch.png"
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    return file_path


def main():
    df = load_data()
    fig, _, _ = plot_dispatch(df)
    file_path = save_plot(fig)
    print(f"Saved plot to {file_path}")


if __name__ == "__main__":
    main()
