from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = "C:/Users/L1165683/GitHub_Repos/data-processing/inputs/price_inputs"
OUTPUT_DIR = "C:/Users/L1165683/GitHub_Repos/data-processing/output/ercot_load_hub_price_analysis"
PRICE_PROFILES_DIR = "price_profiles"
HEATMAPS_DIR = "mean_month_hour_heatmaps"
YEARLY_HEATMAPS_GLOBAL_SCALE_DIR = "yearly_month_hour_heatmaps_global_scale"
YEARLY_LINE_PLOTS_DIR = "yearly_line_plots"

REQUIRED_COLUMNS = [
    "Delivery Date",
    "Delivery Hour",
    "Delivery Interval",
    "Repeated Hour Flag",
    "Settlement Point Name",
    "Settlement Point Type",
    "Settlement Point Price",
]


def load_price_profiles(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")

    csv_paths = sorted(input_path.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in input directory: {input_path}")

    profile_chunks: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")

        prepared = prepare_price_data(df)
        for settlement_point_name, group in prepared.groupby("Settlement Point Name", sort=False):
            profile_chunks[settlement_point_name].append(group.reset_index(drop=True))

    return finalize_price_profiles(profile_chunks)


def prepare_price_data(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Delivery Date"] = pd.to_datetime(prepared["Delivery Date"], format="%m/%d/%Y")
    prepared["Delivery Hour"] = pd.to_numeric(prepared["Delivery Hour"], errors="raise").astype(int)
    prepared["Delivery Interval"] = pd.to_numeric(
        prepared["Delivery Interval"], errors="coerce"
    ).astype("Int64")
    prepared["Settlement Point Price"] = pd.to_numeric(
        prepared["Settlement Point Price"], errors="coerce"
    )
    prepared["Repeated Hour Flag"] = prepared["Repeated Hour Flag"].fillna("").astype(str).str.strip()
    prepared["Settlement Point Name"] = (
        prepared["Settlement Point Name"].fillna("").astype(str).str.strip()
    )
    prepared["Settlement Point Type"] = (
        prepared["Settlement Point Type"].fillna("").astype(str).str.strip()
    )

    prepared = prepared.loc[prepared["Settlement Point Name"] != ""].copy()
    prepared["timestamp"] = prepared["Delivery Date"] + pd.to_timedelta(
        prepared["Delivery Hour"] - 1, unit="h"
    )

    return prepared


def finalize_price_profiles(
    profile_chunks: dict[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    profiles: dict[str, pd.DataFrame] = {}

    sort_columns = [
        "Delivery Date",
        "Delivery Hour",
        "Delivery Interval",
        "Repeated Hour Flag",
    ]

    for settlement_point_name, chunks in profile_chunks.items():
        profile = pd.concat(chunks, ignore_index=True)
        profile = profile.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        profiles[settlement_point_name] = profile

    return profiles


def ensure_output_dir(output_dir: str | Path = OUTPUT_DIR) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def safe_file_stem(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe_value.strip("._") or "unnamed_settlement_point"


def save_price_profiles(
    profiles: dict[str, pd.DataFrame], output_dir: str | Path = OUTPUT_DIR
) -> Path:
    profiles_output_dir = ensure_output_dir(output_dir) / PRICE_PROFILES_DIR
    profiles_output_dir.mkdir(parents=True, exist_ok=True)

    for settlement_point_name, profile in profiles.items():
        output_path = profiles_output_dir / f"{safe_file_stem(settlement_point_name)}.csv"
        profile.to_csv(output_path, index=False)

    return profiles_output_dir


def build_month_hour_heatmap(profile: pd.DataFrame) -> pd.DataFrame:
    heatmap_df = profile.copy()
    heatmap_df["month"] = heatmap_df["Delivery Date"].dt.month

    matrix = (
        heatmap_df.groupby(["month", "Delivery Hour"])["Settlement Point Price"]
        .mean()
        .unstack()
        .reindex(index=range(1, 13), columns=range(1, 25))
    )

    return matrix


def render_month_hour_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean Settlement Point Price")

    ax.set_title(title)
    ax.set_xlabel("Delivery Hour")
    ax.set_ylabel("Month")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(1, 25))
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_labels)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_month_hour_heatmaps(
    profiles: dict[str, pd.DataFrame], output_dir: str | Path = OUTPUT_DIR
) -> Path:
    heatmaps_output_dir = ensure_output_dir(output_dir) / HEATMAPS_DIR
    heatmaps_output_dir.mkdir(parents=True, exist_ok=True)

    for settlement_point_name, profile in profiles.items():
        matrix = build_month_hour_heatmap(profile)
        output_path = heatmaps_output_dir / f"{safe_file_stem(settlement_point_name)}.png"
        render_month_hour_heatmap(
            matrix=matrix,
            title=f"Mean Price Heatmap: {settlement_point_name}",
            output_path=output_path,
        )

    return heatmaps_output_dir


def compute_global_yearly_heatmap_scale(profiles: dict[str, pd.DataFrame]) -> tuple[float, float]:
    global_min: float | None = None
    global_max: float | None = None

    for profile in profiles.values():
        yearly_profile = profile.copy()
        yearly_profile["year"] = yearly_profile["Delivery Date"].dt.year

        for _, year_df in yearly_profile.groupby("year", sort=True):
            matrix = build_month_hour_heatmap(year_df)
            stacked = matrix.stack(dropna=True)
            if stacked.empty:
                continue

            year_min = float(stacked.min())
            year_max = float(stacked.max())

            global_min = year_min if global_min is None else min(global_min, year_min)
            global_max = year_max if global_max is None else max(global_max, year_max)

    if global_min is None or global_max is None:
        raise ValueError("Unable to compute global yearly heatmap scale from empty profiles.")

    return global_min, global_max


def save_yearly_month_hour_heatmaps_global_scale(
    profiles: dict[str, pd.DataFrame], output_dir: str | Path = OUTPUT_DIR
) -> Path:
    yearly_heatmaps_output_dir = ensure_output_dir(output_dir) / YEARLY_HEATMAPS_GLOBAL_SCALE_DIR
    yearly_heatmaps_output_dir.mkdir(parents=True, exist_ok=True)

    global_min, global_max = compute_global_yearly_heatmap_scale(profiles)

    for settlement_point_name, profile in profiles.items():
        node_output_dir = yearly_heatmaps_output_dir / safe_file_stem(settlement_point_name)
        node_output_dir.mkdir(parents=True, exist_ok=True)

        yearly_profile = profile.copy()
        yearly_profile["year"] = yearly_profile["Delivery Date"].dt.year

        for year, year_df in yearly_profile.groupby("year", sort=True):
            matrix = build_month_hour_heatmap(year_df)
            output_path = node_output_dir / f"{year}.png"
            render_month_hour_heatmap(
                matrix=matrix,
                title=f"Mean Price Heatmap: {settlement_point_name} ({year})",
                output_path=output_path,
                vmin=global_min,
                vmax=global_max,
            )

    return yearly_heatmaps_output_dir


def save_yearly_line_plots(
    profiles: dict[str, pd.DataFrame], output_dir: str | Path = OUTPUT_DIR
) -> Path:
    yearly_output_dir = ensure_output_dir(output_dir) / YEARLY_LINE_PLOTS_DIR
    yearly_output_dir.mkdir(parents=True, exist_ok=True)

    for settlement_point_name, profile in profiles.items():
        node_output_dir = yearly_output_dir / safe_file_stem(settlement_point_name)
        node_output_dir.mkdir(parents=True, exist_ok=True)

        yearly_profile = profile.copy()
        yearly_profile["year"] = yearly_profile["Delivery Date"].dt.year

        for year, year_df in yearly_profile.groupby("year", sort=True):
            year_df = year_df.sort_values("timestamp", kind="stable")

            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(year_df["timestamp"], year_df["Settlement Point Price"], linewidth=0.8)
            ax.set_title(f"Settlement Point Price: {settlement_point_name} ({year})")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Settlement Point Price")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            output_path = node_output_dir / f"{year}.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)

    return yearly_output_dir


def main() -> None:
    profiles = load_price_profiles(INPUT_DIR)
    profiles_output_dir = save_price_profiles(profiles)
    heatmaps_output_dir = save_month_hour_heatmaps(profiles)
    yearly_heatmaps_output_dir = save_yearly_month_hour_heatmaps_global_scale(profiles)
    yearly_line_plots_output_dir = save_yearly_line_plots(profiles)
    print(f"Loaded {len(profiles)} settlement point profiles from {INPUT_DIR}")
    print(f"Saved price profiles to {profiles_output_dir}")
    print(f"Saved heatmaps to {heatmaps_output_dir}")
    print(f"Saved yearly global-scale heatmaps to {yearly_heatmaps_output_dir}")
    print(f"Saved yearly line plots to {yearly_line_plots_output_dir}")


if __name__ == "__main__":
    main()
