from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import re

import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EiaInputPaths:
    input_dir: Path


def default_input_paths() -> EiaInputPaths:
    # Hardcode your input directory here.
    return EiaInputPaths(
        input_dir=Path("C:/Users/L1165683/GitHub_Repos/data-processing/inputs/eia_data")
    )


def load_csvs_from_dir(directory: Path) -> Dict[str, pd.DataFrame]:
    pattern = re.compile(r"^heatrateseia(\d{4})\.csv$", re.IGNORECASE)
    data: Dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        year = match.group(1)
        data[year] = pd.read_csv(path)
    return data


def process_yearly_dataframes(
    raw_by_year: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    processed: Dict[str, pd.DataFrame] = {}
    for year, df in raw_by_year.items():
        working = df.copy()
        working.columns = [str(col).replace("\n", " ").strip() for col in working.columns]

        tx_only = working[working["Plant State"] == "TX"].copy()

        elec_prefix = "Elec_MMBtu "
        netgen_prefix = "Netgen "
        elec_months = {
            col[len(elec_prefix) :]
            for col in tx_only.columns
            if col.startswith(elec_prefix)
        }
        netgen_months = {
            col[len(netgen_prefix) :]
            for col in tx_only.columns
            if col.startswith(netgen_prefix)
        }
        months = sorted(elec_months & netgen_months)

        for month in months:
            elec_col = f"{elec_prefix}{month}"
            netgen_col = f"{netgen_prefix}{month}"

            elec = pd.to_numeric(tx_only[elec_col], errors="coerce").fillna(0.0)
            netgen = pd.to_numeric(tx_only[netgen_col], errors="coerce").fillna(0.0)
            denom = netgen * 1000.0
            heat_rate = elec / denom
            heat_rate = heat_rate.mask(denom == 0.0)

            tx_only[f"Heat Rate {month}"] = heat_rate

        processed[year] = tx_only
    return processed


def plot_heat_rate_boxplots_by_fuel(
    processed_by_year: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for year, df in processed_by_year.items():
        year_dir = output_dir / "eia_data" / "boxplot_yearlies" / year
        year_dir.mkdir(parents=True, exist_ok=True)

        fuel_types = (
            df["Reported Fuel Type Code"]
            .astype(str)
            .str.strip()
            .replace({"nan": None})
            .dropna()
            .unique()
        )

        heat_rate_cols = [col for col in df.columns if col.startswith("Heat Rate ")]
        months_in_data = [col.replace("Heat Rate ", "") for col in heat_rate_cols]
        months = [m for m in month_order if m in months_in_data]

        for fuel_type in fuel_types:
            fuel_df = df[df["Reported Fuel Type Code"].astype(str).str.strip() == fuel_type]

            box_data = []
            for month in months:
                values = fuel_df[f"Heat Rate {month}"].dropna()
                box_data.append(values)

            if all(series.empty for series in box_data):
                continue

            plt.figure(figsize=(12, 6))
            plt.boxplot(box_data, labels=months, showfliers=True)
            plt.xlabel("Month")
            plt.ylabel("MMBtu/kWh")
            plt.title(f"Heat Rate by Month - {fuel_type} ({year})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            safe_fuel = re.sub(r"[^A-Za-z0-9._-]+", "_", fuel_type).strip("_") or "unknown"
            output_path = year_dir / f"heat_rate_boxplot_{safe_fuel}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()


def plot_heat_rate_boxplot_all_years(
    processed_by_year: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    if not processed_by_year:
        return

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    years = sorted(int(y) for y in processed_by_year.keys())
    min_year, max_year = years[0], years[-1]

    combined = pd.concat(processed_by_year.values(), ignore_index=True)
    heat_rate_cols = [col for col in combined.columns if col.startswith("Heat Rate ")]
    months_in_data = [col.replace("Heat Rate ", "") for col in heat_rate_cols]
    months = [m for m in month_order if m in months_in_data]

    fuel_types = (
        combined["Reported Fuel Type Code"]
        .astype(str)
        .str.strip()
        .replace({"nan": None})
        .dropna()
        .unique()
    )

    for fuel_type in fuel_types:
        fuel_df = combined[
            combined["Reported Fuel Type Code"].astype(str).str.strip() == fuel_type
        ]

        box_data = []
        for month in months:
            values = fuel_df[f"Heat Rate {month}"].dropna()
            box_data.append(values)

        if all(series.empty for series in box_data):
            continue

        safe_fuel = re.sub(r"[^A-Za-z0-9._-]+", "_", fuel_type).strip("_") or "unknown"
        output_path = (
            output_dir
            / "eia_data"
            / "boxplot_all_years_by_fuel"
            / safe_fuel
            / "heat_rate_boxplot.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.boxplot(box_data, labels=months, showfliers=True)
        plt.xlabel("Month")
        plt.ylabel("MMBtu/kWh")
        plt.title(f"Heat Rate by Month - {fuel_type} ({min_year}\u2013{max_year})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def plot_heat_rate_distribution_all_years_by_fuel(
    processed_by_year: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    if not processed_by_year:
        return

    years = sorted(int(y) for y in processed_by_year.keys())
    min_year, max_year = years[0], years[-1]

    combined = pd.concat(processed_by_year.values(), ignore_index=True)
    heat_rate_cols = [col for col in combined.columns if col.startswith("Heat Rate ")]
    if not heat_rate_cols:
        return

    fuel_types = (
        combined["Reported Fuel Type Code"]
        .astype(str)
        .str.strip()
        .replace({"nan": None})
        .dropna()
        .unique()
    )

    for fuel_type in fuel_types:
        fuel_df = combined[
            combined["Reported Fuel Type Code"].astype(str).str.strip() == fuel_type
        ]
        if fuel_df.empty:
            continue

        values = fuel_df[heat_rate_cols].stack().dropna()
        if values.empty:
            continue

        safe_fuel = re.sub(r"[^A-Za-z0-9._-]+", "_", fuel_type).strip("_") or "unknown"
        output_path = (
            output_dir
            / "eia_data"
            / "distribution_all_years_by_fuel"
            / safe_fuel
            / "heat_rate_distribution.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins="auto", edgecolor="black")
        plt.xlabel("MMBtu/kWh")
        plt.ylabel("Count")
        plt.title(f"Heat Rate Distribution - {fuel_type} ({min_year}\u2013{max_year})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def plot_avg_monthly_heat_rate_by_fuel_time_series(
    processed_by_year: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    if not processed_by_year:
        return

    allowed_fuels = {"BFG", "SUN", "WND", "WAT", "NG", "NUC", "LIG", "RC", "DFO", "PC"}

    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    long_frames = []
    for year, df in processed_by_year.items():
        heat_rate_cols = [col for col in df.columns if col.startswith("Heat Rate ")]
        if not heat_rate_cols:
            continue
        subset = df[["Reported Fuel Type Code", *heat_rate_cols]].copy()
        subset["Reported Fuel Type Code"] = (
            subset["Reported Fuel Type Code"].astype(str).str.strip().str.upper()
        )
        subset = subset[subset["Reported Fuel Type Code"].isin(allowed_fuels)]
        if subset.empty:
            continue
        melted = subset.melt(
            id_vars=["Reported Fuel Type Code"],
            value_vars=heat_rate_cols,
            var_name="Month",
            value_name="Heat Rate",
        )
        melted["Year"] = int(year)
        melted["Month"] = melted["Month"].str.replace("Heat Rate ", "", regex=False)
        melted["MonthNum"] = melted["Month"].map(month_map)
        long_frames.append(melted)

    if not long_frames:
        return

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["Heat Rate", "MonthNum"])
    long_df["YearMonth"] = pd.to_datetime(
        dict(year=long_df["Year"], month=long_df["MonthNum"], day=1)
    )

    grouped = (
        long_df.groupby(["Reported Fuel Type Code", "YearMonth"])["Heat Rate"]
        .mean()
        .reset_index()
    )

    years = sorted(int(y) for y in processed_by_year.keys())
    min_year, max_year = years[0], years[-1]
    full_index = pd.date_range(
        start=f"{min_year}-01-01", end=f"{max_year}-12-01", freq="MS"
    )

    output_path = (
        output_dir
        / "eia_data"
        / "avg_monthly_heat_rate_by_fuel"
        / "avg_monthly_heat_rate_by_fuel.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for fuel_type, fuel_group in grouped.groupby("Reported Fuel Type Code"):
        series = fuel_group.set_index("YearMonth")["Heat Rate"].reindex(full_index)
        plt.plot(full_index, series.values, label=str(fuel_type))

    plt.xlabel("Month")
    plt.ylabel("MMBtu/kWh")
    plt.title(f"Average Monthly Heat Rate by Fuel ({min_year}\u2013{max_year})")
    plt.legend(title="Fuel Type", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    inputs = default_input_paths()
    dataframes = load_csvs_from_dir(inputs.input_dir)
    processed = process_yearly_dataframes(dataframes)

    # TODO: Add plotting functions and call them here.
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_heat_rate_boxplots_by_fuel(processed, output_dir)
    plot_heat_rate_boxplot_all_years(processed, output_dir)
    plot_heat_rate_distribution_all_years_by_fuel(processed, output_dir)
    plot_avg_monthly_heat_rate_by_fuel_time_series(processed, output_dir)


if __name__ == "__main__":
    main()
