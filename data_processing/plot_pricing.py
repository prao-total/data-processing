"""Utilities for loading and plotting pricing data across years."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt

# Hardcoded CSV paths by year.
CSV_BY_YEAR = {
    2023: Path("C:/Users/L1165683/GitHub_Repos/yes-energy-extract/tests/outputs/HB_HUBAVG_rtlmp_2023-01-01_to_2023-12-31_hourly.csv"),
    2024: Path("C:/Users/L1165683/GitHub_Repos/yes-energy-extract/tests/outputs/HB_HUBAVG_rtlmp_2024-01-01_to_2024-12-31_hourly.csv"),
    2025: Path("C:/Users/L1165683/GitHub_Repos/yes-energy-extract/tests/outputs/HB_HUBAVG_rtlmp_2025-01-01_to_2025-12-31_hourly.csv"),
}


def read_yearly_csvs(
    csv_by_year: dict[int, Path] | None = None,
    *,
    date_col: str = "DATETIME",
    value_col: str = "AVGVALUE",
    parse_dates: bool = True,
) -> dict[int, pd.DataFrame]:
    """Read each CSV into a DataFrame.

    Args:
        csv_by_year: Mapping of year -> CSV path. Defaults to `CSV_BY_YEAR`.
        date_col: Column containing the date/day values.
        value_col: Column containing the metric to plot.
        parse_dates: If True, attempt to parse date_col as datetime.
    """
    csv_by_year = csv_by_year or CSV_BY_YEAR

    data_by_year: dict[int, pd.DataFrame] = {}
    for year, path in csv_by_year.items():
        df = pd.read_csv(path)
        if parse_dates and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="ignore")
        if date_col in df.columns:
            df = df.sort_values(date_col)
        if value_col not in df.columns:
            raise ValueError(f"Missing required column '{value_col}' in {path}")
        data_by_year[year] = df

    return data_by_year


def plot_days(
    days: Iterable,
    data_by_year: dict[int, pd.DataFrame],
    *,
    title: str,
    x_label: str,
    y_label: str,
    date_col: str = "DATETIME",
    value_col: str = "AVGVALUE",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot selected days as a line plot for each year.

    Args:
        days: Iterable of day values (dates or numbers) to plot.
        data_by_year: Output of `read_yearly_csvs`.
        title: Plot title.
        x_label: X-axis title.
        y_label: Y-axis title.
    """
    days_list = list(days)
    if not days_list:
        raise ValueError("days must contain at least one day")

    plt.figure(figsize=figsize)

    for year, df in data_by_year.items():
        if date_col not in df.columns:
            raise ValueError(f"Missing required column '{date_col}' for year {year}")

        # If date_col is datetime-like, coerce days to datetime for matching.
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            days_coerced = pd.to_datetime(days_list, errors="coerce")
            mask = df[date_col].isin(days_coerced)
            x_values = df.loc[mask, date_col]
        else:
            mask = df[date_col].isin(days_list)
            x_values = df.loc[mask, date_col]

        y_values = df.loc[mask, value_col]

        if len(x_values) == 0:
            continue

        plt.plot(x_values, y_values, marker="o", label=str(year))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    data = read_yearly_csvs()
    plot_days(
        ["2024-01-01 00:00:00", "2024-01-15 00:00:00", "2024-02-01 00:00:00"],
        data,
        title="Selected Days Pricing",
        x_label="Date",
        y_label="Average Value",
    )
