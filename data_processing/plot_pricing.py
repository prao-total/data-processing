"""Utilities for loading and plotting pricing data across years."""

from __future__ import annotations

from pathlib import Path
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
    days,
    data_by_year: dict[int, pd.DataFrame],
    *,
    title: str,
    x_label: str,
    y_label: str,
    output_dir: Path | str = "output",
    date_col: str = "DATETIME",
    value_col: str = "AVGVALUE",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot multiple days as separate line plots for each year.

    Args:
        days: Iterable of day values (string or datetime). Time will be truncated to date.
        data_by_year: Output of `read_yearly_csvs`.
        title: Plot title.
        x_label: X-axis title (hour of day).
        y_label: Y-axis title.
    """
    if days is None:
        raise ValueError("days must be provided")
    days_list = list(days)
    if not days_list:
        raise ValueError("days must contain at least one day")

    plt.figure(figsize=figsize)

    day_windows: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for day in days_list:
        day_ts = pd.to_datetime(day, errors="raise")
        day_date = day_ts.date()
        day_start = pd.Timestamp(day_date)
        day_end = day_start + pd.Timedelta(days=1)
        label = day_start.strftime("%Y-%m-%d")
        day_windows.append((day_start, day_end, label))

    for year, df in data_by_year.items():
        if date_col not in df.columns:
            raise ValueError(f"Missing required column '{date_col}' for year {year}")

        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        for day_start, day_end, label in day_windows:
            mask = (df[date_col] >= day_start) & (df[date_col] < day_end)
            day_df = df.loc[mask, [date_col, value_col]]
            if day_df.empty:
                continue

            day_df = day_df.copy()
            day_df["hour"] = day_df[date_col].dt.hour
            hourly = day_df.groupby("hour", as_index=True)[value_col].mean()
            hourly = hourly.reindex(range(24))

            x_values = hourly.index
            y_values = hourly.values

            plt.plot(
                x_values,
                y_values,
                marker="o",
                label=f"{year} {label}",
            )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()

    output_base = Path(output_dir)
    plots_dir = output_base / "pricing_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pricing_plot_{timestamp}.png"
    output_path = plots_dir / filename
    plt.savefig(output_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Example usage
    data = read_yearly_csvs()
    plot_days(
        ["2023-01-31 00:00:00", "2024-01-16 00:00:00", "2025-02-20 00:00:00"],
        data,
        title="Winter Peak Day Pricing",
        x_label="Time",
        y_label="Price (USD/MWh)",
        output_dir="C:/Users/L1165683/GitHub_Repos/data-processing/data_processing/output",
    )
