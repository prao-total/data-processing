from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# TODO: Replace with real input file path.
CSV_PATH = "C:/path/to/your/input.csv"

# TODO: Replace with real output directory.
OUTPUT_DIR = "C:/path/to/your/output"


OFF_PEAK_END_HOUR = 6  # inclusive
MONTH_START_HOUR = {
    11: 18,  # Nov
    12: 18,  # Dec
    1: 18,  # Jan
    10: 19,  # Oct
    2: 19,  # Feb
    9: 20,  # Sep
    4: 20,  # Apr
}
DEFAULT_START_HOUR = 21  # Mar, May-Aug


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def ensure_output_dir(output_dir=OUTPUT_DIR, subdir=None):
    output_root = Path(output_dir).expanduser().resolve()
    output_path = output_root if subdir is None else output_root / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _off_peak_start_hour(month):
    return MONTH_START_HOUR.get(month, DEFAULT_START_HOUR)


def filter_off_peak(df):
    ts = df["ts"]
    month = ts.dt.month
    hour = ts.dt.hour
    start_hour = month.map(_off_peak_start_hour)
    is_off_peak = (hour >= start_hour) | (hour <= OFF_PEAK_END_HOUR)
    return df[is_off_peak].copy()


def run_analysis(df, output_path):
    off_peak_df = filter_off_peak(df)

    summary_path = output_path / "summary.csv"
    summary = pd.DataFrame(
        {
            "rows_total": [len(df)],
            "rows_off_peak": [len(off_peak_df)],
            "columns": [len(df.columns)],
        }
    )
    summary.to_csv(summary_path, index=False)
    return off_peak_df, summary_path


def save_resource_hour_counts(off_peak_df, output_dir):
    df = off_peak_df.copy()
    df["Base Point"] = pd.to_numeric(df["Base Point"], errors="coerce")
    agg = (
        df.groupby("resource_type", dropna=False)
        .agg(
            hours_count=("ts", "count"),
            base_point_sum=("Base Point", "sum"),
        )
        .reset_index()
    )
    output_path = ensure_output_dir(output_dir, "hour_count_basepoint_sum")
    out_path = output_path / "off_peak_resource_hours_base_point.csv"
    agg.to_csv(out_path, index=False)
    return out_path


def make_plots(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, "Plots pending", ha="center", va="center")
    plot_path = output_path / "placeholder_plot.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_load_duration_curve(off_peak_df, output_dir):
    df = off_peak_df.copy()
    df["Base Point"] = pd.to_numeric(df["Base Point"], errors="coerce")
    series = df["Base Point"].dropna().sort_values(ascending=False).reset_index(drop=True)

    output_path = ensure_output_dir(output_dir, "load_duration_curve")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index + 1, series.values, linewidth=1.5)
    ax.set_xlabel("Hours")
    ax.set_ylabel("MW")
    ax.set_title("Load Duration Curve (Off-Peak)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plot_path = output_path / "load_duration_curve.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    df = load_data()
    output_path = ensure_output_dir()
    off_peak_df, summary_path = run_analysis(df, output_path)
    agg_path = save_resource_hour_counts(off_peak_df, OUTPUT_DIR)
    ldc_path = save_load_duration_curve(off_peak_df, OUTPUT_DIR)
    plot_path = make_plots(off_peak_df, output_path)
    print(f"Saved summary to {summary_path}")
    print(f"Saved resource hours/base point to {agg_path}")
    print(f"Saved load duration curve to {ldc_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
