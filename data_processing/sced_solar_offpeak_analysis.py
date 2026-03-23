from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# TODO: Replace with real input file path.
CSV_PATH = "C:/Users/L1165683/OneDrive - TotalEnergies/Documents/db/plots/Base Point_mean_hourly_202301010000_202312312359/aggregated.csv"

# TODO: Replace with real output directory.
OUTPUT_DIR = "C:/Users/L1165683/GitHub_Repos/data-processing/data_processing/output/solar_offpeak_analysis"


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
NEIGHBOR_OFFSET = pd.Timedelta(hours=1)


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


def exclude_plus_minus_one_hour_noise(df, group_col="resource_name"):
    df = df.copy()
    df = df.dropna(subset=["ts"]).sort_values([group_col, "ts"]).reset_index(drop=True)

    prior_lookup = df[[group_col, "ts"]].copy()
    prior_lookup["ts"] = prior_lookup["ts"] + NEIGHBOR_OFFSET
    prior_lookup["has_prior_hour"] = True

    next_lookup = df[[group_col, "ts"]].copy()
    next_lookup["ts"] = next_lookup["ts"] - NEIGHBOR_OFFSET
    next_lookup["has_next_hour"] = True

    df = df.merge(prior_lookup, on=[group_col, "ts"], how="left")
    df = df.merge(next_lookup, on=[group_col, "ts"], how="left")

    is_kept = df["has_prior_hour"].fillna(False) & df["has_next_hour"].fillna(False)
    return df.loc[is_kept].drop(columns=["has_prior_hour", "has_next_hour"]).copy()


def filter_off_peak(df):
    ts = df["ts"]
    month = ts.dt.month
    hour = ts.dt.hour
    start_hour = month.map(_off_peak_start_hour)
    is_off_peak = (hour >= start_hour) | (hour <= OFF_PEAK_END_HOUR)
    off_peak_df = df[is_off_peak].copy()
    return exclude_plus_minus_one_hour_noise(off_peak_df)


def run_analysis(df, output_path):
    off_peak_df = filter_off_peak(df)
    raw_off_peak_df = df[((df["ts"].dt.hour >= df["ts"].dt.month.map(_off_peak_start_hour)) | (df["ts"].dt.hour <= OFF_PEAK_END_HOUR))].copy()

    summary_path = output_path / "summary.csv"
    summary = pd.DataFrame(
        {
            "rows_total": [len(df)],
            "rows_off_peak_raw": [len(raw_off_peak_df)],
            "rows_off_peak_reduced_noise": [len(off_peak_df)],
            "columns": [len(df.columns)],
        }
    )
    summary.to_csv(summary_path, index=False)
    return off_peak_df, summary_path


def save_resource_hour_counts(off_peak_df, output_dir):
    df = off_peak_df.copy()
    df["Base Point"] = pd.to_numeric(df["Base Point"], errors="coerce")
    agg = (
        df.groupby("resource_name", dropna=False)
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




def save_load_duration_curve(off_peak_df, output_dir):
    df = off_peak_df.copy()
    df["Base Point"] = pd.to_numeric(df["Base Point"], errors="coerce")
    df = df.dropna(subset=["Base Point"]).sort_values("Base Point", ascending=False).reset_index(drop=True)
    df["Hours"] = df.index + 1

    output_path = ensure_output_dir(output_dir, "load_duration_curve")
    data_path = output_path / "load_duration_curve_data.csv"
    df.to_csv(data_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Hours"], df["Base Point"], linewidth=1.5)
    ax.set_xlabel("# of 5-minute intervals")
    ax.set_ylabel("MW")
    ax.set_title("Load Duration Curve (Off-Peak)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plot_path = output_path / "load_duration_curve.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path, data_path


def main():
    df = load_data()
    output_path = ensure_output_dir()
    off_peak_df, summary_path = run_analysis(df, output_path)
    agg_path = save_resource_hour_counts(off_peak_df, OUTPUT_DIR)
    ldc_path, ldc_data_path = save_load_duration_curve(off_peak_df, OUTPUT_DIR)
    print(f"Saved summary to {summary_path}")
    print(f"Saved resource hours/base point to {agg_path}")
    print(f"Saved load duration curve to {ldc_path}")
    print(f"Saved load duration curve data to {ldc_data_path}")


if __name__ == "__main__":
    main()
