from pathlib import Path

import pandas as pd

# TODO: Replace with real input file paths.
CSV_PATHS = [
    "C:/Users/L1165683/OneDrive - TotalEnergies/Documents/db/input/file_1.csv",
    "C:/Users/L1165683/OneDrive - TotalEnergies/Documents/db/input/file_2.csv",
    "C:/Users/L1165683/OneDrive - TotalEnergies/Documents/db/input/file_3.csv",
]

# TODO: Replace with real output directory.
OUTPUT_DIR = "C:/Users/L1165683/GitHub_Repos/data-processing/data_processing/output/sced_unique_names"
OUTPUT_FILE_NAME = "resource_name_fuel_type_avg_base_point.csv"

REQUIRED_COLUMNS = ["resource_name", "fuel_type", "avg_base_point"]
GROUP_COLUMNS = ["resource_name", "fuel_type"]


def ensure_output_dir(output_dir=OUTPUT_DIR):
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
    return df


def aggregate_resource_fuel_rows(csv_paths=CSV_PATHS):
    frames = []
    for csv_path in csv_paths:
        df = load_csv(csv_path)
        reduced_df = df[REQUIRED_COLUMNS].copy()
        reduced_df["avg_base_point"] = pd.to_numeric(reduced_df["avg_base_point"], errors="coerce")
        frames.append(reduced_df)

    combined = pd.concat(frames, ignore_index=True)
    aggregated_rows = (
        combined.groupby(GROUP_COLUMNS, dropna=False, as_index=False)
        .agg(avg_base_point=("avg_base_point", "mean"))
        .sort_values(GROUP_COLUMNS, na_position="last")
        .reset_index(drop=True)
    )
    return aggregated_rows


def save_aggregated_rows(df, output_dir=OUTPUT_DIR, output_file_name=OUTPUT_FILE_NAME):
    output_path = ensure_output_dir(output_dir)
    out_file = output_path / output_file_name
    df.to_csv(out_file, index=False)
    return out_file


def main():
    aggregated_rows = aggregate_resource_fuel_rows()
    output_file = save_aggregated_rows(aggregated_rows)
    print(f"Saved aggregated resource/fuel rows to {output_file}")


if __name__ == "__main__":
    main()
