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
OUTPUT_FILE_NAME = "unique_resource_name_fuel_type.csv"

REQUIRED_COLUMNS = ["resource_name", "fuel_type", "avg_base_point"]
UNIQUE_COLUMNS = ["resource_name", "fuel_type"]


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


def extract_unique_resource_fuel_rows(csv_paths=CSV_PATHS):
    frames = []
    for csv_path in csv_paths:
        df = load_csv(csv_path)
        frames.append(df[UNIQUE_COLUMNS].copy())

    combined = pd.concat(frames, ignore_index=True)
    unique_rows = (
        combined.drop_duplicates(subset=UNIQUE_COLUMNS)
        .sort_values(UNIQUE_COLUMNS, na_position="last")
        .reset_index(drop=True)
    )
    return unique_rows


def save_unique_rows(df, output_dir=OUTPUT_DIR, output_file_name=OUTPUT_FILE_NAME):
    output_path = ensure_output_dir(output_dir)
    out_file = output_path / output_file_name
    df.to_csv(out_file, index=False)
    return out_file


def main():
    unique_rows = extract_unique_resource_fuel_rows()
    output_file = save_unique_rows(unique_rows)
    print(f"Saved unique resource/fuel rows to {output_file}")


if __name__ == "__main__":
    main()
