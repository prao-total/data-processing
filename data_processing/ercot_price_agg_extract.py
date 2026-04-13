from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


INPUT_DIR = "/Users/pradyrao/Downloads"
FILE_NAME_FILTER = ""
OUTPUT_CSV_PATH = (
    "/Users/pradyrao/VSCode/data-processing/data_processing/output/ercot_price_agg_wide.csv"
)

REQUIRED_COLUMNS = ["DATETIME", "AVGVALUE", "OBJECTNAME"]
EXPECTED_HEADERS = [
    "DATETIME",
    "DAYOFWEEK",
    "OBJECTID",
    "OBJECTNAME",
    "DATATYPE",
    "AGG_LEVEL",
    "MINVALUE",
    "MAXVALUE",
    "AVGVALUE",
    "TIMEZONE",
    "COUNTVALUE",
    "SUMVALUE",
    "STDDEVVALUE",
    "HOURENDING",
    "MARKETDAY",
    "PEAKTYPE",
    "MONTH",
    "YEAR",
]


def find_matching_csvs(input_dir: str, name_filter: str) -> list[Path]:
    directory = Path(input_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    filter_text = name_filter.lower()
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and filter_text in path.name.lower()
    )


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")


def has_expected_header_row(csv_path: Path) -> bool:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, [])

    normalized_row = [value.strip() for value in first_row]
    return normalized_row == EXPECTED_HEADERS


def repair_missing_header(csv_path: Path) -> None:
    if has_expected_header_row(csv_path):
        return

    df = pd.read_csv(csv_path, header=None, names=EXPECTED_HEADERS)
    df.to_csv(csv_path, index=False)


def first_object_name(df: pd.DataFrame, csv_path: Path) -> str:
    non_null_names = df["OBJECTNAME"].dropna().astype(str).str.strip()
    non_empty_names = non_null_names[non_null_names != ""]
    if non_empty_names.empty:
        return csv_path.stem
    return non_empty_names.iloc[0]


def build_series_frame(csv_path: Path, used_headers: set[str]) -> pd.DataFrame:
    repair_missing_header(csv_path)
    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)

    value_header = first_object_name(df, csv_path)
    if value_header in used_headers:
        value_header = f"{value_header}_{csv_path.stem}"
    used_headers.add(value_header)

    result = df.loc[:, ["DATETIME", "AVGVALUE"]].copy()
    result["DATETIME"] = pd.to_datetime(result["DATETIME"], errors="coerce")
    result = result.dropna(subset=["DATETIME"])
    result = result.drop_duplicates(subset=["DATETIME"], keep="first")
    result = result.rename(columns={"AVGVALUE": value_header})
    return result


def build_wide_dataframe(input_dir: str, name_filter: str) -> pd.DataFrame:
    csv_paths = find_matching_csvs(input_dir, name_filter)
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {Path(input_dir).expanduser().resolve()} "
            f"with '{name_filter}' in the filename."
        )

    used_headers: set[str] = set()
    merged_df: pd.DataFrame | None = None

    for csv_path in csv_paths:
        current_df = build_series_frame(csv_path, used_headers)
        if merged_df is None:
            merged_df = current_df
        else:
            merged_df = merged_df.merge(current_df, on="DATETIME", how="outer")

    if merged_df is None:
        raise RuntimeError("No data was loaded from the matched CSV files.")

    merged_df = merged_df.sort_values("DATETIME").reset_index(drop=True)
    return merged_df


def write_output(df: pd.DataFrame, output_csv_path: str) -> Path:
    output_path = Path(output_csv_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    wide_df = build_wide_dataframe(INPUT_DIR, FILE_NAME_FILTER)
    output_path = write_output(wide_df, OUTPUT_CSV_PATH)
    print(f"Wrote {len(wide_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
