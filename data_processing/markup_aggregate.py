from pathlib import Path

import pandas as pd


INPUT_CSV = Path("data_processing/output/markup.csv")


def hourly_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_hourly{input_csv.suffix}")


def aggregate_to_hourly(input_csv: Path) -> Path:
    df = pd.read_csv(input_csv)

    if "ts" not in df.columns:
        raise ValueError("Input CSV must have a 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"], errors="raise")

    hourly = (
        df.set_index("ts")
        .sort_index()
        .resample("h")
        .mean(numeric_only=True)
        .reset_index()
    )

    output_csv = hourly_output_path(input_csv)
    hourly.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    output_path = aggregate_to_hourly(INPUT_CSV)
    print(f"Wrote {output_path}")
