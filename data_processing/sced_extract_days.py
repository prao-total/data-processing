#!/usr/bin/env python3
"""
Extract selected days and/or months from wide 5-minute timeseries CSVs.

INPUT FILES:
- Named like either:
    - aggregate_<metric>.csv
    - aggregation_<metric>.csv
- Columns:
    1) "Resource Name"
    2) "Resource Type"
    3+) timestamp columns in 5-minute resolution (parseable datetime strings)

OUTPUT:
- For each requested DAY (YYYY-MM-DD), write:
      OUTPUT_DIR/<YYYY-MM-DD>/<original_filename>
- For each requested MONTH (YYYY-MM), write:
      OUTPUT_DIR/<YYYY-MM>/<original_filename>
- Only the timestamp columns within the chosen period are kept, in chronological order.

USAGE (hardcoded variables or CLI flags):
    python extract_periods_from_aggregate_csvs.py \
      --input-dir "/path/to/input" \
      --output-dir "/path/to/output" \
      --days 2025-01-15 2025-01-16 \
      --months 2025-01 2025-02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd


# ------------------------- Hardcoded defaults (edit these) -------------------------
INPUT_DIR  = r"/path/to/input_dir"     # folder containing aggregate_*.csv or aggregation_*.csv
OUTPUT_DIR = r"/path/to/output_dir"    # destination root folder
DAYS       = []                        # e.g. ["2025-01-15", "2025-01-16"]
MONTHS     = []                        # e.g. ["2025-01", "2025-02"]
# -----------------------------------------------------------------------------------


def normalize_key_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to find the two key columns: Resource Name, Resource Type.
    Returns the *actual* column names in the dataframe (to preserve exact casing).
    Raises ValueError if not found.
    """
    def norm(s: str) -> str:
        return s.strip().lower()

    cols_norm = {norm(c): c for c in df.columns}

    name_col = cols_norm.get("resource name")
    type_col = cols_norm.get("resource type")

    if name_col is None or type_col is None:
        raise ValueError(
            f"Could not find required columns 'Resource Name' and 'Resource Type'. "
            f"Columns found: {list(df.columns)}"
        )
    return name_col, type_col


def detect_timestamp_columns(df: pd.DataFrame, key_cols: Tuple[str, str]) -> Dict[pd.Timestamp, str]:
    """
    Attempt to parse remaining column headers as datetimes.
    Returns a mapping {parsed_timestamp: original_column_name}
    Only includes columns that successfully parse to a Timestamp.
    Drops timezone info (localized to naive) for date/month filtering.
    """
    name_col, type_col = key_cols
    ts_map: Dict[pd.Timestamp, str] = {}

    for col in df.columns:
        if col in (name_col, type_col):
            continue
        ts = pd.to_datetime(col, errors="coerce", utc=False)
        if pd.isna(ts):
            continue
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        ts_map[pd.Timestamp(ts)] = col

    return ts_map


def subset_for_day(df: pd.DataFrame, key_cols: Tuple[str, str], ts_map: Dict[pd.Timestamp, str], day_str: str) -> pd.DataFrame:
    """
    Select only the timestamp columns that fall on a specific day (YYYY-MM-DD).
    Returns a new DataFrame with key columns + sorted timestamp columns.
    """
    target_date = pd.to_datetime(day_str).date()
    cols = [key_cols[0], key_cols[1]]
    day_ts = sorted([ts for ts in ts_map.keys() if ts.date() == target_date])
    cols += [ts_map[ts] for ts in day_ts if ts_map[ts] in df.columns]
    return df.loc[:, cols]


def subset_for_month(df: pd.DataFrame, key_cols: Tuple[str, str], ts_map: Dict[pd.Timestamp, str], month_str: str) -> pd.DataFrame:
    """
    Select only the timestamp columns that fall in a specific month (YYYY-MM).
    Returns a new DataFrame with key columns + sorted timestamp columns.
    """
    # Parse YYYY-MM robustly by appending '-01'
    target = pd.to_datetime(f"{month_str}-01")
    year, month = target.year, target.month

    cols = [key_cols[0], key_cols[1]]
    month_ts = sorted([ts for ts in ts_map.keys() if ts.year == year and ts.month == month])
    cols += [ts_map[ts] for ts in month_ts if ts_map[ts] in df.columns]
    return df.loc[:, cols]


def write_subset(out_folder: Path, filename: str, subset_df: pd.DataFrame, period_label: str) -> None:
    """
    Write the subset DataFrame to out_folder/filename.
    Emits helpful logs when only key columns are present (i.e., no matching timestamps).
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename

    if subset_df.shape[1] <= 2:
        print(f"[INFO] {filename}: No matching timestamp columns for {period_label}. Writing keys only.", file=sys.stderr)

    try:
        subset_df.to_csv(out_path, index=False)
        print(f"[OK] Wrote {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}", file=sys.stderr)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    days: List[str],
    months: List[str],
    patterns: List[str] = None,
) -> None:
    """
    For each CSV in input_dir matching patterns (default: aggregate_*.csv and aggregation_*.csv),
    write one output per requested day into output_dir/<YYYY-MM-DD>/ with the same filename,
    and one output per requested month into output_dir/<YYYY-MM>/ with the same filename.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if patterns is None:
        patterns = ["aggregate_*.csv", "aggregation_*.csv"]

    csv_paths: List[Path] = []
    for pat in patterns:
        csv_paths.extend(sorted(input_dir.glob(pat)))

    if not csv_paths:
        print(f"[WARN] No files found matching {patterns} in {input_dir}", file=sys.stderr)
        return

    # Normalize day strings: keep as YYYY-MM-DD
    days_norm = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in days] if days else []
    # Normalize months: keep as YYYY-MM
    months_norm = [pd.to_datetime(f"{m}-01").strftime("%Y-%m") for m in months] if months else []

    for csv_path in csv_paths:
        # Read as strings to preserve exact values/headers
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path.name}: {e}", file=sys.stderr)
            continue

        # Detect key columns and timestamp columns
        try:
            key_cols = normalize_key_columns(df)
        except ValueError as e:
            print(f"[ERROR] {csv_path.name}: {e}", file=sys.stderr)
            continue

        ts_map = detect_timestamp_columns(df, key_cols)
        if not ts_map:
            print(f"[WARN] {csv_path.name}: No timestamp columns detected; skipping.", file=sys.stderr)
            continue

        # Per-day outputs
        for day in days_norm:
            day_df = subset_for_day(df, key_cols, ts_map, day)
            day_folder = output_dir / day
            write_subset(day_folder, csv_path.name, day_df, period_label=f"day {day}")

        # Per-month outputs
        for month in months_norm:
            month_df = subset_for_month(df, key_cols, ts_map, month)
            month_folder = output_dir / month
            write_subset(month_folder, csv_path.name, month_df, period_label=f"month {month}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract selected days and/or months from wide 5-minute CSVs.")
    p.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Folder with aggregate_*.csv or aggregation_*.csv")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Destination root folder")
    p.add_argument("--days", type=str, nargs="*", default=DAYS, help='Days to extract (YYYY-MM-DD), e.g., 2025-01-15 2025-01-16')
    p.add_argument("--months", type=str, nargs="*", default=MONTHS, help='Months to extract (YYYY-MM), e.g., 2025-01 2025-02')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        days=args.days,
        months=args.months,
    )


if __name__ == "__main__":
    main()
