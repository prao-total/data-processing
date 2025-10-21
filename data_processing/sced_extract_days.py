#!/usr/bin/env python3
"""
Extract selected days from wide 5-minute timeseries CSVs.

INPUT FORMAT (per file):
- Name: aggregate_<metric>.csv
- Columns:
    1) "Resource Name"
    2) "Resource Type"
    3+) timestamp columns in 5-minute resolution (any standard datetime string)

OUTPUT:
- For each requested day (YYYY-MM-DD), create a subfolder:
      OUTPUT_DIR/<YYYY-MM-DD>/
  and write one CSV per input with only that day's timestamp columns preserved,
  along with "Resource Name" and "Resource Type".

NOTES:
- Timestamp column names are parsed with pandas.to_datetime(errors="coerce"):
  non-datetime headers are ignored (besides the first two key columns).
- Case/whitespace tolerant detection for the two key columns.
- Columns for a day are written in chronological order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd


# ------------------------- Hardcoded defaults (edit these) -------------------------
INPUT_DIR  = r"/path/to/input_dir"     # folder containing aggregate_*.csv
OUTPUT_DIR = r"/path/to/output_dir"    # destination root folder
DAYS       = ["2025-01-15", "2025-01-16"]  # list of days (YYYY-MM-DD)

# If you prefer CLI, leave hardcoded values as-is and supply args; CLI overrides them.
# -----------------------------------------------------------------------------------


def normalize_key_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to find the two key columns: Resource Name, Resource Type.
    Returns the *actual* column names in the dataframe (to preserve exact casing).
    Raises ValueError if not found.
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("  ", " ")

    cols_norm = {norm(c): c for c in df.columns}

    name_candidates = ["resource name"]
    type_candidates = ["resource type"]

    name_col = next((cols_norm[c] for c in name_candidates if c in cols_norm), None)
    type_col = next((cols_norm[c] for c in type_candidates if c in cols_norm), None)

    if name_col is None or type_col is None:
        # Helpful diagnostics
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
    """
    name_col, type_col = key_cols
    ts_map: Dict[pd.Timestamp, str] = {}

    for col in df.columns:
        if col in (name_col, type_col):
            continue
        # Try to parse header as datetime
        ts = pd.to_datetime(col, errors="coerce", utc=False)
        if pd.isna(ts):
            # Not a timestamp column; ignore silently
            continue
        # Normalize to pandas Timestamp without timezone (naive) for date filtering
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        ts_map[pd.Timestamp(ts)] = col

    return ts_map


def subset_day(
    df: pd.DataFrame,
    key_cols: Tuple[str, str],
    ts_map: Dict[pd.Timestamp, str],
    day_str: str
) -> pd.DataFrame:
    """
    Select only the timestamp columns that fall on a specific day (YYYY-MM-DD).
    Returns a new DataFrame with key columns + sorted timestamp columns.
    """
    target_date = pd.to_datetime(day_str).date()
    # Pick the timestamps whose date matches
    day_ts_sorted = sorted([ts for ts in ts_map.keys() if ts.date() == target_date])

    # If no timestamps for this day, return just the two key cols (or empty if absent)
    selected_cols = [key_cols[0], key_cols[1]] + [ts_map[ts] for ts in day_ts_sorted]
    # Filter by existing columns (in case of any oddities)
    selected_cols = [c for c in selected_cols if c in df.columns]

    return df.loc[:, selected_cols]


def process_directory(
    input_dir: Path,
    output_dir: Path,
    days: List[str],
    filename_prefix: str = "aggregate_",
) -> None:
    """
    For each CSV in input_dir with the given prefix, write one output per requested day
    into output_dir/<YYYY-MM-DD>/ with the same filename.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob(f"{filename_prefix}*.csv"))
    if not csv_paths:
        print(f"[WARN] No files found matching {filename_prefix}*.csv in {input_dir}", file=sys.stderr)
        return

    # Normalize day strings
    days_norm = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in days]

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, dtype=str)  # read as strings to preserve exact values
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path.name}: {e}", file=sys.stderr)
            continue

        try:
            key_cols = normalize_key_columns(df)
        except ValueError as e:
            print(f"[ERROR] {csv_path.name}: {e}", file=sys.stderr)
            continue

        ts_map = detect_timestamp_columns(df, key_cols)
        if not ts_map:
            print(f"[WARN] {csv_path.name}: No timestamp columns detected; skipping.", file=sys.stderr)
            continue

        for day in days_norm:
            day_folder = output_dir / day
            day_folder.mkdir(parents=True, exist_ok=True)
            out_path = day_folder / csv_path.name

            day_df = subset_day(df, key_cols, ts_map, day)

            # If a file for that day has zero timestamp columns (i.e., only keys), warn.
            if day_df.shape[1] <= 2:
                print(f"[INFO] {csv_path.name}: No data columns found for {day}. Writing keys only.", file=sys.stderr)

            try:
                day_df.to_csv(out_path, index=False)
            except Exception as e:
                print(f"[ERROR] Failed to write {out_path}: {e}", file=sys.stderr)
                continue

            print(f"[OK] Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract selected days from 5-minute wide CSVs.")
    p.add_argument("--input-dir", type=str, default=INPUT_DIR, help="Folder with aggregate_*.csv")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Destination root folder")
    p.add_argument(
        "--days",
        type=str,
        nargs="+",
        default=DAYS,
        help='Days to extract in YYYY-MM-DD (e.g., 2025-01-15 2025-01-16)',
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="aggregate_",
        help="Filename prefix to match (default: aggregate_)."
    )
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
        filename_prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
