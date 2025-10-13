"""
Aggregate 'Base Point' across nested zip archives.

- Scans a folder of .zip files (and nested zips) for CSVs whose names match a pattern
  (default: contains 'GEN_RESOURCE_DATA', case-insensitive optionally).
- Loads those CSVs into pandas DataFrames.
- Aggregates into a single matrix:
    rows = Resource Name
    cols = SCED Time Stamp
    vals = Base Point
- If a cell already has a value, skip (do not overwrite) and log a message,
  unless --on-conflict overwrite is chosen.

Usage (basic):
    python aggregate_resource_basepoint.py "/path/to/zip_folder"

Use 'RESOURCE_GEN' instead of default pattern:
    python aggregate_resource_basepoint.py "/path/to/zip_folder" --pattern RESOURCE_GEN --ignore-case

Save outputs:
    python aggregate_resource_basepoint.py "/path/to/zip_folder" \
        --save-agg aggregated_base_point_matrix.csv \
        --save-log aggregation_log.txt
"""

import os
import io
import zipfile
from typing import Dict, Optional, Callable, Any, Tuple, List
import argparse

import pandas as pd
import numpy as np


# -----------------------
# Loader: nested zip CSVs
# -----------------------

def find_target_csvs_in_nested_zips(
    root_folder: str,
    pattern: str = "GEN_RESOURCE_DATA",     # substring to match in CSV filenames
    ignore_case: bool = False,
    csv_read_kwargs: Optional[dict] = None,
    max_depth: Optional[int] = None,
    on_error: Optional[Callable[[Exception, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Recursively scan a folder of ZIP files (including ZIPs inside ZIPs), find CSVs whose
    filename contains `pattern`, and return them as a dict of DataFrames.

    Keys in the returned dict are composite paths that show where the file came from:
        outer.zip::inner.zip::path/in/zip/GEN_RESOURCE_DATA_2024.csv
    """
    csv_read_kwargs = csv_read_kwargs or {}
    results: Dict[str, pd.DataFrame] = {}

    def log_error(exc: Exception, context: str):
        if on_error is not None:
            try:
                on_error(exc, context)
            except Exception:
                pass

    def matches(name: str) -> bool:
        if ignore_case:
            return (pattern.lower() in name.lower()) and name.lower().endswith(".csv")
        return (pattern in name) and name.lower().endswith(".csv")

    def unique_key(proposed_key: str) -> str:
        if proposed_key not in results:
            return proposed_key
        i = 2
        while f"{proposed_key} ({i})" in results:
            i += 1
        return f"{proposed_key} ({i})"

    def process_zipfile_obj(zf: zipfile.ZipFile, composite_prefix: str, depth: int):
        if max_depth is not None and depth > max_depth:
            return

        for info in zf.infolist():
            if hasattr(info, "is_dir") and info.is_dir():
                continue

            inner_name = info.filename  # path within zip
            inner_key = f"{composite_prefix}::{inner_name}"

            try:
                with zf.open(info, "r") as fh:
                    data = fh.read()

                # Nested ZIP detection: by .zip extension or by file signature
                nested = False
                if inner_name.lower().endswith(".zip"):
                    nested = True
                else:
                    try:
                        nested = zipfile.is_zipfile(io.BytesIO(data))
                    except Exception:
                        nested = False

                if nested:
                    try:
                        with zipfile.ZipFile(io.BytesIO(data)) as nested_zf:
                            process_zipfile_obj(nested_zf, inner_key, depth + 1)
                    except Exception as nested_exc:
                        log_error(nested_exc, f"Opening nested zip: {inner_key}")
                    continue

                # CSV match?
                base = os.path.basename(inner_name)
                if matches(base):
                    try:
                        df = pd.read_csv(io.BytesIO(data), **csv_read_kwargs)
                        key = unique_key(inner_key)
                        results[key] = df
                    except Exception as read_exc:
                        log_error(read_exc, f"Reading CSV: {inner_key}")

            except Exception as outer_exc:
                log_error(outer_exc, f"Reading member: {inner_key}")

    for entry in os.scandir(root_folder):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(".zip"):
            continue

        top_key = entry.name
        try:
            with zipfile.ZipFile(entry.path, "r") as zf:
                process_zipfile_obj(zf, top_key, depth=1)
        except Exception as e:
            log_error(e, f"Opening top-level zip: {entry.path}")

    return results


# ----------------------------
# Aggregator: Resource x Time
# ----------------------------

def aggregate_base_point_matrix(
    dfs: Dict[str, pd.DataFrame],
    resource_col: str = "Resource Name",
    timestamp_col: str = "SCED Time Stamp",
    value_col: str = "Base Point",
    case_insensitive_cols: bool = True,
    trim_resource: bool = True,
    coerce_value_numeric: bool = True,
    timestamp_format: Optional[str] = None,  # e.g., "%m/%d/%Y %H:%M:%S"
    on_conflict: str = "skip",               # "skip" or "overwrite"
    max_messages: Optional[int] = 2000,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combine CSV DataFrames into a wide matrix:
      index   = Resource Name
      columns = SCED Time Stamp
      values  = Base Point

    If a cell already has a value, we skip it and log a message (unless on_conflict="overwrite").
    """
    messages: List[str] = []

    def log(msg: str):
        if max_messages is None or len(messages) < max_messages:
            messages.append(msg)

    # Normalize / match column names flexibly
    def find_col(df: pd.DataFrame, desired: str) -> Optional[str]:
        if desired in df.columns:
            return desired
        if not case_insensitive_cols:
            return None

        def norm(s: str) -> str:
            return "".join(ch for ch in str(s).lower() if ch.isalnum())

        desired_norm = norm(desired)
        candidates = {c: norm(c) for c in df.columns}

        # Exact normalized match
        for c, cn in candidates.items():
            if cn == desired_norm:
                return c

        # Heuristics / aliases
        aliases = {
            "resourcename": {"resourcename", "resource"},
            "scedtimestamp": {"scedtimestamp", "timestamp", "time", "scedtime", "time_stamp"},
            "basepoint": {"basepoint", "base_point", "setpoint", "set_point", "mw", "value"},
        }
        if desired_norm in aliases:
            for c, cn in candidates.items():
                if cn in aliases[desired_norm]:
                    return c
        return None

    cells: Dict[Tuple[str, pd.Timestamp], float] = {}
    source_of: Dict[Tuple[str, pd.Timestamp], str] = {}

    for src_name, df in dfs.items():
        # Identify columns
        rc = find_col(df, resource_col) or resource_col
        tc = find_col(df, timestamp_col) or timestamp_col
        vc = find_col(df, value_col) or value_col

        # Validate presence
        missing = [c for c in [rc, tc, vc] if c not in df.columns]
        if missing:
            log(f"[WARN] {src_name}: missing required columns (found: {df.columns.tolist()}); skipping.")
            continue

        work = df[[rc, tc, vc]].copy()

        # Clean Resource
        if trim_resource:
            work[rc] = work[rc].astype(str).str.strip()

        # Parse timestamp (optional exact format)
        if timestamp_format:
            work[tc] = pd.to_datetime(work[tc], errors="coerce", format=timestamp_format)
        else:
            work[tc] = pd.to_datetime(work[tc], errors="coerce")

        # Coerce value
        if coerce_value_numeric:
            work[vc] = pd.to_numeric(work[vc], errors="coerce")

        # Drop rows with missing essentials
        before = len(work)
        work = work.dropna(subset=[rc, tc, vc])
        after = len(work)
        if after < before:
            log(f"[INFO] {src_name}: dropped {before - after} row(s) with null {rc}/{tc}/{vc}.")

        # Populate cells with conflict handling
        for res, ts, val in work.itertuples(index=False, name=None):
            key = (str(res), pd.Timestamp(ts))

            if key not in cells or (pd.isna(cells[key]) and not pd.isna(val)):
                cells[key] = val
                source_of[key] = src_name
                continue

            existing = cells[key]
            # If new value is NaN, nothing to add
            if pd.isna(val):
                continue

            # If existing is NaN but new has value
            if pd.isna(existing):
                cells[key] = val
                source_of[key] = src_name
                continue

            # Both non-null: conflict
            if on_conflict == "skip":
                log(f"[INFO] Duplicate for (Resource='{res}', Time='{ts}') from {src_name}; "
                    f"value already present from {source_of[key]}; skipped.")
            elif on_conflict == "overwrite":
                log(f"[WARN] Overwriting (Resource='{res}', Time='{ts}'): "
                    f"{existing} (from {source_of[key]}) -> {val} (from {src_name}).")
                cells[key] = val
                source_of[key] = src_name

    if not cells:
        empty = pd.DataFrame(columns=[resource_col]).set_index(resource_col)
        return empty, messages

    tidy = pd.DataFrame(
        [(r, t, v) for (r, t), v in cells.items()],
        columns=[resource_col, timestamp_col, value_col],
    )
    agg_df = tidy.pivot(index=resource_col, columns=timestamp_col, values=value_col)

    # Sort for readability
    try:
        agg_df = agg_df.sort_index(axis=0)  # resources
    except Exception:
        pass
    try:
        agg_df = agg_df.sort_index(axis=1)  # timestamps
    except Exception:
        pass

    return agg_df, messages


# -----------
# CLI / Main
# -----------

def main():
    parser = argparse.ArgumentParser(
        description="Scan nested zip files for target CSVs and aggregate Base Point into a Resource x Time matrix."
    )
    parser.add_argument("root_folder", help="Path to folder containing top-level zip files.")
    parser.add_argument("--pattern", default="GEN_RESOURCE_DATA",
                        help="Substring to match in CSV filenames (e.g., 'GEN_RESOURCE_DATA' or 'RESOURCE_GEN').")
    parser.add_argument("--ignore-case", action="store_true",
                        help="Case-insensitive match for filename pattern.")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Maximum nested-zip depth to traverse (None = unlimited).")
    parser.add_argument("--encoding", default=None,
                        help="CSV encoding for pandas.read_csv (e.g., 'utf-8', 'latin-1').")
    parser.add_argument("--timestamp-format", default=None,
                        help="Optional strftime format for timestamps, e.g., '%%m/%%d/%%Y %%H:%%M:%%S'.")
    parser.add_argument("--on-conflict", choices=["skip", "overwrite"], default="skip",
                        help="When duplicate (Resource, Timestamp) occurs: 'skip' (default) or 'overwrite'.")
    parser.add_argument("--save-agg", default=None,
                        help="Path to save aggregated matrix (CSV or XLSX by extension).")
    parser.add_argument("--save-log", default=None,
                        help="Path to save aggregation log messages (text file).")

    args = parser.parse_args()

    csv_kwargs: Dict[str, Any] = {}
    if args.encoding:
        csv_kwargs["encoding"] = args.encoding

    def log_error(exc: Exception, ctx: str):
        print(f"[WARN] {ctx} -> {exc}")

    print(f"Scanning '{args.root_folder}' for CSVs with pattern '{args.pattern}' "
          f"(ignore_case={args.ignore_case})...")

    dfs = find_target_csvs_in_nested_zips(
        root_folder=args.root_folder,
        pattern=args.pattern,
        ignore_case=args.ignore_case,
        csv_read_kwargs=csv_kwargs,
        max_depth=args.max_depth,
        on_error=log_error,
    )
    print(f"Found {len(dfs)} matching CSV(s).")

    agg_df, msgs = aggregate_base_point_matrix(
        dfs,
        timestamp_format=args.timestamp_format,
        on_conflict=args.on_conflict,
    )
    print(f"Aggregated matrix shape: {agg_df.shape}")

    # Save aggregated matrix if requested
    if args.save_agg:
        out = args.save_agg
        ext = os.path.splitext(out)[1].lower()
        if ext in (".xlsx", ".xlsm"):
            # Excel
            try:
                agg_df.to_excel(out, merge_cells=False, engine="openpyxl")
            except Exception as e:
                print(f"[ERROR] Failed to save Excel '{out}': {e}")
        else:
            # Default to CSV
            try:
                agg_df.to_csv(out, index=True)
            except Exception as e:
                print(f"[ERROR] Failed to save CSV '{out}': {e}")
        print(f"Saved aggregated matrix to {out}")

    # Save logs if requested
    if args.save_log:
        try:
            with open(args.save_log, "w", encoding="utf-8") as f:
                f.write("\n".join(msgs))
            print(f"Saved log to {args.save_log}")
        except Exception as e:
            print(f"[ERROR] Failed to save log '{args.save_log}': {e}")

    # Brief on-screen messages (cap to keep console readable)
    to_show = 30
    print(f"\n--- Sample messages (showing up to {to_show} of {len(msgs)}) ---")
    for m in msgs[:to_show]:
        print(m)
    if len(msgs) > to_show:
        print(f"... ({len(msgs) - to_show} more)")


if __name__ == "__main__":
    main()