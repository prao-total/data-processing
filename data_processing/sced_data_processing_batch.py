#!/usr/bin/env python
"""
SCED data processing: Option A aggregator

Reads one or more ERCOT 60-day SCED generator CSVs and builds
an "aggregated matrix" for each value column (e.g. Base Point, HSL).
Each output CSV has Resource Name as rows and SCED timestamps as columns.

Environment variables expected:

  ROOT_FOLDER       (required)  -> root directory containing SCED CSVs
  SAVE_AGG_DIR      (preferred) -> directory to write aggregated CSVs into
  SAVE_AGG_PATH     (legacy)    -> if set, we use its *directory* as SAVE_AGG_DIR
  VALUE_COLS                    -> comma-separated list of columns, e.g. "Base Point, HSL"
  FILENAME_CONTAINS             -> optional filter substring for CSV names
                                   default: "SCED_Gen_Resource_Data"
  TIMESTAMP_COLUMN              -> optional; default: "SCED Time Stamp"
  AGG_LOG_PATH                  -> optional log file path
  VERBOSE                       -> "0"/"false" to silence stdout logs (default on)

Usage:

  python sced_data_processing.py
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


# ---------------------------------------------------------------------------
# Config + env helpers
# ---------------------------------------------------------------------------

@dataclass
class Config:
    root_folder: Path
    save_agg_dir: Path
    value_cols: List[str]
    filename_contains: Optional[str] = None
    timestamp_col: str = "SCED Time Stamp"
    log_path: Optional[Path] = None
    verbose: bool = True


def env_get(key: str, default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    val = os.getenv(key, default)
    if required and (val is None or str(val).strip() == ""):
        raise SystemExit(f"ERROR: environment variable {key!r} is required but not set.")
    return val


def load_config() -> Config:
    # Load .env if python-dotenv is installed
    if load_dotenv is not None:
        load_dotenv()

    root_folder = Path(env_get("ROOT_FOLDER", required=True))  # type: ignore[arg-type]

    save_agg_dir_str = env_get("SAVE_AGG_DIR")
    save_agg_path_str = env_get("SAVE_AGG_PATH")

    if save_agg_dir_str:
        save_agg_dir = Path(save_agg_dir_str)
    elif save_agg_path_str:
        # Back-compat: if old single-file path is given, use its directory as SAVE_AGG_DIR
        save_agg_dir = Path(save_agg_path_str).parent
    else:
        raise SystemExit("ERROR: either SAVE_AGG_DIR or SAVE_AGG_PATH must be set in the environment.")

    value_cols_raw = env_get("VALUE_COLS", "Base Point")
    assert value_cols_raw is not None
    value_cols = [c.strip() for c in value_cols_raw.split(",") if c.strip()]
    if not value_cols:
        raise SystemExit("ERROR: VALUE_COLS is empty after parsing.")

    filename_contains = env_get("FILENAME_CONTAINS", "SCED_Gen_Resource_Data")
    timestamp_col = env_get("TIMESTAMP_COLUMN", "SCED Time Stamp") or "SCED Time Stamp"

    log_path_str = env_get("AGG_LOG_PATH")
    log_path = Path(log_path_str) if log_path_str else None

    verbose_env = env_get("VERBOSE", "1")
    verbose = str(verbose_env).strip() not in {"0", "false", "False", ""}

    return Config(
        root_folder=root_folder,
        save_agg_dir=save_agg_dir,
        value_cols=value_cols,
        filename_contains=filename_contains,
        timestamp_col=timestamp_col,
        log_path=log_path,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Logging + file discovery
# ---------------------------------------------------------------------------

def log(msg: str, cfg: Config, *, end: str = "\n") -> None:
    msg = msg.rstrip("\n")
    if cfg.verbose:
        print(msg, end=end, flush=True)
    if cfg.log_path is not None:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def find_input_csvs(cfg: Config) -> List[Path]:
    if not cfg.root_folder.is_dir():
        raise SystemExit(f"ERROR: ROOT_FOLDER {cfg.root_folder} does not exist or is not a directory.")

    all_csvs = list(cfg.root_folder.rglob("*.csv"))
    if cfg.filename_contains:
        needle = cfg.filename_contains.lower()
        csvs = [p for p in all_csvs if needle in p.name.lower()]
    else:
        csvs = all_csvs

    if not csvs:
        raise SystemExit(
            f"ERROR: No CSV files found under {cfg.root_folder} matching '*.csv' "
            f"and filter {cfg.filename_contains!r}."
        )

    return sorted(csvs)


# ---------------------------------------------------------------------------
# Core aggregation helpers
# ---------------------------------------------------------------------------

def detect_timestamp_column(df: pd.DataFrame, preferred: str, fallback_substring: str = "time") -> Optional[str]:
    """Pick the timestamp column: prefer `preferred`, else first col containing 'time'."""
    if preferred in df.columns:
        return preferred
    for col in df.columns:
        if fallback_substring.lower() in str(col).lower():
            return col
    return None


def safe_value_col_name(value_col: str) -> str:
    """Turn 'Base Point' into 'Base_Point', etc., for filenames."""
    out = re.sub(r"[^0-9A-Za-z]+", "_", value_col).strip("_")
    return out or "value"


def aggregate_for_value_col(
    csv_paths: Sequence[Path],
    value_col: str,
    cfg: Config,
) -> tuple[Optional[pd.DataFrame], List[str]]:
    """
    Build a wide matrix for `value_col`:

      index   -> Resource Name
      columns -> SCED timestamps
      values  -> `value_col` values

    Returns (matrix, messages).
    """
    msgs: List[str] = []
    series_list: List[pd.Series] = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            msgs.append(f"[WARN] Failed to read {path.name}: {e}")
            continue

        if "Resource Name" not in df.columns:
            msgs.append(f"[WARN] Skipping {path.name}: missing 'Resource Name' column.")
            continue

        ts_col = detect_timestamp_column(df, cfg.timestamp_col)
        if ts_col is None:
            msgs.append(f"[WARN] Skipping {path.name}: could not find timestamp column.")
            continue

        if value_col not in df.columns:
            msgs.append(f"[WARN] {value_col!r} not found in {path.name}; skipping for this file.")
            continue

        tmp = df[["Resource Name", ts_col, value_col]].copy()
        # Parse timestamps and drop bad rows
        tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
        tmp = tmp[tmp[ts_col].notna()]
        tmp = tmp.dropna(subset=[value_col])

        if tmp.empty:
            msgs.append(f"[INFO] No valid rows for {value_col!r} in {path.name}; all NaN or bad timestamps.")
            continue

        pivot = tmp.pivot_table(
            index="Resource Name",
            columns=ts_col,
            values=value_col,
            aggfunc="mean",  # if duplicates, average
        )

        # Drop all-NaN rows/cols
        pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if pivot.empty:
            msgs.append(f"[INFO] Pivot for {value_col!r} in {path.name} was empty after cleaning.")
            continue

        s = pivot.stack(dropna=True)
        s.name = value_col
        series_list.append(s)

    if not series_list:
        msgs.append(f"[ERROR] No data accumulated for value column {value_col!r}; nothing to aggregate.")
        return None, msgs

    combined = pd.concat(series_list)
    # Deduplicate (Resource Name, timestamp) pairs: keep last seen value
    combined = combined[~combined.index.duplicated(keep="last")]

    matrix = combined.unstack(level=1)

    # Sort by resource then timestamp
    matrix = matrix.sort_index(axis=0)
    try:
        matrix.columns = sorted(matrix.columns)
    except Exception:
        # If timestamps can’t be sorted for some reason, leave them as is
        pass

    return matrix, msgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_config()
    cfg.save_agg_dir.mkdir(parents=True, exist_ok=True)

    log(f"ROOT_FOLDER       = {cfg.root_folder}", cfg)
    log(f"SAVE_AGG_DIR      = {cfg.save_agg_dir}", cfg)
    log(f"VALUE_COLS        = {', '.join(cfg.value_cols)}", cfg)
    log(f"FILENAME_CONTAINS = {cfg.filename_contains}", cfg)
    log(f"TIMESTAMP_COLUMN  = {cfg.timestamp_col}", cfg)
    if cfg.log_path:
        log(f"Logging to: {cfg.log_path}", cfg)

    csv_paths = find_input_csvs(cfg)
    log(f"Discovered {len(csv_paths)} CSV files to process.", cfg)

    for value_col in cfg.value_cols:
        log("", cfg)
        log(f"=== Aggregating column: {value_col!r} ===", cfg)

        matrix, msgs = aggregate_for_value_col(csv_paths, value_col, cfg)
        for m in msgs:
            log(m, cfg)

        if matrix is None or matrix.empty:
            log(f"[WARN] Skipping write for {value_col!r} because the aggregated matrix is empty.", cfg)
            continue

        safe_name = safe_value_col_name(value_col)
        out_path = cfg.save_agg_dir / f"aggregated_{safe_name}.csv"
        matrix.to_csv(out_path, index=True)
        log(f"[OK] Wrote aggregated matrix for {value_col!r} to {out_path}", cfg)

    log("Done.", cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
