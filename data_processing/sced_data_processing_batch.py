#!/usr/bin/env python
"""
SCED data processing: Option A aggregator with nested ZIP support.

Reads ERCOT 60-day SCED generator data stored as:
- Plain CSV files, and/or
- ZIP files containing CSVs, and possibly nested ZIPs

For each value column in VALUE_COLS (e.g. "Base Point, HSL"),
builds an "aggregated matrix" CSV:

  rows    -> Resource Name
  columns -> SCED timestamps
  values  -> value column (e.g. Base Point or HSL)

Env variables (in .env or environment):

  ROOT_FOLDER       (required)  -> root directory of raw SCED data (zips & csvs)
  SAVE_AGG_DIR      (preferred) -> directory to write aggregated CSVs into
  SAVE_AGG_PATH     (legacy)    -> if set, its *directory* is used as SAVE_AGG_DIR
  VALUE_COLS                    -> comma-separated list of columns, e.g. "Base Point, HSL"
  FILENAME_CONTAINS             -> optional substring filter for filenames
                                   default: "SCED_Gen_Resource_Data"
  TIMESTAMP_COLUMN              -> optional preferred timestamp column
                                   default: "SCED Time Stamp"
  AGG_LOG_PATH                  -> optional log file path
  VERBOSE                       -> "0"/"false" to silence stdout logs (default on)

Usage:

  python sced_data_processing.py
"""

from __future__ import annotations

import io
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

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
    if load_dotenv is not None:
        load_dotenv()

    root_folder_str = env_get("ROOT_FOLDER", required=True)
    root_folder = Path(root_folder_str)  # type: ignore[arg-type]

    save_agg_dir_str = env_get("SAVE_AGG_DIR")
    save_agg_path_str = env_get("SAVE_AGG_PATH")

    if save_agg_dir_str:
        save_agg_dir = Path(save_agg_dir_str)
    elif save_agg_path_str:
        # backwards compatible: use parent directory of single-file path
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
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, cfg: Config, *, end: str = "\n") -> None:
    msg = msg.rstrip("\n")
    if cfg.verbose:
        print(msg, end=end, flush=True)
    if cfg.log_path is not None:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Input discovery: directories + nested ZIPs
# ---------------------------------------------------------------------------

def detect_timestamp_column(df: pd.DataFrame, preferred: str, fallback_substring: str = "time") -> Optional[str]:
    """Pick the timestamp column: prefer `preferred`, else first col containing 'time'."""
    if preferred in df.columns:
        return preferred
    for col in df.columns:
        if fallback_substring.lower() in str(col).lower():
            return col
    return None


def _filename_matches(name: str, needle: Optional[str]) -> bool:
    if needle is None or needle == "":
        return True
    return needle.lower() in name.lower()


def _iter_dataframes_from_zip(
    zf: zipfile.ZipFile,
    root_label: str,
    needle: Optional[str],
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Recursively iterate through ZIP contents.

    Yields (source_label, df) for every CSV whose name contains `needle`,
    including CSVs that live inside nested ZIPs.
    """
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue

        lower = name.lower()
        label_prefix = f"{root_label}::{name}"

        # Nested ZIP: recurse
        if lower.endswith(".zip"):
            try:
                with zf.open(info) as inner_fp:
                    data = inner_fp.read()
                with zipfile.ZipFile(io.BytesIO(data)) as inner_zf:
                    yield from _iter_dataframes_from_zip(inner_zf, label_prefix, needle)
            except Exception as e:
                # Just log / skip bad nested zips
                # We don't have cfg here, so caller should handle logging
                continue
        elif lower.endswith(".csv") and _filename_matches(name, needle):
            try:
                with zf.open(info) as csv_fp:
                    df = pd.read_csv(csv_fp)

                df.rename(columns=lambda c: str(c).strip(), inplace=True)

                # OPTIONAL: debug print of columns
                # print(f"[DEBUG] Columns in nested CSV {label_prefix}: {list(df.columns)}")

                yield label_prefix, df
            except Exception:
                # Caller logs; skip bad CSVs
                continue



def iter_input_dataframes(cfg: Config) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Iterate over all input dataframes under ROOT_FOLDER, including:

      - CSV files on disk whose names contain filename_contains
      - CSV files inside ZIPs, including nested ZIPs, whose names contain filename_contains

    Yields (source_label, df).
    """
    if not cfg.root_folder.is_dir():
        raise SystemExit(f"ERROR: ROOT_FOLDER {cfg.root_folder} does not exist or is not a directory.")

    needle = cfg.filename_contains

    for path in cfg.root_folder.rglob("*"):
        if not path.is_file():
            continue

        lower = path.name.lower()

        # Plain CSVs
        if lower.endswith(".csv") and _filename_matches(path.name, needle):
            try:
                df = pd.read_csv(path)

                # NEW: trim whitespace from all column names
                df.rename(columns=lambda c: str(c).strip(), inplace=True)

                # OPTIONAL: debug print of columns
                # print(f"[DEBUG] Columns in CSV {path}: {list(df.columns)}")

            except Exception:
                # Caller logs; skip bad file
                continue
            yield str(path), df


        # ZIPs (possibly with nested zips)
        elif lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    for label, df in _iter_dataframes_from_zip(zf, str(path), needle):
                        yield label, df
            except Exception:
                # Skip corrupt or unreadable zips
                continue


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def safe_value_col_name(value_col: str) -> str:
    """Turn 'Base Point' into 'Base_Point', etc., for filenames."""
    out = re.sub(r"[^0-9A-Za-z]+", "_", value_col).strip("_")
    return out or "value"


def aggregate_for_value_col(
    value_col: str,
    cfg: Config,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Build a wide matrix for `value_col`:

      index   -> Resource Name
      columns -> SCED timestamps
      values  -> `value_col` values

    We *stream* over all input dataframes under ROOT_FOLDER (including nested zips).

    Returns (matrix, messages).
    """
    msgs: List[str] = []
    series_list: List[pd.Series] = []

    for source_label, df in iter_input_dataframes(cfg):
        # NEW: ensure trimmed column names here as well
        df.rename(columns=lambda c: str(c).strip(), inplace=True)

        # Basic sanity checks
        if "Resource Name" not in df.columns:
            msgs.append(f"[WARN] {source_label}: missing 'Resource Name' column; skipping.")
            continue


        ts_col = detect_timestamp_column(df, cfg.timestamp_col)
        if ts_col is None:
            msgs.append(f"[WARN] {source_label}: could not find timestamp column; skipping.")
            continue

        if value_col not in df.columns:
            # Not an error; file just doesn't have this metric
            msgs.append(f"[INFO] {source_label}: value column {value_col!r} not found; skipping.")
            continue

        tmp = df[["Resource Name", ts_col, value_col]].copy()

        # Parse timestamps & clean
        tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
        tmp = tmp[tmp[ts_col].notna()]
        tmp = tmp.dropna(subset=[value_col])

        if tmp.empty:
            msgs.append(f"[INFO] {source_label}: no valid rows for {value_col!r} after cleaning.")
            continue

        pivot = tmp.pivot_table(
            index="Resource Name",
            columns=ts_col,
            values=value_col,
            aggfunc="mean",  # average duplicates within file
        )

        # Drop all-NaN rows/columns
        pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if pivot.empty:
            msgs.append(f"[INFO] {source_label}: pivot for {value_col!r} is empty after dropna.")
            continue

        s = pivot.stack(dropna=True)
        s.name = value_col
        series_list.append(s)

    if not series_list:
        msgs.append(f"[ERROR] No data accumulated for value column {value_col!r}; nothing to aggregate.")
        return None, msgs

    combined = pd.concat(series_list)

    # Deduplicate (Resource Name, timestamp) pairs: keep last seen
    combined = combined[~combined.index.duplicated(keep="last")]

    matrix = combined.unstack(level=1)

    # Sort
    matrix = matrix.sort_index(axis=0)
    try:
        matrix.columns = sorted(matrix.columns)
    except Exception:
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

    for value_col in cfg.value_cols:
        log("", cfg)
        log(f"=== Aggregating column: {value_col!r} ===", cfg)

        matrix, msgs = aggregate_for_value_col(value_col, cfg)
        for m in msgs:
            log(m, cfg)

        if matrix is None or matrix.empty:
            log(f"[WARN] Skipping write for {value_col!r} because the aggregated matrix is empty.", cfg)
            continue

        # ------------------------------------------------------------------
        # NEW: Attach Resource Type column
        # ------------------------------------------------------------------
        if "Resource Name" in matrix.index.names:
            type_series_list = []

            for source_label, df in iter_input_dataframes(cfg):
                if "Resource Name" in df.columns and "Resource Type" in df.columns:
                    tmp = df.drop_duplicates("Resource Name")[["Resource Name", "Resource Type"]]
                    tmp = tmp.set_index("Resource Name")["Resource Type"]
                    type_series_list.append(tmp)

            if type_series_list:
                combined_types = pd.concat(type_series_list)
                combined_types = combined_types[~combined_types.index.duplicated(keep="last")]

                # Align to matrix index (matching resource names)
                rt = combined_types.reindex(matrix.index)

                # Insert Resource Type as first column
                matrix.insert(0, "Resource Type", rt)

        # ------------------------------------------------------------------

        safe_name = safe_value_col_name(value_col)
        out_path = cfg.save_agg_dir / f"aggregated_{safe_name}.csv"
        matrix.to_csv(out_path, index=True)
        log(f"[OK] Wrote aggregated matrix for {value_col!r} to {out_path}", cfg)
        log(f"      Shape: {matrix.shape[0]} resources x {matrix.shape[1]} columns", cfg)

    log("Done.", cfg)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
