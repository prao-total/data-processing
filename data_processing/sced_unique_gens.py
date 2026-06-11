#!/usr/bin/env python
"""
Build a unique SCED resource list from nested ERCOT ZIP files.

Reads ROOT_FOLDER and SAVE_AGG_DIR from .env/environment, scans normal ZIPs and
nested ZIPs for:

  - 60d_SCED_Gen_Resource_Data-*.csv
  - 60d_ESR_Data_in_SCED-*.csv

For all matched CSV rows, writes one summary CSV with each unique
Resource Name / Resource Type pair, its first and final SCED timestamp, and
the total source row count. It also computes average, minimum, and maximum
values for selected SCED offer/cost columns.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


TARGET_FILENAME_PATTERNS = (
    re.compile(r"^60d_SCED_Gen_Resource_Data-.*\.csv$", re.IGNORECASE),
    re.compile(r"^60d_ESR_Data_in_SCED-.*\.csv$", re.IGNORECASE),
)
REQUIRED_COLUMNS = ("Resource Name", "Resource Type", "SCED Time Stamp")
METRIC_COLUMNS = (
    "Base Point",
    "Start Up Cold Offer",
    "Start Up Hot Offer",
    "Start Up Inter Offer",
    "Min Gen Cost",
)
OUTPUT_FILE_NAME = "sced_unique_resource_name_type_pairs.csv"


@dataclass(frozen=True)
class Config:
    root_folder: Path
    save_agg_dir: Path
    output_path: Path
    timestamp_format: Optional[str] = None
    output_timestamp_format: str = "%Y-%m-%d %H:%M"
    csv_encoding: Optional[str] = None
    csv_sep: Optional[str] = None
    verbose: bool = True


def load_env_file_if_needed() -> None:
    if load_dotenv is not None:
        load_dotenv()
        return

    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def env_get(key: str, default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    value = os.getenv(key, default)
    if required and (value is None or str(value).strip() == ""):
        raise SystemExit(f"ERROR: environment variable {key!r} is required but not set.")
    return value


def parse_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "n", ""}


def clean_optional_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned or cleaned.startswith("#"):
        return None
    if " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].rstrip()
    return cleaned or None


def load_config() -> Config:
    load_env_file_if_needed()

    root_folder = Path(env_get("ROOT_FOLDER", required=True) or "").expanduser()
    save_agg_dir = Path(env_get("SAVE_AGG_DIR", required=True) or "").expanduser()

    output_file_name = env_get("SCED_UNIQUE_GENS_OUTPUT_FILE", OUTPUT_FILE_NAME) or OUTPUT_FILE_NAME
    output_path = save_agg_dir / output_file_name

    csv_encoding = clean_optional_env_value(env_get("CSV_ENCODING"))

    csv_sep = clean_optional_env_value(env_get("CSV_SEP"))

    return Config(
        root_folder=root_folder,
        save_agg_dir=save_agg_dir,
        output_path=output_path,
        timestamp_format=env_get("TIMESTAMP_FORMAT"),
        output_timestamp_format=env_get("OUTPUT_TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M") or "%Y-%m-%d %H:%M",
        csv_encoding=csv_encoding,
        csv_sep=csv_sep,
        verbose=parse_bool(env_get("VERBOSE", "true")),
    )


def log(message: str, cfg: Config) -> None:
    if cfg.verbose:
        print(message, flush=True)


def target_filename_matches(name: str) -> bool:
    basename = Path(name).name
    return any(pattern.match(basename) for pattern in TARGET_FILENAME_PATTERNS)


def iter_target_csv_streams_from_zip(
    zf: zipfile.ZipFile,
    root_label: str,
) -> Iterator[tuple[str, BinaryIO]]:
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue

        source_label = f"{root_label}::{name}"
        lower_name = name.lower()

        if lower_name.endswith(".zip"):
            try:
                with zf.open(info) as inner_zip_fp:
                    inner_zip_bytes = inner_zip_fp.read()
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zf:
                    yield from iter_target_csv_streams_from_zip(inner_zf, source_label)
            except zipfile.BadZipFile:
                continue
            continue

        if lower_name.endswith(".csv") and target_filename_matches(name):
            with zf.open(info) as csv_fp:
                yield source_label, csv_fp


def iter_target_csv_streams(root_folder: Path) -> Iterator[tuple[str, BinaryIO]]:
    if not root_folder.is_dir():
        raise SystemExit(f"ERROR: ROOT_FOLDER {root_folder} does not exist or is not a directory.")

    for path in sorted(root_folder.rglob("*")):
        if not path.is_file():
            continue

        lower_name = path.name.lower()

        if lower_name.endswith(".csv") and target_filename_matches(path.name):
            with path.open("rb") as csv_fp:
                yield str(path), csv_fp
            continue

        if lower_name.endswith(".zip"):
            try:
                with zipfile.ZipFile(path) as zf:
                    yield from iter_target_csv_streams_from_zip(zf, str(path))
            except zipfile.BadZipFile:
                continue


def summary_columns(include_internal_metric_columns: bool = True) -> list[str]:
    columns = ["Resource Name", "Resource Type", "first_sced_time_stamp", "final_sced_time_stamp", "row_count"]
    for metric in METRIC_COLUMNS:
        if include_internal_metric_columns:
            columns.extend([f"{metric}_sum", f"{metric}_count"])
        columns.extend([f"{metric}_min", f"{metric}_max"])
    return columns


def output_columns() -> list[str]:
    columns = ["Resource Name", "Resource Type", "first_sced_time_stamp", "final_sced_time_stamp", "row_count"]
    for metric in METRIC_COLUMNS:
        columns.extend([f"{metric}_avg", f"{metric}_min", f"{metric}_max"])
    return columns


def read_required_columns(csv_fp: BinaryIO, source_label: str, cfg: Config) -> Optional[pd.DataFrame]:
    wanted_columns = set(REQUIRED_COLUMNS) | set(METRIC_COLUMNS)
    read_kwargs: dict[str, object] = {
        "usecols": lambda col: str(col).strip() in wanted_columns,
    }
    if cfg.csv_encoding:
        read_kwargs["encoding"] = cfg.csv_encoding
    if cfg.csv_sep and cfg.csv_sep.lower() != "auto":
        read_kwargs["sep"] = cfg.csv_sep
    elif cfg.csv_sep and cfg.csv_sep.lower() == "auto":
        read_kwargs["sep"] = None
        read_kwargs["engine"] = "python"

    try:
        df = pd.read_csv(csv_fp, **read_kwargs)
    except Exception as exc:
        log(f"[WARN] {source_label}: failed to read CSV: {exc}", cfg)
        return None

    df.rename(columns=lambda col: str(col).strip(), inplace=True)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        log(f"[WARN] {source_label}: missing columns {missing_columns}; skipping.", cfg)
        return None

    for metric in METRIC_COLUMNS:
        if metric not in df.columns:
            df[metric] = pd.NA

    return df.loc[:, list(REQUIRED_COLUMNS) + list(METRIC_COLUMNS)]


def summarize_resource_pairs(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Resource Name"] = tmp["Resource Name"].astype("string").str.strip()
    tmp["Resource Type"] = tmp["Resource Type"].astype("string").str.strip()

    tmp = tmp[
        tmp["Resource Name"].notna()
        & tmp["Resource Type"].notna()
        & (tmp["Resource Name"] != "")
        & (tmp["Resource Type"] != "")
    ]

    if cfg.timestamp_format:
        tmp["SCED Time Stamp"] = pd.to_datetime(
            tmp["SCED Time Stamp"],
            format=cfg.timestamp_format,
            errors="coerce",
        )
    else:
        tmp["SCED Time Stamp"] = pd.to_datetime(tmp["SCED Time Stamp"], errors="coerce")

    tmp = tmp[tmp["SCED Time Stamp"].notna()]
    if tmp.empty:
        return pd.DataFrame(columns=summary_columns())

    for metric in METRIC_COLUMNS:
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")

    aggregations: dict[str, tuple[str, str]] = {
        "first_sced_time_stamp": ("SCED Time Stamp", "min"),
        "final_sced_time_stamp": ("SCED Time Stamp", "max"),
        "row_count": ("SCED Time Stamp", "size"),
    }
    for metric in METRIC_COLUMNS:
        aggregations[f"{metric}_sum"] = (metric, "sum")
        aggregations[f"{metric}_count"] = (metric, "count")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")

    return (
        tmp.groupby(["Resource Name", "Resource Type"], dropna=False, as_index=False)
        .agg(**aggregations)
    )


def merge_summary(running: pd.DataFrame, next_summary: pd.DataFrame) -> pd.DataFrame:
    if running.empty:
        return next_summary
    if next_summary.empty:
        return running

    combined = pd.concat([running, next_summary], ignore_index=True)
    aggregations: dict[str, tuple[str, str]] = {
        "first_sced_time_stamp": ("first_sced_time_stamp", "min"),
        "final_sced_time_stamp": ("final_sced_time_stamp", "max"),
        "row_count": ("row_count", "sum"),
    }
    for metric in METRIC_COLUMNS:
        aggregations[f"{metric}_sum"] = (f"{metric}_sum", "sum")
        aggregations[f"{metric}_count"] = (f"{metric}_count", "sum")
        aggregations[f"{metric}_min"] = (f"{metric}_min", "min")
        aggregations[f"{metric}_max"] = (f"{metric}_max", "max")

    return (
        combined.groupby(["Resource Name", "Resource Type"], dropna=False, as_index=False)
        .agg(**aggregations)
    )


def format_output(summary: pd.DataFrame, timestamp_format: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=output_columns())

    output = summary.sort_values(["Resource Name", "Resource Type"], kind="stable").reset_index(drop=True)
    for column in ("first_sced_time_stamp", "final_sced_time_stamp"):
        output[column] = pd.to_datetime(output[column], errors="coerce").dt.strftime(timestamp_format)

    for metric in METRIC_COLUMNS:
        metric_count = pd.to_numeric(output[f"{metric}_count"], errors="coerce")
        metric_sum = pd.to_numeric(output[f"{metric}_sum"], errors="coerce")
        output[f"{metric}_avg"] = metric_sum.div(metric_count.where(metric_count != 0))

    return output.loc[:, output_columns()]


def build_unique_resource_summary(cfg: Config) -> tuple[pd.DataFrame, int]:
    running_summary = pd.DataFrame(columns=summary_columns())
    matched_csv_count = 0

    for source_label, csv_fp in iter_target_csv_streams(cfg.root_folder):
        matched_csv_count += 1
        df = read_required_columns(csv_fp, source_label, cfg)
        if df is None:
            continue

        file_summary = summarize_resource_pairs(df, cfg)
        running_summary = merge_summary(running_summary, file_summary)
        log(f"[INFO] processed {source_label}", cfg)

    return running_summary, matched_csv_count


def main() -> None:
    cfg = load_config()
    cfg.save_agg_dir.mkdir(parents=True, exist_ok=True)

    log(f"ROOT_FOLDER  = {cfg.root_folder}", cfg)
    log(f"SAVE_AGG_DIR = {cfg.save_agg_dir}", cfg)
    log(f"OUTPUT       = {cfg.output_path}", cfg)

    summary, matched_csv_count = build_unique_resource_summary(cfg)
    output = format_output(summary, cfg.output_timestamp_format)
    output.to_csv(cfg.output_path, index=False)

    log(f"[DONE] Matched CSV files: {matched_csv_count}", cfg)
    log(f"[DONE] Unique Resource Name / Resource Type pairs: {len(output)}", cfg)
    log(f"[DONE] Wrote {cfg.output_path}", cfg)


if __name__ == "__main__":
    main()
