"""
SCED Gen Resource Data Aggregator (combined features)

Features:
- Scans a folder of .zip files (and nested zips) for CSV files whose name contains a pattern.
- Loads those CSVs with robust encoding/sep handling.
- Aggregates into a wide matrix: rows=Resource Name, columns=SCED Time Stamp, values=<VALUE_COL or each VALUE_COLS>.
- Optional passthrough column "Resource Type" (or any name set by RESOURCE_TYPE_COL) inserted after index.
- Supports batching (BATCH_SIZE) across top-level zips.
- Supports multi-output: VALUE_COLS + SAVE_AGG_DIR => one CSV per column (filenames include the column name).
- Legacy single-output: VALUE_COL + SAVE_AGG_PATH.
- Conflict policy: ON_CONFLICT=skip|overwrite.
- Output timestamp header formatting: OUTPUT_TIMESTAMP_FORMAT.

Env keys (examples at bottom of file).
"""

import os
import io
import re
import zipfile
import codecs
from typing import Dict, Optional, Callable, Any, Tuple, List

import pandas as pd
import numpy as np


# =========================
# .env loader (no CLI args)
# =========================

def load_env_file(env_path: str = ".env") -> None:
    """
    Load key=value pairs from a .env file into os.environ.
    Supports inline comments after # and strips surrounding quotes.
    Uses python-dotenv if available; otherwise simple parser.
    """
    # Try python-dotenv if installed
    try:
        from dotenv import load_dotenv as _dotenv_load
        _dotenv_load(env_path, override=False)
        return
    except Exception:
        pass

    # Fallback parser with inline-comment stripping
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                # strip inline comments
                if "#" in val:
                    val = val.split("#", 1)[0]
                val = val.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, val)
    except Exception:
        # best effort
        pass


def env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def env_get_bool(key: str, default: bool = False) -> bool:
    s = os.environ.get(key)
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_get_int(key: str, default: Optional[int] = None) -> Optional[int]:
    s = os.environ.get(key)
    if s is None or str(s).strip() == "":
        return default
    try:
        return int(str(s).strip())
    except Exception:
        return default


def sanitize_encoding(enc: Optional[str]) -> Optional[str]:
    """
    Returns a valid codec name or None.
    Treats '', 'auto', 'none', 'null', 'na' as None.
    Strips inline comments if present (defensive double-sanitize).
    Validates using codecs.lookup.
    """
    if enc is None:
        return None
    enc = enc.strip().strip('"').strip("'")
    if "#" in enc:
        enc = enc.split("#", 1)[0].strip()
    if enc.lower() in {"", "auto", "none", "null", "na"}:
        return None
    try:
        codecs.lookup(enc)
        return enc
    except Exception:
        return None


# =========================
# Zip CSV loader
# =========================

def find_target_csvs_in_nested_zips(
    root_folder: str,
    pattern: str = "SCED_Gen_Resource_Data",
    ignore_case: bool = True,
    csv_read_kwargs: Optional[dict] = None,
    max_depth: Optional[int] = None,
    on_error: Optional[Callable[[Exception, str], None]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Recursively scan a folder of ZIP files (including ZIPs inside ZIPs), find CSV files whose
    filename contains `pattern`, and return a dict of DataFrames.

    Returns (results, matched_count, loaded_count).
    Keys in the returned dict are composite paths:
        outer.zip::inner.zip::path/in/zip/SCED_Gen_Resource_Data-01-JAN-24.csv
    """
    results: Dict[str, pd.DataFrame] = {}
    matched = 0
    loaded = 0

    base_kwargs = (csv_read_kwargs or {}).copy()
    if "sep" not in base_kwargs:
        base_kwargs["sep"] = None  # autodetect
    if "engine" not in base_kwargs and base_kwargs.get("sep", None) is None:
        base_kwargs["engine"] = "python"
    if "dtype_backend" in base_kwargs and base_kwargs["dtype_backend"] is None:
        base_kwargs.pop("dtype_backend")

    if "encoding" in base_kwargs:
        enc = sanitize_encoding(base_kwargs["encoding"])
        if enc is None:
            base_kwargs.pop("encoding", None)
        else:
            base_kwargs["encoding"] = enc

    def log_error(exc: Exception, context: str):
        if on_error is not None:
            try:
                on_error(exc, context)
            except Exception:
                pass
        elif verbose:
            print(f"[WARN] {context} -> {exc}")

    def matches(name: str) -> bool:
        return (pattern.lower() in name.lower() if ignore_case else pattern in name) and name.lower().endswith(".csv")

    def unique_key(proposed_key: str) -> str:
        if proposed_key not in results:
            return proposed_key
        i = 2
        while f"{proposed_key} ({i})" in results:
            i += 1
        return f"{proposed_key} ({i})"

    def read_csv_with_fallbacks(buf: bytes) -> pd.DataFrame:
        # As-is
        try:
            return pd.read_csv(io.BytesIO(buf), **base_kwargs)
        except (UnicodeDecodeError, LookupError):
            pass
        # utf-8, utf-8-sig
        for enc in ("utf-8", "utf-8-sig"):
            try:
                k = {k: v for k, v in base_kwargs.items() if k != "encoding"}
                return pd.read_csv(io.BytesIO(buf), encoding=enc, **k)
            except (UnicodeDecodeError, LookupError):
                continue
        # latin-1 fallback
        k = {k: v for k, v in base_kwargs.items() if k != "encoding"}
        return pd.read_csv(io.BytesIO(buf), encoding="latin-1", **k)

    def process_zipfile_obj(zf: zipfile.ZipFile, composite_prefix: str, depth: int):
        nonlocal matched, loaded
        if max_depth is not None and depth > max_depth:
            return

        for info in zf.infolist():
            if hasattr(info, "is_dir") and info.is_dir():
                continue

            inner_name = info.filename
            inner_key = f"{composite_prefix}::{inner_name}"

            try:
                with zf.open(info, "r") as fh:
                    data = fh.read()

                # Nested zip detection
                is_nested = False
                if inner_name.lower().endswith(".zip"):
                    is_nested = True
                else:
                    try:
                        is_nested = zipfile.is_zipfile(io.BytesIO(data))
                    except Exception:
                        is_nested = False

                if is_nested:
                    try:
                        with zipfile.ZipFile(io.BytesIO(data)) as nested_zf:
                            process_zipfile_obj(nested_zf, inner_key, depth + 1)
                    except Exception as nested_exc:
                        log_error(nested_exc, f"Opening nested zip: {inner_key}")
                    continue

                # CSV match?
                base = os.path.basename(inner_name)
                if matches(base):
                    matched += 1
                    try:
                        df = read_csv_with_fallbacks(data)
                        # Trim header whitespace
                        try:
                            df.rename(columns=lambda c: str(c).strip(), inplace=True)
                        except Exception:
                            pass
                        results[unique_key(inner_key)] = df
                        loaded += 1
                    except Exception as read_exc:
                        log_error(read_exc, f"Reading CSV: {inner_key}")
            except Exception as e:
                log_error(e, f"Reading member: {inner_key}")

    # Top-level scan
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

    return results, matched, loaded


def find_target_csvs_in_given_zips(
    zip_paths: List[str],
    pattern: str = "SCED_Gen_Resource_Data",
    ignore_case: bool = True,
    csv_read_kwargs: Optional[dict] = None,
    max_depth: Optional[int] = None,
    on_error: Optional[Callable[[Exception, str], None]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], int, int]:
    """Same as above but restricted to a provided list of top-level zip paths."""
    results: Dict[str, pd.DataFrame] = {}
    matched = 0
    loaded = 0

    base_kwargs = (csv_read_kwargs or {}).copy()
    if "sep" not in base_kwargs:
        base_kwargs["sep"] = None
    if "engine" not in base_kwargs and base_kwargs.get("sep", None) is None:
        base_kwargs["engine"] = "python"
    if "dtype_backend" in base_kwargs and base_kwargs["dtype_backend"] is None:
        base_kwargs.pop("dtype_backend")

    if "encoding" in base_kwargs:
        enc = sanitize_encoding(base_kwargs["encoding"])
        if enc is None:
            base_kwargs.pop("encoding", None)
        else:
            base_kwargs["encoding"] = enc

    def log_error(exc: Exception, context: str):
        if on_error is not None:
            try:
                on_error(exc, context)
            except Exception:
                pass
        elif verbose:
            print(f"[WARN] {context} -> {exc}")

    def matches(name: str) -> bool:
        return (pattern.lower() in name.lower() if ignore_case else pattern in name) and name.lower().endswith(".csv")

    def unique_key(proposed_key: str) -> str:
        if proposed_key not in results:
            return proposed_key
        i = 2
        while f"{proposed_key} ({i})" in results:
            i += 1
        return f"{proposed_key} ({i})"

    def read_csv_with_fallbacks(buf: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(io.BytesIO(buf), **base_kwargs)
        except (UnicodeDecodeError, LookupError):
            pass
        for enc in ("utf-8", "utf-8-sig"):
            try:
                k = {k: v for k, v in base_kwargs.items() if k != "encoding"}
                return pd.read_csv(io.BytesIO(buf), encoding=enc, **k)
            except (UnicodeDecodeError, LookupError):
                continue
        k = {k: v for k, v in base_kwargs.items() if k != "encoding"}
        return pd.read_csv(io.BytesIO(buf), encoding="latin-1", **k)

    def process_zipfile_obj(zf: zipfile.ZipFile, composite_prefix: str, depth: int):
        nonlocal matched, loaded
        if max_depth is not None and depth > max_depth:
            return

        for info in zf.infolist():
            if hasattr(info, "is_dir") and info.is_dir():
                continue
            inner_name = info.filename
            inner_key = f"{composite_prefix}::{inner_name}"

            try:
                with zf.open(info, "r") as fh:
                    data = fh.read()

                is_nested = False
                if inner_name.lower().endswith(".zip"):
                    is_nested = True
                else:
                    try:
                        is_nested = zipfile.is_zipfile(io.BytesIO(data))
                    except Exception:
                        is_nested = False

                if is_nested:
                    try:
                        with zipfile.ZipFile(io.BytesIO(data)) as nested_zf:
                            process_zipfile_obj(nested_zf, inner_key, depth + 1)
                    except Exception as nested_exc:
                        log_error(nested_exc, f"Opening nested zip: {inner_key}")
                    continue

                base = os.path.basename(inner_name)
                if matches(base):
                    matched += 1
                    try:
                        df = read_csv_with_fallbacks(data)
                        try:
                            df.rename(columns=lambda c: str(c).strip(), inplace=True)
                        except Exception:
                            pass
                        results[unique_key(inner_key)] = df
                        loaded += 1
                    except Exception as read_exc:
                        log_error(read_exc, f"Reading CSV: {inner_key}")
            except Exception as e:
                log_error(e, f"Reading member: {inner_key}")

    for zp in zip_paths:
        if not os.path.isfile(zp) or not zp.lower().endswith(".zip"):
            continue
        top_key = os.path.basename(zp)
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                process_zipfile_obj(zf, top_key, depth=1)
        except Exception as e:
            log_error(e, f"Opening top-level zip: {zp}")

    return results, matched, loaded


# =========================
# Aggregator
# =========================

def aggregate_base_point_matrix(
    dfs: Dict[str, pd.DataFrame],
    resource_col: str = "Resource Name",
    timestamp_col: str = "SCED Time Stamp",
    value_col: str = "Base Point",
    case_insensitive_cols: bool = True,
    trim_resource: bool = True,
    coerce_value_numeric: bool = True,
    timestamp_format: Optional[str] = "%m/%d/%Y %H:%M",
    on_conflict: str = "skip",
    max_messages: Optional[int] = 5000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combine CSV DataFrames into a wide matrix:
        index   = resource_col (e.g., Resource Name)
        columns = timestamp_col (e.g., SCED Time Stamp)
        values  = value_col (e.g., Base Point, HSL, ...)

    Also captures an optional RESOURCE_TYPE_COL (default "Resource Type") and inserts it
    as the first data column after the index.
    """
    messages: List[str] = []

    def log(msg: str):
        if verbose:
            messages.append(msg)

    def norm(s: Any) -> str:
        return str(s).strip().lower()

    def resolve_col(df: pd.DataFrame, desired: str) -> Optional[str]:
        # Direct
        if desired in df.columns:
            return desired
        # Strip + direct
        stripped = {str(c).strip(): c for c in df.columns}
        if desired in stripped:
            return stripped[desired]
        if not case_insensitive_cols:
            return None
        desired_norm = norm(desired)
        candidates = {c: norm(c) for c in df.columns}
        for c, cn in candidates.items():
            if cn == desired_norm:
                return c
        return None

    def parse_timestamps(series: pd.Series) -> pd.Series:
        if timestamp_format:
            parsed = pd.to_datetime(series, errors="coerce", format=timestamp_format)
            if parsed.isna().mean() > 0.5:
                log("[INFO] High NaT rate with provided TIMESTAMP_FORMAT; falling back to general parsing.")
                parsed = pd.to_datetime(series, errors="coerce")
            return parsed
        return pd.to_datetime(series, errors="coerce")

    cells: Dict[Tuple[str, pd.Timestamp], float] = {}
    source_of: Dict[Tuple[str, pd.Timestamp], str] = {}
    # capture a single resource type per resource name
    rtype_map: Dict[str, Any] = {}

    for src_name, df in dfs.items():
        # Ensure header whitespace is trimmed
        try:
            df = df.rename(columns=lambda c: str(c).strip())
        except Exception:
            pass

        rc = resolve_col(df, resource_col) or resource_col
        tc = resolve_col(df, timestamp_col) or timestamp_col
        vc = resolve_col(df, value_col) or value_col

        missing = [c for c in (rc, tc, vc) if c not in df.columns]
        if missing:
            log(f"[WARN] {src_name}: missing required columns (found: {list(df.columns)[:12]}...); skipping.")
            continue

        # Optional resource type column
        rtype_env = env_get("RESOURCE_TYPE_COL", "Resource Type")
        rtype_col = resolve_col(df, rtype_env)

        cols_sel = [rc, tc, vc] + ([rtype_col] if rtype_col else [])
        work = df[cols_sel].copy()

        # Clean resource names
        if trim_resource:
            work[rc] = work[rc].astype(str).str.strip()

        # Parse timestamps
        work[tc] = parse_timestamps(work[tc])

        # Coerce value column to numeric
        if coerce_value_numeric:
            work[vc] = pd.to_numeric(work[vc], errors="coerce")

        # Drop rows missing essentials
        before = len(work)
        work = work.dropna(subset=[rc, tc, vc])
        dropped = before - len(work)
        if dropped > 0:
            log(f"[INFO] {src_name}: dropped {dropped} row(s) with null {rc}/{tc}/{vc}.")

        # Insert with conflict handling; capture resource type
        for row in work.itertuples(index=False, name=None):
            res, ts, val = row[0], row[1], row[2]
            key = (str(res), pd.Timestamp(ts))

            # Record resource type (first non-empty wins)
            if rtype_col and len(row) > 3:
                rt = row[3]
                if rt is not None and str(rt).strip() != "":
                    if res not in rtype_map or pd.isna(rtype_map[res]) or rtype_map[res] in ("", None):
                        rtype_map[res] = rt

            if key not in cells or (pd.isna(cells[key]) and not pd.isna(val)):
                cells[key] = val
                source_of[key] = src_name
                continue

            existing = cells[key]
            if pd.isna(val):
                continue
            if pd.isna(existing):
                cells[key] = val
                source_of[key] = src_name
                continue

            # Both non-null: conflict
            if on_conflict == "skip":
                log(
                    f"[INFO] Duplicate for (Resource='{res}', Time='{ts}') from {src_name}; "
                    f"value already present from {source_of[key]}; skipped."
                )
            elif on_conflict == "overwrite":
                log(
                    f"[WARN] Overwriting (Resource='{res}', Time='{ts}'): "
                    f"{existing} (from {source_of[key]}) -> {val} (from {src_name})."
                )
                cells[key] = val
                source_of[key] = src_name

    if not cells:
        empty = pd.DataFrame(columns=[resource_col]).set_index(resource_col)
        # If any resource types were captured for empty set, attach blank col as placeholder
        return empty, messages

    # Tidy then pivot
    tidy = pd.DataFrame(
        [(r, t, v) for (r, t), v in cells.items()],
        columns=[resource_col, timestamp_col, value_col],
    )
    agg_df = tidy.pivot(index=resource_col, columns=timestamp_col, values=value_col)

    # Attach Resource Type column after index (as first data column)
    if rtype_map:
        rt_series = pd.Series(rtype_map, name=env_get("RESOURCE_TYPE_COL", "Resource Type"))
        rt_aligned = rt_series.reindex(agg_df.index)
        agg_df.insert(0, rt_series.name, rt_aligned)

    # Sort for readability
    try:
        agg_df = agg_df.sort_index(axis=0)
    except Exception:
        pass
    try:
        agg_df = agg_df.sort_index(axis=1)
    except Exception:
        pass

    return agg_df, messages


# =========================
# Main
# =========================

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def safe_name(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))
    return re.sub(r"_+", "_", out).strip("_")


def main():
    # Load env file
    env_file = env_get("ENV_FILE", ".env")
    load_env_file(env_file)

    # Required
    root_folder = env_get("ROOT_FOLDER")
    save_agg_path = env_get("SAVE_AGG_PATH")  # legacy single-output
    save_agg_dir = env_get("SAVE_AGG_DIR")    # multi-output directory
    save_agg_basename = env_get("SAVE_AGG_BASENAME", "aggregation")

    if not root_folder or not os.path.isdir(root_folder):
        raise ValueError("ROOT_FOLDER is missing or not a directory (set it in .env).")
    if not save_agg_path and not save_agg_dir:
        raise ValueError("Provide SAVE_AGG_PATH (single output) or SAVE_AGG_DIR (multiple outputs) in .env.")
    if save_agg_path and not save_agg_path.lower().endswith(".csv"):
        save_agg_path = save_agg_path + ".csv"

    # Common env options
    pattern = env_get("FILENAME_PATTERN", "SCED_Gen_Resource_Data")
    ignore_case = env_get_bool("IGNORE_CASE", True)
    verbose = env_get_bool("VERBOSE", True)
    max_depth = env_get_int("MAX_NESTED_DEPTH", None)

    # CSV reading
    csv_engine = env_get("CSV_ENGINE", "python")
    csv_sep = env_get("CSV_SEP", "auto")
    raw_encoding = env_get("CSV_ENCODING", None)
    csv_encoding = sanitize_encoding(raw_encoding)
    csv_kwargs: Dict[str, Any] = {"engine": csv_engine}
    if csv_sep and csv_sep.lower() != "auto":
        csv_kwargs["sep"] = "\t" if str(csv_sep).strip().lower() in {"\\t", "tab", "tsv"} else csv_sep
    else:
        csv_kwargs["sep"] = None  # autodetect
    if csv_encoding:
        csv_kwargs["encoding"] = csv_encoding

    # Aggregation/env options
    on_conflict = env_get("ON_CONFLICT", "skip").lower()
    if on_conflict not in {"skip", "overwrite"}:
        on_conflict = "skip"

    case_insensitive_cols = env_get_bool("CASE_INSENSITIVE_COLS", True)
    trim_resource = env_get_bool("TRIM_RESOURCE", True)
    coerce_value_numeric = env_get_bool("COERCE_VALUE_NUMERIC", True)
    timestamp_format = env_get("TIMESTAMP_FORMAT", "%m/%d/%Y %H:%M") or None
    output_ts_fmt = env_get("OUTPUT_TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M")

    # Value columns
    value_cols_raw = env_get("VALUE_COLS", "").strip() if env_get("VALUE_COLS") is not None else ""
    if value_cols_raw:
        value_cols = [c.strip() for c in value_cols_raw.split(",") if c.strip()]
    else:
        value_cols = [env_get("VALUE_COL", "Base Point")]

    # Batching
    batch_size = int(env_get("BATCH_SIZE", "0") or "0")
    zip_sort = env_get("ZIP_SORT", "name").lower()  # name | mtime

    # Merge helper
    def merge_wide(existing: Optional[pd.DataFrame], incoming: pd.DataFrame) -> pd.DataFrame:
        if existing is None or len(existing) == 0:
            return incoming.copy()
        idx = existing.index.union(incoming.index)
        cols = existing.columns.union(incoming.columns)
        a = existing.reindex(index=idx, columns=cols)
        b = incoming.reindex(index=idx, columns=cols)
        if on_conflict == "overwrite":
            out = a.copy()
            mask = b.notna()
            out[mask] = b[mask]
            return out
        return a.combine_first(b)

    # Enumerate top-level zips
    all_zips = []
    for entry in os.scandir(root_folder):
        if entry.is_file() and entry.name.lower().endswith(".zip"):
            all_zips.append(entry.path)
    if zip_sort == "mtime":
        all_zips.sort(key=lambda p: os.path.getmtime(p))
    else:
        all_zips.sort(key=lambda p: os.path.basename(p).lower())

    # Prepare per-column aggregation structures
    agg_by_col: Dict[str, Optional[pd.DataFrame]] = {vc: None for vc in value_cols}
    msgs_by_col: Dict[str, List[str]] = {vc: [] for vc in value_cols}
    total_matched = 0
    total_loaded = 0

    def process_dfs_and_merge(dfs: Dict[str, pd.DataFrame]):
        nonlocal total_matched, total_loaded
        if not dfs:
            return
        for vc in value_cols:
            batch_agg, batch_msgs = aggregate_base_point_matrix(
                dfs,
                resource_col=env_get("RESOURCE_COL", "Resource Name"),
                timestamp_col=env_get("TIMESTAMP_COL", "SCED Time Stamp"),
                value_col=vc,
                case_insensitive_cols=case_insensitive_cols,
                trim_resource=trim_resource,
                coerce_value_numeric=coerce_value_numeric,
                timestamp_format=timestamp_format,
                on_conflict=on_conflict,
                max_messages=int(env_get("MAX_MESSAGES", "5000")),
                verbose=verbose,
            )
            agg_by_col[vc] = merge_wide(agg_by_col[vc], batch_agg)
            msgs_by_col[vc].extend(batch_msgs)

    if not batch_size:
        # Single-shot path
        def log_error(exc: Exception, ctx: str):
            if verbose:
                print(f"[WARN] {ctx} -> {exc}")
        dfs, matched, loaded = find_target_csvs_in_nested_zips(
            root_folder=root_folder,
            pattern=pattern,
            ignore_case=ignore_case,
            csv_read_kwargs=csv_kwargs,
            max_depth=max_depth,
            on_error=log_error,
            verbose=verbose,
        )
        total_matched += matched
        total_loaded += loaded
        process_dfs_and_merge(dfs)
    else:
        # Batched over top-level zips
        for i in range(0, len(all_zips), batch_size):
            batch_paths = all_zips[i:i+batch_size]
            if verbose:
                print(f"[INFO] Processing batch {i//batch_size + 1} with {len(batch_paths)} zip(s)")
            dfs, matched, loaded = find_target_csvs_in_given_zips(
                batch_paths,
                pattern=pattern,
                ignore_case=ignore_case,
                csv_read_kwargs=csv_kwargs,
                max_depth=max_depth,
                on_error=None,
                verbose=verbose,
            )
            total_matched += matched
            total_loaded += loaded
            process_dfs_and_merge(dfs)

    # Saving
    def finalize_cols_for_csv(df: pd.DataFrame) -> pd.DataFrame:
        # Keep passthrough Resource Type as the first data column if present
        if df is not None and "Resource Type" in df.columns:
            cols = ["Resource Type"] + [c for c in df.columns if c != "Resource Type"]
            df = df[cols]
        # Format datetime headers
        cols = df.columns
        if isinstance(cols, pd.DatetimeIndex):
            try:
                df.columns = cols.strftime(output_ts_fmt)
            except Exception:
                df.columns = cols.strftime("%Y-%m-%d %H:%M:%S")
        return df

    if len(value_cols) == 1 and save_agg_path:
        vc = value_cols[0]
        ensure_parent_dir(save_agg_path)
        df_out = finalize_cols_for_csv(agg_by_col[vc])
        try:
            df_out.to_csv(save_agg_path, index=True)
            if verbose:
                print(f"[INFO] Saved aggregated matrix to '{save_agg_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to save CSV to '{save_agg_path}': {e}")
    else:
        if not save_agg_dir:
            raise ValueError("Multiple value columns detected. Please set SAVE_AGG_DIR for per-column outputs.")
        os.makedirs(save_agg_dir, exist_ok=True)
        base = save_agg_basename or "aggregation"
        for vc in value_cols:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(vc))
            safe = re.sub(r"_+", "_", safe).strip("_")
            out_path = os.path.join(save_agg_dir, f"{base}_{safe}.csv")
            ensure_parent_dir(out_path)
            df_out = finalize_cols_for_csv(agg_by_col[vc])
            try:
                df_out.to_csv(out_path, index=True)
                if verbose:
                    print(f"[INFO] Saved aggregated matrix for '{vc}' to '{out_path}'")
            except Exception as e:
                raise RuntimeError(f"Failed to save CSV to '{out_path}': {e}")

    # Optional aggregation log (combined)
    agg_log_path = env_get("AGG_LOG_PATH", None)
    if agg_log_path:
        ensure_parent_dir(agg_log_path)
        try:
            with open(agg_log_path, "w", encoding="utf-8") as fh:
                fh.write("=== Aggregation summary ===\n")
                fh.write(f"Matched CSVs: {total_matched}\nLoaded CSVs: {total_loaded}\n")
                fh.write(f"Value columns: {', '.join(value_cols)}\n")
                for vc in value_cols:
                    fh.write(f"\n--- Messages for '{vc}' ---\n")
                    for m in msgs_by_col[vc]:
                        fh.write(m + "\n")
            if verbose:
                print(f"[INFO] Saved aggregation log to '{agg_log_path}'")
        except Exception as e:
            print(f"[WARN] Failed to save log '{agg_log_path}': {e}")


if __name__ == "__main__":
    main()