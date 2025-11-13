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

import csv
import io
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# =========================
# Env helpers
# =========================

def env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key, default)
    if v is None:
        return None
    return str(v).strip() if isinstance(v, str) else str(v)


def env_get_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, None)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def load_env_file(path: str) -> None:
    """
    Load key=value pairs from a .env style file into os.environ (without overriding existing).
    """
    if not path:
        return
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key and key not in os.environ:
                os.environ[key] = val


# =========================
# CSV loading helpers
# =========================

@dataclass
class CsvLoadResult:
    df: Optional[pd.DataFrame]
    msg: str


def try_read_csv(
    file_like: io.BytesIO,
    path_ctx: str,
    csv_read_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> CsvLoadResult:
    """
    Attempt to read CSV using a couple of strategies:
    - user-provided kwargs (if any),
    - default utf-8, sep=',',
    - fallback engine/encoding combos.
    """
    csv_read_kwargs = csv_read_kwargs or {}
    attempts = []

    def log(msg: str):
        if verbose:
            print(msg)

    # Strategy 1: user-provided kwargs first
    attempts.append(("user_kwargs", csv_read_kwargs.copy()))

    # Strategy 2: simple defaults
    attempts.append(("utf8_comma", {"encoding": "utf-8", "sep": ","}))

    # Strategy 3: latin-1 fallback
    attempts.append(("latin1_comma", {"encoding": "latin-1", "sep": ","}))

    for label, kwargs in attempts:
        try:
            file_like.seek(0)
            df = pd.read_csv(file_like, **kwargs)
            return CsvLoadResult(df=df, msg=f"[INFO] Loaded CSV ({label}) from {path_ctx}")
        except MemoryError as e:
            log(f"[WARN] Reading CSV: {path_ctx} -> {e}")
            return CsvLoadResult(df=None, msg=f"[WARN] MemoryError while reading {path_ctx}: {e}")
        except Exception as e:
            log(f"[WARN] Reading CSV ({label}): {path_ctx} -> {e}")

    return CsvLoadResult(df=None, msg=f"[ERROR] Failed to read CSV after multiple attempts: {path_ctx}")


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
    filename contains `pattern`, and return a dict of DataFrames keyed by a unique path.

    Returns:
        dfs, matched_count, loaded_count
    """
    pattern_cmp = pattern.lower() if ignore_case and pattern else pattern
    dfs: Dict[str, pd.DataFrame] = {}
    matched = 0
    loaded = 0

    def log(msg: str):
        if verbose:
            print(msg)

    def name_matches(name: str) -> bool:
        if not pattern_cmp:
            return True
        if ignore_case:
            return pattern_cmp in name.lower()
        return pattern in name

    def walk_zip(zf: zipfile.ZipFile, parent_ctx: str, depth: int):
        nonlocal matched, loaded
        if max_depth is not None and depth > max_depth:
            return

        for info in zf.infolist():
            name = info.filename
            ctx = f"{parent_ctx}::{name}"
            if info.is_dir():
                continue
            if name.lower().endswith(".zip"):
                # Nested zip
                try:
                    with zf.open(info) as nested_bytes:
                        nested_data = nested_bytes.read()
                    with zipfile.ZipFile(io.BytesIO(nested_data)) as nested_zf:
                        walk_zip(nested_zf, ctx, depth + 1)
                except Exception as e:
                    if on_error:
                        on_error(e, ctx)
                    else:
                        log(f"[WARN] Failed to open nested zip {ctx}: {e}")
                continue
            if not name.lower().endswith(".csv"):
                continue
            if not name_matches(name):
                continue

            matched += 1
            try:
                with zf.open(info) as csv_bytes:
                    data = csv_bytes.read()
                bio = io.BytesIO(data)
                res = try_read_csv(bio, ctx, csv_read_kwargs=csv_read_kwargs, verbose=verbose)
                if res.df is not None:
                    dfs[ctx] = res.df
                    loaded += 1
                    if verbose:
                        print(res.msg)
            except Exception as e:
                if on_error:
                    on_error(e, ctx)
                else:
                    log(f"[WARN] Failed to load CSV {ctx}: {e}")

    # Scan top-level zips
    for entry in os.scandir(root_folder):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(".zip"):
            continue
        zip_path = entry.path
        ctx_root = zip_path
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                walk_zip(zf, ctx_root, depth=1)
        except Exception as e:
            if on_error:
                on_error(e, ctx_root)
            else:
                log(f"[WARN] Failed to open zip {ctx_root}: {e}")

    return dfs, matched, loaded


def find_target_csvs_in_given_zips(
    zip_paths: Iterable[str],
    pattern: str = "SCED_Gen_Resource_Data",
    ignore_case: bool = True,
    csv_read_kwargs: Optional[dict] = None,
    max_depth: Optional[int] = None,
    on_error: Optional[Callable[[Exception, str], None]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Variant of find_target_csvs_in_nested_zips that only scans a given list of zip file paths.
    """
    pattern_cmp = pattern.lower() if ignore_case and pattern else pattern
    dfs: Dict[str, pd.DataFrame] = {}
    matched = 0
    loaded = 0

    def log(msg: str):
        if verbose:
            print(msg)

    def name_matches(name: str) -> bool:
        if not pattern_cmp:
            return True
        if ignore_case:
            return pattern_cmp in name.lower()
        return pattern in name

    def walk_zip(zf: zipfile.ZipFile, parent_ctx: str, depth: int):
        nonlocal matched, loaded
        if max_depth is not None and depth > max_depth:
            return

        for info in zf.infolist():
            name = info.filename
            ctx = f"{parent_ctx}::{name}"
            if info.is_dir():
                continue
            if name.lower().endswith(".zip"):
                try:
                    with zf.open(info) as nested_bytes:
                        nested_data = nested_bytes.read()
                    with zipfile.ZipFile(io.BytesIO(nested_data)) as nested_zf:
                        walk_zip(nested_zf, ctx, depth + 1)
                except Exception as e:
                    if on_error:
                        on_error(e, ctx)
                    else:
                        log(f"[WARN] Failed to open nested zip {ctx}: {e}")
                continue
            if not name.lower().endswith(".csv"):
                continue
            if not name_matches(name):
                continue

            matched += 1
            try:
                with zf.open(info) as csv_bytes:
                    data = csv_bytes.read()
                bio = io.BytesIO(data)
                res = try_read_csv(bio, ctx, csv_read_kwargs=csv_read_kwargs, verbose=verbose)
                if res.df is not None:
                    dfs[ctx] = res.df
                    loaded += 1
                    if verbose:
                        print(res.msg)
            except Exception as e:
                if on_error:
                    on_error(e, ctx)
                else:
                    log(f"[WARN] Failed to load CSV {ctx}: {e}")

    for zip_path in zip_paths:
        ctx_root = zip_path
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                walk_zip(zf, ctx_root, depth=1)
        except Exception as e:
            if on_error:
                on_error(e, ctx_root)
            else:
                log(f"[WARN] Failed to open zip {ctx_root}: {e}")

    return dfs, matched, loaded


# =========================
# Aggregation logic
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
        columns = timestamp_col (e.g., SCED Time Stamp, parsed as datetime)
        values  = value_col (e.g., Base Point)

    The main logic:
        1) Normalize column names (case-insensitive if requested).
        2) Extract (resource, timestamp, value) tuples into a dict.
        3) Build a tidy DataFrame from these tuples.
        4) Pivot to wide (resource x timestamp).
        5) Attach a passthrough Resource Type column if present.

    Note: On large batches, this can be memory heavy if called with many dfs at once.
    In streaming mode we typically call it on *small groups* or single CSVs.
    """
    messages: List[str] = []

    def log(msg: str):
        if verbose and (max_messages is None or len(messages) < max_messages):
            messages.append(msg)
            if verbose:
                print(msg)

    def resolve_col(df: pd.DataFrame, col_name: str) -> Optional[str]:
        cols = list(df.columns)
        if case_insensitive_cols:
            lower_map = {str(c).lower().strip(): c for c in cols}
            return lower_map.get(str(col_name).lower().strip(), None)
        return col_name if col_name in cols else None

    def parse_ts(series: pd.Series) -> pd.Series:
        if timestamp_format:
            try:
                return pd.to_datetime(series, format=timestamp_format, errors="coerce")
            except Exception:
                pass
        return pd.to_datetime(series, errors="coerce")

    # Global dict of (resource, timestamp) -> value, with conflict resolution
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
            try:
                work[rc] = work[rc].astype(str).str.strip()
            except Exception:
                pass

        # Parse timestamps
        work[tc] = parse_ts(work[tc])

        # Coerce numeric
        if coerce_value_numeric:
            work[vc] = pd.to_numeric(work[vc], errors="coerce")

        before = len(work)
        # Drop rows without resource/timestamp/value
        work = work.dropna(subset=[rc, tc, vc])
        dropped = before - len(work)
        if dropped > 0:
            log(f"[INFO] {src_name}: dropped {dropped} row(s) with null {rc}/{tc}/{vc}.")

        # Track resource type (first non-null wins)
        if rtype_col and rtype_col in work.columns:
            r_g = (
                work[[rc, rtype_col]]
                .dropna(subset=[rc])
                .drop_duplicates(subset=[rc], keep="first")
            )
            for rname, rtype_val in zip(r_g[rc].tolist(), r_g[rtype_col].tolist()):
                if rname not in rtype_map or (rtype_map[rname] in (None, "") and pd.notna(rtype_val)):
                    if pd.notna(rtype_val) and str(rtype_val).strip() != "":
                        rtype_map[rname] = rtype_val

        # Accumulate into cells
        for row in work.itertuples(index=False):
            rname = getattr(row, rc)
            tstamp = getattr(row, tc)
            val = getattr(row, vc)
            if pd.isna(tstamp) or pd.isna(val):
                continue
            key = (rname, tstamp)
            if key in cells:
                # Resolve conflict based on policy
                if on_conflict == "overwrite":
                    cells[key] = val
                    source_of[key] = src_name
            else:
                cells[key] = val
                source_of[key] = src_name

    if not cells:
        # No data aggregated
        return pd.DataFrame(), messages

    # Build tidy DataFrame from cells
    tidy_rows = []
    for (rname, tstamp), val in cells.items():
        tidy_rows.append((rname, tstamp, val))
    tidy = pd.DataFrame(tidy_rows, columns=[resource_col, timestamp_col, value_col])

    # Pivot to wide
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
        raise ValueError("SAVE_AGG_PATH must end with .csv")

    # Config
    pattern = env_get("CSV_PATTERN", "SCED_Gen_Resource_Data")
    ignore_case = env_get_bool("IGNORE_CASE", True)
    on_conflict = env_get("ON_CONFLICT", "skip").lower()
    if on_conflict not in ("skip", "overwrite"):
        raise ValueError("ON_CONFLICT must be 'skip' or 'overwrite'")
    max_depth_raw = env_get("MAX_DEPTH", None)
    max_depth = int(max_depth_raw) if max_depth_raw not in (None, "", "None") else None
    case_insensitive_cols = env_get_bool("CASE_INSENSITIVE_COLS", True)
    trim_resource = env_get_bool("TRIM_RESOURCE", True)
    coerce_value_numeric = env_get_bool("COERCE_VALUE_NUMERIC", True)
    timestamp_format = env_get("TIMESTAMP_FORMAT", "%m/%d/%Y %H:%M") or None
    output_ts_fmt = env_get("OUTPUT_TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M")

    # Value columns
    value_cols_raw = env_get("VALUE_COLS", "").strip()
    value_col_single = env_get("VALUE_COL", "").strip()
    if value_cols_raw:
        value_cols = [v.strip() for v in value_cols_raw.split(",") if v.strip()]
    elif value_col_single:
        value_cols = [value_col_single]
    else:
        # Default
        value_cols = ["Base Point"]

    verbose = env_get_bool("VERBOSE", True)

    # Batching
    batch_size = int(env_get("BATCH_SIZE", "0") or "0")
    zip_sort = env_get("ZIP_SORT", "name").lower()  # name | mtime


    # Decide output path(s) for each value column (streaming mode)
    if len(value_cols) == 1 and save_agg_path:
        # Legacy single-output: one CSV for the single metric
        out_paths: Dict[str, str] = {value_cols[0]: save_agg_path}
    else:
        if not save_agg_dir:
            raise ValueError(
                "Multiple value columns detected. Please set SAVE_AGG_DIR for per-column outputs."
            )
        os.makedirs(save_agg_dir, exist_ok=True)
        base = save_agg_basename or "aggregation"
        out_paths = {}
        for vc in value_cols:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(vc))
            safe = re.sub(r"_+", "_", safe).strip("_")
            out_paths[vc] = os.path.join(save_agg_dir, f"{base}_{safe}.csv")

    # Merge helper (kept for compatibility; not used in streaming mode)
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

    # We no longer keep big agg_by_col matrices in memory – everything streams to disk.
    agg_by_col: Dict[str, Optional[pd.DataFrame]] = {vc: None for vc in value_cols}
    msgs_by_col: Dict[str, List[str]] = {vc: [] for vc in value_cols}
    total_matched = 0
    total_loaded = 0

    def process_dfs_and_merge(dfs: Dict[str, pd.DataFrame]):
        """Process each CSV immediately and stream results to disk.

        For each DataFrame in `dfs` and for each value column:
          - Aggregate into a wide matrix for that single file
          - Merge into the on-disk CSV for that metric
        """
        nonlocal total_matched, total_loaded
        if not dfs:
            return

        for src_name, df in dfs.items():
            # Note: matched/loaded counters are already maintained by the
            # zip-scanning helpers; we don't adjust them here.
            single = {src_name: df}
            for vc in value_cols:
                batch_agg, batch_msgs = aggregate_base_point_matrix(
                    single,
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
                msgs_by_col[vc].extend(batch_msgs)
                merge_and_save_incremental(vc, batch_agg)
    if not batch_size:
        # Single-shot path
        def log_error(exc: Exception, ctx: str):
            if verbose:
                print(f"[WARN] {ctx} -> {exc}")
        dfs, matched, loaded = find_target_csvs_in_nested_zips(
            root_folder=root_folder,
            pattern=pattern,
            ignore_case=ignore_case,
            csv_read_kwargs={"engine": "c"},
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
                zip_paths=batch_paths,
                pattern=pattern,
                ignore_case=ignore_case,
                csv_read_kwargs={"engine": "c"},
                max_depth=max_depth,
                on_error=None,
                verbose=verbose,
            )
            total_matched += matched
            total_loaded += loaded
            process_dfs_and_merge(dfs)

    def finalize_cols_for_csv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleanup for writing CSV:
        - Keep "Resource Type" as the first column if present.
        - Convert any Timestamp column labels using OUTPUT_TIMESTAMP_FORMAT.
        """
        if df is None or df.empty:
            return df

        if "Resource Type" in df.columns:
            cols = ["Resource Type"] + [c for c in df.columns if c != "Resource Type"]
            df = df[cols]

        # Convert datetime-like column labels to strings (even if not a DatetimeIndex)
        new_cols = []
        for c in df.columns:
            if isinstance(c, pd.Timestamp):
                try:
                    new_cols.append(c.strftime(output_ts_fmt))
                except Exception:
                    new_cols.append(c.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                new_cols.append(c)
        df.columns = new_cols
        return df

    def merge_and_save_incremental(vc: str, batch_agg: pd.DataFrame) -> None:
        """
        Merge a per-file wide matrix for value column `vc` into the on-disk
        aggregated CSV.

        - If no data in this batch: no-op.
        - If output doesn't exist yet: write fresh file.
        - If it exists: union index/columns and respect ON_CONFLICT policy.
        """
        if batch_agg is None or batch_agg.empty:
            return

        out_path = out_paths[vc]
        ensure_parent_dir(out_path)

        df_new = finalize_cols_for_csv(batch_agg)

        if os.path.exists(out_path):
            df_old = pd.read_csv(out_path, index_col=0)

            # Align index & columns
            all_index = df_old.index.union(df_new.index)
            all_cols = df_old.columns.union(df_new.columns)

            df_old = df_old.reindex(index=all_index, columns=all_cols)
            df_new = df_new.reindex(index=all_index, columns=all_cols)

            if on_conflict == "overwrite":
                mask = df_new.notna()
                df_old[mask] = df_new[mask]
            else:  # "skip"
                mask = df_old.isna() & df_new.notna()
                df_old[mask] = df_new[mask]

            df_out = df_old
        else:
            df_out = df_new

        df_out.to_csv(out_path, index=True)
        if verbose:
            print(f"[INFO] Incrementally updated '{vc}' -> '{out_path}'")

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