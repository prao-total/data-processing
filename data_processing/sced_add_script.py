#!/usr/bin/env python3
"""
add_resource_type.py

- Scans nested ZIPs (outer ZIPs containing inner ZIPs) for CSVs with
  'SCED_Gen_Resource_Data' in their filenames and builds a first-match map
  {Resource Name -> Fuel}.
- Amends target CSVs named 'aggregation_*.csv' (first column = 'Resource Name',
  remaining columns are timestamps) by inserting a 'Resource Type' column
  after 'Resource Name'.
"""

# ==========================================================
# ----------- CONFIGURATION SECTION (EDIT THESE) -----------
# ==========================================================
TARGET_DIR = "/path/to/target_csvs"   # directory with aggregation_*.csv
ZIP_ROOT   = "/path/to/zip_tree"      # directory containing outer ZIPs
OUT_DIR    =  "/path/to/output_csvs"  # where amended CSVs are written
INPLACE    = False                    # True = overwrite target files
LIMIT_ZIPS = None                     # e.g., 3 for quick testing, or None
# ==========================================================

import sys
import csv
import io
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import pandas as pd

RESOURCE_COL_CANDIDATES = [
    "resource name", "resource_name", "resource", "unit", "unit name",
    "ercot_unitcode", "ercot unitcode", "ercot unit code"
]
FUEL_COL_CANDIDATES = [
    "fuel", "fuel type", "fuel_type", "resourcetype", "resource type"
]

TARGET_RESOURCE_COL = "Resource Name"
OUTPUT_FUEL_COL    = "Resource Type"


def best_column(df_columns, candidates):
    lower_map = {c.lower().strip(): c for c in df_columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        for c in df_columns:
            if cand.lower().strip() in c.lower().strip():
                return c
    return None


def read_sced_csv_from_bytes(data: bytes) -> pd.DataFrame | None:
    # Try common delimiters
    for sep in [",", "|", "\t", ";"]:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


def build_resource_fuel_map(zip_root: Path, limit: int | None = None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    processed_outer = 0
    outer_zips = sorted([p for p in zip_root.rglob("*.zip") if p.is_file()])

    for outer_zip_path in outer_zips:
        if limit is not None and processed_outer >= limit:
            break
        processed_outer += 1

        try:
            with ZipFile(outer_zip_path, "r") as outer_z:
                for inner_name in outer_z.namelist():
                    if not inner_name.lower().endswith(".zip"):
                        continue
                    try:
                        inner_bytes = outer_z.read(inner_name)
                    except Exception:
                        continue

                    try:
                        with ZipFile(io.BytesIO(inner_bytes), "r") as inner_z:
                            csv_members = [
                                m for m in inner_z.namelist()
                                if m.lower().endswith(".csv")
                                and "sced_gen_resource_data" in m.lower()
                            ]
                            for csv_name in csv_members:
                                try:
                                    csv_bytes = inner_z.read(csv_name)
                                except Exception:
                                    continue
                                df = read_sced_csv_from_bytes(csv_bytes)
                                if df is None or df.empty:
                                    continue

                                res_col = best_column(df.columns, RESOURCE_COL_CANDIDATES)
                                fuel_col = best_column(df.columns, FUEL_COL_CANDIDATES)
                                if res_col is None or fuel_col is None:
                                    continue

                                res_series = df[res_col].astype(str).str.strip()
                                fuel_series = df[fuel_col].astype(str).str.strip()

                                for r, f in zip(res_series, fuel_series):
                                    if not r or r.lower() in ("nan", "none"):
                                        continue
                                    if not f or f.lower() in ("nan", "none"):
                                        continue
                                    # First match wins
                                    if r not in mapping:
                                        mapping[r] = f
                    except BadZipFile:
                        continue
                    except Exception:
                        continue
        except BadZipFile:
            continue
        except Exception:
            continue

    return mapping


def ensure_resource_name_first(df: pd.DataFrame) -> pd.DataFrame:
    """Rename/position the resource column so that TARGET_RESOURCE_COL is first."""
    if df.columns[0] == TARGET_RESOURCE_COL:
        return df
    # Try to find a column that matches "Resource Name" intent and move/rename it
    guess = best_column(df.columns, [TARGET_RESOURCE_COL] + RESOURCE_COL_CANDIDATES)
    if guess is None:
        raise ValueError(f"Missing '{TARGET_RESOURCE_COL}' or equivalent first column.")
    if guess != TARGET_RESOURCE_COL:
        df = df.rename(columns={guess: TARGET_RESOURCE_COL})
    # Move to front if not already
    cols = list(df.columns)
    cols.remove(TARGET_RESOURCE_COL)
    df = df[[TARGET_RESOURCE_COL] + cols]
    return df


def amend_target_csv(target_path: Path, res2fuel: dict[str, str], inplace: bool, out_dir: Path | None):
    df = pd.read_csv(target_path, dtype=str, low_memory=False)
    if df.empty:
        amended = df
    else:
        df = ensure_resource_name_first(df)

        # Build Resource Type from map; Unknown if not found
        resource_types = df[TARGET_RESOURCE_COL].astype(str).str.strip().map(lambda r: res2fuel.get(r, "Unknown"))

        if OUTPUT_FUEL_COL in df.columns:
            df[OUTPUT_FUEL_COL] = resource_types
            amended = df
        else:
            amended = df.copy()
            amended.insert(1, OUTPUT_FUEL_COL, resource_types)

    # Write
    if inplace:
        amended.to_csv(target_path, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        amended.to_csv(out_dir / target_path.name, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    target_dir = Path(TARGET_DIR)
    zip_root   = Path(ZIP_ROOT)
    out_dir    = Path(OUT_DIR)

    if not target_dir.exists() or not zip_root.exists():
        print("Check your paths at the top of the script.", file=sys.stderr)
        sys.exit(1)

    print("Building Resource->Fuel map from SCED ZIPs...")
    res2fuel = build_resource_fuel_map(zip_root, LIMIT_ZIPS)
    print(f"Found {len(res2fuel)} unique resources with fuels.")

    # ONLY process files named aggregation_*.csv
    target_csvs = sorted(target_dir.glob("aggregation_*.csv"))
    if not target_csvs:
        print(f"No files matching 'aggregation_*.csv' in {target_dir}", file=sys.stderr)
        sys.exit(0)

    amended = 0
    for csv_path in target_csvs:
        try:
            amend_target_csv(csv_path, res2fuel, INPLACE, out_dir)
            amended += 1
            print(f"✓ Amended {csv_path.name}")
        except Exception as e:
            print(f"[WARN] Skipped {csv_path.name}: {e}", file=sys.stderr)

    print(f"Done. Amended {amended}/{len(target_csvs)} files.")


if __name__ == "__main__":
    main()
