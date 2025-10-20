#!/usr/bin/env python3
"""
add_resource_type.py

This script:
1. Scans nested ZIPs (outer ZIPs containing inner ZIPs) for CSVs with
   'SCED_Gen_Resource_Data' in their filenames.
2. Extracts the first-found mapping of {Resource Name -> Fuel}.
3. Amends each target CSV (where first column = 'Resource Name') by adding
   a 'Resource Type' column with the fuel type.
4. Writes amended CSVs to the output directory.
"""

# ==========================================================
# ----------- CONFIGURATION SECTION (EDIT THESE) -----------
# ==========================================================

# Path to your target CSVs (each starting with "Resource Name")
TARGET_DIR = "/path/to/target_csvs"

# Path to the root directory that contains the outer ZIP files
ZIP_ROOT = "/path/to/zip_tree"

# Path where amended CSVs will be written
OUT_DIR = "/path/to/output_csvs"

# If True, overwrites target CSVs directly
INPLACE = False

# Optional: limit the number of outer zips processed for faster testing
LIMIT_ZIPS = None  # e.g. 3 or None for no limit

# ==========================================================
# --------------------- SCRIPT LOGIC -----------------------
# ==========================================================

import os
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
OUTPUT_FUEL_COL = "Resource Type"


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
    for sep in [",", "|", "\t", ";"]:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


def build_resource_fuel_map(zip_root: Path, limit: int | None = None) -> dict[str, str]:
    mapping = {}
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
                        with ZipFile(io.BytesIO(inner_bytes), "r") as inner_z:
                            csv_members = [
                                m for m in inner_z.namelist()
                                if m.lower().endswith(".csv")
                                and "sced_gen_resource_data" in m.lower()
                            ]
                            for csv_name in csv_members:
                                try:
                                    csv_bytes = inner_z.read(csv_name)
                                    df = read_sced_csv_from_bytes(csv_bytes)
                                except Exception:
                                    continue
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
                                    if r not in mapping:
                                        mapping[r] = f
                    except Exception:
                        continue
        except BadZipFile:
            continue
    return mapping


def amend_target_csv(target_path: Path, res2fuel: dict[str, str], inplace: bool, out_dir: Path | None):
    df = pd.read_csv(target_path, dtype=str, low_memory=False)
    if df.empty:
        amended = df
    else:
        first_col = df.columns[0]
        if first_col != TARGET_RESOURCE_COL:
            guess = best_column(df.columns, [TARGET_RESOURCE_COL])
            if guess:
                df = df.rename(columns={guess: TARGET_RESOURCE_COL})
        if TARGET_RESOURCE_COL not in df.columns:
            raise ValueError(f"{target_path.name}: Missing '{TARGET_RESOURCE_COL}' column.")
        resource_types = df[TARGET_RESOURCE_COL].astype(str).str.strip().map(lambda r: res2fuel.get(r, "Unknown"))
        if OUTPUT_FUEL_COL in df.columns:
            df[OUTPUT_FUEL_COL] = resource_types
            amended = df
        else:
            amended = df.copy()
            amended.insert(1, OUTPUT_FUEL_COL, resource_types)
    if inplace:
        amended.to_csv(target_path, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        amended.to_csv(out_dir / target_path.name, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    target_dir = Path(TARGET_DIR)
    zip_root = Path(ZIP_ROOT)
    out_dir = Path(OUT_DIR)

    if not target_dir.exists() or not zip_root.exists():
        print("Check your directory paths at the top of the script!", file=sys.stderr)
        sys.exit(1)

    print("Building Resource->Fuel map from SCED ZIPs...")
    res2fuel = build_resource_fuel_map(zip_root, LIMIT_ZIPS)
    print(f"Found {len(res2fuel)} resource-fuel pairs")

    target_csvs = sorted(target_dir.glob("*.csv"))
    for csv_path in target_csvs:
        try:
            amend_target_csv(csv_path, res2fuel, INPLACE, out_dir)
            print(f"✓ Amended {csv_path.name}")
        except Exception as e:
            print(f"[WARN] Skipped {csv_path.name}: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
