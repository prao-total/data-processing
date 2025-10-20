#!/usr/bin/env python3
"""
merge_resource_type_hardcoded.py

Reads:
  - TARGET_PATH: CSV with columns ['Resource Name', 'Resource Type'] to build a mapping
  - INPUT_PATH:  CSV to amend (must contain 'Resource Name')

Writes:
  - OUTPUT_PATH: INPUT + 'Resource Type' inserted immediately after 'Resource Name'
"""

# ================== EDIT THESE PATHS ==================
TARGET_PATH = "/path/to/target_mapping.csv"   # has 'Resource Name' and 'Resource Type'
INPUT_PATH  = "/path/to/input.csv"            # has 'Resource Name' + other columns
OUTPUT_PATH = "/path/to/output.csv"           # destination file
# Optional: fail if any Resource Name is unmapped
HARD_FAIL_ON_UNKNOWN = False
# =====================================================

import csv
from pathlib import Path
import pandas as pd
import sys

TARGET_RESOURCE_COL = "Resource Name"
OUTPUT_FUEL_COL    = "Resource Type"


def load_mapping(target_path: Path) -> dict[str, str]:
    """Load mapping: 'Resource Name' -> 'Resource Type' (first occurrence wins)."""
    df = pd.read_csv(target_path, dtype=str, low_memory=False)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    if TARGET_RESOURCE_COL.lower() not in cols_lower or OUTPUT_FUEL_COL.lower() not in cols_lower:
        raise ValueError(
            f"{target_path}: must contain columns '{TARGET_RESOURCE_COL}' and '{OUTPUT_FUEL_COL}'."
        )
    rcol = cols_lower[TARGET_RESOURCE_COL.lower()]
    fcol = cols_lower[OUTPUT_FUEL_COL.lower()]

    sub = (
        df[[rcol, fcol]]
        .dropna(how="any")
        .astype(str)
        .apply(lambda s: s.str.strip())
        .drop_duplicates(subset=[rcol], keep="first")
    )
    return dict(zip(sub[rcol], sub[fcol]))


def add_resource_type(input_path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    """Return input with 'Resource Type' inserted after 'Resource Name'."""
    df = pd.read_csv(input_path, dtype=str, low_memory=False)

    cols_lower = {c.lower().strip(): c for c in df.columns}
    if TARGET_RESOURCE_COL.lower() not in cols_lower:
        raise ValueError(f"{input_path}: missing required '{TARGET_RESOURCE_COL}' column.")
    rcol = cols_lower[TARGET_RESOURCE_COL.lower()]

    names = df[rcol].astype(str).str.strip()
    types = names.map(lambda x: mapping.get(x, None))

    # Optionally hard-fail if any unknowns
    if HARD_FAIL_ON_UNKNOWN and types.isna().any():
        unknowns = names[types.isna()].dropna().unique().tolist()
        raise ValueError(f"{len(unknowns)} resource(s) had no mapping. Examples: {unknowns[:10]}")

    # Default unknowns to 'Unknown'
    types = types.fillna("Unknown")

    # Replace if already present; else insert immediately after Resource Name
    if OUTPUT_FUEL_COL in df.columns:
        df[OUTPUT_FUEL_COL] = types
        return df

    cols = list(df.columns)
    insert_at = cols.index(rcol) + 1
    out = df.copy()
    out.insert(insert_at, OUTPUT_FUEL_COL, types)
    return out


def main():
    target = Path(TARGET_PATH)
    inp    = Path(INPUT_PATH)
    out    = Path(OUTPUT_PATH)

    if not target.exists():
        print(f"Target mapping CSV not found: {target}", file=sys.stderr); sys.exit(1)
    if not inp.exists():
        print(f"Input CSV not found: {inp}", file=sys.stderr); sys.exit(1)

    mapping = load_mapping(target)
    amended = add_resource_type(inp, mapping)

    out.parent.mkdir(parents=True, exist_ok=True)
    amended.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)

    # Report unknowns (informational)
    n_unknown = int((amended[OUTPUT_FUEL_COL] == "Unknown").sum())
    if n_unknown:
        sample = (
            amended.loc[amended[OUTPUT_FUEL_COL] == "Unknown", TARGET_RESOURCE_COL]
            .dropna().astype(str).head(20).tolist()
        )
        print(f"[INFO] {n_unknown} resource(s) lacked a mapped type. Examples: {sample}")
    print(f"[OK] Wrote: {out}")


if __name__ == "__main__":
    main()
