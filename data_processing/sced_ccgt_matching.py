from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class CcgtMatchConfig:
    input_csv: Path
    output_dir: Path | None = None


def default_config() -> CcgtMatchConfig:
    # Hardcode your input CSV path here.
    return CcgtMatchConfig(
        input_csv=Path("C:/Users/L1165683/OneDrive - TotalEnergies/Documents/db/plots/plants_by_fuel.csv"),
        output_dir=Path("C:/Users/L1165683/GitHub_Repos/data-processing/data_processing/output/buckets"),
    )


def load_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def bucket_similar_names(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Bucket by the plant base name (substring before the first underscore).
    """
    if "resource_name" not in df.columns:
        raise KeyError("Column 'resource_name' not found in input CSV")

    buckets: Dict[str, List[str]] = {}
    raw_names = df["resource_name"].astype(str).fillna("")
    normalized = raw_names.str.strip()
    base_names = normalized.str.split("_", n=1).str[0]

    for raw, base in zip(raw_names, base_names):
        key = base if base else "__MISSING__"
        buckets.setdefault(key, []).append(raw)

    return buckets


def add_bucket_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a plant-level bucket id column derived from base names.
    """
    working = df.copy()
    if "resource_name" not in working.columns:
        raise KeyError("Column 'resource_name' not found in input CSV")
    raw_names = working["resource_name"].astype(str).fillna("")
    normalized = raw_names.str.strip()
    base_names = normalized.str.split("_", n=1).str[0]
    bucket_id = base_names.where(base_names != "", "__MISSING__")
    working["bucket_id"] = bucket_id
    return working


def main() -> None:
    config = default_config()
    df = load_input_csv(config.input_csv)

    bucketed = add_bucket_column(df)
    buckets = bucket_similar_names(bucketed)

    print(f"Loaded {len(df)} rows from {config.input_csv}")
    print(f"Identified {len(buckets)} buckets (placeholder logic)")

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.output_dir / "sced_ccgt_buckets.csv"
        bucketed.to_csv(output_path, index=False)
        print(f"Wrote bucketed output to {output_path}")


if __name__ == "__main__":
    main()
