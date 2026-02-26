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


def summarize_units_per_fuel(df: pd.DataFrame) -> pd.Series:
    if "fuel_type" not in df.columns:
        raise KeyError("Column 'fuel_type' not found in input CSV")
    return df.groupby("fuel_type", dropna=False).size().sort_values(ascending=False)


def _bucket_fuel_sets(df: pd.DataFrame) -> pd.DataFrame:
    if "bucket_id" not in df.columns:
        raise KeyError("Column 'bucket_id' not found; call add_bucket_column first")
    if "fuel_type" not in df.columns:
        raise KeyError("Column 'fuel_type' not found in input CSV")
    fuels_by_bucket = df.groupby("bucket_id")["fuel_type"].agg(
        lambda s: tuple(sorted(set(s.astype(str))))
    )
    return fuels_by_bucket.to_frame(name="bucket_fuels")


def summarize_units_with_target_fuel(
    df: pd.DataFrame, target_fuels: List[str]
) -> pd.DataFrame:
    fuels_by_bucket = _bucket_fuel_sets(df)
    df_with_bucket = df.merge(fuels_by_bucket, left_on="bucket_id", right_index=True)

    rows = []
    for target in target_fuels:
        mask = df_with_bucket["bucket_fuels"].apply(lambda fuels: target in fuels)
        counts = (
            df_with_bucket[mask]
            .groupby("fuel_type", dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        for fuel, count in counts.items():
            rows.append({"target_fuel": target, "fuel_type": fuel, "unit_count": int(count)})

    return pd.DataFrame(rows)


def summarize_bucket_types(df: pd.DataFrame) -> pd.DataFrame:
    fuels_by_bucket = _bucket_fuel_sets(df)
    bucket_type = fuels_by_bucket["bucket_fuels"].apply(lambda fuels: "|".join(fuels))
    counts = bucket_type.value_counts().rename_axis("bucket_type").reset_index(name="bucket_count")
    return counts


def main() -> None:
    config = default_config()
    df = load_input_csv(config.input_csv)

    bucketed = add_bucket_column(df)
    buckets = bucket_similar_names(bucketed)

    print(f"Loaded {len(df)} rows from {config.input_csv}")
    print(f"Identified {len(buckets)} buckets (placeholder logic)")

    print("\nUnits per fuel:")
    print(summarize_units_per_fuel(bucketed).to_string())

    targets = ["CCGT90", "SCLE90"]
    print("\nUnits per fuel in buckets containing target fuels:")
    target_counts = summarize_units_with_target_fuel(bucketed, targets)
    if target_counts.empty:
        print("No matching buckets for target fuels.")
    else:
        print(target_counts.to_string(index=False))

    print("\nBucket types (unique fuel combinations) with count:")
    print(summarize_bucket_types(bucketed).to_string(index=False))

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.output_dir / "sced_ccgt_buckets.csv"
        bucketed.to_csv(output_path, index=False)
        print(f"Wrote bucketed output to {output_path}")


if __name__ == "__main__":
    main()
