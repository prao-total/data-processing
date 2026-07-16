"""Enrich HCA cluster assignments with SB metadata and explain the clusters.

The clustering input file contains standardized (``z_``) behavioral features.
This script joins those resources to the SB list using ``resource_name`` and
``cdr_unit_code``, then produces:

* an enriched row-level cluster assignment file;
* an audit file for all non-unique or unmatched joins;
* per-cluster match coverage and capacity statistics;
* fuel/technology/zone/status enrichment tables;
* behavioral feature profiles and cluster-vs-rest contrasts; and
* a compact Markdown interpretation report.

Capacity and asset metadata are descriptive variables. Unless they were also
used in the original clustering feature matrix, they should not be described
as causes of the cluster assignments.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CLUSTERS = Path("/Users/pradyrao/Downloads/cluster_assignments.csv")
DEFAULT_SB_LIST = Path("/Users/pradyrao/Downloads/sb_list.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "hca_cluster_analysis"

CLUSTER_REQUIRED_COLUMNS = {"resource_name", "cluster_id"}
SB_REQUIRED_COLUMNS = {"cdr_unit_code", "cdr_capacity_mw"}
SB_METADATA_COLUMNS = [
    "unit_name",
    "cdr_unit_code",
    "cdr_gen_id",
    "cdr_capacity_mw",
    "resolved_capacity_mw",
    "cdr_fuel",
    "cdr_technology",
    "cdr_zone",
    "cdr_status",
    "county",
    "type",
]
CATEGORY_COLUMNS = ["cdr_fuel", "cdr_technology", "cdr_zone", "cdr_status"]

FEATURE_FAMILIES = {
    "operating range": (
        "base_point",
        "normalized_base_point",
        "normalized_hsl",
        "normalized_lsl",
    ),
    "offer-price level": (
        "average_weighted_offer_price",
        "10th_percentile_offer_price",
        "90th_percentile_offer_price",
    ),
    "offer-price volatility": (
        "weighted_stdev_offer_price",
        "normalized_curve_std",
    ),
    "scarcity-price behavior": ("fraction_intervals_at_5000",),
    "commitment economics": (
        "min_gen_cost",
        "hot_startup_cost",
        "cold_startup_cost",
        "inter_startup_cost",
    ),
    "offer-curve shape": ("normalized_offer_curve",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join HCA clusters to the SB list and profile each cluster."
    )
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--sb-list", type=Path, default=DEFAULT_SB_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-features",
        type=int,
        default=8,
        help="Number of cluster-vs-rest feature contrasts included in the report.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().strip('"').upper())


def normalize_code(value: object) -> str:
    """Normalize harmless separators while preserving the identifier's content."""
    return re.sub(r"[^A-Z0-9]+", "", normalize_text(value))


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def clean_metadata_value(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def canonical_identity(row: pd.Series) -> tuple[str, ...]:
    """Core CDR identity used to decide whether duplicate SB rows are equivalent."""
    fields = ["unit_name", "cdr_unit_code", "cdr_gen_id", "cdr_capacity_mw"]
    return tuple(normalize_text(row.get(field, "")) for field in fields)


def choose_capacity(row: pd.Series) -> float:
    cdr = pd.to_numeric(row.get("cdr_capacity_mw"), errors="coerce")
    if pd.notna(cdr):
        return float(cdr)
    resolved = pd.to_numeric(row.get("resolved_capacity_mw"), errors="coerce")
    return float(resolved) if pd.notna(resolved) else np.nan


def prepare_sb_lookup(sb: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prepared = sb.copy()
    prepared["_code_text"] = prepared["cdr_unit_code"].map(normalize_text)
    prepared["_code_key"] = prepared["cdr_unit_code"].map(normalize_code)
    prepared["_capacity_mw"] = prepared.apply(choose_capacity, axis=1)
    prepared = prepared[prepared["_code_key"] != ""].copy()
    return {
        key: group.copy()
        for key, group in prepared.groupby("_code_key", sort=False, dropna=False)
    }


def resolve_candidate_group(
    resource_name: str, candidates: pd.DataFrame
) -> tuple[dict[str, object], list[dict[str, object]]]:
    exact_text = candidates[
        candidates["_code_text"] == normalize_text(resource_name)
    ].copy()
    considered = exact_text if not exact_text.empty else candidates.copy()
    method = "exact_code" if not exact_text.empty else "normalized_code"

    identities = considered.apply(canonical_identity, axis=1)
    unique_identity_count = identities.nunique(dropna=False)
    capacities = considered["_capacity_mw"].dropna().round(6).unique()
    equivalent = unique_identity_count == 1 and len(capacities) <= 1

    audit_rows = []
    for _, candidate in considered.iterrows():
        audit_rows.append(
            {
                "resource_name": resource_name,
                "candidate_cdr_unit_code": candidate.get("cdr_unit_code", ""),
                "candidate_unit_name": candidate.get("unit_name", ""),
                "candidate_cdr_gen_id": candidate.get("cdr_gen_id", ""),
                "candidate_capacity_mw": candidate.get("_capacity_mw", np.nan),
                "candidate_cdr_fuel": candidate.get("cdr_fuel", ""),
                "candidate_cdr_technology": candidate.get("cdr_technology", ""),
            }
        )

    if len(considered) == 1:
        status = "matched"
    elif equivalent:
        status = "matched_duplicate_equivalent"
    else:
        return (
            {
                "capacity_match_status": "ambiguous",
                "capacity_match_method": method,
                "capacity_candidate_count": int(len(considered)),
                "capacity_needs_review": True,
            },
            audit_rows,
        )

    selected = considered.iloc[0]
    result: dict[str, object] = {
        "capacity_match_status": status,
        "capacity_match_method": method,
        "capacity_candidate_count": int(len(considered)),
        "capacity_needs_review": False,
        "matched_cdr_capacity_mw": selected["_capacity_mw"],
    }
    for column in SB_METADATA_COLUMNS:
        if column in selected.index:
            output_column = f"sb_{column}" if column != "cdr_capacity_mw" else "sb_cdr_capacity_mw"
            result[output_column] = selected[column]
    return result, audit_rows if len(considered) > 1 else []


def enrich_clusters(
    clusters: pd.DataFrame, sb: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clusters["resource_name"].map(normalize_text).duplicated().any():
        duplicates = clusters.loc[
            clusters["resource_name"].map(normalize_text).duplicated(False),
            "resource_name",
        ].tolist()
        raise ValueError(f"Duplicate resource_name values in cluster file: {duplicates[:10]}")

    lookup = prepare_sb_lookup(sb)
    enriched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for cluster_record in clusters.to_dict("records"):
        row = dict(cluster_record)
        resource_name = clean_metadata_value(cluster_record["resource_name"])
        candidates = lookup.get(normalize_code(resource_name))
        if candidates is None:
            row.update(
                {
                    "capacity_match_status": "unmatched",
                    "capacity_match_method": "none",
                    "capacity_candidate_count": 0,
                    "capacity_needs_review": True,
                    "matched_cdr_capacity_mw": np.nan,
                }
            )
        else:
            match, candidate_audit = resolve_candidate_group(resource_name, candidates)
            row.update(match)
            for audit in candidate_audit:
                audit["cluster_id"] = cluster_record["cluster_id"]
                audit["final_match_status"] = match["capacity_match_status"]
            audit_rows.extend(candidate_audit)
        enriched_rows.append(row)

    enriched = pd.DataFrame(enriched_rows)
    audit = pd.DataFrame(audit_rows)
    unmatched = enriched[enriched["capacity_match_status"] == "unmatched"][
        ["resource_name", "cluster_id", "capacity_match_status"]
    ].copy()
    if not unmatched.empty:
        audit = pd.concat([audit, unmatched], ignore_index=True, sort=False)
    return enriched, audit


def matched_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["capacity_match_status"].isin(
        ["matched", "matched_duplicate_equivalent"]
    )


def cluster_coverage(enriched: pd.DataFrame) -> pd.DataFrame:
    working = enriched.assign(_matched=matched_mask(enriched))
    summary = (
        working.groupby("cluster_id", dropna=False)
        .agg(
            resource_count=("resource_name", "size"),
            capacity_matched_count=("_matched", "sum"),
            capacity_ambiguous_count=(
                "capacity_match_status",
                lambda values: int((values == "ambiguous").sum()),
            ),
            capacity_unmatched_count=(
                "capacity_match_status",
                lambda values: int((values == "unmatched").sum()),
            ),
        )
        .reset_index()
    )
    summary["capacity_match_rate"] = (
        summary["capacity_matched_count"] / summary["resource_count"]
    )
    return summary


def capacity_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    matched = enriched[matched_mask(enriched)].copy()
    matched["matched_cdr_capacity_mw"] = pd.to_numeric(
        matched["matched_cdr_capacity_mw"], errors="coerce"
    )
    summary = (
        matched.groupby("cluster_id")["matched_cdr_capacity_mw"]
        .agg(
            capacity_observation_count="count",
            capacity_mw_mean="mean",
            capacity_mw_median="median",
            capacity_mw_min="min",
            capacity_mw_max="max",
            capacity_mw_total="sum",
        )
        .reset_index()
    )
    quartiles = (
        matched.groupby("cluster_id")["matched_cdr_capacity_mw"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "capacity_mw_q1", 0.75: "capacity_mw_q3"})
        .reset_index()
    )
    return summary.merge(quartiles, on="cluster_id", how="left")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("z_")]
    if not columns:
        raise ValueError("No standardized clustering feature columns beginning with 'z_'")
    return columns


def feature_profiles(
    enriched: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = enriched[features].apply(pd.to_numeric, errors="coerce")
    working = pd.concat([enriched[["cluster_id"]], numeric], axis=1)

    profile_rows: list[dict[str, object]] = []
    for cluster_id, group in working.groupby("cluster_id", dropna=False):
        for feature in features:
            values = group[feature]
            profile_rows.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "count": int(values.count()),
                }
            )
    profiles = pd.DataFrame(profile_rows)

    contrasts: list[dict[str, object]] = []
    for cluster_id, group in working.groupby("cluster_id", dropna=False):
        outside = working[working["cluster_id"] != cluster_id]
        for feature in features:
            cluster_median = group[feature].median()
            outside_median = outside[feature].median()
            contrast = cluster_median - outside_median
            contrasts.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "cluster_median": cluster_median,
                    "outside_cluster_median": outside_median,
                    "median_contrast": contrast,
                    "absolute_median_contrast": abs(contrast),
                    "feature_family": classify_feature(feature),
                }
            )
    contrast_frame = pd.DataFrame(contrasts).sort_values(
        ["cluster_id", "absolute_median_contrast"],
        ascending=[True, False],
    )
    return profiles, contrast_frame


def classify_feature(feature: str) -> str:
    normalized = feature[2:] if feature.startswith("z_") else feature
    for family, tokens in FEATURE_FAMILIES.items():
        if any(token in normalized for token in tokens):
            return family
    return "other"


def category_enrichment(
    enriched: pd.DataFrame, category_columns: Iterable[str]
) -> pd.DataFrame:
    matched = enriched[matched_mask(enriched)].copy()
    rows: list[dict[str, object]] = []
    for column in category_columns:
        sb_column = f"sb_{column}"
        if sb_column not in matched.columns:
            continue
        values = matched[sb_column].fillna("UNKNOWN").astype(str).str.strip()
        values = values.mask(values == "", "UNKNOWN")
        overall = values.value_counts(normalize=True)
        for cluster_id, indexes in matched.groupby("cluster_id").groups.items():
            cluster_values = values.loc[indexes]
            counts = cluster_values.value_counts()
            for category, count in counts.items():
                cluster_share = count / len(cluster_values)
                overall_share = float(overall.get(category, 0))
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "attribute": column,
                        "category": category,
                        "resource_count": int(count),
                        "cluster_share": cluster_share,
                        "overall_share": overall_share,
                        "enrichment_ratio": (
                            cluster_share / overall_share if overall_share else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["cluster_id", "attribute", "enrichment_ratio"],
        ascending=[True, True, False],
    )


def fmt_number(value: object, digits: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):,.{digits}f}"


def humanize_feature(feature: str) -> str:
    normalized = feature[2:] if feature.startswith("z_") else feature
    return normalized.replace("_", " ")


def direction(value: float) -> str:
    return "higher" if value > 0 else "lower"


def write_report(
    output_path: Path,
    enriched: pd.DataFrame,
    coverage: pd.DataFrame,
    capacities: pd.DataFrame,
    contrasts: pd.DataFrame,
    enrichments: pd.DataFrame,
    top_features: int,
) -> None:
    capacity_lookup = capacities.set_index("cluster_id").to_dict("index")
    lines = [
        "# HCA cluster interpretation",
        "",
        (
            "Behavioral `z_` features explain why resources were grouped. Capacity, "
            "fuel, technology, zone, and status describe which assets landed in each "
            "group; they are not treated as clustering causes."
        ),
        "",
        "## Join quality",
        "",
        (
            f"- Clustered resources: {len(enriched):,}; confidently capacity-matched: "
            f"{int(matched_mask(enriched).sum()):,} "
            f"({matched_mask(enriched).mean():.1%})."
        ),
        (
            "- Capacity totals can double-count physical equipment when multiple SCED "
            "configurations refer to the same plant equipment."
        ),
        "",
    ]

    for _, cov in coverage.sort_values("cluster_id").iterrows():
        cluster_id = cov["cluster_id"]
        cluster_contrasts = contrasts[contrasts["cluster_id"] == cluster_id].head(
            top_features
        )
        cap = capacity_lookup.get(cluster_id, {})
        lines.extend(
            [
                f"## Cluster {cluster_id}",
                "",
                (
                    f"{int(cov['resource_count']):,} resources; "
                    f"{int(cov['capacity_matched_count']):,} confidently matched "
                    f"({cov['capacity_match_rate']:.1%}), "
                    f"{int(cov['capacity_ambiguous_count']):,} ambiguous, and "
                    f"{int(cov['capacity_unmatched_count']):,} unmatched."
                ),
                "",
            ]
        )
        if cap:
            lines.append(
                "Matched-resource capacity: median "
                f"{fmt_number(cap.get('capacity_mw_median'))} MW "
                f"(IQR {fmt_number(cap.get('capacity_mw_q1'))}–"
                f"{fmt_number(cap.get('capacity_mw_q3'))} MW)."
            )
            lines.append("")

        lines.append("Strongest behavioral contrasts against all other clusters:")
        lines.append("")
        for _, item in cluster_contrasts.iterrows():
            lines.append(
                f"- {humanize_feature(item['feature'])}: "
                f"{direction(item['median_contrast'])} by "
                f"{abs(item['median_contrast']):.2f} standardized units "
                f"({item['feature_family']})."
            )

        if not enrichments.empty:
            distinctive = enrichments[
                (enrichments["cluster_id"] == cluster_id)
                & (enrichments["resource_count"] >= 2)
                & (enrichments["cluster_share"] >= 0.10)
                & (enrichments["enrichment_ratio"] > 1.0)
            ].nlargest(5, "enrichment_ratio")
            if not distinctive.empty:
                lines.extend(["", "Overrepresented matched-resource metadata:", ""])
                for _, item in distinctive.iterrows():
                    lines.append(
                        f"- {item['attribute']} = {item['category']}: "
                        f"{item['cluster_share']:.1%} of matched cluster resources, "
                        f"{item['enrichment_ratio']:.2f}× the matched fleet share."
                    )
        if cov["capacity_match_rate"] < 0.70:
            lines.extend(
                [
                    "",
                    (
                        "Capacity and asset-composition conclusions are provisional "
                        "because fewer than 70% of this cluster's resources matched."
                    ),
                ]
            )
        if int(cov["resource_count"]) < 5:
            lines.extend(
                [
                    "",
                    (
                        "This is a very small cluster and should be interpreted as an "
                        "outlier group rather than a broad fleet archetype."
                    ),
                ]
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    enriched: pd.DataFrame,
    audit: pd.DataFrame,
    top_features: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns(enriched)
    coverage = cluster_coverage(enriched)
    capacities = capacity_summary(enriched)
    profiles, contrasts = feature_profiles(enriched, features)
    enrichments = category_enrichment(enriched, CATEGORY_COLUMNS)

    outputs = {
        "enriched": output_dir / "cluster_assignments_enriched.csv",
        "match_audit": output_dir / "capacity_match_audit.csv",
        "coverage": output_dir / "cluster_match_coverage.csv",
        "capacity": output_dir / "cluster_capacity_summary.csv",
        "feature_profiles": output_dir / "cluster_feature_profiles.csv",
        "feature_contrasts": output_dir / "cluster_feature_contrasts.csv",
        "metadata_enrichment": output_dir / "cluster_metadata_enrichment.csv",
        "report": output_dir / "cluster_interpretation.md",
        "run_summary": output_dir / "run_summary.json",
    }
    enriched.to_csv(outputs["enriched"], index=False)
    audit.to_csv(outputs["match_audit"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    capacities.to_csv(outputs["capacity"], index=False)
    profiles.to_csv(outputs["feature_profiles"], index=False)
    contrasts.to_csv(outputs["feature_contrasts"], index=False)
    enrichments.to_csv(outputs["metadata_enrichment"], index=False)
    write_report(
        outputs["report"],
        enriched,
        coverage,
        capacities,
        contrasts,
        enrichments,
        top_features,
    )

    summary = {
        "clustered_resources": int(len(enriched)),
        "confident_matches": int(matched_mask(enriched).sum()),
        "ambiguous_matches": int(
            (enriched["capacity_match_status"] == "ambiguous").sum()
        ),
        "unmatched_resources": int(
            (enriched["capacity_match_status"] == "unmatched").sum()
        ),
        "overall_match_rate": float(matched_mask(enriched).mean()),
        "cluster_count": int(enriched["cluster_id"].nunique(dropna=False)),
        "feature_count": len(features),
    }
    outputs["run_summary"].write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return outputs


def main() -> None:
    args = parse_args()
    clusters = load_csv(args.clusters.expanduser(), CLUSTER_REQUIRED_COLUMNS)
    sb = load_csv(args.sb_list.expanduser(), SB_REQUIRED_COLUMNS)
    enriched, audit = enrich_clusters(clusters, sb)
    outputs = write_outputs(
        args.output_dir.expanduser(), enriched, audit, args.top_features
    )

    print(f"Loaded {len(clusters):,} clustered resources and {len(sb):,} SB rows.")
    print(
        f"Confident capacity matches: {matched_mask(enriched).sum():,}/"
        f"{len(enriched):,} ({matched_mask(enriched).mean():.1%})."
    )
    print(f"Wrote analysis outputs to {args.output_dir.expanduser().resolve()}:")
    for name, path in outputs.items():
        print(f"  {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
