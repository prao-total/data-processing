"""Match HCA cluster resources to the SB generator list.

Cluster assignments use SCED resource codes.  This script adapts those resources
to the input expected by :mod:`sced_to_sb_matching`, reuses that module's
matching engine, and turns the SB-to-SCED result back into cluster-centric
outputs.  It intentionally does not interpret or describe the clusters.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sced_to_sb_matching as sb_matching


DEFAULT_CLUSTERS = Path("/Users/pradyrao/Downloads/cluster_assignments.csv")
DEFAULT_FEATURE_TABLE = Path("/Users/pradyrao/Downloads/feature_table.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "hca_cluster_matching"

CLUSTER_REQUIRED_COLUMNS = {"resource_name", "cluster_id"}
MATCHED_STATUSES = {"matched", "ambiguous"}
RESOURCE_ATTRIBUTES = [
    "cdr_fuel",
    "cdr_technology",
    "cdr_zone",
    "cdr_status",
    "type",
]
MIN_GEN_FEATURE = "z_median_min_gen_cost"
STARTUP_FEATURES = {
    "hot": "z_median_hot_startup_cost",
    "inter": "z_median_inter_startup_cost",
    "cold": "z_median_cold_startup_cost",
}
CURVE_FEATURES = [f"z_median_normalized_offer_curve_{point:02d}" for point in range(21)]
RAW_COST_FEATURES = [
    "median_min_gen_cost",
    "median_hot_startup_cost",
    "median_inter_startup_cost",
    "median_cold_startup_cost",
]
RAW_CURVE_FEATURES = [
    f"median_normalized_offer_curve_{point:02d}" for point in range(21)
]
RAW_FEATURES = RAW_COST_FEATURES + RAW_CURVE_FEATURES

MATCH_COLUMNS = [
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
    "matched_sced_node",
    "matched_sced_resource_type",
    "matched_sced_plant_descriptor",
    "matched_sced_unit_descriptor",
    "plant_match_method",
    "plant_match_score",
    "unit_assignment_method",
    "unit_assignment_score",
    "sb_to_sced_match_method",
    "sb_to_sced_match_score",
    "sb_to_sced_match_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match SCED-coded HCA cluster assignments to the SB list."
    )
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def validate_clusters(clusters: pd.DataFrame) -> None:
    normalized = clusters["resource_name"].map(sb_matching.normalize_text)
    if normalized.eq("").any():
        raise ValueError("cluster_assignments contains blank resource_name values")
    if normalized.duplicated().any():
        duplicates = clusters.loc[normalized.duplicated(False), "resource_name"].tolist()
        raise ValueError(f"Duplicate cluster resource names: {duplicates[:10]}")


def adapt_clusters_for_matcher(clusters: pd.DataFrame) -> pd.DataFrame:
    """Build the minimal SCED frame required by the shared matching engine.

    The HCA output has no raw resource type, timestamp, or offer averages.  Blank
    or missing values are supplied so these unavailable fields cannot influence
    candidate scoring.  Resource-code, plant-name, and unit-token matching remain
    active.
    """
    return pd.DataFrame(
        {
            "Resource Name": clusters["resource_name"],
            "Resource Type": "",
            "final_sced_time_stamp": "",
            "Base Point_avg": pd.NA,
            "Start Up Cold Offer_avg": pd.NA,
            "Start Up Hot Offer_avg": pd.NA,
            "Start Up Inter Offer_avg": pd.NA,
            "Min Gen Cost_avg": pd.NA,
        }
    )


def run_shared_matcher(
    clusters: pd.DataFrame, sb: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sced_input = adapt_clusters_for_matcher(clusters)
    sced_prepared = sb_matching.prepare_sced_resource_df(sced_input)
    sb_filtered = sb_matching.filter_sb_rows_for_matching(sb)
    return sb_matching.build_sb_matches(sb_filtered, sced_prepared)


def accepted_matches(sb_matches: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    accepted = sb_matches[
        sb_matches["sb_to_sced_match_status"].isin(MATCHED_STATUSES)
        & sb_matches["matched_sced_node"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    available_columns = [column for column in MATCH_COLUMNS if column in accepted.columns]
    accepted = accepted[available_columns]
    cluster_metadata = clusters[["resource_name", "cluster_id"]].rename(
        columns={"resource_name": "matched_sced_node"}
    )
    accepted = cluster_metadata.merge(accepted, on="matched_sced_node", how="inner")
    accepted["match_rank"] = (
        accepted.groupby("matched_sced_node")["sb_to_sced_match_score"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    accepted["accepted_match_count"] = accepted.groupby("matched_sced_node")[
        "matched_sced_node"
    ].transform("size")
    return accepted.sort_values(
        ["cluster_id", "matched_sced_node", "match_rank"], kind="stable"
    ).reset_index(drop=True)


def build_primary_matches(
    clusters: pd.DataFrame, accepted: pd.DataFrame
) -> pd.DataFrame:
    """Amend each cluster row with its primary and complete set of SB matches."""
    if accepted.empty:
        primary = pd.DataFrame(columns=["matched_sced_node"])
        aggregated = pd.DataFrame(columns=["matched_sced_node"])
    else:
        primary = accepted[accepted["match_rank"].eq(1)].copy()
        primary = primary.drop(columns=["cluster_id", "match_rank"], errors="ignore")
        aggregated = (
            accepted.groupby("matched_sced_node", sort=False)
            .agg(
                matched_sb_unit_names=("unit_name", join_distinct),
                matched_sb_cdr_unit_codes=("cdr_unit_code", join_distinct),
                matched_sb_capacities_mw=("cdr_capacity_mw", join_distinct),
                matched_sb_resolved_capacities_mw=(
                    "resolved_capacity_mw",
                    join_distinct,
                ),
            )
            .reset_index()
        )

    output = clusters.merge(
        primary,
        left_on="resource_name",
        right_on="matched_sced_node",
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        aggregated,
        on="matched_sced_node",
        how="left",
        validate="many_to_one",
    )
    output["cluster_to_sb_match_status"] = "unmatched"
    has_match = output["matched_sced_node"].notna()
    output.loc[has_match, "cluster_to_sb_match_status"] = "matched"
    multiple = output["accepted_match_count"].fillna(0).gt(1)
    output.loc[multiple, "cluster_to_sb_match_status"] = "multiple_sb_matches"
    output["accepted_match_count"] = output["accepted_match_count"].fillna(0).astype(int)
    return output


def join_distinct(values: pd.Series) -> str:
    """Join distinct nonblank values while preserving their source order."""
    result: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return " | ".join(result)


def cluster_population_stats(amended: pd.DataFrame) -> pd.DataFrame:
    total_resources = len(amended)
    grouped = amended.groupby("cluster_id", dropna=False)
    summary = grouped.agg(
        resource_count=("resource_name", "size"),
        matched_resource_count=(
            "cluster_to_sb_match_status",
            lambda values: int(values.ne("unmatched").sum()),
        ),
        unmatched_resource_count=(
            "cluster_to_sb_match_status",
            lambda values: int(values.eq("unmatched").sum()),
        ),
    ).reset_index()
    summary["cluster_share"] = summary["resource_count"] / total_resources
    summary["matched_resource_share"] = (
        summary["matched_resource_count"] / summary["resource_count"]
    )

    if "dendrogram_leaf_order" in amended.columns:
        leaf_order = (
            grouped["dendrogram_leaf_order"]
            .agg(dendrogram_leaf_order_min="min", dendrogram_leaf_order_max="max")
            .reset_index()
        )
        summary = summary.merge(leaf_order, on="cluster_id", how="left")
    return summary


def cluster_capacity_stats(amended: pd.DataFrame) -> pd.DataFrame:
    working = amended[["cluster_id", "cdr_capacity_mw"]].copy()
    working["cdr_capacity_mw"] = pd.to_numeric(
        working["cdr_capacity_mw"], errors="coerce"
    )
    grouped = working.groupby("cluster_id", dropna=False)["cdr_capacity_mw"]
    summary = grouped.agg(
        capacity_observation_count="count",
        capacity_mean_mw="mean",
        capacity_median_mw="median",
        capacity_std_mw="std",
        capacity_min_mw="min",
        capacity_max_mw="max",
        capacity_total_mw="sum",
    ).reset_index()
    quartiles = (
        grouped.quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "capacity_q1_mw", 0.75: "capacity_q3_mw"})
        .reset_index()
    )
    return summary.merge(quartiles, on="cluster_id", how="left")


def normalized_category(values: pd.Series) -> pd.Series:
    normalized = values.fillna("UNKNOWN").astype(str).str.strip()
    return normalized.mask(normalized.eq(""), "UNKNOWN")


def cluster_resource_distribution(amended: pd.DataFrame) -> pd.DataFrame:
    """Return complete matched-resource distributions for SB attributes."""
    matched = amended[amended["cluster_to_sb_match_status"].ne("unmatched")].copy()
    rows: list[dict[str, object]] = []
    for attribute in RESOURCE_ATTRIBUTES:
        if attribute not in matched.columns:
            continue
        values = normalized_category(matched[attribute])
        attribute_frame = matched[["cluster_id"]].copy()
        attribute_frame["category"] = values
        counts = (
            attribute_frame.groupby(["cluster_id", "category"], dropna=False)
            .size()
            .rename("resource_count")
            .reset_index()
        )
        totals = counts.groupby("cluster_id")["resource_count"].transform("sum")
        counts["matched_cluster_share"] = counts["resource_count"] / totals
        counts["attribute"] = attribute
        rows.extend(counts.to_dict("records"))
    columns = [
        "cluster_id",
        "attribute",
        "category",
        "resource_count",
        "matched_cluster_share",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["cluster_id", "attribute", "resource_count", "category"],
        ascending=[True, True, False, True],
        kind="stable",
    )


def dominant_resource_stats(distribution: pd.DataFrame) -> pd.DataFrame:
    """Pivot the most common value and its share for every SB attribute."""
    if distribution.empty:
        return pd.DataFrame(columns=["cluster_id"])
    dominant = (
        distribution.sort_values(
            ["cluster_id", "attribute", "resource_count", "category"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .groupby(["cluster_id", "attribute"], as_index=False)
        .first()
    )
    rows: list[dict[str, object]] = []
    for cluster_id, group in dominant.groupby("cluster_id", dropna=False):
        row: dict[str, object] = {"cluster_id": cluster_id}
        for _, item in group.iterrows():
            attribute = item["attribute"]
            row[f"dominant_{attribute}"] = item["category"]
            row[f"dominant_{attribute}_count"] = item["resource_count"]
            row[f"dominant_{attribute}_share"] = item["matched_cluster_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def safe_column_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"


def wide_resource_stats(distribution: pd.DataFrame) -> pd.DataFrame:
    """Put every resource category count and share on the cluster summary row."""
    if distribution.empty:
        return pd.DataFrame(columns=["cluster_id"])
    rows: list[dict[str, object]] = []
    for cluster_id, group in distribution.groupby("cluster_id", dropna=False):
        row: dict[str, object] = {"cluster_id": cluster_id}
        used_names: set[str] = set()
        for _, item in group.iterrows():
            prefix = safe_column_token(item["attribute"])
            category = safe_column_token(item["category"])
            base_name = f"{prefix}_{category}"
            if base_name in used_names:
                raise ValueError(
                    "Resource categories produce duplicate summary columns: "
                    f"{item['attribute']}={item['category']}"
                )
            used_names.add(base_name)
            row[f"{base_name}_count"] = item["resource_count"]
            row[f"{base_name}_share"] = item["matched_cluster_share"]
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def validate_feature_columns(amended: pd.DataFrame) -> None:
    required = {MIN_GEN_FEATURE, *STARTUP_FEATURES.values(), *CURVE_FEATURES}
    missing = sorted(required - set(amended.columns))
    if missing:
        raise ValueError(f"Cluster assignments are missing feature columns: {missing}")


def rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else np.nan


def resource_distances(values: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Normalized Euclidean distance from each resource to a family centroid."""
    distances = []
    for row in values:
        valid = np.isfinite(row) & np.isfinite(centroid)
        if not valid.any():
            distances.append(np.nan)
        else:
            distances.append(float(np.sqrt(np.mean(np.square(row[valid] - centroid[valid])))))
    return np.asarray(distances, dtype=float)


def cluster_feature_stats(amended: pd.DataFrame) -> pd.DataFrame:
    validate_feature_columns(amended)
    rows: list[dict[str, object]] = []

    for cluster_id, group in amended.groupby("cluster_id", dropna=False):
        row: dict[str, object] = {"cluster_id": cluster_id}

        min_gen = pd.to_numeric(group[MIN_GEN_FEATURE], errors="coerce")
        row["min_gen_valid_count"] = int(min_gen.count())
        row["min_gen_centroid"] = min_gen.mean()
        row["min_gen_dispersion"] = min_gen.std(ddof=1)

        for label, feature in STARTUP_FEATURES.items():
            values = pd.to_numeric(group[feature], errors="coerce")
            centroid = values.mean()
            row[f"{label}_startup_valid_count"] = int(values.count())
            row[f"{label}_startup_centroid"] = centroid
            row[f"{label}_startup_distinctiveness"] = abs(centroid)
            row[f"{label}_startup_dispersion"] = values.std(ddof=1)

        curve = group[CURVE_FEATURES].apply(pd.to_numeric, errors="coerce")
        curve_values = curve.to_numpy(dtype=float)
        curve_centroid = np.nanmean(curve_values, axis=0)
        for point, value in enumerate(curve_centroid):
            row[f"curve_centroid_{point:02d}"] = value

        curve_distances = resource_distances(curve_values, curve_centroid)
        valid_distances = curve_distances[np.isfinite(curve_distances)]
        row["curve_valid_resource_count"] = int(len(valid_distances))
        row["curve_level"] = float(np.nanmean(curve_centroid))
        row["curve_distinctiveness"] = rms(curve_centroid)
        row["curve_dispersion"] = (
            float(np.mean(valid_distances)) if valid_distances.size else np.nan
        )
        row["curve_dispersion_stdev"] = (
            float(np.std(valid_distances, ddof=1))
            if valid_distances.size > 1
            else np.nan
        )
        row["curve_start"] = curve_centroid[0]
        row["curve_middle"] = curve_centroid[10]
        row["curve_end"] = curve_centroid[20]
        row["curve_slope"] = curve_centroid[20] - curve_centroid[0]
        row["curve_upper_end_slope"] = curve_centroid[20] - curve_centroid[15]
        if np.isfinite(curve_centroid).any():
            peak_position = int(np.nanargmax(np.abs(curve_centroid)))
            row["curve_peak"] = abs(curve_centroid[peak_position])
            row["curve_peak_value"] = curve_centroid[peak_position]
            row["curve_peak_position"] = peak_position
        else:
            row["curve_peak"] = np.nan
            row["curve_peak_value"] = np.nan
            row["curve_peak_position"] = pd.NA
        rows.append(row)

    return pd.DataFrame(rows)


def raw_feature_stats(
    amended: pd.DataFrame, feature_table: pd.DataFrame
) -> pd.DataFrame:
    """Summarize selected original-scale features within each cluster."""
    required = {"resource_name", *RAW_FEATURES}
    missing = sorted(required - set(feature_table.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    prepared = feature_table[["resource_name", *RAW_FEATURES]].copy()
    prepared["_resource_key"] = prepared["resource_name"].map(
        sb_matching.normalize_text
    )
    if prepared["_resource_key"].eq("").any():
        raise ValueError("Feature table contains blank resource_name values")
    if prepared["_resource_key"].duplicated().any():
        duplicates = prepared.loc[
            prepared["_resource_key"].duplicated(False), "resource_name"
        ].tolist()
        raise ValueError(f"Duplicate feature-table resource names: {duplicates[:10]}")

    cluster_keys = amended[["resource_name", "cluster_id"]].copy()
    cluster_keys["_resource_key"] = cluster_keys["resource_name"].map(
        sb_matching.normalize_text
    )
    working = cluster_keys[["cluster_id", "_resource_key"]].merge(
        prepared.drop(columns="resource_name"),
        on="_resource_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = working.loc[working["_merge"].ne("both"), "_resource_key"].tolist()
    if unmatched:
        raise ValueError(
            f"Feature table did not match {len(unmatched)} cluster resources: "
            f"{unmatched[:10]}"
        )

    rows: list[dict[str, object]] = []
    for cluster_id, group in working.groupby("cluster_id", dropna=False):
        row: dict[str, object] = {"cluster_id": cluster_id}
        for feature in RAW_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce")
            prefix = f"raw_{feature}"
            row[f"{prefix}_mean"] = values.mean()
            row[f"{prefix}_median"] = values.median()
            row[f"{prefix}_stdev"] = values.std(ddof=1)
            row[f"{prefix}_min"] = values.min()
            row[f"{prefix}_max"] = values.max()
            row[f"{prefix}_q1"] = values.quantile(0.25)
            row[f"{prefix}_q3"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def build_cluster_summary(
    amended: pd.DataFrame, feature_table: pd.DataFrame
) -> pd.DataFrame:
    population = cluster_population_stats(amended)
    capacity = cluster_capacity_stats(amended)
    distribution = cluster_resource_distribution(amended)
    dominant = dominant_resource_stats(distribution)
    resource_stats = wide_resource_stats(distribution)
    feature_stats = cluster_feature_stats(amended)
    original_scale_stats = raw_feature_stats(amended, feature_table)
    summary = population.merge(capacity, on="cluster_id", how="left")
    summary = summary.merge(dominant, on="cluster_id", how="left")
    summary = summary.merge(resource_stats, on="cluster_id", how="left")
    summary = summary.merge(feature_stats, on="cluster_id", how="left")
    summary = summary.merge(original_scale_stats, on="cluster_id", how="left")
    return summary.sort_values("cluster_id")


def save_median_curve_plot(cluster_summary: pd.DataFrame, path: Path) -> None:
    """Plot each cluster's raw median offer curve with +/- one stdev bars."""
    if sb_matching.plt is None:
        raise RuntimeError("matplotlib is required to create the median curve plot")

    figure, axis = sb_matching.plt.subplots(figsize=(14, 8))
    curve_points = np.arange(21)
    colors = sb_matching.plt.get_cmap("tab10")

    for color_index, (_, cluster) in enumerate(
        cluster_summary.sort_values("cluster_id").iterrows()
    ):
        medians = np.asarray(
            [
                cluster[f"raw_median_normalized_offer_curve_{point:02d}_median"]
                for point in curve_points
            ],
            dtype=float,
        )
        standard_deviations = np.asarray(
            [
                cluster[f"raw_median_normalized_offer_curve_{point:02d}_stdev"]
                for point in curve_points
            ],
            dtype=float,
        )
        axis.errorbar(
            curve_points,
            medians,
            yerr=standard_deviations,
            label=f"Cluster {cluster['cluster_id']:g}",
            color=colors(color_index % 10),
            linewidth=2,
            marker="o",
            markersize=4,
            elinewidth=1,
            capsize=2,
            alpha=0.85,
        )

    axis.axhline(0, color="#555555", linewidth=0.8, alpha=0.6)
    axis.set_title("Median Offer Curve by Cluster (Error Bars = +/- 1 Stdev)")
    axis.set_xlabel("Normalized offer-curve point")
    axis.set_ylabel("Offer price")
    axis.set_xticks(curve_points)
    axis.grid(True, alpha=0.2)
    axis.legend(title="Cluster", loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    sb_matching.plt.close(figure)


def filter_candidate_audit(
    candidates: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    if candidates.empty or "matched_sced_node" not in candidates.columns:
        return candidates.copy()
    cluster_names = set(clusters["resource_name"].astype(str))
    return candidates[
        candidates["matched_sced_node"].astype(str).isin(cluster_names)
    ].sort_values(
        ["matched_sced_node", "sb_to_sced_match_score"],
        ascending=[True, False],
        kind="stable",
    )


def save_outputs(
    output_dir: Path,
    primary: pd.DataFrame,
    accepted: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_summary = build_cluster_summary(primary, feature_table)
    paths = {
        "amended_clusters": output_dir / "cluster_assignments_with_sb_matches.csv",
        "cluster_summary": output_dir / "cluster_summary.csv",
        "median_curve_plot": output_dir / "cluster_median_curves_with_stdev.png",
        "all_accepted_matches": output_dir / "cluster_to_sb_all_accepted_matches.csv",
        "candidate_audit": output_dir / "cluster_to_sb_candidate_audit.csv",
    }
    primary.to_csv(paths["amended_clusters"], index=False)
    cluster_summary.to_csv(paths["cluster_summary"], index=False)
    save_median_curve_plot(cluster_summary, paths["median_curve_plot"])
    accepted.to_csv(paths["all_accepted_matches"], index=False)
    candidates.to_csv(paths["candidate_audit"], index=False)
    return paths


def main() -> None:
    args = parse_args()
    clusters = load_csv(args.clusters.expanduser(), CLUSTER_REQUIRED_COLUMNS)
    validate_clusters(clusters)
    feature_table = load_csv(
        args.feature_table.expanduser(), {"resource_name", *RAW_FEATURES}
    )
    sb_path = Path(sb_matching.SB_LIST_PATH).expanduser()
    sb = load_csv(sb_path, set(sb_matching.SB_REQUIRED_COLUMNS))

    sb_matches, candidate_rows = run_shared_matcher(clusters, sb)
    accepted = accepted_matches(sb_matches, clusters)
    primary = build_primary_matches(clusters, accepted)
    candidates = filter_candidate_audit(candidate_rows, clusters)
    paths = save_outputs(
        args.output_dir.expanduser(),
        primary,
        accepted,
        candidates,
        feature_table,
    )

    matched = primary["cluster_to_sb_match_status"].ne("unmatched")
    multiple = primary["cluster_to_sb_match_status"].eq("multiple_sb_matches")
    print(f"Loaded {len(clusters):,} clustered SCED resources and {len(sb):,} SB rows.")
    print(f"Matched {matched.sum():,}/{len(primary):,} cluster resources ({matched.mean():.1%}).")
    print(f"Resources with multiple accepted SB rows: {multiple.sum():,}.")
    print(f"Wrote matching outputs to {args.output_dir.expanduser().resolve()}:")
    for label, path in paths.items():
        print(f"  {label}: {path.resolve()}")


if __name__ == "__main__":
    main()
