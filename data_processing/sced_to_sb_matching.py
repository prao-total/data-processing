from __future__ import annotations

import os
import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

import sced_to_price_matching as price_matching


SCED_RESOURCE_LIST_PATH = "/Users/pradyrao/Downloads/sced_unique_resource_name_type_pairs.csv"
SCED_NAME_LIST_PATH = "/Users/pradyrao/Downloads/sced_name_list.csv"
YES_UNITS_LIST_PATH = "/Users/pradyrao/Downloads/ERCOT_YES_units_list (1).csv"
RTLMP_BUS_LIST_PATH = "/Users/pradyrao/Downloads/rtlmp_bus_ercot_list.csv"
RTLMP_LIST_PATH = "/Users/pradyrao/Downloads/rtlmp_ercot_list.csv"
RESOURCE_NODE_MAPPING_PATH = "/Users/pradyrao/Downloads/SP_List_EB_Mapping 2/Resource_Node_to_Unit_02202026_094122.csv"
SB_LIST_PATH = "/Users/pradyrao/Downloads/sb_list.csv"
PUN_GENERATION_REPORT_PATH = "/Users/pradyrao/Downloads/PUN_Generation_Report/PUN_Generation_Report.csv"
FINAL_SB_MATCHES_PATH = None

OUTPUT_DIR = "/Users/pradyrao/VSCode/data-processing/data_processing/output/sced_to_sb_matching"
SCED_PRICE_SIMPLE_MATCHES_FILE_NAME = "sced_to_price_simple_matches.csv"
SCED_PRICE_SUMMARY_FILE_NAME = "sced_to_price_summary.csv"
SB_MATCHES_FILE_NAME = "sb_to_sced_matches.csv"
SB_CANDIDATES_FILE_NAME = "sb_to_sced_candidates.csv"
SB_SUMMARY_FILE_NAME = "sb_to_sced_summary.csv"
SB_CONFIDENCE_BAND_BY_FUEL_FILE_NAME = "sb_to_sced_confidence_band_by_fuel.csv"
SB_CONFIDENCE_BAND_BY_FUEL_PLOT_FILE_NAME = "sb_to_sced_confidence_band_by_fuel.png"
SB_UNMATCHED_BY_CDR_FUEL_FILE_NAME = "sb_to_sced_unmatched_by_cdr_fuel.csv"
SB_UNMATCHED_BY_CDR_FUEL_PLOT_FILE_NAME = "sb_to_sced_unmatched_by_cdr_fuel.png"
SCED_COVERAGE_DETAIL_FILE_NAME = "sced_coverage_from_sb_detail.csv"
SCED_COVERAGE_SUMMARY_FILE_NAME = "sced_coverage_from_sb_summary.csv"
SCED_COVERAGE_PLOT_FILE_NAME = "sced_coverage_from_sb_by_resource_type.png"
PUN_SB_PRESENCE_FILE_NAME = "pun_generation_report_with_sb_flag.csv"

SCED_RESOURCE_REQUIRED_COLUMNS = [
    "Resource Name",
    "Resource Type",
    "final_sced_time_stamp",
    "Base Point_avg",
    "Start Up Cold Offer_avg",
    "Start Up Hot Offer_avg",
    "Start Up Inter Offer_avg",
    "Min Gen Cost_avg",
]
SB_REQUIRED_COLUMNS = ["unit_name", "cdr_unit_code", "cdr_fuel", "county", "cdr_capacity_mw"]
SCED_NAME_REQUIRED_COLUMNS = price_matching.SCED_NAME_REQUIRED_COLUMNS
YES_REQUIRED_COLUMNS = price_matching.YES_REQUIRED_COLUMNS
RTLMP_REQUIRED_COLUMNS = price_matching.RTLMP_REQUIRED_COLUMNS
RESOURCE_NODE_MAPPING_REQUIRED_COLUMNS = price_matching.RESOURCE_NODE_MAPPING_REQUIRED_COLUMNS
PUN_REQUIRED_COLUMNS = ["SubstationName", "UnitName"]

SB_OVERRIDE_KEY_COLUMNS = ["unit_name", "cdr_unit_code", "cdr_gen_id"]
SB_OVERRIDE_COLUMNS = [
    "matched_sced_node",
    "sb_to_sced_match_method",
    "sb_to_sced_match_score",
    "sb_to_sced_match_status",
]

FUZZY_RESOURCE_CODE_CUTOFF = 0.82
FUZZY_UNIT_NAME_CUTOFF = 0.82
STRONG_UNIT_NAME_FUZZY_CUTOFF = 0.94
CONFIDENCE_BAND_ORDER = ["High", "Medium", "Low", "Unmatched"]
UNMATCHED_STATUS_ORDER = [
    "unmatched",
    "unmatched_cancelled",
    "unmatched_retired",
    "unmatched_inactive",
    "unmatched_in_progress",
]
UNMATCHED_STATUS_LABELS = {
    "unmatched": "Unmatched",
    "unmatched_cancelled": "Cancelled",
    "unmatched_retired": "Retired",
    "unmatched_inactive": "Inactive",
    "unmatched_in_progress": "In progress",
}


def load_csv(csv_path: str, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
    return df


def ensure_output_dir(output_dir: str = OUTPUT_DIR) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"').upper()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(value) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalize_text(value))


def node_family(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if "_" in text:
        return text.split("_", 1)[0]
    match = re.match(r"[A-Z0-9]+", text)
    return match.group(0) if match else text


def parse_numeric(value):
    return pd.to_numeric(value, errors="coerce")


def similarity_score(left: str, right: str) -> float:
    left = normalize_text(left)
    right = normalize_text(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def fuel_key(value) -> str:
    text = normalize_text(value)
    aliases = {
        "NATURAL GAS": "GAS",
        "NG": "GAS",
        "GAS": "GAS",
        "COAL": "COAL",
        "LIGNITE": "COAL",
        "NUCLEAR": "NUC",
        "NUC": "NUC",
        "WIND": "WIND",
        "WND": "WIND",
        "SOLAR": "SOLAR",
        "PV": "SOLAR",
        "PVGR": "SOLAR",
        "BATTERY": "BESS",
        "BESS": "BESS",
        "ESR": "BESS",
        "HYDRO": "HYDRO",
    }
    if text in aliases:
        return aliases[text]
    if "WIND" in text:
        return "WIND"
    if "SOLAR" in text or text.startswith("PV"):
        return "SOLAR"
    if "NUC" in text:
        return "NUC"
    if "COAL" in text or "LIGNITE" in text:
        return "COAL"
    if "GAS" in text:
        return "GAS"
    if "BAT" in text or "BESS" in text or "STORAGE" in text:
        return "BESS"
    return text


def extract_resource_unit_tokens(resource_name: str) -> set[str]:
    text = normalize_text(resource_name)
    tokens = set(re.findall(r"[A-Z]+\d+[A-Z]*|\d+[A-Z]*", text))
    suffix_match = re.search(r"(?:_|-)([A-Z]*\d+[A-Z]*)$", text)
    if suffix_match:
        tokens.add(suffix_match.group(1))
    for token in list(tokens):
        number_match = re.search(r"(\d+)$", token)
        if number_match:
            tokens.add(number_match.group(1))
            tokens.add(f"U{number_match.group(1)}")
            tokens.add(f"UNIT{number_match.group(1)}")
    return tokens


def split_plant_unit_descriptor(value: str) -> tuple[str, str]:
    text = normalize_text(value)
    if not text:
        return "", ""
    if "_" in text:
        plant, unit = text.split("_", 1)
        return plant.strip(), unit.strip()

    unit_match = re.search(
        r"\b((?:UNIT|U|GEN|G|GT|CT|ST|TG|GTG|CTG|STG)\s*-?\s*\d+[A-Z]*)\s*$",
        text,
    )
    if unit_match:
        return text[: unit_match.start()].strip(), normalize_text(unit_match.group(1)).replace(" ", "")
    return text, ""


def unit_tokens_from_text(value: str) -> set[str]:
    text = normalize_text(value)
    tokens = extract_resource_unit_tokens(text)
    unit_match = re.search(
        r"\b((?:UNIT|U|GEN|G|GT|CT|ST|TG|GTG|CTG|STG)\s*-?\s*\d+[A-Z]*)\b",
        text,
    )
    if unit_match:
        compact = normalize_text(unit_match.group(1)).replace(" ", "")
        tokens.add(compact)
        number_match = re.search(r"(\d+)", compact)
        if number_match:
            tokens.add(number_match.group(1))
            tokens.add(f"U{number_match.group(1)}")
            tokens.add(f"UNIT{number_match.group(1)}")
    return tokens


def derive_sb_plant_descriptor(row: pd.Series) -> str:
    cdr_unit_code = normalize_text(row.get("cdr_unit_code", ""))
    if "_" in cdr_unit_code:
        return cdr_unit_code.split("_", 1)[0]

    unit_name = normalize_text(row.get("unit_name", ""))
    cdr_gen_id = normalize_text(row.get("cdr_gen_id", "")).replace(" ", "")
    if cdr_gen_id and unit_name.endswith(cdr_gen_id):
        return unit_name[: -len(cdr_gen_id)].strip()

    plant, _ = split_plant_unit_descriptor(unit_name)
    return plant


def derive_sb_unit_descriptor(row: pd.Series) -> str:
    cdr_gen_id = normalize_text(row.get("cdr_gen_id", "")).replace(" ", "")
    if cdr_gen_id:
        return cdr_gen_id
    cdr_unit_code = normalize_text(row.get("cdr_unit_code", ""))
    if "_" in cdr_unit_code:
        return cdr_unit_code.split("_", 1)[1].strip()
    _, unit = split_plant_unit_descriptor(row.get("unit_name", ""))
    return unit


def prepare_sced_resource_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.rename(
        columns={
            "Resource Name": "resource_name",
            "Resource Type": "resource_type",
            "Base Point_avg": "avg_base_point",
            "Start Up Cold Offer_avg": "avg_start_up_cold_offer",
            "Start Up Hot Offer_avg": "avg_start_up_hot_offer",
            "Start Up Inter Offer_avg": "avg_start_up_inter_offer",
            "Min Gen Cost_avg": "avg_min_gen_cost",
        }
    )
    prepared["resource_name_norm"] = prepared["resource_name"].map(normalize_text)
    prepared["resource_name_key"] = prepared["resource_name"].map(normalize_key)
    prepared["resource_family"] = prepared["resource_name_norm"].map(node_family)
    prepared["resource_type_norm"] = prepared["resource_type"].map(fuel_key)
    name_parts = prepared["resource_name"].map(split_plant_unit_descriptor)
    prepared["sced_plant_descriptor"] = name_parts.map(lambda parts: parts[0])
    prepared["sced_plant_descriptor_key"] = prepared["sced_plant_descriptor"].map(normalize_key)
    prepared["sced_unit_descriptor"] = name_parts.map(lambda parts: parts[1])
    prepared["resource_unit_tokens"] = prepared["resource_name"].map(extract_resource_unit_tokens)
    prepared["resource_unit_tokens"] = [
        tokens.union(unit_tokens_from_text(unit_descriptor))
        for tokens, unit_descriptor in zip(prepared["resource_unit_tokens"], prepared["sced_unit_descriptor"])
    ]

    for column in [
        "avg_base_point",
        "avg_start_up_cold_offer",
        "avg_start_up_hot_offer",
        "avg_start_up_inter_offer",
        "avg_min_gen_cost",
    ]:
        prepared[column] = parse_numeric(prepared[column])
    return prepared


def prepare_sb_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["sb_unit_name_norm"] = prepared["unit_name"].map(normalize_text)
    prepared["sb_unit_name_key"] = prepared["unit_name"].map(normalize_key)
    prepared["sb_cdr_unit_code_norm"] = prepared["cdr_unit_code"].map(normalize_text)
    prepared["sb_cdr_unit_code_key"] = prepared["cdr_unit_code"].map(normalize_key)
    prepared["sb_cdr_gen_id_norm"] = prepared["cdr_gen_id"].map(normalize_text) if "cdr_gen_id" in prepared.columns else ""
    prepared["sb_cdr_inr_norm"] = prepared["cdr_inr"].map(normalize_text) if "cdr_inr" in prepared.columns else ""
    prepared["sb_county_norm"] = prepared["county"].map(normalize_text)
    prepared["sb_fuel_norm"] = prepared["cdr_fuel"].map(fuel_key)
    prepared["sb_capacity_mw"] = parse_numeric(prepared["cdr_capacity_mw"])
    prepared["sb_plant_descriptor"] = prepared.apply(derive_sb_plant_descriptor, axis=1)
    prepared["sb_plant_descriptor_key"] = prepared["sb_plant_descriptor"].map(normalize_key)
    prepared["sb_unit_descriptor"] = prepared.apply(derive_sb_unit_descriptor, axis=1)

    for source_column, norm_column, key_column in [
        ("eia_unit_code", "sb_eia_unit_code_norm", "sb_eia_unit_code_key"),
        ("eia_generator_id", "sb_eia_generator_id_norm", "sb_eia_generator_id_key"),
        ("cnl_inr", "sb_cnl_inr_norm", "sb_cnl_inr_key"),
        ("gis_inr", "sb_gis_inr_norm", "sb_gis_inr_key"),
    ]:
        if source_column in prepared.columns:
            prepared[norm_column] = prepared[source_column].map(normalize_text)
            prepared[key_column] = prepared[source_column].map(normalize_key)
        else:
            prepared[norm_column] = ""
            prepared[key_column] = ""
    return prepared


def filter_sb_rows_for_matching(sb_df: pd.DataFrame) -> pd.DataFrame:
    unit_name_present = sb_df["unit_name"].fillna("").astype(str).str.strip().ne("")
    capacity_positive = parse_numeric(sb_df["cdr_capacity_mw"]).fillna(0).gt(0)
    return sb_df[unit_name_present & capacity_positive].copy().reset_index(drop=True)


def build_sced_lookups(sced_df: pd.DataFrame) -> dict[str, object]:
    rows = sced_df.to_dict("records")
    by_resource_name: dict[str, list[dict]] = {}
    by_resource_key: dict[str, list[dict]] = {}
    by_resource_family: dict[str, list[dict]] = {}
    by_plant_key: dict[str, list[dict]] = {}
    resource_names: list[str] = []
    plant_keys: list[str] = []

    for row in rows:
        if row["resource_name_norm"]:
            by_resource_name.setdefault(row["resource_name_norm"], []).append(row)
            if row["resource_name_norm"] not in resource_names:
                resource_names.append(row["resource_name_norm"])
        if row["resource_name_key"]:
            by_resource_key.setdefault(row["resource_name_key"], []).append(row)
        if row["resource_family"]:
            by_resource_family.setdefault(row["resource_family"], []).append(row)
        if row["sced_plant_descriptor_key"]:
            by_plant_key.setdefault(row["sced_plant_descriptor_key"], []).append(row)
            if row["sced_plant_descriptor_key"] not in plant_keys:
                plant_keys.append(row["sced_plant_descriptor_key"])

    return {
        "by_resource_name": by_resource_name,
        "by_resource_key": by_resource_key,
        "by_resource_family": by_resource_family,
        "by_plant_key": by_plant_key,
        "resource_names": resource_names,
        "plant_keys": plant_keys,
    }


def sb_code_fields(sb_row: dict) -> list[tuple[str, str, str]]:
    fields = [
        ("cdr_unit_code", sb_row.get("sb_cdr_unit_code_norm", ""), sb_row.get("sb_cdr_unit_code_key", "")),
        ("eia_unit_code", sb_row.get("sb_eia_unit_code_norm", ""), sb_row.get("sb_eia_unit_code_key", "")),
        ("cnl_inr", sb_row.get("sb_cnl_inr_norm", ""), sb_row.get("sb_cnl_inr_key", "")),
        ("gis_inr", sb_row.get("sb_gis_inr_norm", ""), sb_row.get("sb_gis_inr_key", "")),
    ]
    return [(name, norm_value, key_value) for name, norm_value, key_value in fields if norm_value or key_value]


def sb_unit_hint_tokens(sb_row: dict) -> set[str]:
    tokens: set[str] = set()
    for value in [
        sb_row.get("sb_unit_name_norm", ""),
        sb_row.get("sb_unit_descriptor", ""),
        sb_row.get("sb_cdr_gen_id_norm", ""),
        sb_row.get("sb_cdr_inr_norm", ""),
        sb_row.get("sb_eia_generator_id_norm", ""),
    ]:
        text = normalize_text(value).replace(" ", "")
        if not text:
            continue
        tokens.add(text)
        tokens.update(unit_tokens_from_text(text))
        number_match = re.search(r"(\d+)$", text)
        if number_match:
            tokens.add(number_match.group(1))
            tokens.add(f"U{number_match.group(1)}")
            tokens.add(f"UNIT{number_match.group(1)}")
    return tokens


def matched_sced_fields(sced_row: dict | None) -> dict:
    if not sced_row:
        return {
            "alternative_sced_node": "",
            "matched_sced_node": "",
            "matched_sced_resource_type": "",
            "matched_sced_plant_descriptor": "",
            "matched_sced_unit_descriptor": "",
            "matched_sced_final_sced_time_stamp": "",
            "matched_sced_base_point_avg": pd.NA,
            "matched_sced_start_up_cold_offer_avg": pd.NA,
            "matched_sced_start_up_hot_offer_avg": pd.NA,
            "matched_sced_start_up_inter_offer_avg": pd.NA,
            "matched_sced_min_gen_cost_avg": pd.NA,
        }
    return {
        "alternative_sced_node": "",
        "matched_sced_node": sced_row.get("resource_name", ""),
        "matched_sced_resource_type": sced_row.get("resource_type", ""),
        "matched_sced_plant_descriptor": sced_row.get("sced_plant_descriptor", ""),
        "matched_sced_unit_descriptor": sced_row.get("sced_unit_descriptor", ""),
        "matched_sced_final_sced_time_stamp": sced_row.get("final_sced_time_stamp", ""),
        "matched_sced_base_point_avg": sced_row.get("avg_base_point", pd.NA),
        "matched_sced_start_up_cold_offer_avg": sced_row.get("avg_start_up_cold_offer", pd.NA),
        "matched_sced_start_up_hot_offer_avg": sced_row.get("avg_start_up_hot_offer", pd.NA),
        "matched_sced_start_up_inter_offer_avg": sced_row.get("avg_start_up_inter_offer", pd.NA),
        "matched_sced_min_gen_cost_avg": sced_row.get("avg_min_gen_cost", pd.NA),
    }


def score_sb_candidate(sb_row: dict, sced_row: dict, base_score: int, method: str) -> dict:
    score = base_score
    if sb_row.get("sb_fuel_norm") and sb_row["sb_fuel_norm"] == sced_row.get("resource_type_norm", ""):
        score += 12

    unit_hint_tokens = sb_unit_hint_tokens(sb_row)
    if unit_hint_tokens and unit_hint_tokens.intersection(sced_row.get("resource_unit_tokens", set())):
        score += 15

    capacity = sb_row.get("sb_capacity_mw", pd.NA)
    base_point = sced_row.get("avg_base_point", pd.NA)
    if pd.notna(capacity) and pd.notna(base_point) and float(capacity) > 0:
        ratio = abs(float(capacity) - float(base_point)) / float(capacity)
        if ratio <= 0.10:
            score += 8
        elif ratio <= 0.25:
            score += 4

    score += int(18 * similarity_score(sb_row.get("sb_cdr_unit_code_norm", ""), sced_row.get("resource_name_norm", "")))
    score += int(10 * similarity_score(sb_row.get("sb_unit_name_norm", ""), sced_row.get("resource_name_norm", "")))

    return {
        **matched_sced_fields(sced_row),
        "sb_plant_descriptor": sb_row.get("sb_plant_descriptor", ""),
        "sb_unit_descriptor": sb_row.get("sb_unit_descriptor", ""),
        "plant_match_method": "",
        "plant_match_score": pd.NA,
        "unit_assignment_method": method,
        "unit_assignment_score": score,
        "sb_to_sced_match_method": method,
        "sb_to_sced_match_score": score,
    }


def collect_sb_candidates(sb_row: dict, lookups: dict[str, object]) -> list[dict]:
    candidates: list[dict] = []
    seen_nodes: set[tuple[str, str]] = set()

    def add_candidates(rows: list[dict], base_score: int, method: str):
        for sced_row in rows:
            key = (method, sced_row["resource_name"])
            if key in seen_nodes:
                continue
            seen_nodes.add(key)
            candidates.append(score_sb_candidate(sb_row, sced_row, base_score, method))

    for field_name, norm_value, key_value in sb_code_fields(sb_row):
        if norm_value:
            add_candidates(lookups["by_resource_name"].get(norm_value, []), 230, f"{field_name}_exact")
        if key_value:
            add_candidates(lookups["by_resource_key"].get(key_value, []), 220, f"{field_name}_key_exact")

    cdr_unit_code_norm = sb_row.get("sb_cdr_unit_code_norm", "")
    if cdr_unit_code_norm:
        for resource_name in get_close_matches(
            cdr_unit_code_norm,
            lookups["resource_names"],
            n=5,
            cutoff=FUZZY_RESOURCE_CODE_CUTOFF,
        ):
            add_candidates(lookups["by_resource_name"].get(resource_name, []), 165, "cdr_unit_code_fuzzy")

    unit_name_norm = sb_row.get("sb_unit_name_norm", "")
    if unit_name_norm:
        add_candidates(lookups["by_resource_key"].get(normalize_key(unit_name_norm), []), 205, "unit_name_key_exact")
        for resource_name in get_close_matches(
            unit_name_norm,
            lookups["resource_names"],
            n=5,
            cutoff=FUZZY_UNIT_NAME_CUTOFF,
        ):
            add_candidates(lookups["by_resource_name"].get(resource_name, []), 135, "unit_name_fuzzy")
        for resource_name in get_close_matches(
            unit_name_norm,
            lookups["resource_names"],
            n=3,
            cutoff=STRONG_UNIT_NAME_FUZZY_CUTOFF,
        ):
            add_candidates(lookups["by_resource_name"].get(resource_name, []), 190, "unit_name_strong_fuzzy")

    for _, norm_value, _ in sb_code_fields(sb_row):
        family = node_family(norm_value)
        if family:
            add_candidates(lookups["by_resource_family"].get(family, []), 120, "resource_family")

    return candidates


def choose_best_sb_match(candidate_rows: list[dict]) -> dict:
    if not candidate_rows:
        return {
            **matched_sced_fields(None),
            "sb_plant_descriptor": "",
            "sb_unit_descriptor": "",
            "plant_match_method": "",
            "plant_match_score": pd.NA,
            "unit_assignment_method": "",
            "unit_assignment_score": pd.NA,
            "sb_to_sced_match_method": "",
            "sb_to_sced_match_score": pd.NA,
            "sb_to_sced_match_status": "unmatched",
        }

    sorted_rows = sorted(
        candidate_rows,
        key=lambda row: (row["sb_to_sced_match_score"], row["matched_sced_node"]),
        reverse=True,
    )
    best = sorted_rows[0].copy()
    top_score = best["sb_to_sced_match_score"]
    top_nodes = []
    for row in sorted_rows:
        if row["sb_to_sced_match_score"] != top_score:
            break
        if row["matched_sced_node"] and row["matched_sced_node"] not in top_nodes:
            top_nodes.append(row["matched_sced_node"])
    best["sb_to_sced_match_status"] = "ambiguous" if len(top_nodes) > 1 else "matched"
    return best


def is_strong_direct_match(match: dict) -> bool:
    method = str(match.get("sb_to_sced_match_method", ""))
    score = match.get("sb_to_sced_match_score", pd.NA)
    if method.endswith("_exact") or method.endswith("_key_exact"):
        return True
    if method == "unit_name_strong_fuzzy" and pd.notna(score) and float(score) >= 205:
        return True
    return False


def plant_match_score(sb_group: list[dict], sced_rows: list[dict], method: str, base_score: int) -> int:
    score = base_score
    sb_fuels = {row.get("sb_fuel_norm", "") for row in sb_group if row.get("sb_fuel_norm", "")}
    sced_fuels = {row.get("resource_type_norm", "") for row in sced_rows if row.get("resource_type_norm", "")}
    if sb_fuels.intersection(sced_fuels):
        score += 10
    for sb_row in sb_group:
        for sced_row in sced_rows:
            if normalize_key(sb_row.get("sb_cdr_unit_code_norm", "")) == sced_row.get("resource_name_key", ""):
                score += 30
                break
    return score


def choose_sced_plant_group(sb_group: list[dict], lookups: dict[str, object]) -> tuple[list[dict], str, int]:
    if not sb_group:
        return [], "", 0

    plant_key = sb_group[0].get("sb_plant_descriptor_key", "")
    candidate_groups: list[tuple[list[dict], str, int]] = []
    if plant_key:
        exact_rows = lookups["by_plant_key"].get(plant_key, [])
        if exact_rows:
            candidate_groups.append((exact_rows, "plant_descriptor_exact", plant_match_score(sb_group, exact_rows, "plant_descriptor_exact", 180)))

        fuzzy_keys = get_close_matches(plant_key, lookups["plant_keys"], n=3, cutoff=0.88)
        for fuzzy_key in fuzzy_keys:
            if fuzzy_key == plant_key:
                continue
            rows = lookups["by_plant_key"].get(fuzzy_key, [])
            candidate_groups.append((rows, "plant_descriptor_fuzzy", plant_match_score(sb_group, rows, "plant_descriptor_fuzzy", 145)))

    if not candidate_groups:
        return [], "", 0

    best_rows, best_method, best_score = sorted(candidate_groups, key=lambda item: item[2], reverse=True)[0]
    return best_rows, best_method, best_score


def score_unit_pair(sb_row: dict, sced_row: dict) -> int:
    score = 0
    sb_tokens = sb_unit_hint_tokens(sb_row)
    sced_tokens = sced_row.get("resource_unit_tokens", set())
    if sb_tokens and sced_tokens and sb_tokens.intersection(sced_tokens):
        score += 80
    if normalize_key(sb_row.get("sb_cdr_unit_code_norm", "")) == sced_row.get("resource_name_key", ""):
        score += 120
    if sb_row.get("sb_fuel_norm") and sb_row["sb_fuel_norm"] == sced_row.get("resource_type_norm", ""):
        score += 15
    score += int(20 * similarity_score(sb_row.get("sb_cdr_unit_code_norm", ""), sced_row.get("resource_name_norm", "")))

    capacity = sb_row.get("sb_capacity_mw", pd.NA)
    base_point = sced_row.get("avg_base_point", pd.NA)
    if pd.notna(capacity) and pd.notna(base_point) and float(capacity) > 0:
        ratio = abs(float(capacity) - float(base_point)) / float(capacity)
        if ratio <= 0.10:
            score += 10
        elif ratio <= 0.25:
            score += 5
    return score


def best_basepoint_sced_row(sced_rows: list[dict], excluded_nodes: set[str] | None = None) -> dict | None:
    excluded_nodes = excluded_nodes or set()
    available_rows = [row for row in sced_rows if row.get("resource_name", "") not in excluded_nodes]
    if not available_rows:
        available_rows = sced_rows
    if not available_rows:
        return None
    return sorted(
        available_rows,
        key=lambda row: (
            -1 if pd.isna(row.get("avg_base_point", pd.NA)) else float(row.get("avg_base_point", 0)),
            row.get("resource_name", ""),
        ),
        reverse=True,
    )[0]


def build_plant_unit_match(
    sb_row: dict,
    sced_row: dict,
    plant_method: str,
    plant_score: int,
    unit_method: str,
    unit_score: int,
) -> dict:
    total_score = plant_score + unit_score
    return {
        **matched_sced_fields(sced_row),
        "sb_plant_descriptor": sb_row.get("sb_plant_descriptor", ""),
        "sb_unit_descriptor": sb_row.get("sb_unit_descriptor", ""),
        "plant_match_method": plant_method,
        "plant_match_score": plant_score,
        "unit_assignment_method": unit_method,
        "unit_assignment_score": unit_score,
        "sb_to_sced_match_method": f"{plant_method}:{unit_method}",
        "sb_to_sced_match_score": total_score,
        "sb_to_sced_match_status": "matched",
    }


def assign_units_within_plant(sb_group: list[dict], sced_rows: list[dict], plant_method: str, plant_score: int) -> dict[int, dict]:
    assignments: dict[int, dict] = {}
    used_sced_nodes: set[str] = set()
    pair_scores = []

    for sb_row in sb_group:
        for sced_row in sced_rows:
            pair_scores.append((score_unit_pair(sb_row, sced_row), sb_row, sced_row))

    for unit_score, sb_row, sced_row in sorted(pair_scores, key=lambda item: item[0], reverse=True):
        if unit_score <= 0:
            break
        if sb_row["_sb_match_index"] in assignments:
            continue
        if sced_row["resource_name"] in used_sced_nodes:
            continue
        assignments[sb_row["_sb_match_index"]] = build_plant_unit_match(
            sb_row,
            sced_row,
            plant_method,
            plant_score,
            "distinct_unit_token_match",
            unit_score,
        )
        used_sced_nodes.add(sced_row["resource_name"])

    remaining_sb_rows = [row for row in sb_group if row["_sb_match_index"] not in assignments]
    remaining_sced_count = len([row for row in sced_rows if row["resource_name"] not in used_sced_nodes])
    reuse_highest_basepoint = len(remaining_sb_rows) > remaining_sced_count

    for sb_row in remaining_sb_rows:
        sced_row = best_basepoint_sced_row(sced_rows, set() if reuse_highest_basepoint else used_sced_nodes)
        if not sced_row:
            continue
        unit_score = score_unit_pair(sb_row, sced_row)
        method = "highest_basepoint_reuse" if reuse_highest_basepoint else "highest_basepoint_unit"
        assignments[sb_row["_sb_match_index"]] = build_plant_unit_match(
            sb_row,
            sced_row,
            plant_method,
            plant_score,
            method,
            unit_score,
        )
        if not reuse_highest_basepoint:
            used_sced_nodes.add(sced_row["resource_name"])

    return assignments


def build_sb_matches(sb_df: pd.DataFrame, sced_prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sb_prepared = prepare_sb_df(sb_df)
    sb_prepared["_sb_match_index"] = range(len(sb_prepared))
    sced_lookups = build_sced_lookups(sced_prepared_df)

    match_by_index: dict[int, dict] = {}
    candidate_rows = []

    sb_records = sb_prepared.to_dict("records")
    for row in sb_records:
        row_index = row["_sb_match_index"]
        candidates = collect_sb_candidates(row, sced_lookups)
        for candidate in candidates:
            candidate_rows.append(
                {
                    "sb_row_index": row_index,
                    "unit_name": row.get("unit_name", ""),
                    "cdr_unit_code": row.get("cdr_unit_code", ""),
                    **candidate,
                }
            )
        best_match = choose_best_sb_match(candidates)
        if is_strong_direct_match(best_match):
            match_by_index[row_index] = best_match

    remaining_rows = [row for row in sb_records if row["_sb_match_index"] not in match_by_index]
    plant_groups: dict[str, list[dict]] = {}
    for row in remaining_rows:
        plant_key = row.get("sb_plant_descriptor_key", "")
        if plant_key:
            plant_groups.setdefault(plant_key, []).append(row)

    for sb_group in plant_groups.values():
        sced_rows, plant_method, plant_score = choose_sced_plant_group(sb_group, sced_lookups)
        if not sced_rows:
            continue
        match_by_index.update(assign_units_within_plant(sb_group, sced_rows, plant_method, plant_score))

    match_rows = []
    for row in sb_records:
        row_index = row["_sb_match_index"]
        if row_index in match_by_index:
            match_rows.append(match_by_index[row_index])
            continue
        fallback_match = choose_best_sb_match(collect_sb_candidates(row, sced_lookups))
        fallback_match["sb_plant_descriptor"] = row.get("sb_plant_descriptor", "")
        fallback_match["sb_unit_descriptor"] = row.get("sb_unit_descriptor", "")
        if not fallback_match.get("plant_match_method"):
            fallback_match["plant_match_method"] = ""
        if "plant_match_score" not in fallback_match:
            fallback_match["plant_match_score"] = pd.NA
        if not fallback_match.get("unit_assignment_method"):
            fallback_match["unit_assignment_method"] = fallback_match.get("sb_to_sced_match_method", "")
        if "unit_assignment_score" not in fallback_match:
            fallback_match["unit_assignment_score"] = fallback_match.get("sb_to_sced_match_score", pd.NA)
        match_rows.append(fallback_match)

    match_df = pd.DataFrame(match_rows)
    original_columns = list(sb_df.columns)
    output_df = pd.concat([sb_df.reset_index(drop=True), match_df.reset_index(drop=True)], axis=1)
    appended_columns = [column for column in output_df.columns if column not in original_columns]
    return output_df[original_columns + appended_columns], pd.DataFrame(candidate_rows)


def build_override_key(df: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    return df.apply(
        lambda row: "||".join(normalize_key(row.get(column, "")) for column in key_columns),
        axis=1,
    )


def manual_override_has_value(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def apply_sb_match_overrides(sb_matches_df: pd.DataFrame, sced_prepared_df: pd.DataFrame) -> pd.DataFrame:
    if not FINAL_SB_MATCHES_PATH:
        return sb_matches_df

    override_df = load_csv(FINAL_SB_MATCHES_PATH, SB_OVERRIDE_KEY_COLUMNS)
    override_columns = [column for column in SB_OVERRIDE_COLUMNS if column in override_df.columns]
    if not override_columns:
        return sb_matches_df

    merged_df = sb_matches_df.copy()
    merged_df["_override_key"] = build_override_key(merged_df, SB_OVERRIDE_KEY_COLUMNS)
    override_df["_override_key"] = build_override_key(override_df, SB_OVERRIDE_KEY_COLUMNS)
    override_df = override_df.drop_duplicates(subset="_override_key", keep="last").set_index("_override_key")

    for column in override_columns:
        override_values = merged_df["_override_key"].map(override_df[column])
        merged_df[column] = [
            override if manual_override_has_value(override) else current
            for current, override in zip(merged_df[column], override_values)
        ]

    merged_df = merged_df.drop(columns="_override_key")
    return hydrate_matched_sced_fields(merged_df, sced_prepared_df)


def hydrate_matched_sced_fields(sb_matches_df: pd.DataFrame, sced_prepared_df: pd.DataFrame) -> pd.DataFrame:
    output_df = sb_matches_df.copy()
    existing_alternative = (
        output_df["alternative_sced_node"].copy()
        if "alternative_sced_node" in output_df.columns
        else pd.Series([""] * len(output_df), index=output_df.index)
    )
    sced_lookup = {
        normalize_key(row["resource_name"]): row
        for row in sced_prepared_df.to_dict("records")
        if normalize_key(row.get("resource_name", ""))
    }

    sced_field_columns = list(matched_sced_fields(None).keys())
    hydrated_rows = []
    for _, row in output_df.iterrows():
        sced_row = sced_lookup.get(normalize_key(row.get("matched_sced_node", "")))
        hydrated_rows.append(matched_sced_fields(sced_row))

    hydrated_df = pd.DataFrame(hydrated_rows)
    for column in sced_field_columns:
        output_df[column] = hydrated_df[column]
    output_df["alternative_sced_node"] = existing_alternative.fillna("")

    manually_matched = output_df["matched_sced_node"].fillna("").astype(str).str.strip().ne("")
    output_df.loc[
        manually_matched & output_df["sb_to_sced_match_status"].eq("unmatched"),
        "sb_to_sced_match_status",
    ] = "matched"
    return output_df


def choose_esr_replacement(sb_row: pd.Series, esr_rows: list[dict]) -> dict | None:
    if not esr_rows:
        return None

    sb_tokens = sb_unit_hint_tokens(sb_row.to_dict())
    capacity = parse_numeric(pd.Series([sb_row.get("cdr_capacity_mw", pd.NA)])).iloc[0]

    def replacement_score(esr_row: dict) -> tuple[int, float, str]:
        score = 0
        if sb_tokens and sb_tokens.intersection(esr_row.get("resource_unit_tokens", set())):
            score += 100
        if pd.notna(capacity) and pd.notna(esr_row.get("avg_base_point", pd.NA)) and float(capacity) > 0:
            ratio = abs(float(capacity) - float(esr_row.get("avg_base_point", 0))) / float(capacity)
            if ratio <= 0.10:
                score += 20
            elif ratio <= 0.25:
                score += 10
        base_point = 0.0 if pd.isna(esr_row.get("avg_base_point", pd.NA)) else float(esr_row.get("avg_base_point", 0))
        return score, base_point, esr_row.get("resource_name", "")

    return sorted(esr_rows, key=replacement_score, reverse=True)[0]


def replace_storage_pwrstr_matches_with_esr(
    sb_matches_df: pd.DataFrame,
    sced_prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = sb_matches_df.copy()
    if "alternative_sced_node" not in output_df.columns:
        output_df["alternative_sced_node"] = ""

    esr_by_plant_key: dict[str, list[dict]] = {}
    for sced_row in sced_prepared_df.to_dict("records"):
        if normalize_text(sced_row.get("resource_type", "")) != "ESR":
            continue
        plant_key = sced_row.get("sced_plant_descriptor_key", "")
        if plant_key:
            esr_by_plant_key.setdefault(plant_key, []).append(sced_row)

    replacement_count = 0
    for index, row in output_df.iterrows():
        if normalize_text(row.get("cdr_fuel", "")) != "STORAGE":
            continue
        if normalize_text(row.get("matched_sced_resource_type", "")) != "PWRSTR":
            continue
        matched_node = row.get("matched_sced_node", "")
        if not normalize_text(matched_node):
            continue

        plant_key = normalize_key(row.get("matched_sced_plant_descriptor", "")) or normalize_key(split_plant_unit_descriptor(matched_node)[0])
        esr_row = choose_esr_replacement(row, esr_by_plant_key.get(plant_key, []))
        if not esr_row:
            continue

        original_node = row["matched_sced_node"]
        for column, value in matched_sced_fields(esr_row).items():
            if column == "alternative_sced_node":
                continue
            output_df.at[index, column] = value
        output_df.at[index, "alternative_sced_node"] = original_node
        output_df.at[index, "sb_to_sced_match_method"] = f"{row.get('sb_to_sced_match_method', '')}:storage_esr_replacement"
        output_df.at[index, "unit_assignment_method"] = f"{row.get('unit_assignment_method', '')}:storage_esr_replacement"
        replacement_count += 1

    if replacement_count:
        output_df["storage_esr_replacement_applied"] = output_df["alternative_sced_node"].fillna("").astype(str).str.strip().ne("")
    else:
        output_df["storage_esr_replacement_applied"] = False
    return output_df


def mark_unmatched_in_progress(
    sb_matches_df: pd.DataFrame,
    sced_prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    max_sced_timestamp = pd.to_datetime(
        sced_prepared_df["final_sced_time_stamp"],
        errors="coerce",
    ).max()
    if pd.isna(max_sced_timestamp):
        return sb_matches_df

    output_df = sb_matches_df.copy()
    is_unmatched = (
        output_df["matched_sced_node"].fillna("").astype(str).str.strip().eq("")
        & output_df["sb_to_sced_match_status"].eq("unmatched")
    )

    cdr_sync_date = pd.to_datetime(output_df.get("cdr_sync_date", pd.Series(index=output_df.index)), errors="coerce")
    gis_projected_cod = pd.to_datetime(output_df.get("gis_projected_cod", pd.Series(index=output_df.index)), errors="coerce")
    eia_operating_year = pd.to_numeric(
        output_df.get("eia_operating_year", pd.Series(index=output_df.index)),
        errors="coerce",
    )

    date_after_sced = cdr_sync_date.gt(max_sced_timestamp) | gis_projected_cod.gt(max_sced_timestamp)
    year_after_sced = eia_operating_year.gt(max_sced_timestamp.year)
    output_df.loc[is_unmatched & (date_after_sced | year_after_sced), "sb_to_sced_match_status"] = "unmatched_in_progress"
    return output_df


def mark_unmatched_retired(
    sb_matches_df: pd.DataFrame,
    sced_prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    max_sced_timestamp = pd.to_datetime(
        sced_prepared_df["final_sced_time_stamp"],
        errors="coerce",
    ).max()
    if pd.isna(max_sced_timestamp):
        return sb_matches_df

    output_df = sb_matches_df.copy()
    is_unmatched = (
        output_df["matched_sced_node"].fillna("").astype(str).str.strip().eq("")
        & output_df["sb_to_sced_match_status"].eq("unmatched")
    )
    planned_retirement_year = pd.to_numeric(
        output_df.get("eia_planned_retirement_year", pd.Series(index=output_df.index)),
        errors="coerce",
    )
    output_df.loc[
        is_unmatched & planned_retirement_year.lt(max_sced_timestamp.year),
        "sb_to_sced_match_status",
    ] = "unmatched_retired"
    return output_df


def mark_unmatched_cancelled(sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    output_df = sb_matches_df.copy()
    is_unmatched = (
        output_df["matched_sced_node"].fillna("").astype(str).str.strip().eq("")
        & output_df["sb_to_sced_match_status"].eq("unmatched")
    )
    if "cnl_cancel_date" not in output_df.columns:
        return output_df

    cancel_present = output_df["cnl_cancel_date"].fillna("").astype(str).str.strip().ne("")
    output_df.loc[is_unmatched & cancel_present, "sb_to_sced_match_status"] = "unmatched_cancelled"
    return output_df


def mark_unmatched_inactive(sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    output_df = sb_matches_df.copy()
    is_unmatched = (
        output_df["matched_sced_node"].fillna("").astype(str).str.strip().eq("")
        & output_df["sb_to_sced_match_status"].eq("unmatched")
    )

    inactive_present = pd.Series(False, index=output_df.index)
    for column in ["cnl_inactive", "cnl_inactive_date"]:
        if column in output_df.columns:
            inactive_present = inactive_present | output_df[column].fillna("").astype(str).str.strip().ne("")

    output_df.loc[is_unmatched & inactive_present, "sb_to_sced_match_status"] = "unmatched_inactive"
    return output_df


def apply_unmatched_status_priority(
    sb_matches_df: pd.DataFrame,
    sced_prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = mark_unmatched_cancelled(sb_matches_df)
    output_df = mark_unmatched_retired(output_df, sced_prepared_df)
    output_df = mark_unmatched_inactive(output_df)
    output_df = mark_unmatched_in_progress(output_df, sced_prepared_df)
    return output_df


def build_sced_price_matches(sced_prepared_df: pd.DataFrame) -> pd.DataFrame:
    sced_price_input_df = sced_prepared_df[["resource_name", "resource_type"]].rename(
        columns={"resource_type": "fuel_type"}
    )
    sced_name_df = price_matching.load_csv(
        SCED_NAME_LIST_PATH,
        SCED_NAME_REQUIRED_COLUMNS,
    )
    yes_df = price_matching.load_csv(
        YES_UNITS_LIST_PATH,
        YES_REQUIRED_COLUMNS,
    )
    rtlmp_bus_df = price_matching.load_csv(
        RTLMP_BUS_LIST_PATH,
        RTLMP_REQUIRED_COLUMNS,
    )
    rtlmp_df = price_matching.load_csv(
        RTLMP_LIST_PATH,
        RTLMP_REQUIRED_COLUMNS,
    )
    resource_node_mapping_df = price_matching.load_csv(
        RESOURCE_NODE_MAPPING_PATH,
        RESOURCE_NODE_MAPPING_REQUIRED_COLUMNS,
    )

    best_matches_df, _, _ = price_matching.build_match_tables(
        sced_price_input_df,
        sced_name_df,
        yes_df,
        rtlmp_bus_df,
        rtlmp_df,
        resource_node_mapping_df,
    )
    generated_simple_matches_df = price_matching.build_simple_matches(best_matches_df)
    return price_matching.choose_simple_matches_source(generated_simple_matches_df)


def add_price_nodes_to_sb_matches(
    sb_matches_df: pd.DataFrame,
    sced_price_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    price_lookup = (
        sced_price_matches_df.drop_duplicates(subset="resource_name", keep="last")
        .set_index("resource_name")[["price_code", "price_node_source"]]
        .to_dict("index")
    )

    output_df = sb_matches_df.copy()
    output_df["matched_price_node"] = output_df["matched_sced_node"].map(
        lambda resource_name: price_lookup.get(resource_name, {}).get("price_code", "")
        if pd.notna(resource_name)
        else ""
    )
    output_df["matched_price_node_source"] = output_df["matched_sced_node"].map(
        lambda resource_name: price_lookup.get(resource_name, {}).get("price_node_source", "")
        if pd.notna(resource_name)
        else ""
    )
    return output_df


def pun_combo_keys(pun_df: pd.DataFrame) -> set[str]:
    pun_combo_series = (
        pun_df["SubstationName"].fillna("").astype(str).str.strip()
        + "_"
        + pun_df["UnitName"].fillna("").astype(str).str.strip()
    )
    return {normalize_key(value) for value in pun_combo_series if normalize_key(value)}


def add_pun_presence_flag_to_sb_matches(pun_df: pd.DataFrame, sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    pun_keys = pun_combo_keys(pun_df)
    output_df = sb_matches_df.copy()
    output_df["in_pun_generation_report"] = [
        "Y"
        if (
            normalize_key(row.get("cdr_unit_code", "")) in pun_keys
            or normalize_key(row.get("unit_name", "")) in pun_keys
            or normalize_key(row.get("matched_sced_node", "")) in pun_keys
        )
        else "N"
        for _, row in output_df.iterrows()
    ]
    return output_df


def build_sb_pun_presence_output(pun_df: pd.DataFrame, sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    output_df = pun_df.copy()
    output_df["pun_combo"] = (
        output_df["SubstationName"].fillna("").astype(str).str.strip()
        + "_"
        + output_df["UnitName"].fillna("").astype(str).str.strip()
    )
    lookup_values = pd.concat(
        [
            sb_matches_df["cdr_unit_code"],
            sb_matches_df["unit_name"],
            sb_matches_df["matched_sced_node"],
        ],
        ignore_index=True,
    )
    lookup_keys = {normalize_key(value) for value in lookup_values.dropna().astype(str) if normalize_key(value)}
    output_df["in_sb_match_list"] = output_df["pun_combo"].map(
        lambda value: "Y" if normalize_key(value) in lookup_keys else "N"
    )
    return output_df


def build_sb_summary(sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    status_counts = sb_matches_df["sb_to_sced_match_status"].value_counts(dropna=False).to_dict()
    total_rows = len(sb_matches_df)
    return pd.DataFrame(
        [
            {
                "rows_total": total_rows,
                "rows_matched": int(status_counts.get("matched", 0)),
                "rows_ambiguous": int(status_counts.get("ambiguous", 0)),
                "rows_unmatched": int(status_counts.get("unmatched", 0)),
                "rows_unmatched_cancelled": int(status_counts.get("unmatched_cancelled", 0)),
                "rows_unmatched_retired": int(status_counts.get("unmatched_retired", 0)),
                "rows_unmatched_inactive": int(status_counts.get("unmatched_inactive", 0)),
                "rows_unmatched_in_progress": int(status_counts.get("unmatched_in_progress", 0)),
                "pct_matched_or_ambiguous": round(
                    (status_counts.get("matched", 0) + status_counts.get("ambiguous", 0)) / total_rows,
                    4,
                )
                if total_rows
                else 0.0,
            }
        ]
    )


def confidence_band(row: pd.Series) -> str:
    if row.get("sb_to_sced_match_status") == "unmatched" or pd.isna(row.get("sb_to_sced_match_score")):
        return "Unmatched"
    score = float(row["sb_to_sced_match_score"])
    if score >= 220:
        return "High"
    if score >= 165:
        return "Medium"
    return "Low"


def build_sb_confidence_band_by_fuel(sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = sb_matches_df[
        sb_matches_df["matched_sced_node"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    df["fuel"] = df["matched_sced_resource_type"].fillna("").astype(str).str.strip().replace("", "UNKNOWN")
    df["confidence_band"] = df.apply(confidence_band, axis=1)

    grouped = (
        df.groupby(["fuel", "confidence_band"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    totals = grouped.groupby("fuel", dropna=False)["rows"].sum().rename("fuel_total").reset_index()
    grouped = grouped.merge(totals, on="fuel", how="left")
    grouped["pct_of_fuel"] = (grouped["rows"] / grouped["fuel_total"]).round(4)

    all_fuels = sorted(df["fuel"].dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [all_fuels, CONFIDENCE_BAND_ORDER],
        names=["fuel", "confidence_band"],
    )
    grouped = (
        grouped.set_index(["fuel", "confidence_band"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    grouped["fuel_total"] = grouped.groupby("fuel")["rows"].transform("sum")
    grouped["pct_of_fuel"] = grouped["rows"].div(grouped["fuel_total"].where(grouped["fuel_total"].ne(0))).fillna(0).round(4)
    grouped["confidence_band"] = pd.Categorical(
        grouped["confidence_band"],
        categories=CONFIDENCE_BAND_ORDER,
        ordered=True,
    )
    return grouped.sort_values(["fuel", "confidence_band"]).reset_index(drop=True)


def save_sb_confidence_band_plot(confidence_df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the SB confidence band plot")

    output_path = ensure_output_dir(output_dir)
    plot_path = output_path / SB_CONFIDENCE_BAND_BY_FUEL_PLOT_FILE_NAME
    plot_df = confidence_df.pivot(index="fuel", columns="confidence_band", values="rows").fillna(0)
    plot_df = plot_df.reindex(columns=CONFIDENCE_BAND_ORDER, fill_value=0)

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 7),
        width=0.82,
        color={
            "High": "#2f7d32",
            "Medium": "#f0a202",
            "Low": "#c44900",
            "Unmatched": "#777777",
        },
    )
    ax.set_title("Matched SB to SCED Confidence Bands by SCED Fuel")
    ax.set_xlabel("Matched SCED fuel")
    ax.set_ylabel("SB rows")
    ax.legend(title="Confidence band")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def build_sb_unmatched_by_cdr_fuel(sb_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = sb_matches_df.copy()
    df["cdr_fuel_group"] = df["cdr_fuel"].fillna("").astype(str).str.strip().replace("", "UNKNOWN")
    df = df[df["sb_to_sced_match_status"].isin(UNMATCHED_STATUS_ORDER)].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cdr_fuel_group",
                "unmatched_status",
                "status_label",
                "rows",
                "fuel_unmatched_total",
                "pct_of_unmatched_fuel",
            ]
        )

    grouped = (
        df.groupby(["cdr_fuel_group", "sb_to_sced_match_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .rename(columns={"sb_to_sced_match_status": "unmatched_status"})
    )

    full_index = pd.MultiIndex.from_product(
        [sorted(df["cdr_fuel_group"].unique()), UNMATCHED_STATUS_ORDER],
        names=["cdr_fuel_group", "unmatched_status"],
    )
    grouped = (
        grouped.set_index(["cdr_fuel_group", "unmatched_status"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    grouped["status_label"] = grouped["unmatched_status"].map(UNMATCHED_STATUS_LABELS)
    grouped["fuel_unmatched_total"] = grouped.groupby("cdr_fuel_group")["rows"].transform("sum")
    grouped["pct_of_unmatched_fuel"] = (
        grouped["rows"]
        .div(grouped["fuel_unmatched_total"].where(grouped["fuel_unmatched_total"].ne(0)))
        .fillna(0)
        .round(4)
    )
    grouped["unmatched_status"] = pd.Categorical(
        grouped["unmatched_status"],
        categories=UNMATCHED_STATUS_ORDER,
        ordered=True,
    )
    return grouped.sort_values(["fuel_unmatched_total", "cdr_fuel_group", "unmatched_status"], ascending=[False, True, True]).reset_index(drop=True)


def save_sb_unmatched_by_cdr_fuel_plot(unmatched_df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the SB unmatched by CDR fuel plot")

    output_path = ensure_output_dir(output_dir)
    plot_path = output_path / SB_UNMATCHED_BY_CDR_FUEL_PLOT_FILE_NAME
    plot_df = unmatched_df[unmatched_df["fuel_unmatched_total"] > 0].copy()
    plot_df["status_label"] = pd.Categorical(
        plot_df["status_label"],
        categories=[UNMATCHED_STATUS_LABELS[status] for status in UNMATCHED_STATUS_ORDER],
        ordered=True,
    )
    pivot_df = (
        plot_df.pivot(index="cdr_fuel_group", columns="status_label", values="rows")
        .fillna(0)
    )
    fuel_order = (
        plot_df[["cdr_fuel_group", "fuel_unmatched_total"]]
        .drop_duplicates()
        .sort_values(["fuel_unmatched_total", "cdr_fuel_group"], ascending=[False, True])["cdr_fuel_group"]
    )
    status_columns = [UNMATCHED_STATUS_LABELS[status] for status in UNMATCHED_STATUS_ORDER]
    pivot_df = pivot_df.reindex(index=fuel_order, columns=status_columns, fill_value=0)

    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 7),
        width=0.82,
        color={
            "Unmatched": "#777777",
            "Cancelled": "#7b3294",
            "Retired": "#b35806",
            "Inactive": "#d95f02",
            "In progress": "#1b9e77",
        },
    )
    ax.set_title("Unmatched SB Rows by CDR Fuel and Status")
    ax.set_xlabel("SB CDR fuel")
    ax.set_ylabel("Unmatched SB rows")
    ax.legend(title="Unmatched status")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def build_sced_coverage_outputs(
    sced_prepared_df: pd.DataFrame,
    sb_matches_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    used_sced_keys = set()
    for column in ["matched_sced_node", "alternative_sced_node"]:
        if column not in sb_matches_df.columns:
            continue
        used_sced_keys.update(
            normalize_key(value)
            for value in sb_matches_df[column].dropna().astype(str)
            if normalize_key(value)
        )

    detail_df = sced_prepared_df[
        [
            "resource_name",
            "resource_type",
            "final_sced_time_stamp",
            "avg_base_point",
            "avg_start_up_cold_offer",
            "avg_start_up_hot_offer",
            "avg_start_up_inter_offer",
            "avg_min_gen_cost",
        ]
    ].copy()
    detail_df["sced_resource_key"] = detail_df["resource_name"].map(normalize_key)
    detail_df["is_used_by_sb"] = detail_df["sced_resource_key"].isin(used_sced_keys)
    detail_df["sced_to_sb_status"] = detail_df["is_used_by_sb"].map(lambda used: "used" if used else "missing")

    summary_df = (
        detail_df.groupby("resource_type", dropna=False)
        .agg(
            sced_rows_total=("resource_name", "size"),
            sced_rows_used=("is_used_by_sb", "sum"),
            sced_rows_missing=("is_used_by_sb", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    summary_df["pct_used"] = summary_df["sced_rows_used"].div(summary_df["sced_rows_total"]).fillna(0).round(4)
    summary_df["pct_missing"] = summary_df["sced_rows_missing"].div(summary_df["sced_rows_total"]).fillna(0).round(4)

    total_rows = len(detail_df)
    total_used = int(detail_df["is_used_by_sb"].sum())
    total_missing = total_rows - total_used
    total_df = pd.DataFrame(
        [
            {
                "resource_type": "ALL",
                "sced_rows_total": total_rows,
                "sced_rows_used": total_used,
                "sced_rows_missing": total_missing,
                "pct_used": round(total_used / total_rows, 4) if total_rows else 0.0,
                "pct_missing": round(total_missing / total_rows, 4) if total_rows else 0.0,
            }
        ]
    )
    summary_df = pd.concat([total_df, summary_df], ignore_index=True)
    summary_df = summary_df.sort_values(
        ["resource_type"],
        key=lambda series: series.eq("ALL").map({True: "", False: "Z"}) + series.astype(str),
    ).reset_index(drop=True)

    return detail_df, summary_df


def save_sced_coverage_plot(sced_coverage_summary_df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to save the SCED coverage plot")

    output_path = ensure_output_dir(output_dir)
    plot_path = output_path / SCED_COVERAGE_PLOT_FILE_NAME
    plot_df = sced_coverage_summary_df[sced_coverage_summary_df["resource_type"].ne("ALL")].copy()
    plot_df = plot_df.sort_values(["sced_rows_total", "resource_type"], ascending=[False, True])
    plot_df = plot_df.set_index("resource_type")[["sced_rows_used", "sced_rows_missing"]]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 7),
        width=0.82,
        color={
            "sced_rows_used": "#2f7d32",
            "sced_rows_missing": "#777777",
        },
    )
    ax.set_title("SCED Resources Used vs Missing from SB Matches by Resource Type")
    ax.set_xlabel("SCED resource type")
    ax.set_ylabel("SCED rows")
    ax.legend(["Used", "Missing"], title="SCED coverage")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def build_sced_price_summary(sced_price_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = sced_price_matches_df.copy()
    df["has_price_node"] = df["price_code"].fillna("").astype(str).str.strip().ne("")
    df["price_node_source"] = df["price_node_source"].fillna("").astype(str).str.strip().replace("", "unmatched")
    df["fuel_type"] = df["fuel_type"].fillna("").astype(str).str.strip().replace("", "UNKNOWN")

    total_rows = len(df)
    matched_rows = int(df["has_price_node"].sum())
    summary_rows = [
        {
            "section": "overall",
            "group": "ALL",
            "rows_total": total_rows,
            "rows_matched": matched_rows,
            "rows_unmatched": total_rows - matched_rows,
            "pct_matched": round(matched_rows / total_rows, 4) if total_rows else 0.0,
        }
    ]

    for source, source_df in df.groupby("price_node_source", dropna=False):
        rows_total = len(source_df)
        rows_matched = int(source_df["has_price_node"].sum())
        summary_rows.append(
            {
                "section": "price_node_source",
                "group": source,
                "rows_total": rows_total,
                "rows_matched": rows_matched,
                "rows_unmatched": rows_total - rows_matched,
                "pct_matched": round(rows_matched / rows_total, 4) if rows_total else 0.0,
            }
        )

    for fuel_type, fuel_df in df.groupby("fuel_type", dropna=False):
        rows_total = len(fuel_df)
        rows_matched = int(fuel_df["has_price_node"].sum())
        summary_rows.append(
            {
                "section": "fuel_type",
                "group": fuel_type,
                "rows_total": rows_total,
                "rows_matched": rows_matched,
                "rows_unmatched": rows_total - rows_matched,
                "pct_matched": round(rows_matched / rows_total, 4) if rows_total else 0.0,
            }
        )

    return pd.DataFrame(summary_rows)


def save_sb_outputs(
    sced_price_matches_df: pd.DataFrame,
    sced_price_summary_df: pd.DataFrame,
    sb_matches_df: pd.DataFrame,
    sb_candidates_df: pd.DataFrame,
    sb_summary_df: pd.DataFrame,
    sb_confidence_band_by_fuel_df: pd.DataFrame,
    sb_unmatched_by_cdr_fuel_df: pd.DataFrame,
    sced_coverage_detail_df: pd.DataFrame,
    sced_coverage_summary_df: pd.DataFrame,
    pun_presence_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    output_path = ensure_output_dir(output_dir)
    sced_price_matches_path = output_path / SCED_PRICE_SIMPLE_MATCHES_FILE_NAME
    sced_price_summary_path = output_path / SCED_PRICE_SUMMARY_FILE_NAME
    sb_matches_path = output_path / SB_MATCHES_FILE_NAME
    sb_candidates_path = output_path / SB_CANDIDATES_FILE_NAME
    sb_summary_path = output_path / SB_SUMMARY_FILE_NAME
    sb_confidence_band_by_fuel_path = output_path / SB_CONFIDENCE_BAND_BY_FUEL_FILE_NAME
    sb_unmatched_by_cdr_fuel_path = output_path / SB_UNMATCHED_BY_CDR_FUEL_FILE_NAME
    sced_coverage_detail_path = output_path / SCED_COVERAGE_DETAIL_FILE_NAME
    sced_coverage_summary_path = output_path / SCED_COVERAGE_SUMMARY_FILE_NAME
    pun_presence_path = output_path / PUN_SB_PRESENCE_FILE_NAME

    sced_price_matches_df.to_csv(sced_price_matches_path, index=False)
    sced_price_summary_df.to_csv(sced_price_summary_path, index=False)
    sb_matches_df.to_csv(sb_matches_path, index=False)
    sb_candidates_df.to_csv(sb_candidates_path, index=False)
    sb_summary_df.to_csv(sb_summary_path, index=False)
    sb_confidence_band_by_fuel_df.to_csv(sb_confidence_band_by_fuel_path, index=False)
    sb_unmatched_by_cdr_fuel_df.to_csv(sb_unmatched_by_cdr_fuel_path, index=False)
    sced_coverage_detail_df.to_csv(sced_coverage_detail_path, index=False)
    sced_coverage_summary_df.to_csv(sced_coverage_summary_path, index=False)
    pun_presence_df.to_csv(pun_presence_path, index=False)
    return (
        sced_price_matches_path,
        sced_price_summary_path,
        sb_matches_path,
        sb_candidates_path,
        sb_summary_path,
        sb_confidence_band_by_fuel_path,
        sb_unmatched_by_cdr_fuel_path,
        sced_coverage_detail_path,
        sced_coverage_summary_path,
        pun_presence_path,
    )


def main():
    sced_resource_df = load_csv(SCED_RESOURCE_LIST_PATH, SCED_RESOURCE_REQUIRED_COLUMNS)
    sb_df = filter_sb_rows_for_matching(load_csv(SB_LIST_PATH, SB_REQUIRED_COLUMNS))
    pun_df = load_csv(PUN_GENERATION_REPORT_PATH, PUN_REQUIRED_COLUMNS)

    sced_prepared_df = prepare_sced_resource_df(sced_resource_df)
    generated_sb_matches_df, sb_candidates_df = build_sb_matches(sb_df, sced_prepared_df)
    sb_matches_df = apply_sb_match_overrides(generated_sb_matches_df, sced_prepared_df)
    sb_matches_df = replace_storage_pwrstr_matches_with_esr(sb_matches_df, sced_prepared_df)
    sb_matches_df = apply_unmatched_status_priority(sb_matches_df, sced_prepared_df)
    sced_price_matches_df = build_sced_price_matches(sced_prepared_df)
    sced_price_summary_df = build_sced_price_summary(sced_price_matches_df)
    sb_matches_df = add_price_nodes_to_sb_matches(sb_matches_df, sced_price_matches_df)
    sb_matches_df = add_pun_presence_flag_to_sb_matches(pun_df, sb_matches_df)
    pun_presence_df = build_sb_pun_presence_output(pun_df, sb_matches_df)
    sb_summary_df = build_sb_summary(sb_matches_df)
    sb_confidence_band_by_fuel_df = build_sb_confidence_band_by_fuel(sb_matches_df)
    sb_unmatched_by_cdr_fuel_df = build_sb_unmatched_by_cdr_fuel(sb_matches_df)
    sced_coverage_detail_df, sced_coverage_summary_df = build_sced_coverage_outputs(
        sced_prepared_df,
        sb_matches_df,
    )

    (
        sced_price_matches_path,
        sced_price_summary_path,
        sb_matches_path,
        sb_candidates_path,
        sb_summary_path,
        sb_confidence_band_by_fuel_path,
        sb_unmatched_by_cdr_fuel_path,
        sced_coverage_detail_path,
        sced_coverage_summary_path,
        pun_presence_path,
    ) = save_sb_outputs(
        sced_price_matches_df,
        sced_price_summary_df,
        sb_matches_df,
        sb_candidates_df,
        sb_summary_df,
        sb_confidence_band_by_fuel_df,
        sb_unmatched_by_cdr_fuel_df,
        sced_coverage_detail_df,
        sced_coverage_summary_df,
        pun_presence_df,
    )
    sb_confidence_band_by_fuel_plot_path = save_sb_confidence_band_plot(sb_confidence_band_by_fuel_df)
    sb_unmatched_by_cdr_fuel_plot_path = save_sb_unmatched_by_cdr_fuel_plot(sb_unmatched_by_cdr_fuel_df)
    sced_coverage_plot_path = save_sced_coverage_plot(sced_coverage_summary_df)

    print(f"Saved SCED price matches to {sced_price_matches_path}")
    print(f"Saved SCED price summary to {sced_price_summary_path}")
    print(f"Saved SB matches to {sb_matches_path}")
    print(f"Saved SB candidates to {sb_candidates_path}")
    print(f"Saved SB summary to {sb_summary_path}")
    print(f"Saved SB confidence band by fuel data to {sb_confidence_band_by_fuel_path}")
    print(f"Saved SB confidence band by fuel plot to {sb_confidence_band_by_fuel_plot_path}")
    print(f"Saved SB unmatched by CDR fuel data to {sb_unmatched_by_cdr_fuel_path}")
    print(f"Saved SB unmatched by CDR fuel plot to {sb_unmatched_by_cdr_fuel_plot_path}")
    print(f"Saved SCED coverage detail to {sced_coverage_detail_path}")
    print(f"Saved SCED coverage summary to {sced_coverage_summary_path}")
    print(f"Saved SCED coverage plot to {sced_coverage_plot_path}")
    print(f"Saved PUN presence output to {pun_presence_path}")


if __name__ == "__main__":
    main()
