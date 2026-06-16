from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


SCED_PLANT_LIST_PATH = "/Users/pradyrao/Downloads/sced_unique_resource_name_type_pairs.csv"
SCED_NAME_LIST_PATH = "/Users/pradyrao/Downloads/sced_name_list.csv"
YES_UNITS_LIST_PATH = "/Users/pradyrao/Downloads/ERCOT_YES_units_list (1).csv"
RTLMP_BUS_LIST_PATH = "/Users/pradyrao/Downloads/rtlmp_bus_ercot_list.csv"
RTLMP_LIST_PATH = "/Users/pradyrao/Downloads/rtlmp_ercot_list.csv"
RESOURCE_NODE_MAPPING_PATH = "/Users/pradyrao/Downloads/SP_List_EB_Mapping 2/Resource_Node_to_Unit_02202026_094122.csv"
SB_LIST_PATH = "/Users/pradyrao/Downloads/sb_list.csv"
ERCOT_CDR_PATH = "/Users/pradyrao/Downloads/ercot_cdr_july.csv"
PUN_GENERATION_REPORT_PATH = "/Users/pradyrao/Downloads/PUN_Generation_Report/PUN_Generation_Report.csv"
FINAL_SIMPLE_MATCHES_PATH = None
FINAL_PLEXOS_MATCHES_PATH = None

OUTPUT_DIR = "/Users/pradyrao/VSCode/data-processing/data_processing/output/sced_to_sb_matching"
BEST_MATCHES_FILE_NAME = "sced_price_code_matches.csv"
ALL_CANDIDATES_FILE_NAME = "sced_price_code_candidates.csv"
SUMMARY_FILE_NAME = "sced_price_code_summary.csv"
SIMPLE_MATCHES_FILE_NAME = "sced_to_price_simple_matches.csv"
SB_MATCHES_FILE_NAME = "sb_to_sced_matches.csv"
SB_CANDIDATES_FILE_NAME = "sb_to_sced_candidates.csv"
SB_SUMMARY_FILE_NAME = "sb_to_sced_summary.csv"
PUN_SB_PRESENCE_FILE_NAME = "pun_generation_report_with_sb_flag.csv"
PLEXOS_MATCHES_FILE_NAME = "plexos_to_sced_price_matches.csv"
PLEXOS_TECH_SUMMARY_FILE_NAME = "plexos_match_summary_by_technology.csv"
SCED_PLEXOS_COVERAGE_DETAIL_FILE_NAME = "sced_coverage_from_plexos_detail.csv"
SCED_PLEXOS_COVERAGE_SUMMARY_FILE_NAME = "sced_coverage_from_plexos_summary.csv"
SCED_PLEXOS_DUPLICATES_FILE_NAME = "sced_coverage_from_plexos_duplicates.csv"
PUN_PLEXOS_PRESENCE_FILE_NAME = "pun_generation_report_with_plexos_flag.csv"
PLANT_RECONCILIATION_FILE_NAME = "plant_capacity_vs_basepoint_reconciliation.csv"
PLANT_RECONCILIATION_FLAGS_FILE_NAME = "plant_capacity_vs_basepoint_discrepancies.csv"
SCED_COVERAGE_PLOT_FILE_NAME = "sced_coverage_from_plexos_summary.png"

SCED_PLANT_REQUIRED_COLUMNS = [
    "Resource Name",
    "Resource Type",
    "final_sced_time_stamp",
    "Base Point_avg",
    "Start Up Cold Offer_avg",
    "Start Up Hot Offer_avg",
    "Start Up Inter Offer_avg",
    "Min Gen Cost_avg",
]
SCED_NAME_REQUIRED_COLUMNS = [
    "Unit Code",
    "Generator Station Code",
    "Generator Station Description",
    "Generator Type",
    "Nameplate Capacity (MW)",
]
YES_REQUIRED_COLUMNS = [
    "PLANTNAME",
    "UNITNAME",
    "NODENAME",
    "PRIMARYFUEL",
    "PRIMEMOVER",
    "NAMEPLATECAPACITY",
]
RTLMP_REQUIRED_COLUMNS = ["OBJECTTYPE", "OBJECTID", "NAME", "ISO"]
RESOURCE_NODE_MAPPING_REQUIRED_COLUMNS = ["RESOURCE_NODE", "UNIT_SUBSTATION", "UNIT_NAME"]
PLEXOS_REQUIRED_COLUMNS = ["Class", "Name", "CDR Name", "ERCOT_UnitCode", "County", "Fuel Reporting"]
SB_REQUIRED_COLUMNS = ["unit_name", "cdr_unit_code", "cdr_fuel", "county", "cdr_capacity_mw"]
ERCOT_CDR_REQUIRED_COLUMNS = ["UNIT NAME", "UNIT CODE", "COUNTY", "FUEL"]
PUN_REQUIRED_COLUMNS = ["SubstationName", "UnitName"]
FINAL_PLEXOS_MATCHES_REQUIRED_COLUMNS = [
    "Class",
    "Name",
    "Category",
    "Description",
    "EIA Plant Code",
    "EIA Gen Code",
    "EIA_UnitCode",
    "EIA_PM",
    "CDR Name",
    "ERCOT_INR Code",
    "ERCOT_UnitCode",
    "County",
    "CDR Zone",
    "Comments",
    "Fuel Reporting",
    "matched_sced_node",
    "matched_sced_station_code",
    "matched_sced_station_description",
    "matched_sced_fuel_type",
    "matched_sced_capacity_mw",
    "matched_price_node",
    "matched_price_node_source",
    "plexos_to_sced_match_method",
    "plexos_to_sced_match_score",
    "plexos_to_sced_match_status",
]
PLEXOS_OVERRIDE_KEY_COLUMNS = ["Class", "Name", "ERCOT_UnitCode", "CDR Name"]
PLEXOS_OVERRIDE_COLUMNS = [
    "matched_sced_node",
    "matched_sced_station_code",
    "matched_sced_station_description",
    "matched_sced_fuel_type",
    "matched_sced_capacity_mw",
    "matched_price_node",
    "matched_price_node_source",
    "plexos_to_sced_match_method",
    "plexos_to_sced_match_score",
    "plexos_to_sced_match_status",
]

FUZZY_PLANT_MATCH_CUTOFF = 0.88
FUZZY_STATION_TO_RTLMP_CUTOFF = 0.70
FUZZY_UNMATCHED_DESC_TO_YES_CUTOFF = 0.82
MAX_ALT_MATCHES = 5
MAX_YES_FAMILY_ROWS = 12
MAX_RTLMP_FAMILY_ROWS = 20
MAX_STATION_FUZZY_POOL = 80
MAX_RESOURCE_NODE_FUZZY_POOL = 50
PLEXOS_ERCOT_UNIT_FUZZY_CUTOFF = 0.82
PLEXOS_CDR_NAME_FUZZY_CUTOFF = 0.86
PLEXOS_NAME_FUZZY_CUTOFF = 0.82
PLEXOS_CDR_UNIT_NAME_FUZZY_CUTOFF = 0.84
PLEXOS_YES_PLANT_FUZZY_CUTOFF = 0.84
PLEXOS_ERCOT_UNIT_LATE_FUZZY_CUTOFF = 0.72
PLANT_DISCREPANCY_MW_THRESHOLD = 20.0
PLANT_DISCREPANCY_PCT_THRESHOLD = 0.25
WIND_BASEPOINT_CAPACITY_RATIO_THRESHOLD = 0.60
SOLAR_BASEPOINT_CAPACITY_RATIO_THRESHOLD = 0.50


@dataclass(frozen=True)
class CandidateScore:
    score: int
    method: str
    stage: str


def ensure_output_dir(output_dir: str = OUTPUT_DIR) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_csv(csv_path: str, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
    return df


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"').upper()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(value) -> str:
    text = normalize_text(value)
    return re.sub(r"[^A-Z0-9]+", "", text)


def node_family(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if "_" in text:
        return text.split("_", 1)[0]
    return re.match(r"[A-Z0-9]+", text).group(0) if re.match(r"[A-Z0-9]+", text) else text


def parse_numeric(value):
    return pd.to_numeric(value, errors="coerce")


def similarity_score(left: str, right: str) -> float:
    left = normalize_text(left)
    right = normalize_text(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def is_terminal_plexos_unit_descriptor(inner: str, full_text: str) -> bool:
    inner_norm = normalize_text(inner)
    compact = inner_norm.replace(" ", "")
    if not compact:
        return False
    if compact == "ERCOT":
        return True
    if re.fullmatch(r"(?:U)?\d+", compact):
        return True
    if re.fullmatch(r"[A-Z]{1,8}-?\d+[A-Z]*", compact):
        return True
    if "(" in full_text[: full_text.rfind("(")] and re.fullmatch(r"[A-Z0-9-]{1,12}", compact):
        return True
    return False


def parse_plexos_name_parts(name: str) -> dict[str, str]:
    raw = "" if pd.isna(name) else str(name).strip()
    normalized = normalize_text(raw)
    unit_hint = ""
    base = raw

    while True:
        parenthetical_match = re.search(r"\(([^)]+)\)\s*$", base)
        if not parenthetical_match:
            break
        inner = parenthetical_match.group(1).strip().upper()
        if is_terminal_plexos_unit_descriptor(inner, base):
            if not unit_hint and normalize_text(inner).replace(" ", "") != "ERCOT":
                unit_hint = normalize_text(inner).replace(" ", "")
            base = base[: parenthetical_match.start()].strip()
            continue
        break

    if not unit_hint:
        trailing_match = re.search(
            r"\b((?:U|UNIT|CT|ST|GT|CC|GEN|TG|HRSG|GTG|CTG|STG)\s*-?\s*\d+[A-Z]*)\s*$",
            raw,
            re.IGNORECASE,
        )
        if trailing_match:
            unit_hint = normalize_text(trailing_match.group(1)).replace(" ", "")
            base = raw[: trailing_match.start()].strip()

    if not unit_hint:
        trailing_numeric = re.search(r"\b(\d+)\s*$", raw)
        if trailing_numeric and "(" not in raw[trailing_numeric.start() - 1 :]:
            unit_hint = trailing_numeric.group(1)
            base = raw[: trailing_numeric.start()].strip()

    return {
        "name_base_norm": normalize_text(base if base else raw),
        "unit_hint": normalize_text(unit_hint).replace(" ", ""),
        "full_name_norm": normalized,
    }


def extract_resource_unit_tokens(resource_name: str) -> set[str]:
    text = normalize_text(resource_name)
    tokens: set[str] = set()
    if not text:
        return tokens

    for match in re.finditer(r"(?:_|^)(UNIT|U|CT|ST|GT|CC)?(\d+)(?:_|$)", text):
        prefix = (match.group(1) or "").upper()
        number = match.group(2)
        tokens.add(number)
        if prefix:
            tokens.add(f"{prefix}{number}")
        if prefix == "UNIT":
            tokens.add(f"U{number}")
        if prefix == "U":
            tokens.add(f"UNIT{number}")

    for match in re.finditer(r"([A-Z]+)(\d+)$", text):
        prefix = match.group(1).upper()
        number = match.group(2)
        if prefix in {"UNIT", "U", "CT", "ST", "GT", "CC", "G"}:
            tokens.add(number)
            tokens.add(f"{prefix}{number}")
    return tokens


def is_unit_like_token(token: str) -> bool:
    token = normalize_text(token)
    if not token:
        return False
    return bool(
        re.fullmatch(r"\d+", token)
        or re.fullmatch(r"(?:UNIT|U|CT|ST|GT|CC|G|GEN|TG|HRSG)\d+[A-Z]*", token)
    )


def derive_sced_plant_id(resource_name: str) -> str:
    text = normalize_text(resource_name)
    if not text:
        return ""
    parts = text.split("_")
    if len(parts) >= 3 and parts[1] and is_unit_like_token(parts[2]):
        return "_".join(parts[:2])
    return parts[0]


def fuel_key(value) -> str:
    text = normalize_text(value)
    if text in {"PVGR", "SOLAR"}:
        return "SOLAR"
    if "SOLAR" in text or "PHOTOVOLTAIC" in text:
        return "SOLAR"
    if text in {"WIND", "WINDG"} or "WIND" in text:
        return "WIND"
    if text in {"BESS", "STORAGE"} or "STORAGE" in text or "BATTERY" in text:
        return "STORAGE"
    if text in {"SCLE90", "NATURAL GAS", "GAS"} or "GAS" in text:
        return "GAS"
    if "COAL" in text:
        return "COAL"
    if "NUCLEAR" in text:
        return "NUCLEAR"
    if "HYDRO" in text:
        return "HYDRO"
    if "OIL" in text:
        return "OIL"
    return text


def extract_stems(resource_name: str) -> list[str]:
    text = normalize_text(resource_name)
    if not text:
        return []

    stems: list[str] = []

    def add(candidate: str):
        candidate = normalize_text(candidate)
        if candidate and candidate not in stems:
            stems.append(candidate)

    add(text)

    patterns = [
        r"_[A-Z]+[0-9]+$",
        r"_[0-9]+$",
        r"_UNIT[0-9A-Z]+$",
        r"_BESS[0-9A-Z]+$",
        r"_SOLAR[0-9A-Z]+$",
        r"_WIND[0-9A-Z]+$",
        r"_[A-Z]+$",
    ]

    reduced = text
    for pattern in patterns:
        next_value = re.sub(pattern, "", reduced)
        if next_value != reduced:
            add(next_value)
            reduced = next_value

    tokens = text.split("_")
    for size in range(len(tokens) - 1, 0, -1):
        add("_".join(tokens[:size]))
    add(tokens[0])

    return stems


def joined_examples(values, limit: int = 10) -> str:
    ordered = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= limit:
            break
    return " | ".join(ordered)


def prepare_sced_name_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["unit_code_norm"] = prepared["Unit Code"].map(normalize_text)
    prepared["station_code_norm"] = prepared["Generator Station Code"].map(normalize_text)
    prepared["station_desc_norm"] = prepared["Generator Station Description"].map(normalize_text)
    prepared["generator_type_norm"] = prepared["Generator Type"].map(normalize_text)
    prepared["capacity_mw"] = prepared["Nameplate Capacity (MW)"].map(parse_numeric)
    return prepared


def prepare_sced_plant_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["resource_name_norm"] = prepared["resource_name"].map(normalize_text)
    prepared["fuel_type_norm"] = prepared["fuel_type"].map(fuel_key)
    prepared["derived_stems"] = prepared["resource_name"].map(extract_stems)
    return prepared


def prepare_yes_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["plant_name_norm"] = prepared["PLANTNAME"].map(normalize_text)
    prepared["plant_name_key"] = prepared["PLANTNAME"].map(normalize_key)
    prepared["unit_name_norm"] = prepared["UNITNAME"].map(normalize_text)
    prepared["node_name_norm"] = prepared["NODENAME"].map(normalize_text)
    prepared["node_name_key"] = prepared["NODENAME"].map(normalize_key)
    prepared["yes_fuel_norm"] = prepared["PRIMARYFUEL"].map(fuel_key)
    prepared["yes_mover_norm"] = prepared["PRIMEMOVER"].map(normalize_text)
    prepared["yes_capacity_mw"] = prepared["NAMEPLATECAPACITY"].map(parse_numeric)
    prepared["node_family"] = prepared["node_name_norm"].map(node_family)
    return prepared


def prepare_rtlmp_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    prepared = df.copy()
    prepared["rtlmp_name_norm"] = prepared["NAME"].map(normalize_text)
    prepared["rtlmp_name_key"] = prepared["NAME"].map(normalize_key)
    prepared["rtlmp_family"] = prepared["rtlmp_name_norm"].map(node_family)
    prepared["price_node_source"] = source_label
    return prepared


def prepare_resource_node_mapping_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["resource_node_norm"] = prepared["RESOURCE_NODE"].map(normalize_text)
    prepared["unit_substation_norm"] = prepared["UNIT_SUBSTATION"].map(normalize_text)
    prepared["unit_name_norm"] = prepared["UNIT_NAME"].map(normalize_text)
    prepared["sced_key_norm"] = (
        prepared["unit_substation_norm"] + "_" + prepared["unit_name_norm"]
    )
    prepared["resource_node_family"] = prepared["resource_node_norm"].map(node_family)
    return prepared


def prepare_plexos_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["plexos_class_norm"] = prepared["Class"].map(normalize_text)
    prepared["plexos_name_norm"] = prepared["Name"].map(normalize_text)
    prepared["plexos_cdr_name_norm"] = prepared["CDR Name"].map(normalize_text)
    prepared["plexos_ercot_unit_norm"] = prepared["ERCOT_UnitCode"].map(normalize_text)
    prepared["plexos_fuel_norm"] = prepared["Fuel Reporting"].map(fuel_key)
    prepared["plexos_county_norm"] = prepared["County"].map(normalize_text) if "County" in prepared.columns else ""
    name_parts = prepared["Name"].map(parse_plexos_name_parts)
    prepared["plexos_name_base_norm"] = name_parts.map(lambda parts: parts["name_base_norm"])
    prepared["plexos_name_unit_hint"] = name_parts.map(lambda parts: parts["unit_hint"])
    return prepared


def prepare_sced_unique_df(df: pd.DataFrame) -> pd.DataFrame:
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
    prepared["fuel_type"] = prepared["resource_type"]
    prepared["resource_name_norm"] = prepared["resource_name"].map(normalize_text)
    prepared["resource_name_key"] = prepared["resource_name"].map(normalize_key)
    prepared["resource_family"] = prepared["resource_name_norm"].map(node_family)
    prepared["resource_type_norm"] = prepared["resource_type"].map(fuel_key)
    prepared["resource_unit_tokens"] = prepared["resource_name"].map(extract_resource_unit_tokens)
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
    prepared["sb_type_norm"] = prepared["type"].map(normalize_text) if "type" in prepared.columns else ""
    prepared["sb_capacity_mw"] = parse_numeric(prepared["cdr_capacity_mw"])

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


def build_sced_unique_lookups(sced_df: pd.DataFrame) -> dict[str, object]:
    rows = sced_df.to_dict("records")
    by_resource_name: dict[str, list[dict]] = {}
    by_resource_key: dict[str, list[dict]] = {}
    by_resource_family: dict[str, list[dict]] = {}
    resource_names: list[str] = []

    for row in rows:
        if row["resource_name_norm"]:
            by_resource_name.setdefault(row["resource_name_norm"], []).append(row)
            if row["resource_name_norm"] not in resource_names:
                resource_names.append(row["resource_name_norm"])
        if row["resource_name_key"]:
            by_resource_key.setdefault(row["resource_name_key"], []).append(row)
        if row["resource_family"]:
            by_resource_family.setdefault(row["resource_family"], []).append(row)

    return {
        "by_resource_name": by_resource_name,
        "by_resource_key": by_resource_key,
        "by_resource_family": by_resource_family,
        "resource_names": resource_names,
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
        sb_row.get("sb_cdr_gen_id_norm", ""),
        sb_row.get("sb_cdr_inr_norm", ""),
        sb_row.get("sb_eia_generator_id_norm", ""),
    ]:
        text = normalize_text(value).replace(" ", "")
        if not text:
            continue
        tokens.add(text)
        number_match = re.search(r"(\d+)$", text)
        if number_match:
            tokens.add(number_match.group(1))
            tokens.add(f"U{number_match.group(1)}")
            tokens.add(f"UNIT{number_match.group(1)}")
    return tokens


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
        "matched_sced_node": sced_row.get("resource_name", ""),
        "matched_sced_resource_type": sced_row.get("resource_type", ""),
        "matched_sced_final_sced_time_stamp": sced_row.get("final_sced_time_stamp", ""),
        "matched_sced_base_point_avg": sced_row.get("avg_base_point", pd.NA),
        "matched_sced_start_up_cold_offer_avg": sced_row.get("avg_start_up_cold_offer", pd.NA),
        "matched_sced_start_up_hot_offer_avg": sced_row.get("avg_start_up_hot_offer", pd.NA),
        "matched_sced_start_up_inter_offer_avg": sced_row.get("avg_start_up_inter_offer", pd.NA),
        "matched_sced_min_gen_cost_avg": sced_row.get("avg_min_gen_cost", pd.NA),
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
            add_candidates(
                lookups["by_resource_name"].get(norm_value, []),
                230,
                f"{field_name}_exact",
            )
        if key_value:
            add_candidates(
                lookups["by_resource_key"].get(key_value, []),
                220,
                f"{field_name}_key_exact",
            )

    cdr_unit_code_norm = sb_row.get("sb_cdr_unit_code_norm", "")
    if cdr_unit_code_norm:
        fuzzy_resource_names = get_close_matches(
            cdr_unit_code_norm,
            lookups["resource_names"],
            n=5,
            cutoff=PLEXOS_ERCOT_UNIT_FUZZY_CUTOFF,
        )
        for resource_name in fuzzy_resource_names:
            add_candidates(
                lookups["by_resource_name"].get(resource_name, []),
                165,
                "cdr_unit_code_fuzzy",
            )

    unit_name_norm = sb_row.get("sb_unit_name_norm", "")
    if unit_name_norm:
        fuzzy_resource_names = get_close_matches(
            unit_name_norm,
            lookups["resource_names"],
            n=5,
            cutoff=PLEXOS_NAME_FUZZY_CUTOFF,
        )
        for resource_name in fuzzy_resource_names:
            add_candidates(
                lookups["by_resource_name"].get(resource_name, []),
                135,
                "unit_name_fuzzy",
            )

    for _, norm_value, _ in sb_code_fields(sb_row):
        family = node_family(norm_value)
        if family:
            add_candidates(
                lookups["by_resource_family"].get(family, []),
                120,
                "resource_family",
            )

    return candidates


def choose_best_sb_match(candidate_rows: list[dict]) -> dict:
    if not candidate_rows:
        return {
            "matched_sced_node": "",
            "matched_sced_resource_type": "",
            "matched_sced_final_sced_time_stamp": "",
            "matched_sced_base_point_avg": pd.NA,
            "matched_sced_start_up_cold_offer_avg": pd.NA,
            "matched_sced_start_up_hot_offer_avg": pd.NA,
            "matched_sced_start_up_inter_offer_avg": pd.NA,
            "matched_sced_min_gen_cost_avg": pd.NA,
            "sb_to_sced_match_method": "",
            "sb_to_sced_match_score": pd.NA,
            "sb_to_sced_match_status": "unmatched",
        }

    sorted_rows = sorted(
        candidate_rows,
        key=lambda row: (
            row["sb_to_sced_match_score"],
            row["matched_sced_node"],
        ),
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


def build_sb_matches(
    sb_df: pd.DataFrame,
    sced_unique_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sb_prepared = prepare_sb_df(sb_df)
    sced_prepared = prepare_sced_unique_df(sced_unique_df)
    sced_lookups = build_sced_unique_lookups(sced_prepared)

    match_rows = []
    candidate_rows = []
    for row_index, row in enumerate(sb_prepared.to_dict("records")):
        candidates = collect_sb_candidates(row, sced_lookups)
        for candidate in candidates:
            candidate_copy = {
                "sb_row_index": row_index,
                "unit_name": row.get("unit_name", ""),
                "cdr_unit_code": row.get("cdr_unit_code", ""),
                **candidate,
            }
            candidate_rows.append(candidate_copy)
        match_rows.append(choose_best_sb_match(candidates))

    match_df = pd.DataFrame(match_rows)
    original_columns = list(sb_df.columns)
    output_df = pd.concat([sb_df.reset_index(drop=True), match_df.reset_index(drop=True)], axis=1)
    appended_columns = [column for column in output_df.columns if column not in original_columns]
    candidates_df = pd.DataFrame(candidate_rows)
    return output_df[original_columns + appended_columns], candidates_df


def add_pun_presence_flag_to_sb_matches(
    pun_df: pd.DataFrame,
    sb_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    pun_combo_series = (
        pun_df["SubstationName"].fillna("").astype(str).str.strip()
        + "_"
        + pun_df["UnitName"].fillna("").astype(str).str.strip()
    )
    pun_combo_keys = {
        normalize_key(value)
        for value in pun_combo_series
        if normalize_key(value)
    }

    output_df = sb_matches_df.copy()
    comparison_series = pd.concat(
        [
            output_df["cdr_unit_code"],
            output_df["unit_name"],
            output_df["matched_sced_node"],
        ],
        ignore_index=True,
    )
    comparison_keys = [
        normalize_key(value) if pd.notna(value) else ""
        for value in comparison_series
    ]
    row_count = len(output_df)
    cdr_keys = comparison_keys[:row_count]
    unit_name_keys = comparison_keys[row_count : row_count * 2]
    sced_keys = comparison_keys[row_count * 2 :]
    output_df["in_pun_generation_report"] = [
        "Y" if cdr_key in pun_combo_keys or unit_name_key in pun_combo_keys or sced_key in pun_combo_keys else "N"
        for cdr_key, unit_name_key, sced_key in zip(cdr_keys, unit_name_keys, sced_keys)
    ]
    return output_df


def build_sb_pun_presence_output(
    pun_df: pd.DataFrame,
    sb_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = pun_df.copy()
    output_df["pun_combo"] = (
        output_df["SubstationName"].fillna("").astype(str).str.strip()
        + "_"
        + output_df["UnitName"].fillna("").astype(str).str.strip()
    )
    pun_combo_keys = output_df["pun_combo"].map(normalize_key)

    lookup_series = pd.concat(
        [
            sb_matches_df["cdr_unit_code"],
            sb_matches_df["unit_name"],
            sb_matches_df["matched_sced_node"],
        ],
        ignore_index=True,
    )
    lookup_keys = {
        normalize_key(value)
        for value in lookup_series.dropna().astype(str)
        if normalize_key(value)
    }
    output_df["in_sb_match_list"] = pun_combo_keys.map(
        lambda value: "Y" if value in lookup_keys else "N"
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
                "pct_matched_or_ambiguous": round(
                    (status_counts.get("matched", 0) + status_counts.get("ambiguous", 0)) / total_rows,
                    4,
                )
                if total_rows
                else 0.0,
            }
        ]
    )


def save_sb_outputs(
    sb_matches_df: pd.DataFrame,
    sb_candidates_df: pd.DataFrame,
    sb_summary_df: pd.DataFrame,
    pun_presence_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    output_path = ensure_output_dir(output_dir)
    sb_matches_path = output_path / SB_MATCHES_FILE_NAME
    sb_candidates_path = output_path / SB_CANDIDATES_FILE_NAME
    sb_summary_path = output_path / SB_SUMMARY_FILE_NAME
    pun_presence_path = output_path / PUN_SB_PRESENCE_FILE_NAME

    sb_matches_df.to_csv(sb_matches_path, index=False)
    sb_candidates_df.to_csv(sb_candidates_path, index=False)
    sb_summary_df.to_csv(sb_summary_path, index=False)
    pun_presence_df.to_csv(pun_presence_path, index=False)
    return sb_matches_path, sb_candidates_path, sb_summary_path, pun_presence_path


def prepare_ercot_cdr_df(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["cdr_unit_name_norm"] = prepared["UNIT NAME"].map(normalize_text)
    prepared["cdr_unit_code_norm"] = prepared["UNIT CODE"].map(normalize_text)
    prepared["cdr_county_norm"] = prepared["COUNTY"].map(normalize_text)
    prepared["cdr_fuel_norm"] = prepared["FUEL"].map(fuel_key)
    return prepared


def build_lookups(
    yes_df: pd.DataFrame,
    rtlmp_df: pd.DataFrame,
    resource_node_mapping_df: pd.DataFrame,
) -> dict[str, object]:
    yes_rows = yes_df.to_dict("records")
    rtlmp_rows = rtlmp_df.to_dict("records")

    yes_by_plant: dict[str, list[dict]] = {}
    yes_by_plant_key: dict[str, list[dict]] = {}
    yes_by_node: dict[str, list[dict]] = {}
    yes_by_node_family: dict[str, list[dict]] = {}

    for row in yes_rows:
        if row["plant_name_norm"]:
            yes_by_plant.setdefault(row["plant_name_norm"], []).append(row)
        if row["plant_name_key"]:
            yes_by_plant_key.setdefault(row["plant_name_key"], []).append(row)
        if row["node_name_norm"]:
            yes_by_node.setdefault(row["node_name_norm"], []).append(row)
        if row["node_family"]:
            yes_by_node_family.setdefault(row["node_family"], []).append(row)

    rtlmp_by_name = {}
    rtlmp_by_name_key = {}
    rtlmp_by_family = {}
    for row in rtlmp_rows:
        if row["rtlmp_name_norm"]:
            rtlmp_by_name.setdefault(row["rtlmp_name_norm"], []).append(row)
        if row["rtlmp_name_key"]:
            rtlmp_by_name_key.setdefault(row["rtlmp_name_key"], []).append(row)
        if row["rtlmp_family"]:
            rtlmp_by_family.setdefault(row["rtlmp_family"], []).append(row)

    plant_names = sorted(yes_by_plant.keys())
    plant_name_by_key = {normalize_key(name): name for name in plant_names}
    rtlmp_names = sorted(rtlmp_by_name.keys())
    rtlmp_prefix_index: dict[str, list[dict]] = {}
    for row in rtlmp_rows:
        normalized = row["rtlmp_name_norm"]
        if not normalized:
            continue
        for prefix_len in (3, 4, 5, 6):
            if len(normalized) >= prefix_len:
                rtlmp_prefix_index.setdefault(normalized[:prefix_len], []).append(row)

    resource_mapping_by_sced_key = {}
    for row in resource_node_mapping_df.to_dict("records"):
        if row["sced_key_norm"]:
            resource_mapping_by_sced_key.setdefault(row["sced_key_norm"], []).append(row)

    return {
        "yes_by_plant": yes_by_plant,
        "yes_by_plant_key": yes_by_plant_key,
        "yes_by_node": yes_by_node,
        "yes_by_node_family": yes_by_node_family,
        "rtlmp_by_name": rtlmp_by_name,
        "rtlmp_by_name_key": rtlmp_by_name_key,
        "rtlmp_by_family": rtlmp_by_family,
        "rtlmp_names": rtlmp_names,
        "rtlmp_prefix_index": rtlmp_prefix_index,
        "resource_mapping_by_sced_key": resource_mapping_by_sced_key,
        "plant_names": plant_names,
        "plant_name_by_key": plant_name_by_key,
    }


def score_fuel_match(sced_fuel: str, yes_fuel: str, generator_type: str) -> int:
    score = 0
    if sced_fuel and yes_fuel and sced_fuel == yes_fuel:
        score += 10
    if generator_type == "SOLAR" and yes_fuel == "SOLAR":
        score += 5
    if generator_type == "WIND" and yes_fuel == "WIND":
        score += 5
    return score


def score_capacity_match(sced_capacity, yes_capacity) -> int:
    if pd.isna(sced_capacity) or pd.isna(yes_capacity):
        return 0
    diff = abs(float(sced_capacity) - float(yes_capacity))
    if diff <= 1:
        return 8
    if diff <= 5:
        return 5
    if diff <= 20:
        return 2
    return 0


def score_station_to_rtlmp(station_code: str, rtlmp_name: str) -> CandidateScore | None:
    station_code = normalize_text(station_code)
    rtlmp_name = normalize_text(rtlmp_name)
    if not station_code or not rtlmp_name:
        return None

    station_key = normalize_key(station_code)
    rtlmp_key = normalize_key(rtlmp_name)
    station_family = node_family(station_code)
    rtlmp_family = node_family(rtlmp_name)

    if station_code == rtlmp_name:
        return CandidateScore(180, "station_code_exact_to_rtlmp", "station_to_rtlmp")
    if station_key == rtlmp_key:
        return CandidateScore(170, "station_code_key_to_rtlmp", "station_to_rtlmp")
    if station_code in rtlmp_name or rtlmp_name in station_code:
        return CandidateScore(150, "station_code_substring_to_rtlmp", "station_to_rtlmp")
    if station_family and station_family == rtlmp_family:
        return CandidateScore(135, "station_code_family_to_rtlmp", "station_to_rtlmp")
    if station_key and rtlmp_key.startswith(station_key):
        return CandidateScore(145, "station_code_prefix_to_rtlmp", "station_to_rtlmp")
    if station_key and station_key.startswith(rtlmp_key):
        return CandidateScore(140, "station_code_reverse_prefix_to_rtlmp", "station_to_rtlmp")
    return None


def add_station_rtlmp_candidates(
    candidate_rows: list[dict],
    seen_keys: set[tuple[str, str, str]],
    base_row: dict,
    lookups: dict[str, object],
):
    station_code_norm = normalize_text(base_row.get("station_code_norm", ""))
    if not station_code_norm:
        return

    direct_rows = lookups["rtlmp_by_name"].get(station_code_norm, [])
    for rtlmp_row in direct_rows:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            None,
            rtlmp_row,
            CandidateScore(180, "station_code_exact_to_rtlmp", "station_to_rtlmp"),
        )

    direct_key_rows = lookups["rtlmp_by_name_key"].get(normalize_key(station_code_norm), [])
    for rtlmp_row in direct_key_rows:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            None,
            rtlmp_row,
            CandidateScore(170, "station_code_key_to_rtlmp", "station_to_rtlmp"),
        )

    family_rows = lookups["rtlmp_by_family"].get(node_family(station_code_norm), [])[:MAX_RTLMP_FAMILY_ROWS]
    for rtlmp_row in family_rows:
        score = score_station_to_rtlmp(station_code_norm, rtlmp_row["rtlmp_name_norm"])
        if score is not None:
            add_candidate(
                candidate_rows,
                seen_keys,
                base_row,
                None,
                rtlmp_row,
                score,
            )

    fuzzy_pool: list[dict] = []
    seen_rtlmp_names: set[str] = set()
    station_prefixes = [station_code_norm[:size] for size in (6, 5, 4, 3) if len(station_code_norm) >= size]
    for prefix in station_prefixes:
        for rtlmp_row in lookups["rtlmp_prefix_index"].get(prefix, []):
            rtlmp_name = rtlmp_row["rtlmp_name_norm"]
            if rtlmp_name and rtlmp_name not in seen_rtlmp_names:
                fuzzy_pool.append(rtlmp_row)
                seen_rtlmp_names.add(rtlmp_name)
            if len(fuzzy_pool) >= MAX_STATION_FUZZY_POOL:
                break
        if len(fuzzy_pool) >= MAX_STATION_FUZZY_POOL:
            break

    if not fuzzy_pool:
        fuzzy_pool = family_rows[:]
        for rtlmp_row in direct_rows + direct_key_rows:
            rtlmp_name = rtlmp_row["rtlmp_name_norm"]
            if rtlmp_name and rtlmp_name not in seen_rtlmp_names:
                fuzzy_pool.append(rtlmp_row)
                seen_rtlmp_names.add(rtlmp_name)

    fuzzy_names = [row["rtlmp_name_norm"] for row in fuzzy_pool if row["rtlmp_name_norm"]]
    fuzzy_matches = get_close_matches(
        station_code_norm,
        fuzzy_names,
        n=8,
        cutoff=FUZZY_STATION_TO_RTLMP_CUTOFF,
    )
    for rtlmp_name in fuzzy_matches:
        for rtlmp_row in lookups["rtlmp_by_name"].get(rtlmp_name, []):
            score = score_station_to_rtlmp(station_code_norm, rtlmp_row["rtlmp_name_norm"])
            if score is None:
                score = CandidateScore(125, "station_code_fuzzy_to_rtlmp", "station_to_rtlmp")
            else:
                score = CandidateScore(
                    max(score.score, 125),
                    f"{score.method}_fuzzy",
                    "station_to_rtlmp",
                )
            add_candidate(
                candidate_rows,
                seen_keys,
                base_row,
                None,
                rtlmp_row,
                score,
            )


def add_candidate(
    candidate_rows: list[dict],
    seen_keys: set[tuple[str, str, str]],
    base_row: dict,
    yes_row: dict | None,
    rtlmp_row: dict | None,
    candidate_score: CandidateScore,
):
    rtlmp_name = rtlmp_row["NAME"] if rtlmp_row is not None else ""
    yes_node = yes_row["NODENAME"] if yes_row is not None else ""
    dedupe_key = (candidate_score.method, str(yes_node), str(rtlmp_name))
    if dedupe_key in seen_keys:
        return
    seen_keys.add(dedupe_key)

    yes_fuel = yes_row["yes_fuel_norm"] if yes_row is not None else ""
    extra_score = score_fuel_match(
        base_row["fuel_type_norm"],
        yes_fuel,
        base_row.get("generator_type_norm", ""),
    )
    extra_score += score_capacity_match(
        base_row.get("capacity_mw"),
        yes_row["yes_capacity_mw"] if yes_row is not None else pd.NA,
    )

    total_score = candidate_score.score + extra_score
    candidate_rows.append(
        {
            "resource_name": base_row["resource_name"],
            "fuel_type": base_row["fuel_type"],
            "station_metadata_source": base_row["station_metadata_source"],
            "station_code": base_row.get("Generator Station Code", ""),
            "station_desc": base_row.get("Generator Station Description", ""),
            "generator_type": base_row.get("Generator Type", ""),
            "capacity_mw": base_row.get("capacity_mw"),
            "derived_stems": " | ".join(base_row["derived_stems"]),
            "yes_plantname": yes_row["PLANTNAME"] if yes_row is not None else "",
            "yes_unitname": yes_row["UNITNAME"] if yes_row is not None else "",
            "yes_nodename": yes_row["NODENAME"] if yes_row is not None else "",
            "yes_primaryfuel": yes_row["PRIMARYFUEL"] if yes_row is not None else "",
            "yes_primemover": yes_row["PRIMEMOVER"] if yes_row is not None else "",
            "yes_capacity_mw": yes_row["yes_capacity_mw"] if yes_row is not None else pd.NA,
            "rtlmp_name": rtlmp_row["NAME"] if rtlmp_row is not None else "",
            "rtlmp_objectid": rtlmp_row["OBJECTID"] if rtlmp_row is not None else "",
            "price_node_source": rtlmp_row["price_node_source"] if rtlmp_row is not None else "",
            "match_stage": candidate_score.stage,
            "match_method": candidate_score.method,
            "match_score": total_score,
        }
            )


def add_resource_node_candidates(
    candidate_rows: list[dict],
    seen_keys: set[tuple[str, str, str]],
    base_row: dict,
    resource_node_row: dict,
    lookups: dict[str, object],
    exact_score: int = 160,
    family_score: int = 130,
    fuzzy_score: int = 120,
):
    resource_node_norm = resource_node_row["resource_node_norm"]
    if not resource_node_norm:
        return

    direct_rows = lookups["rtlmp_by_name"].get(resource_node_norm, [])
    for rtlmp_row in direct_rows:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            None,
            rtlmp_row,
            CandidateScore(exact_score, "resource_node_exact_to_rtlmp", "resource_node_bridge"),
        )

    direct_key_rows = lookups["rtlmp_by_name_key"].get(normalize_key(resource_node_norm), [])
    for rtlmp_row in direct_key_rows:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            None,
            rtlmp_row,
            CandidateScore(exact_score - 5, "resource_node_key_to_rtlmp", "resource_node_bridge"),
        )

    family_rows = lookups["rtlmp_by_family"].get(
        resource_node_row["resource_node_family"], []
    )[:MAX_RTLMP_FAMILY_ROWS]
    for rtlmp_row in family_rows:
        rtlmp_name_norm = rtlmp_row["rtlmp_name_norm"]
        resource_node_key = normalize_key(resource_node_norm)
        rtlmp_key = normalize_key(rtlmp_name_norm)
        if resource_node_norm in rtlmp_name_norm or rtlmp_name_norm in resource_node_norm:
            method = "resource_node_substring_to_rtlmp"
            score = exact_score - 10
        elif resource_node_key and rtlmp_key.startswith(resource_node_key):
            method = "resource_node_prefix_to_rtlmp"
            score = exact_score - 15
        elif resource_node_key and resource_node_key.startswith(rtlmp_key):
            method = "resource_node_reverse_prefix_to_rtlmp"
            score = exact_score - 20
        else:
            method = "resource_node_family_to_rtlmp"
            score = family_score
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            None,
            rtlmp_row,
            CandidateScore(score, method, "resource_node_bridge"),
        )

    fuzzy_pool = family_rows[:]
    if not fuzzy_pool:
        prefixes = [resource_node_norm[:size] for size in (6, 5, 4, 3) if len(resource_node_norm) >= size]
        seen_names: set[str] = set()
        for prefix in prefixes:
            for rtlmp_row in lookups["rtlmp_prefix_index"].get(prefix, []):
                rtlmp_name = rtlmp_row["rtlmp_name_norm"]
                if rtlmp_name and rtlmp_name not in seen_names:
                    fuzzy_pool.append(rtlmp_row)
                    seen_names.add(rtlmp_name)
                if len(fuzzy_pool) >= MAX_RESOURCE_NODE_FUZZY_POOL:
                    break
            if len(fuzzy_pool) >= MAX_RESOURCE_NODE_FUZZY_POOL:
                break

    fuzzy_names = [row["rtlmp_name_norm"] for row in fuzzy_pool if row["rtlmp_name_norm"]]
    fuzzy_matches = get_close_matches(
        resource_node_norm,
        fuzzy_names,
        n=5,
        cutoff=0.70,
    )
    for rtlmp_name in fuzzy_matches:
        for rtlmp_row in lookups["rtlmp_by_name"].get(rtlmp_name, []):
            add_candidate(
                candidate_rows,
                seen_keys,
                base_row,
                None,
                rtlmp_row,
                CandidateScore(fuzzy_score, "resource_node_fuzzy_to_rtlmp", "resource_node_bridge"),
            )


def add_resource_node_mapping_candidates(
    candidate_rows: list[dict],
    seen_keys: set[tuple[str, str, str]],
    base_row: dict,
    lookups: dict[str, object],
):
    resource_name_norm = normalize_text(base_row.get("resource_name_norm", ""))
    if not resource_name_norm:
        return

    for mapping_row in lookups["resource_mapping_by_sced_key"].get(resource_name_norm, []):
        add_resource_node_candidates(candidate_rows, seen_keys, base_row, mapping_row, lookups)


def add_yes_rtlmp_candidates(
    candidate_rows: list[dict],
    seen_keys: set[tuple[str, str, str]],
    base_row: dict,
    yes_row: dict,
    lookups: dict[str, object],
    exact_method: str,
    family_method: str,
    exact_score: int,
    family_score: int,
):
    rtlmp_exact = lookups["rtlmp_by_name"].get(yes_row["node_name_norm"], [])
    for rtlmp_row in rtlmp_exact:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            yes_row,
            rtlmp_row,
            CandidateScore(exact_score, exact_method, "yes_to_rtlmp"),
        )

    rtlmp_exact_key = lookups["rtlmp_by_name_key"].get(yes_row["node_name_key"], [])
    for rtlmp_row in rtlmp_exact_key:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            yes_row,
            rtlmp_row,
            CandidateScore(exact_score - 5, f"{exact_method}_key", "yes_to_rtlmp"),
        )

    rtlmp_family = lookups["rtlmp_by_family"].get(yes_row["node_family"], [])[:MAX_RTLMP_FAMILY_ROWS]
    for rtlmp_row in rtlmp_family:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            yes_row,
            rtlmp_row,
            CandidateScore(family_score, family_method, "yes_to_rtlmp"),
        )


def add_unmatched_description_fallback_candidates(
    candidate_rows: list[dict],
    seen_keys: set[tuple[str, str, str]],
    base_row: dict,
    lookups: dict[str, object],
):
    station_desc_norm = normalize_text(base_row.get("station_desc_norm", ""))
    if not station_desc_norm:
        return

    desc_candidates: list[tuple[str, list[dict], CandidateScore]] = []

    exact_rows = [
        yes_row
        for yes_row in lookups["yes_by_plant"].get(station_desc_norm, [])
        if yes_row["node_name_norm"]
    ]
    if exact_rows:
        desc_candidates.append(
            (
                station_desc_norm,
                exact_rows,
                CandidateScore(90, "unmatched_desc_exact_to_yes", "unmatched_desc_fallback"),
            )
        )

    fuzzy_names = get_close_matches(
        station_desc_norm,
        lookups["plant_names"],
        n=3,
        cutoff=FUZZY_UNMATCHED_DESC_TO_YES_CUTOFF,
    )
    for plant_name in fuzzy_names:
        if plant_name == station_desc_norm:
            continue
        fuzzy_rows = [
            yes_row
            for yes_row in lookups["yes_by_plant"].get(plant_name, [])
            if yes_row["node_name_norm"]
        ]
        if fuzzy_rows:
            desc_candidates.append(
                (
                    plant_name,
                    fuzzy_rows,
                    CandidateScore(75, "unmatched_desc_fuzzy_to_yes", "unmatched_desc_fallback"),
                )
            )

    for _, yes_rows, base_score in desc_candidates:
        for yes_row in yes_rows:
            add_yes_rtlmp_candidates(
                candidate_rows,
                seen_keys,
                base_row,
                yes_row,
                lookups,
                exact_method=base_score.method + "_node_exact",
                family_method=base_score.method + "_node_family",
                exact_score=base_score.score,
                family_score=base_score.score - 20,
            )


def collect_candidates_for_row(base_row: dict, lookups: dict[str, object]) -> list[dict]:
    candidate_rows: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    processed_yes_families: set[str] = set()

    station_desc_norm = normalize_text(base_row.get("station_desc_norm", ""))
    station_code_norm = normalize_text(base_row.get("station_code_norm", ""))

    add_station_rtlmp_candidates(candidate_rows, seen_keys, base_row, lookups)
    add_resource_node_mapping_candidates(candidate_rows, seen_keys, base_row, lookups)

    if station_desc_norm:
        for yes_row in lookups["yes_by_plant"].get(station_desc_norm, []):
            add_yes_rtlmp_candidates(
                candidate_rows,
                seen_keys,
                base_row,
                yes_row,
                lookups,
                exact_method="desc_exact_to_yes_node",
                family_method="desc_exact_to_yes_node_family",
                exact_score=95,
                family_score=70,
            )

        fuzzy_matches = get_close_matches(
            station_desc_norm,
            lookups["plant_names"],
            n=3,
            cutoff=FUZZY_PLANT_MATCH_CUTOFF,
        )
        for plant_name in fuzzy_matches:
            if plant_name == station_desc_norm:
                continue
            for yes_row in lookups["yes_by_plant"].get(plant_name, []):
                add_yes_rtlmp_candidates(
                    candidate_rows,
                    seen_keys,
                    base_row,
                    yes_row,
                    lookups,
                    exact_method="desc_fuzzy_to_yes_node",
                    family_method="desc_fuzzy_to_yes_node_family",
                    exact_score=70,
                    family_score=50,
                )

    node_patterns: list[str] = []
    if station_code_norm:
        node_patterns.extend(
            [
                station_code_norm,
                f"{station_code_norm}_ALL",
                f"{station_code_norm}_RN",
            ]
        )

    for stem in base_row["derived_stems"]:
        node_patterns.extend([stem, f"{stem}_ALL", f"{stem}_RN"])

    unique_patterns: list[str] = []
    for pattern in node_patterns:
        pattern = normalize_text(pattern)
        if pattern and pattern not in unique_patterns:
            unique_patterns.append(pattern)

    for pattern in unique_patterns:
        for yes_row in lookups["yes_by_node"].get(pattern, []):
            add_yes_rtlmp_candidates(
                candidate_rows,
                seen_keys,
                base_row,
                yes_row,
                lookups,
                exact_method="node_exact_to_yes_node",
                family_method="node_exact_to_yes_node_family",
                exact_score=85,
                family_score=60,
            )

        family_name = node_family(pattern)
        if not family_name or family_name in processed_yes_families:
            continue
        processed_yes_families.add(family_name)

        node_family_rows = lookups["yes_by_node_family"].get(family_name, [])[:MAX_YES_FAMILY_ROWS]
        for yes_row in node_family_rows:
            add_yes_rtlmp_candidates(
                candidate_rows,
                seen_keys,
                base_row,
                yes_row,
                lookups,
                exact_method="node_family_to_yes_node",
                family_method="node_family_to_yes_node_family",
                exact_score=55,
                family_score=40,
            )

    if not candidate_rows:
        add_candidate(
            candidate_rows,
            seen_keys,
            base_row,
            None,
            None,
            CandidateScore(0, "unmatched", "none"),
        )

    return sorted(
        candidate_rows,
        key=lambda row: (
            row["match_score"],
            bool(row["rtlmp_name"]),
            bool(row["yes_nodename"]),
            row["rtlmp_name"],
        ),
        reverse=True,
    )


def classify_best_match(candidate_rows: list[dict]) -> dict:
    best_row = candidate_rows[0].copy()
    real_candidates = [row for row in candidate_rows if row["rtlmp_name"]]

    if not real_candidates:
        best_row["match_status"] = "unmatched"
        best_row["candidate_count"] = 0
        best_row["alt_rtlmp_names"] = ""
        return best_row

    best_score = real_candidates[0]["match_score"]
    top_score_candidates = [row for row in real_candidates if row["match_score"] == best_score]
    top_score_rtlmp_names = []
    for row in top_score_candidates:
        if row["rtlmp_name"] and row["rtlmp_name"] not in top_score_rtlmp_names:
            top_score_rtlmp_names.append(row["rtlmp_name"])
    unique_rtlmp_names = []
    for row in real_candidates:
        if row["rtlmp_name"] and row["rtlmp_name"] not in unique_rtlmp_names:
            unique_rtlmp_names.append(row["rtlmp_name"])

    if len(top_score_rtlmp_names) > 1:
        best_row["match_status"] = "ambiguous"
    else:
        best_row["match_status"] = "matched"

    best_row["candidate_count"] = len(unique_rtlmp_names)
    best_row["alt_rtlmp_names"] = " | ".join(unique_rtlmp_names[1 : 1 + MAX_ALT_MATCHES])
    return best_row


def resolve_station_level_matches(best_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = best_matches_df.copy()
    has_station = df["station_code"].fillna("").astype(str).str.strip() != ""
    station_df = df.loc[has_station].copy()

    station_best_rows = []
    for station_code, group in station_df.groupby("station_code", dropna=False):
        group_sorted = group.sort_values(
            ["match_score", "match_status", "candidate_count", "resource_name"],
            ascending=[False, True, True, True],
        )
        station_best_rows.append(group_sorted.iloc[0].to_dict())

    station_best_df = pd.DataFrame(station_best_rows)
    if station_best_df.empty:
        return df

    station_best_df = station_best_df.rename(
        columns={
            "rtlmp_name": "station_group_rtlmp_name",
            "price_node_source": "station_group_price_node_source",
            "match_method": "station_group_match_method",
            "match_score": "station_group_match_score",
            "match_status": "station_group_match_status",
        }
    )

    station_columns = [
        "station_code",
        "station_group_rtlmp_name",
        "station_group_price_node_source",
        "station_group_match_method",
        "station_group_match_score",
        "station_group_match_status",
    ]
    df = df.merge(station_best_df[station_columns], on="station_code", how="left")

    replace_mask = (
        has_station
        & df["station_group_rtlmp_name"].fillna("").astype(str).ne("")
        & (
            df["rtlmp_name"].fillna("").astype(str).eq("")
            | (
                df["station_group_match_score"].fillna(-1)
                > df["match_score"].fillna(-1)
            )
        )
    )

    df.loc[replace_mask, "rtlmp_name"] = df.loc[replace_mask, "station_group_rtlmp_name"]
    df.loc[replace_mask, "price_node_source"] = df.loc[replace_mask, "station_group_price_node_source"]
    df.loc[replace_mask, "match_method"] = (
        df.loc[replace_mask, "station_group_match_method"].astype(str) + "_station_group"
    )
    df.loc[replace_mask, "match_score"] = df.loc[replace_mask, "station_group_match_score"]
    df.loc[replace_mask, "match_status"] = df.loc[replace_mask, "station_group_match_status"]

    same_station_mask = (
        has_station
        & df["station_group_rtlmp_name"].fillna("").astype(str).ne("")
        & df["rtlmp_name"].fillna("").astype(str).eq(df["station_group_rtlmp_name"].fillna("").astype(str))
    )
    df.loc[same_station_mask, "match_status"] = "matched"

    return df


def apply_unmatched_description_fallback(
    best_matches_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    lookups: dict[str, object],
) -> pd.DataFrame:
    df = best_matches_df.copy()
    merged_lookup = {
        normalize_text(row["resource_name"]): row
        for row in merged_df.to_dict("records")
    }

    for idx, row in df.iterrows():
        if row["match_status"] != "unmatched":
            continue

        base_row = merged_lookup.get(normalize_text(row["resource_name"]))
        if not base_row:
            continue

        candidate_rows: list[dict] = []
        seen_keys: set[tuple[str, str, str]] = set()
        add_unmatched_description_fallback_candidates(candidate_rows, seen_keys, base_row, lookups)
        candidate_rows = [candidate for candidate in candidate_rows if candidate["rtlmp_name"]]
        if not candidate_rows:
            continue

        fallback_best = classify_best_match(
            sorted(
                candidate_rows,
                key=lambda candidate: (
                    candidate["match_score"],
                    bool(candidate["rtlmp_name"]),
                    bool(candidate["yes_nodename"]),
                    candidate["rtlmp_name"],
                ),
                reverse=True,
            )
        )

        df.at[idx, "yes_plantname"] = fallback_best.get("yes_plantname", "")
        df.at[idx, "yes_unitname"] = fallback_best.get("yes_unitname", "")
        df.at[idx, "yes_nodename"] = fallback_best.get("yes_nodename", "")
        df.at[idx, "yes_primaryfuel"] = fallback_best.get("yes_primaryfuel", "")
        df.at[idx, "yes_primemover"] = fallback_best.get("yes_primemover", "")
        df.at[idx, "rtlmp_name"] = fallback_best.get("rtlmp_name", "")
        df.at[idx, "rtlmp_objectid"] = fallback_best.get("rtlmp_objectid", "")
        df.at[idx, "price_node_source"] = fallback_best.get("price_node_source", "")
        df.at[idx, "match_stage"] = fallback_best.get("match_stage", "")
        df.at[idx, "match_method"] = fallback_best.get("match_method", "")
        df.at[idx, "match_score"] = fallback_best.get("match_score", 0)
        df.at[idx, "alt_rtlmp_names"] = fallback_best.get("alt_rtlmp_names", "")
        df.at[idx, "candidate_count"] = fallback_best.get("candidate_count", 0)
        df.at[idx, "match_status"] = fallback_best.get("match_status", "unmatched")

    return df


def apply_secondary_rtlmp_fallback(
    best_matches_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    lookups: dict[str, object],
) -> pd.DataFrame:
    df = best_matches_df.copy()
    merged_lookup = {
        normalize_text(row["resource_name"]): row
        for row in merged_df.to_dict("records")
    }

    for idx, row in df.iterrows():
        if row["match_status"] != "unmatched":
            continue

        base_row = merged_lookup.get(normalize_text(row["resource_name"]))
        if not base_row:
            continue

        candidate_rows = [
            candidate
            for candidate in collect_candidates_for_row(base_row, lookups)
            if candidate["rtlmp_name"]
        ]
        if not candidate_rows:
            continue

        fallback_best = classify_best_match(candidate_rows)
        for field in [
            "yes_plantname",
            "yes_unitname",
            "yes_nodename",
            "yes_primaryfuel",
            "yes_primemover",
            "rtlmp_name",
            "rtlmp_objectid",
            "price_node_source",
            "match_stage",
            "match_method",
            "match_score",
            "alt_rtlmp_names",
            "candidate_count",
            "match_status",
        ]:
            df.at[idx, field] = fallback_best.get(field, df.at[idx, field] if field in df.columns else "")

    return df


def build_match_tables(
    sced_plant_df: pd.DataFrame,
    sced_name_df: pd.DataFrame,
    yes_df: pd.DataFrame,
    rtlmp_bus_df: pd.DataFrame,
    rtlmp_df: pd.DataFrame,
    resource_node_mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sced_name_prepared = prepare_sced_name_df(sced_name_df)
    sced_plant_prepared = prepare_sced_plant_df(sced_plant_df)
    yes_prepared = prepare_yes_df(yes_df)
    rtlmp_bus_prepared = prepare_rtlmp_df(rtlmp_bus_df, "rtlmp_bus")
    rtlmp_prepared = prepare_rtlmp_df(rtlmp_df, "rtlmp")
    resource_node_mapping_prepared = prepare_resource_node_mapping_df(resource_node_mapping_df)

    sced_name_columns = [
        "Unit Code",
        "Generator Station Code",
        "Generator Station Description",
        "Generator Type",
        "capacity_mw",
        "unit_code_norm",
        "station_code_norm",
        "station_desc_norm",
        "generator_type_norm",
    ]

    merged = sced_plant_prepared.merge(
        sced_name_prepared[sced_name_columns],
        left_on="resource_name_norm",
        right_on="unit_code_norm",
        how="left",
    )
    merged["station_metadata_source"] = merged["Unit Code"].notna().map(
        lambda matched: "sced_name_list" if matched else "derived_from_resource_name"
    )

    bus_lookups = build_lookups(yes_prepared, rtlmp_bus_prepared, resource_node_mapping_prepared)
    rtlmp_lookups = build_lookups(yes_prepared, rtlmp_prepared, resource_node_mapping_prepared)

    all_candidate_rows: list[dict] = []
    best_rows: list[dict] = []
    station_candidate_cache: dict[str, list[dict]] = {}

    for base_row in merged.to_dict("records"):
        cache_key = normalize_text(base_row.get("station_code_norm", ""))
        if cache_key:
            cached_candidates = station_candidate_cache.get(cache_key)
            if cached_candidates is None:
                cached_candidates = collect_candidates_for_row(base_row, bus_lookups)
                station_candidate_cache[cache_key] = cached_candidates
            candidate_rows = []
            for row in cached_candidates:
                cloned_row = row.copy()
                cloned_row["resource_name"] = base_row["resource_name"]
                cloned_row["fuel_type"] = base_row["fuel_type"]
                cloned_row["station_metadata_source"] = base_row["station_metadata_source"]
                cloned_row["station_code"] = base_row.get("Generator Station Code", "")
                cloned_row["station_desc"] = base_row.get("Generator Station Description", "")
                cloned_row["generator_type"] = base_row.get("Generator Type", "")
                cloned_row["capacity_mw"] = base_row.get("capacity_mw")
                cloned_row["derived_stems"] = " | ".join(base_row["derived_stems"])
                candidate_rows.append(cloned_row)
        else:
            candidate_rows = collect_candidates_for_row(base_row, bus_lookups)
        all_candidate_rows.extend(candidate_rows)
        best_rows.append(classify_best_match(candidate_rows))

    candidates_df = pd.DataFrame(all_candidate_rows)
    best_matches_df = pd.DataFrame(best_rows).sort_values(
        ["match_status", "match_score", "resource_name"],
        ascending=[True, False, True],
    )
    best_matches_df = resolve_station_level_matches(best_matches_df)
    best_matches_df = apply_unmatched_description_fallback(best_matches_df, merged, bus_lookups)
    best_matches_df = apply_secondary_rtlmp_fallback(best_matches_df, merged, rtlmp_lookups)

    summary_df = pd.DataFrame(
        {
            "metric": [
                "total_resources",
                "matched",
                "ambiguous",
                "unmatched",
                "with_station_metadata",
                "without_station_metadata",
            ],
            "value": [
                len(best_matches_df),
                int((best_matches_df["match_status"] == "matched").sum()),
                int((best_matches_df["match_status"] == "ambiguous").sum()),
                int((best_matches_df["match_status"] == "unmatched").sum()),
                int((best_matches_df["station_metadata_source"] == "sced_name_list").sum()),
                int((best_matches_df["station_metadata_source"] == "derived_from_resource_name").sum()),
            ],
        }
    )

    return best_matches_df, candidates_df, summary_df


def build_simple_matches(best_matches_df: pd.DataFrame) -> pd.DataFrame:
    simple_df = best_matches_df[
        ["resource_name", "fuel_type", "capacity_mw", "rtlmp_name", "price_node_source"]
    ].copy()
    simple_df = simple_df.rename(columns={"rtlmp_name": "price_code"})
    simple_df["price_code"] = simple_df["price_code"].where(
        best_matches_df["match_status"].isin(["matched", "ambiguous"]),
        "",
    )
    simple_df["price_node_source"] = simple_df["price_node_source"].where(
        best_matches_df["match_status"].isin(["matched", "ambiguous"]),
        "",
    )
    if "yes_capacity_mw" in best_matches_df.columns:
        simple_df["capacity_mw"] = simple_df["capacity_mw"].where(
            simple_df["capacity_mw"].notna(),
            best_matches_df["yes_capacity_mw"],
        )
    return simple_df.sort_values("resource_name").reset_index(drop=True)


def choose_simple_matches_source(generated_simple_matches_df: pd.DataFrame) -> pd.DataFrame:
    if not FINAL_SIMPLE_MATCHES_PATH:
        return generated_simple_matches_df

    final_df = load_csv(
        FINAL_SIMPLE_MATCHES_PATH,
        ["resource_name", "fuel_type", "capacity_mw", "price_code", "price_node_source"],
    ).copy()
    return final_df


def build_override_key(df: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    normalized_parts = [df[column].map(normalize_text) for column in key_columns]
    return pd.Series(["||".join(values) for values in zip(*normalized_parts)], index=df.index)


def manual_override_has_value(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def apply_plexos_match_overrides(generated_plexos_matches_df: pd.DataFrame) -> pd.DataFrame:
    if not FINAL_PLEXOS_MATCHES_PATH:
        return generated_plexos_matches_df

    override_df = load_csv(
        FINAL_PLEXOS_MATCHES_PATH,
        FINAL_PLEXOS_MATCHES_REQUIRED_COLUMNS,
    ).copy()
    merged_df = generated_plexos_matches_df.copy()
    merged_df["_override_key"] = build_override_key(merged_df, PLEXOS_OVERRIDE_KEY_COLUMNS)
    override_df["_override_key"] = build_override_key(override_df, PLEXOS_OVERRIDE_KEY_COLUMNS)

    override_df = override_df.drop_duplicates(subset="_override_key", keep="last")
    override_df = override_df.set_index("_override_key")

    for column in PLEXOS_OVERRIDE_COLUMNS:
        override_values = merged_df["_override_key"].map(override_df[column])
        merged_df[column] = [
            override if manual_override_has_value(override) else current
            for current, override in zip(merged_df[column], override_values)
        ]

    merged_df = merged_df.drop(columns="_override_key")
    return merged_df


def build_sced_reference_df(
    best_matches_df: pd.DataFrame,
    simple_matches_df: pd.DataFrame,
    ercot_cdr_df: pd.DataFrame,
) -> pd.DataFrame:
    sced_reference_base = best_matches_df[
        ["resource_name", "station_code", "station_desc", "generator_type", "capacity_mw", "fuel_type"]
    ].copy()

    simple_reference = simple_matches_df[
        ["resource_name", "fuel_type", "capacity_mw", "price_code", "price_node_source"]
    ].copy()
    simple_reference = simple_reference.rename(
        columns={
            "fuel_type": "matched_fuel_type",
            "capacity_mw": "matched_capacity_mw",
        }
    )

    reference_df = sced_reference_base.merge(
        simple_reference,
        on="resource_name",
        how="left",
    )
    cdr_prepared = prepare_ercot_cdr_df(ercot_cdr_df)[
        ["UNIT CODE", "UNIT NAME", "cdr_unit_code_norm", "cdr_unit_name_norm", "cdr_county_norm", "cdr_fuel_norm"]
    ].copy()
    reference_df["resource_name_norm"] = reference_df["resource_name"].map(normalize_text)
    reference_df["resource_name_key"] = reference_df["resource_name"].map(normalize_key)
    reference_df["station_code_norm"] = reference_df["station_code"].map(normalize_text)
    reference_df["station_desc_norm"] = reference_df["station_desc"].map(normalize_text)
    reference_df["capacity_mw"] = reference_df["capacity_mw"].where(
        reference_df["capacity_mw"].notna(),
        reference_df["matched_capacity_mw"],
    )
    reference_df["matched_sced_fuel_norm"] = reference_df["fuel_type"].where(
        reference_df["fuel_type"].fillna("").astype(str).str.strip().ne(""),
        reference_df["generator_type"],
    ).map(fuel_key)
    reference_df["resource_unit_tokens"] = reference_df["resource_name"].map(extract_resource_unit_tokens)
    reference_df["price_code_norm"] = reference_df["price_code"].map(normalize_text)
    reference_df["price_code_key"] = reference_df["price_code"].map(normalize_key)
    reference_df["price_code_family"] = reference_df["price_code_norm"].map(node_family)
    reference_df = reference_df.merge(
        cdr_prepared,
        left_on="resource_name_norm",
        right_on="cdr_unit_code_norm",
        how="left",
    )
    return reference_df


def score_plexos_candidate(
    plexos_row: dict,
    sced_row: dict,
    base_score: int,
    method: str,
    unit_hint_bonus: int = 0,
) -> dict:
    score = base_score + unit_hint_bonus
    if plexos_row["plexos_fuel_norm"] and plexos_row["plexos_fuel_norm"] == sced_row["matched_sced_fuel_norm"]:
        score += 10
    if plexos_row["plexos_fuel_norm"] and plexos_row["plexos_fuel_norm"] == sced_row.get("cdr_fuel_norm", ""):
        score += 8
    if plexos_row.get("plexos_county_norm", "") and plexos_row.get("plexos_county_norm", "") == sced_row.get("cdr_county_norm", ""):
        score += 12
    if sced_row.get("price_code"):
        score += 5

    ercot_unit_bonus = int(20 * similarity_score(
        plexos_row.get("plexos_ercot_unit_norm", ""),
        sced_row.get("resource_name_norm", ""),
    ))
    cdr_name_bonus = int(12 * similarity_score(
        plexos_row.get("plexos_cdr_name_norm", ""),
        sced_row.get("station_desc_norm", ""),
    ))
    name_bonus = int(8 * similarity_score(
        plexos_row.get("plexos_name_norm", ""),
        sced_row.get("station_desc_norm", ""),
    ))
    cdr_unit_name_bonus = int(14 * similarity_score(
        plexos_row.get("plexos_name_base_norm", ""),
        sced_row.get("cdr_unit_name_norm", ""),
    ))
    score += ercot_unit_bonus + cdr_name_bonus + name_bonus + cdr_unit_name_bonus

    return {
        "matched_sced_node": sced_row["resource_name"],
        "matched_sced_station_code": sced_row.get("station_code", ""),
        "matched_sced_station_description": sced_row.get("station_desc", ""),
        "matched_sced_fuel_type": sced_row.get("fuel_type", ""),
        "matched_sced_capacity_mw": sced_row.get("capacity_mw", pd.NA),
        "matched_price_node": sced_row.get("price_code", ""),
        "matched_price_node_source": sced_row.get("price_node_source", ""),
        "plexos_to_sced_match_method": method,
        "plexos_to_sced_match_score": score,
    }


def score_unit_hint_against_sced(plexos_row: dict, sced_row: dict, station_group_size: int) -> int:
    unit_hint = plexos_row.get("plexos_name_unit_hint", "")
    if not unit_hint:
        return 10 if station_group_size == 1 else 0

    normalized_hint = normalize_text(unit_hint).replace(" ", "")
    hint_tokens = {normalized_hint}
    number_match = re.search(r"(\d+)$", normalized_hint)
    if number_match:
        hint_tokens.add(number_match.group(1))
        if normalized_hint.startswith("U"):
            hint_tokens.add(f"UNIT{number_match.group(1)}")
        if normalized_hint.startswith("UNIT"):
            hint_tokens.add(f"U{number_match.group(1)}")

    resource_tokens = sced_row.get("resource_unit_tokens", set())
    if normalized_hint in resource_tokens:
        return 35
    if hint_tokens & resource_tokens:
        return 25
    if station_group_size == 1:
        return 10
    return 0


def build_plexos_reference_lookups(sced_reference_df: pd.DataFrame) -> dict[str, object]:
    rows = sced_reference_df.to_dict("records")
    by_resource_name = {}
    by_resource_key = {}
    by_station_desc = {}
    by_cdr_unit_name = {}
    by_price_code = {}
    by_price_code_key = {}
    by_price_code_family = {}
    resource_names = []
    station_descs = []
    cdr_unit_names = []

    for row in rows:
        if row["resource_name_norm"]:
            by_resource_name.setdefault(row["resource_name_norm"], []).append(row)
            if row["resource_name_norm"] not in resource_names:
                resource_names.append(row["resource_name_norm"])
        if row["resource_name_key"]:
            by_resource_key.setdefault(row["resource_name_key"], []).append(row)
        if row["station_desc_norm"]:
            by_station_desc.setdefault(row["station_desc_norm"], []).append(row)
            if row["station_desc_norm"] not in station_descs:
                station_descs.append(row["station_desc_norm"])
        if row.get("cdr_unit_name_norm", ""):
            by_cdr_unit_name.setdefault(row["cdr_unit_name_norm"], []).append(row)
            if isinstance(row["cdr_unit_name_norm"], str) and row["cdr_unit_name_norm"] and row["cdr_unit_name_norm"] not in cdr_unit_names:
                cdr_unit_names.append(row["cdr_unit_name_norm"])
        if row.get("price_code_norm", ""):
            by_price_code.setdefault(row["price_code_norm"], []).append(row)
        if row.get("price_code_key", ""):
            by_price_code_key.setdefault(row["price_code_key"], []).append(row)
        if row.get("price_code_family", ""):
            by_price_code_family.setdefault(row["price_code_family"], []).append(row)

    return {
        "by_resource_name": by_resource_name,
        "by_resource_key": by_resource_key,
        "by_station_desc": by_station_desc,
        "by_cdr_unit_name": by_cdr_unit_name,
        "by_price_code": by_price_code,
        "by_price_code_key": by_price_code_key,
        "by_price_code_family": by_price_code_family,
        "resource_names": resource_names,
        "station_descs": station_descs,
        "cdr_unit_names": cdr_unit_names,
    }


def build_yes_plexos_lookups(yes_df: pd.DataFrame) -> dict[str, object]:
    yes_prepared = prepare_yes_df(yes_df)
    yes_rows = yes_prepared.to_dict("records")
    by_plant_name = {}
    plant_names = []

    for row in yes_rows:
        plant_name = row.get("plant_name_norm", "")
        if not plant_name:
            continue
        by_plant_name.setdefault(plant_name, []).append(row)
        if plant_name not in plant_names:
            plant_names.append(plant_name)

    return {
        "by_plant_name": by_plant_name,
        "plant_names": plant_names,
    }


def choose_best_plexos_match(candidate_rows: list[dict]) -> dict:
    if not candidate_rows:
        return {
            "matched_sced_node": "",
            "matched_sced_station_code": "",
            "matched_sced_station_description": "",
            "matched_sced_fuel_type": "",
            "matched_sced_capacity_mw": pd.NA,
            "matched_price_node": "",
            "matched_price_node_source": "",
            "plexos_to_sced_match_method": "",
            "plexos_to_sced_match_score": pd.NA,
            "plexos_to_sced_match_status": "unmatched",
        }

    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            row["plexos_to_sced_match_score"],
            bool(row["matched_price_node"]),
            row["matched_sced_node"],
        ),
        reverse=True,
    )
    best = candidate_rows[0].copy()
    top_score = best["plexos_to_sced_match_score"]
    top_nodes = []
    for row in candidate_rows:
        if row["plexos_to_sced_match_score"] != top_score:
            continue
        if row["matched_sced_node"] and row["matched_sced_node"] not in top_nodes:
            top_nodes.append(row["matched_sced_node"])
    best["plexos_to_sced_match_status"] = "ambiguous" if len(top_nodes) > 1 else "matched"
    return best


def collect_plexos_candidates(
    plexos_row: dict,
    lookups: dict[str, object],
) -> list[dict]:
    candidates: list[dict] = []
    seen_nodes: set[tuple[str, str]] = set()

    def add_candidates(rows: list[dict], base_score: int, method: str, use_unit_hint: bool = False):
        station_group_size = len(rows)
        for sced_row in rows:
            key = (method, sced_row["resource_name"])
            if key in seen_nodes:
                continue
            seen_nodes.add(key)
            unit_hint_bonus = score_unit_hint_against_sced(plexos_row, sced_row, station_group_size) if use_unit_hint else 0
            candidates.append(
                score_plexos_candidate(plexos_row, sced_row, base_score, method, unit_hint_bonus)
            )

    ercot_unit_norm = plexos_row.get("plexos_ercot_unit_norm", "")
    if ercot_unit_norm:
        add_candidates(
            lookups["by_resource_name"].get(ercot_unit_norm, []),
            220,
            "ercot_unitcode_exact",
        )
        add_candidates(
            lookups["by_resource_key"].get(normalize_key(ercot_unit_norm), []),
            210,
            "ercot_unitcode_key_exact",
        )
        fuzzy_resource_names = get_close_matches(
            ercot_unit_norm,
            lookups["resource_names"],
            n=5,
            cutoff=PLEXOS_ERCOT_UNIT_FUZZY_CUTOFF,
        )
        for resource_name in fuzzy_resource_names:
            add_candidates(
                lookups["by_resource_name"].get(resource_name, []),
                165,
                "ercot_unitcode_fuzzy",
            )
        if not candidates:
            late_fuzzy_resource_names = get_close_matches(
                ercot_unit_norm,
                lookups["resource_names"],
                n=8,
                cutoff=PLEXOS_ERCOT_UNIT_LATE_FUZZY_CUTOFF,
            )
            for resource_name in late_fuzzy_resource_names:
                add_candidates(
                    lookups["by_resource_name"].get(resource_name, []),
                    135,
                    "ercot_unitcode_fuzzy_late",
                )

    name_base_norm = plexos_row.get("plexos_name_base_norm", "")
    if name_base_norm:
        add_candidates(
            lookups["by_cdr_unit_name"].get(name_base_norm, []),
            185,
            "cdr_unit_name_exact",
            use_unit_hint=True,
        )
        fuzzy_cdr_unit_names = get_close_matches(
            name_base_norm,
            lookups["cdr_unit_names"],
            n=5,
            cutoff=PLEXOS_CDR_UNIT_NAME_FUZZY_CUTOFF,
        )
        for cdr_unit_name in fuzzy_cdr_unit_names:
            add_candidates(
                lookups["by_cdr_unit_name"].get(cdr_unit_name, []),
                150,
                "cdr_unit_name_fuzzy",
                use_unit_hint=True,
            )

    if name_base_norm:
        add_candidates(
            lookups["by_station_desc"].get(name_base_norm, []),
            180,
            "name_base_exact_station_desc",
            use_unit_hint=True,
        )
        fuzzy_station_descs = get_close_matches(
            name_base_norm,
            lookups["station_descs"],
            n=5,
            cutoff=PLEXOS_NAME_FUZZY_CUTOFF,
        )
        for station_desc in fuzzy_station_descs:
            add_candidates(
                lookups["by_station_desc"].get(station_desc, []),
                145,
                "name_base_fuzzy_station_desc",
                use_unit_hint=True,
            )

    cdr_name_norm = plexos_row.get("plexos_cdr_name_norm", "")
    if cdr_name_norm:
        add_candidates(
            lookups["by_station_desc"].get(cdr_name_norm, []),
            150,
            "cdr_name_exact_station_desc",
            use_unit_hint=True,
        )
        fuzzy_station_descs = get_close_matches(
            cdr_name_norm,
            lookups["station_descs"],
            n=5,
            cutoff=PLEXOS_CDR_NAME_FUZZY_CUTOFF,
        )
        for station_desc in fuzzy_station_descs:
            add_candidates(
                lookups["by_station_desc"].get(station_desc, []),
                125,
                "cdr_name_fuzzy_station_desc",
                use_unit_hint=True,
            )

    return candidates


def collect_plexos_yes_price_candidates(
    plexos_row: dict,
    sced_lookups: dict[str, object],
    yes_lookups: dict[str, object],
) -> list[dict]:
    candidates: list[dict] = []
    seen_nodes: set[tuple[str, str]] = set()

    def add_sced_rows(rows: list[dict], base_score: int, method: str):
        station_group_size = len(rows)
        for sced_row in rows:
            key = (method, sced_row["resource_name"])
            if key in seen_nodes:
                continue
            seen_nodes.add(key)
            unit_hint_bonus = score_unit_hint_against_sced(plexos_row, sced_row, station_group_size)
            candidates.append(
                score_plexos_candidate(plexos_row, sced_row, base_score, method, unit_hint_bonus)
            )

    def resolve_yes_row(yes_row: dict, base_score: int, method_prefix: str):
        node_norm = yes_row.get("node_name_norm", "")
        node_key = yes_row.get("node_name_key", "")
        node_family = yes_row.get("node_family", "")
        if not node_norm:
            return

        exact_rows = sced_lookups["by_price_code"].get(node_norm, [])
        if exact_rows:
            add_sced_rows(exact_rows, base_score + 20, f"{method_prefix}_price_exact")

        key_rows = sced_lookups["by_price_code_key"].get(node_key, []) if node_key else []
        if key_rows:
            add_sced_rows(key_rows, base_score + 15, f"{method_prefix}_price_key")

        family_rows = sced_lookups["by_price_code_family"].get(node_family, []) if node_family else []
        if family_rows:
            add_sced_rows(family_rows, base_score, f"{method_prefix}_price_family")

    candidate_names: list[tuple[str, int, str]] = []
    if plexos_row.get("plexos_name_base_norm", ""):
        candidate_names.append((plexos_row["plexos_name_base_norm"], 145, "yes_name_base"))
    if plexos_row.get("plexos_cdr_name_norm", ""):
        candidate_names.append((plexos_row["plexos_cdr_name_norm"], 135, "yes_cdr_name"))

    seen_plant_matches: set[tuple[str, str]] = set()
    for source_name, exact_score, method_prefix in candidate_names:
        for yes_row in yes_lookups["by_plant_name"].get(source_name, []):
            plant_key = (method_prefix, yes_row.get("plant_name_norm", ""))
            if plant_key in seen_plant_matches:
                continue
            seen_plant_matches.add(plant_key)
            resolve_yes_row(yes_row, exact_score, f"{method_prefix}_exact")

        fuzzy_names = get_close_matches(
            source_name,
            yes_lookups["plant_names"],
            n=5,
            cutoff=PLEXOS_YES_PLANT_FUZZY_CUTOFF,
        )
        for plant_name in fuzzy_names:
            for yes_row in yes_lookups["by_plant_name"].get(plant_name, []):
                plant_key = (method_prefix, yes_row.get("plant_name_norm", ""))
                if plant_key in seen_plant_matches:
                    continue
                seen_plant_matches.add(plant_key)
                resolve_yes_row(yes_row, exact_score - 20, f"{method_prefix}_fuzzy")

    return candidates


def build_plexos_matches(
    plexos_df: pd.DataFrame,
    sced_reference_df: pd.DataFrame,
    yes_df: pd.DataFrame,
) -> pd.DataFrame:
    plexos_prepared = prepare_plexos_df(plexos_df)
    sced_lookups = build_plexos_reference_lookups(sced_reference_df)
    yes_lookups = build_yes_plexos_lookups(yes_df)

    appended_rows = []
    for row in plexos_prepared.to_dict("records"):
        primary_candidates = collect_plexos_candidates(row, sced_lookups)
        match = choose_best_plexos_match(primary_candidates)
        if match["plexos_to_sced_match_status"] == "unmatched":
            fallback_candidates = collect_plexos_yes_price_candidates(row, sced_lookups, yes_lookups)
            if fallback_candidates:
                match = choose_best_plexos_match(fallback_candidates)
        appended_rows.append(match)

    match_df = pd.DataFrame(appended_rows)
    original_columns = list(plexos_df.columns)
    output_df = pd.concat([plexos_df.reset_index(drop=True), match_df.reset_index(drop=True)], axis=1)
    appended_columns = [column for column in output_df.columns if column not in original_columns]
    return output_df[original_columns + appended_columns]


def build_plexos_technology_summary(plexos_matches_df: pd.DataFrame) -> pd.DataFrame:
    df = plexos_matches_df.copy()
    df["technology"] = df["Category"].map(normalize_text).replace("", "UNKNOWN")
    df["is_sced_matched"] = df["plexos_to_sced_match_status"].isin(["matched", "ambiguous"])
    df["is_sced_exact"] = df["plexos_to_sced_match_method"].fillna("").isin(
        ["ercot_unitcode_exact", "ercot_unitcode_key_exact"]
    )
    df["is_sced_ambiguous"] = df["plexos_to_sced_match_status"].eq("ambiguous")
    df["is_sced_unmatched"] = df["plexos_to_sced_match_status"].eq("unmatched")
    df["is_price_matched"] = df["matched_price_node"].fillna("").astype(str).str.strip().ne("")
    df["is_price_from_rtlmp_bus"] = df["matched_price_node_source"].fillna("").eq("rtlmp_bus")
    df["is_price_from_rtlmp"] = df["matched_price_node_source"].fillna("").eq("rtlmp")
    df["is_price_blank"] = ~df["is_price_matched"]

    grouped = (
        df.groupby("technology", dropna=False)
        .agg(
            rows_total=("technology", "size"),
            rows_sced_matched=("is_sced_matched", "sum"),
            rows_sced_exact=("is_sced_exact", "sum"),
            rows_sced_ambiguous=("is_sced_ambiguous", "sum"),
            rows_sced_unmatched=("is_sced_unmatched", "sum"),
            rows_price_matched=("is_price_matched", "sum"),
            rows_price_from_rtlmp_bus=("is_price_from_rtlmp_bus", "sum"),
            rows_price_from_rtlmp=("is_price_from_rtlmp", "sum"),
            rows_price_blank=("is_price_blank", "sum"),
        )
        .reset_index()
    )

    for numerator, pct_col in [
        ("rows_sced_matched", "pct_sced_matched"),
        ("rows_sced_exact", "pct_sced_exact"),
        ("rows_sced_ambiguous", "pct_sced_ambiguous"),
        ("rows_price_matched", "pct_price_matched"),
    ]:
        grouped[pct_col] = (grouped[numerator] / grouped["rows_total"]).round(4)

    overall = pd.DataFrame(
        {
            "technology": ["ALL"],
            "rows_total": [len(df)],
            "rows_sced_matched": [int(df["is_sced_matched"].sum())],
            "rows_sced_exact": [int(df["is_sced_exact"].sum())],
            "rows_sced_ambiguous": [int(df["is_sced_ambiguous"].sum())],
            "rows_sced_unmatched": [int(df["is_sced_unmatched"].sum())],
            "rows_price_matched": [int(df["is_price_matched"].sum())],
            "rows_price_from_rtlmp_bus": [int(df["is_price_from_rtlmp_bus"].sum())],
            "rows_price_from_rtlmp": [int(df["is_price_from_rtlmp"].sum())],
            "rows_price_blank": [int(df["is_price_blank"].sum())],
            "pct_sced_matched": [round(float(df["is_sced_matched"].mean()), 4)],
            "pct_sced_exact": [round(float(df["is_sced_exact"].mean()), 4)],
            "pct_sced_ambiguous": [round(float(df["is_sced_ambiguous"].mean()), 4)],
            "pct_price_matched": [round(float(df["is_price_matched"].mean()), 4)],
        }
    )

    summary_df = pd.concat(
        [grouped.sort_values(["rows_total", "technology"], ascending=[False, True]), overall],
        ignore_index=True,
    )
    return summary_df


def build_sced_plexos_coverage_outputs(
    sced_reference_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sced_detail = sced_reference_df[
        [
            "resource_name",
            "station_code",
            "station_desc",
            "fuel_type",
            "generator_type",
            "capacity_mw",
            "price_code",
            "price_node_source",
        ]
    ].copy()
    sced_detail["sced_fuel"] = sced_detail["fuel_type"].where(
        sced_detail["fuel_type"].fillna("").astype(str).str.strip().ne(""),
        sced_detail["generator_type"],
    )
    sced_detail["sced_fuel"] = sced_detail["sced_fuel"].map(normalize_text).replace("", "UNKNOWN")

    matched_plexos = plexos_matches_df[
        plexos_matches_df["plexos_to_sced_match_status"].isin(["matched", "ambiguous"])
        & plexos_matches_df["matched_sced_node"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    plexos_counts = (
        matched_plexos.groupby("matched_sced_node", dropna=False)
        .agg(
            plexos_match_count=("matched_sced_node", "size"),
            plexos_exact_count=("plexos_to_sced_match_method", lambda s: int(s.fillna("").isin(["ercot_unitcode_exact", "ercot_unitcode_key_exact"]).sum())),
            plexos_ambiguous_count=("plexos_to_sced_match_status", lambda s: int((s == "ambiguous").sum())),
            plexos_rows=("Name", lambda s: " | ".join(sorted(dict.fromkeys(str(v) for v in s if pd.notna(v)))[:10])),
        )
        .reset_index()
        .rename(columns={"matched_sced_node": "resource_name"})
    )

    sced_detail = sced_detail.merge(plexos_counts, on="resource_name", how="left")
    sced_detail["plexos_match_count"] = sced_detail["plexos_match_count"].fillna(0).astype(int)
    sced_detail["plexos_exact_count"] = sced_detail["plexos_exact_count"].fillna(0).astype(int)
    sced_detail["plexos_ambiguous_count"] = sced_detail["plexos_ambiguous_count"].fillna(0).astype(int)
    sced_detail["plexos_rows"] = sced_detail["plexos_rows"].fillna("")
    sced_detail["is_matched_in_plexos"] = sced_detail["plexos_match_count"] > 0
    sced_detail["is_duplicate_in_plexos"] = sced_detail["plexos_match_count"] > 1

    summary_df = (
        sced_detail.groupby("sced_fuel", dropna=False)
        .agg(
            sced_rows_total=("resource_name", "size"),
            sced_rows_matched=("is_matched_in_plexos", "sum"),
            sced_rows_unmatched=("is_matched_in_plexos", lambda s: int((~s).sum())),
            sced_rows_duplicated=("is_duplicate_in_plexos", "sum"),
            plexos_links_total=("plexos_match_count", "sum"),
        )
        .reset_index()
        .rename(columns={"sced_fuel": "fuel"})
    )
    summary_df["pct_sced_rows_matched"] = (summary_df["sced_rows_matched"] / summary_df["sced_rows_total"]).round(4)
    summary_df["pct_sced_rows_duplicated"] = (summary_df["sced_rows_duplicated"] / summary_df["sced_rows_total"]).round(4)

    overall = pd.DataFrame(
        {
            "fuel": ["ALL"],
            "sced_rows_total": [len(sced_detail)],
            "sced_rows_matched": [int(sced_detail["is_matched_in_plexos"].sum())],
            "sced_rows_unmatched": [int((~sced_detail["is_matched_in_plexos"]).sum())],
            "sced_rows_duplicated": [int(sced_detail["is_duplicate_in_plexos"].sum())],
            "plexos_links_total": [int(sced_detail["plexos_match_count"].sum())],
            "pct_sced_rows_matched": [round(float(sced_detail["is_matched_in_plexos"].mean()), 4)],
            "pct_sced_rows_duplicated": [round(float(sced_detail["is_duplicate_in_plexos"].mean()), 4)],
        }
    )
    summary_df = pd.concat(
        [summary_df.sort_values(["sced_rows_total", "fuel"], ascending=[False, True]), overall],
        ignore_index=True,
    )

    sced_detail = sced_detail.sort_values(
        ["is_matched_in_plexos", "plexos_match_count", "resource_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    sced_duplicates_df = sced_detail[sced_detail["is_duplicate_in_plexos"]].copy().reset_index(drop=True)
    return sced_detail, summary_df, sced_duplicates_df


def add_pun_presence_flag_to_plexos_matches(
    pun_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    pun_combo_series = (
        pun_df["SubstationName"].fillna("").astype(str).str.strip()
        + "_"
        + pun_df["UnitName"].fillna("").astype(str).str.strip()
    )
    pun_combo_keys = {
        normalize_key(value)
        for value in pun_combo_series
        if normalize_key(value)
    }

    output_df = plexos_matches_df.copy()
    comparison_series = pd.concat(
        [
            output_df["ERCOT_UnitCode"],
            output_df["matched_sced_node"],
        ],
        ignore_index=True,
    )
    comparison_keys = [
        normalize_key(value) if pd.notna(value) else ""
        for value in comparison_series
    ]
    first_half = comparison_keys[: len(output_df)]
    second_half = comparison_keys[len(output_df) :]
    output_df["in_pun_generation_report"] = [
        "Y" if eia_key in pun_combo_keys or sced_key in pun_combo_keys else "N"
        for eia_key, sced_key in zip(first_half, second_half)
    ]
    return output_df


def build_pun_presence_output(
    pun_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = pun_df.copy()
    output_df["pun_combo"] = (
        output_df["SubstationName"].fillna("").astype(str).str.strip()
        + "_"
        + output_df["UnitName"].fillna("").astype(str).str.strip()
    )
    pun_combo_keys = output_df["pun_combo"].map(normalize_key)

    lookup_series = pd.concat(
        [
            plexos_matches_df["ERCOT_UnitCode"],
            plexos_matches_df["matched_sced_node"],
        ],
        ignore_index=True,
    )
    lookup_keys = {
        normalize_key(value)
        for value in lookup_series.dropna().astype(str)
        if normalize_key(value)
    }
    output_df["in_plexos_match_list"] = pun_combo_keys.map(
        lambda value: "Y" if value in lookup_keys else "N"
    )
    return output_df


def add_sced_plant_units_to_plexos_matches(
    sced_plant_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    sced_units = sced_plant_df[["resource_name"]].copy()
    sced_units["sced_plant_id"] = sced_units["resource_name"].map(derive_sced_plant_id)
    plant_units_lookup = (
        sced_units.groupby("sced_plant_id", dropna=False)["resource_name"]
        .agg(joined_examples)
        .to_dict()
    )

    output_df = plexos_matches_df.copy()
    output_df["matched_sced_plant_units"] = output_df["matched_sced_node"].map(
        lambda node: plant_units_lookup.get(derive_sced_plant_id(node), "") if pd.notna(node) and str(node).strip() else ""
    )
    return output_df


def add_duplicate_sced_node_reference(
    sced_plant_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
) -> pd.DataFrame:
    sced_lookup = sced_plant_df[["resource_name", "avg_base_point"]].copy()
    sced_lookup["avg_base_point"] = parse_numeric(sced_lookup["avg_base_point"])
    avg_base_point_lookup = sced_lookup.set_index("resource_name")["avg_base_point"].to_dict()

    output_df = plexos_matches_df.copy()
    duplicate_counts = output_df["matched_sced_node"].fillna("").astype(str).str.strip()
    duplicate_counts = duplicate_counts[duplicate_counts.ne("")].value_counts()

    def build_duplicate_reference(node) -> str:
        if pd.isna(node):
            return ""
        node_text = str(node).strip()
        if not node_text or duplicate_counts.get(node_text, 0) <= 1:
            return ""
        avg_base_point = avg_base_point_lookup.get(node_text)
        if pd.notna(avg_base_point):
            return f"{node_text} | avg_base_point={avg_base_point:.6f}"
        return node_text

    output_df["duplicate_matched_sced_node_reference"] = output_df["matched_sced_node"].map(build_duplicate_reference)
    return output_df


def build_plant_basepoint_reconciliation_outputs(
    sced_plant_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sced_units = sced_plant_df.copy()
    sced_units["sced_plant_id"] = sced_units["resource_name"].map(derive_sced_plant_id)
    sced_units["avg_base_point"] = parse_numeric(sced_units["avg_base_point"])
    sced_units["sced_fuel_norm"] = sced_units["fuel_type"].map(fuel_key).replace("", "UNKNOWN")

    sced_plants = (
        sced_units.groupby("sced_plant_id", dropna=False)
        .agg(
            sced_avg_base_point_sum=("avg_base_point", "sum"),
            sced_avg_base_point_mean=("avg_base_point", "mean"),
            sced_unit_count=("resource_name", "size"),
            sced_fuel=("sced_fuel_norm", lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN"),
            sced_example_nodes=("resource_name", joined_examples),
        )
        .reset_index()
    )

    plexos_units = plexos_matches_df.copy()
    name_parts = plexos_units["Name"].map(parse_plexos_name_parts)
    plexos_units["plexos_plant_id"] = name_parts.map(lambda parts: parts["name_base_norm"])
    plexos_units["linked_sced_plant_id"] = plexos_units["matched_sced_node"].map(derive_sced_plant_id)
    plexos_units["matched_sced_capacity_mw"] = parse_numeric(plexos_units["matched_sced_capacity_mw"])
    plexos_units["has_sced_match"] = plexos_units["matched_sced_node"].fillna("").astype(str).str.strip().ne("")

    linked_units = plexos_units[plexos_units["has_sced_match"]].copy()
    plant_links = (
        linked_units.groupby(["plexos_plant_id", "linked_sced_plant_id"], dropna=False)
        .agg(linked_unit_count=("matched_sced_node", "size"))
        .reset_index()
    )
    plant_links = plant_links.sort_values(
        ["plexos_plant_id", "linked_unit_count", "linked_sced_plant_id"],
        ascending=[True, False, True],
    )
    dominant_links = plant_links.drop_duplicates(subset="plexos_plant_id", keep="first").rename(
        columns={"linked_sced_plant_id": "dominant_sced_plant_id"}
    )
    plant_link_counts = (
        plant_links.groupby("plexos_plant_id", dropna=False)
        .agg(
            distinct_sced_plants=("linked_sced_plant_id", "nunique"),
            total_linked_units=("linked_unit_count", "sum"),
            linked_sced_plants=("linked_sced_plant_id", joined_examples),
        )
        .reset_index()
    )

    plexos_plants = (
        plexos_units.groupby("plexos_plant_id", dropna=False)
        .agg(
            plexos_unit_count=("Name", "size"),
            matched_unit_count=("has_sced_match", "sum"),
            plexos_capacity_sum_mw=("matched_sced_capacity_mw", "sum"),
            plexos_example_names=("Name", joined_examples),
            plexos_matched_sced_nodes=("matched_sced_node", joined_examples),
            plexos_categories=("Category", joined_examples),
            plexos_price_nodes=("matched_price_node", joined_examples),
        )
        .reset_index()
    )
    plexos_plants = plexos_plants.merge(dominant_links, on="plexos_plant_id", how="left")
    plexos_plants = plexos_plants.merge(plant_link_counts, on="plexos_plant_id", how="left")
    plexos_plants["distinct_sced_plants"] = plexos_plants["distinct_sced_plants"].fillna(0).astype(int)
    plexos_plants["total_linked_units"] = plexos_plants["total_linked_units"].fillna(0).astype(int)
    plexos_plants["linked_sced_plants"] = plexos_plants["linked_sced_plants"].fillna("")
    plexos_plants["plant_match_status"] = plexos_plants["distinct_sced_plants"].map(
        lambda count: "unmatched" if count == 0 else ("resolved" if count == 1 else "ambiguous")
    )

    reconciliation_df = plexos_plants.merge(
        sced_plants,
        left_on="dominant_sced_plant_id",
        right_on="sced_plant_id",
        how="left",
    )
    reconciliation_df["difference_mw"] = (
        reconciliation_df["plexos_capacity_sum_mw"] - reconciliation_df["sced_avg_base_point_sum"]
    )
    reconciliation_df["difference_pct"] = reconciliation_df["difference_mw"] / reconciliation_df[
        "plexos_capacity_sum_mw"
    ].where(reconciliation_df["plexos_capacity_sum_mw"].abs() > 0)
    reconciliation_df["basepoint_to_capacity_ratio"] = reconciliation_df["sced_avg_base_point_sum"] / reconciliation_df[
        "plexos_capacity_sum_mw"
    ].where(reconciliation_df["plexos_capacity_sum_mw"].abs() > 0)
    reconciliation_df["abs_difference_mw"] = reconciliation_df["difference_mw"].abs()
    reconciliation_df["abs_difference_pct"] = reconciliation_df["difference_pct"].abs()
    reconciliation_df["has_comparable_values"] = (
        (reconciliation_df["plant_match_status"] == "resolved")
        & reconciliation_df["plexos_capacity_sum_mw"].notna()
        & reconciliation_df["sced_avg_base_point_sum"].notna()
        & (reconciliation_df["plexos_capacity_sum_mw"] > 0)
    )
    reconciliation_df["discrepancy_rule_type"] = "other_over_capacity"
    reconciliation_df.loc[reconciliation_df["sced_fuel"] == "WIND", "discrepancy_rule_type"] = "wind_ratio"
    reconciliation_df.loc[reconciliation_df["sced_fuel"] == "SOLAR", "discrepancy_rule_type"] = "solar_ratio"
    reconciliation_df["discrepancy_threshold"] = pd.NA
    reconciliation_df.loc[reconciliation_df["discrepancy_rule_type"] == "wind_ratio", "discrepancy_threshold"] = WIND_BASEPOINT_CAPACITY_RATIO_THRESHOLD
    reconciliation_df.loc[reconciliation_df["discrepancy_rule_type"] == "solar_ratio", "discrepancy_threshold"] = SOLAR_BASEPOINT_CAPACITY_RATIO_THRESHOLD

    other_rule_mask = (
        reconciliation_df["has_comparable_values"]
        & (reconciliation_df["discrepancy_rule_type"] == "other_over_capacity")
        & (reconciliation_df["sced_avg_base_point_sum"] > reconciliation_df["plexos_capacity_sum_mw"])
        & (
            ((reconciliation_df["sced_avg_base_point_sum"] - reconciliation_df["plexos_capacity_sum_mw"]) >= PLANT_DISCREPANCY_MW_THRESHOLD)
            | (
                ((reconciliation_df["sced_avg_base_point_sum"] - reconciliation_df["plexos_capacity_sum_mw"])
                / reconciliation_df["plexos_capacity_sum_mw"].where(reconciliation_df["plexos_capacity_sum_mw"].abs() > 0))
                >= PLANT_DISCREPANCY_PCT_THRESHOLD
            )
        )
    )
    wind_rule_mask = (
        reconciliation_df["has_comparable_values"]
        & (reconciliation_df["discrepancy_rule_type"] == "wind_ratio")
        & (reconciliation_df["basepoint_to_capacity_ratio"] >= WIND_BASEPOINT_CAPACITY_RATIO_THRESHOLD)
    )
    solar_rule_mask = (
        reconciliation_df["has_comparable_values"]
        & (reconciliation_df["discrepancy_rule_type"] == "solar_ratio")
        & (reconciliation_df["basepoint_to_capacity_ratio"] >= SOLAR_BASEPOINT_CAPACITY_RATIO_THRESHOLD)
    )
    reconciliation_df["is_discrepant"] = other_rule_mask | wind_rule_mask | solar_rule_mask

    reconciliation_df = reconciliation_df.sort_values(
        ["is_discrepant", "abs_difference_mw", "plexos_plant_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    reconciliation_df = reconciliation_df[
        [
            "plexos_plant_id",
            "dominant_sced_plant_id",
            "plant_match_status",
            "distinct_sced_plants",
            "linked_sced_plants",
            "plexos_unit_count",
            "matched_unit_count",
            "sced_unit_count",
            "plexos_capacity_sum_mw",
            "sced_avg_base_point_sum",
            "sced_avg_base_point_mean",
            "difference_mw",
            "difference_pct",
            "basepoint_to_capacity_ratio",
            "is_discrepant",
            "discrepancy_rule_type",
            "discrepancy_threshold",
            "sced_fuel",
            "plexos_categories",
            "plexos_example_names",
            "plexos_matched_sced_nodes",
            "sced_example_nodes",
            "plexos_price_nodes",
        ]
    ]

    discrepancy_df = reconciliation_df[reconciliation_df["is_discrepant"]].copy().reset_index(drop=True)
    return reconciliation_df, discrepancy_df


def save_sced_coverage_plot(
    sced_plexos_coverage_summary_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> Path:
    output_path = ensure_output_dir(output_dir)
    plot_path = output_path / SCED_COVERAGE_PLOT_FILE_NAME

    plot_df = sced_plexos_coverage_summary_df[
        sced_plexos_coverage_summary_df["fuel"].fillna("").astype(str).str.upper().ne("ALL")
    ].copy()
    plot_df = plot_df.sort_values("sced_rows_total", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = list(range(len(plot_df)))
    bar_width = 0.38

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        plot_df["sced_rows_total"],
        width=bar_width,
        label="Total SCED Units",
        color="#9AA5B1",
    )
    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        plot_df["sced_rows_matched"],
        width=bar_width,
        label="Matched SCED Units",
        color="#2B6CB0",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["fuel"], rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel("Fuel")
    ax.set_title("SCED Coverage by Fuel")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_outputs(
    best_matches_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    simple_matches_df: pd.DataFrame,
    plexos_matches_df: pd.DataFrame,
    plexos_tech_summary_df: pd.DataFrame,
    sced_plexos_coverage_detail_df: pd.DataFrame,
    sced_plexos_coverage_summary_df: pd.DataFrame,
    sced_plexos_duplicates_df: pd.DataFrame,
    pun_presence_df: pd.DataFrame,
    plant_reconciliation_df: pd.DataFrame,
    plant_discrepancies_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    output_path = ensure_output_dir(output_dir)
    best_matches_path = output_path / BEST_MATCHES_FILE_NAME
    candidates_path = output_path / ALL_CANDIDATES_FILE_NAME
    summary_path = output_path / SUMMARY_FILE_NAME
    simple_matches_path = output_path / SIMPLE_MATCHES_FILE_NAME
    plexos_matches_path = output_path / PLEXOS_MATCHES_FILE_NAME
    plexos_tech_summary_path = output_path / PLEXOS_TECH_SUMMARY_FILE_NAME
    sced_plexos_coverage_detail_path = output_path / SCED_PLEXOS_COVERAGE_DETAIL_FILE_NAME
    sced_plexos_coverage_summary_path = output_path / SCED_PLEXOS_COVERAGE_SUMMARY_FILE_NAME
    sced_plexos_duplicates_path = output_path / SCED_PLEXOS_DUPLICATES_FILE_NAME
    pun_plexos_presence_path = output_path / PUN_PLEXOS_PRESENCE_FILE_NAME
    plant_reconciliation_path = output_path / PLANT_RECONCILIATION_FILE_NAME
    plant_discrepancies_path = output_path / PLANT_RECONCILIATION_FLAGS_FILE_NAME

    best_matches_df.to_csv(best_matches_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    simple_matches_df.to_csv(simple_matches_path, index=False)
    plexos_matches_df.to_csv(plexos_matches_path, index=False)
    plexos_tech_summary_df.to_csv(plexos_tech_summary_path, index=False)
    sced_plexos_coverage_detail_df.to_csv(sced_plexos_coverage_detail_path, index=False)
    sced_plexos_coverage_summary_df.to_csv(sced_plexos_coverage_summary_path, index=False)
    sced_plexos_duplicates_df.to_csv(sced_plexos_duplicates_path, index=False)
    pun_presence_df.to_csv(pun_plexos_presence_path, index=False)
    plant_reconciliation_df.to_csv(plant_reconciliation_path, index=False)
    plant_discrepancies_df.to_csv(plant_discrepancies_path, index=False)
    return (
        best_matches_path,
        candidates_path,
        summary_path,
        simple_matches_path,
        plexos_matches_path,
        plexos_tech_summary_path,
        sced_plexos_coverage_detail_path,
        sced_plexos_coverage_summary_path,
        sced_plexos_duplicates_path,
        pun_plexos_presence_path,
        plant_reconciliation_path,
        plant_discrepancies_path,
    )


def main():
    sced_plant_df = load_csv(SCED_PLANT_LIST_PATH, SCED_PLANT_REQUIRED_COLUMNS)
    sb_df = load_csv(SB_LIST_PATH, SB_REQUIRED_COLUMNS)
    pun_df = load_csv(PUN_GENERATION_REPORT_PATH, PUN_REQUIRED_COLUMNS)

    sb_matches_df, sb_candidates_df = build_sb_matches(sb_df, sced_plant_df)
    sb_matches_df = add_pun_presence_flag_to_sb_matches(pun_df, sb_matches_df)
    pun_presence_df = build_sb_pun_presence_output(pun_df, sb_matches_df)
    sb_summary_df = build_sb_summary(sb_matches_df)

    (
        sb_matches_path,
        sb_candidates_path,
        sb_summary_path,
        pun_presence_path,
    ) = save_sb_outputs(
        sb_matches_df,
        sb_candidates_df,
        sb_summary_df,
        pun_presence_df,
    )

    print(f"Saved SB matches to {sb_matches_path}")
    print(f"Saved SB candidates to {sb_candidates_path}")
    print(f"Saved SB summary to {sb_summary_path}")
    print(f"Saved PUN presence output to {pun_presence_path}")


if __name__ == "__main__":
    main()
