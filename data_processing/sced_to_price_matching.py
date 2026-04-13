from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd


SCED_PLANT_LIST_PATH = "/Users/pradyrao/Downloads/sced_plant_list_extracted.csv"
SCED_NAME_LIST_PATH = "/Users/pradyrao/Downloads/sced_name_list.csv"
YES_UNITS_LIST_PATH = "/Users/pradyrao/Downloads/ERCOT_YES_units_list (1).csv"
RTLMP_BUS_LIST_PATH = "/Users/pradyrao/Downloads/rtlmp_bus_ercot_list.csv"
RTLMP_LIST_PATH = "/Users/pradyrao/Downloads/rtlmp_ercot_list.csv"
RESOURCE_NODE_MAPPING_PATH = "/Users/pradyrao/Downloads/SP_List_EB_Mapping 2/Resource_Node_to_Unit_02202026_094122.csv"

OUTPUT_DIR = "/Users/pradyrao/VSCode/data-processing/data_processing/output/sced_to_price_matching"
BEST_MATCHES_FILE_NAME = "sced_price_code_matches.csv"
ALL_CANDIDATES_FILE_NAME = "sced_price_code_candidates.csv"
SUMMARY_FILE_NAME = "sced_price_code_summary.csv"
SIMPLE_MATCHES_FILE_NAME = "sced_to_price_simple_matches.csv"

SCED_PLANT_REQUIRED_COLUMNS = ["resource_name", "fuel_type"]
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

FUZZY_PLANT_MATCH_CUTOFF = 0.88
FUZZY_STATION_TO_RTLMP_CUTOFF = 0.70
FUZZY_UNMATCHED_DESC_TO_YES_CUTOFF = 0.82
MAX_ALT_MATCHES = 5
MAX_YES_FAMILY_ROWS = 12
MAX_RTLMP_FAMILY_ROWS = 20
MAX_STATION_FUZZY_POOL = 80
MAX_RESOURCE_NODE_FUZZY_POOL = 50


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


def save_outputs(
    best_matches_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    simple_matches_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    output_path = ensure_output_dir(output_dir)
    best_matches_path = output_path / BEST_MATCHES_FILE_NAME
    candidates_path = output_path / ALL_CANDIDATES_FILE_NAME
    summary_path = output_path / SUMMARY_FILE_NAME
    simple_matches_path = output_path / SIMPLE_MATCHES_FILE_NAME

    best_matches_df.to_csv(best_matches_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    simple_matches_df.to_csv(simple_matches_path, index=False)
    return best_matches_path, candidates_path, summary_path, simple_matches_path


def main():
    sced_plant_df = load_csv(SCED_PLANT_LIST_PATH, SCED_PLANT_REQUIRED_COLUMNS)
    sced_name_df = load_csv(SCED_NAME_LIST_PATH, SCED_NAME_REQUIRED_COLUMNS)
    yes_df = load_csv(YES_UNITS_LIST_PATH, YES_REQUIRED_COLUMNS)
    rtlmp_bus_df = load_csv(RTLMP_BUS_LIST_PATH, RTLMP_REQUIRED_COLUMNS)
    rtlmp_df = load_csv(RTLMP_LIST_PATH, RTLMP_REQUIRED_COLUMNS)
    resource_node_mapping_df = load_csv(
        RESOURCE_NODE_MAPPING_PATH,
        RESOURCE_NODE_MAPPING_REQUIRED_COLUMNS,
    )

    best_matches_df, candidates_df, summary_df = build_match_tables(
        sced_plant_df,
        sced_name_df,
        yes_df,
        rtlmp_bus_df,
        rtlmp_df,
        resource_node_mapping_df,
    )
    simple_matches_df = build_simple_matches(best_matches_df)
    best_matches_path, candidates_path, summary_path, simple_matches_path = save_outputs(
        best_matches_df,
        candidates_df,
        summary_df,
        simple_matches_df,
    )

    print(f"Saved best matches to {best_matches_path}")
    print(f"Saved all candidates to {candidates_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved simple matches to {simple_matches_path}")


if __name__ == "__main__":
    main()
