import os
import re
import json
import pandas as pd
from typing import Dict, Tuple, Optional

# ===================== ENV =====================
def load_env():
    from dotenv import load_dotenv
    load_dotenv()
    return {
        "gas_file": os.getenv("GAS_RESOURCE_FILE", "gas_resources.xlsx"),
        "agg_file": os.getenv("AGGREGATED_MATRIX_FILE", "aggregated_base_point_matrix.csv"),
        "cdr_file": os.getenv("CDR_LIST_FILE", "cdr_list.csv"),
        "output_file": os.getenv("OUTPUT_FILE", "aggregated_base_point_matrix_RELABELED.csv"),
        "tracker_file": os.getenv("TRACKER_FILE", "resource_name_tracker.csv"),
        "manual_crosswalk_file": os.getenv("MANUAL_CROSSWALK_FILE", "manual_crosswalk.csv"),
        "review_file": os.getenv("REVIEW_FILE", "resource_name_uncertain_review.csv"),
        "coverage_file": os.getenv("COVERAGE_FILE", "resource_name_gas_coverage_summary.csv"),
        "missing_gas_file": os.getenv("MISSING_GAS_FILE", "resource_name_missing_gas_units.csv"),
        "high_thresh": int(os.getenv("FUZZY_HIGH_THRESH", "90")),
        "review_thresh": int(os.getenv("FUZZY_REVIEW_THRESH", "70")),
        "enable_fuzzy": os.getenv("ENABLE_FUZZY", "true").lower() in ("1","true","yes","y"),
    }

# ================= NORMALIZATION =================
GENERIC_SUFFIXES = [
    r"\bLLC\b", r"\bINC\b", r"\bENERGY\b", r"\bPOWER\b", r"\bGENERATION\b", r"\bCOMPANY\b",
    r"\bSTATION\b", r"\bPLANT\b", r"\bHOLDINGS\b", r"\bPARTNERS\b", r"\bLP\b", r"\bCO\b"
]
UNIT_PATTERNS = [r"\bUNIT\s*\d+\b", r"\bU\s*\d+\b", r"\bBLOCK\s*\d+\b", r"\bSTG?\s*\d+\b"]
ABBREV_MAP = {
    "&": " AND ",
    "-": " ",
    "_": " ",
    " COMB TURBINE ": " CT ",
    " COMBUSTION TURBINE ": " CT ",
    " STEAM TURBINE ": " ST ",
    " STEAM ": " ST ",
    " COMBINED CYCLE ": " CC ",
}

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).upper()
    for k, v in ABBREV_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    for pat in GENERIC_SUFFIXES + UNIT_PATTERNS:
        s = re.sub(pat, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

try:
    from rapidfuzz import fuzz
    RF_AVAILABLE = True
except Exception:
    import difflib
    RF_AVAILABLE = False

def token_score(a: str, b: str) -> int:
    if RF_AVAILABLE:
        return fuzz.token_set_ratio(a, b)
    import difflib
    return int(difflib.SequenceMatcher(a=a, b=b).ratio() * 100)

# ================ LOADERS =================
GAS_FIELDS = ["ERCOT_UnitCode", "Name", "CDR Name", "ERCOT_INR Code"]

def load_gas_resources(excel_path: str) -> pd.DataFrame:
    gas = pd.read_excel(excel_path, sheet_name=0)
    if "ERCOT_UnitCode" not in gas.columns:
        raise ValueError("gas_resources missing required column: ERCOT_UnitCode")
    return gas

def load_cdr_list(cdr_path: str) -> pd.DataFrame:
    cdr = pd.read_csv(cdr_path)
    cols = {c.upper().strip(): c for c in cdr.columns}
    need = ["UNIT CODE", "UNIT NAME", "FUEL"]
    missing = [k for k in need if k not in cols]
    if missing:
        raise ValueError(f"cdr_list missing columns: {missing}. Found: {list(cdr.columns)}")
    cdr = cdr.rename(columns={
        cols["UNIT CODE"]: "CDR_UNIT_CODE",
        cols["UNIT NAME"]: "CDR_UNIT_NAME",
        cols["FUEL"]: "CDR_FUEL",
    })
    cdr["CDR_UNIT_CODE"] = cdr["CDR_UNIT_CODE"].astype(str).str.strip()
    cdr["CDR_UNIT_NAME"] = cdr["CDR_UNIT_NAME"].astype(str).str.strip()
    cdr["CDR_FUEL"] = cdr["CDR_FUEL"].astype(str).str.strip().str.upper()
    # Natural Gas filter (strict)
    cdr = cdr[cdr["CDR_FUEL"].isin(["GAS", "NATURAL GAS"])]
    cdr["CDR_UNIT_NAME_N"] = cdr["CDR_UNIT_NAME"].map(normalize)
    cdr["CDR_UNIT_CODE_UP"] = cdr["CDR_UNIT_CODE"].str.upper()
    return cdr

def apply_manual_crosswalk(names: pd.Series, manual_path: str) -> Dict[str, str]:
    try:
        if os.path.exists(manual_path):
            xw = pd.read_csv(manual_path)
            xw = xw.dropna(subset=["Resource Name", "ERCOT_UnitCode"])
            return dict(zip(xw["Resource Name"].astype(str), xw["ERCOT_UnitCode"].astype(str)))
    except Exception as e:
        print(f"⚠️ Manual crosswalk load failed: {e}")
    return {}

# ========== BUILD 1:1 ALIAS MAP (deterministic) ==========
def build_deterministic_alias_map(gas_df: pd.DataFrame, cdr_df: pd.DataFrame) -> Tuple[Dict[str, str], set]:
    """
    Returns:
      alias_map: normalized alias -> ERCOT_UnitCode (only aliases mapping to exactly ONE unit kept)
      gas_units: set of ERCOT_UnitCode in gas (for coverage)
    """
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))
    # Restrict CDR to gas units AND Natural Gas
    cdr_gas = cdr_df[cdr_df["CDR_UNIT_CODE"].isin(gas_units)].copy()

    # Collect aliases per unit
    alias_to_units = {}
    def add_alias(alias_raw: str, unit: str):
        if not alias_raw:
            return
        key = normalize(alias_raw)
        if not key:
            return
        alias_to_units.setdefault(key, set()).add(unit)

    # CDR authoritative names
    for _, r in cdr_gas.iterrows():
        u = r["CDR_UNIT_CODE"]
        add_alias(r["CDR_UNIT_NAME"], u)

    # Gas dataframe fields
    for _, r in gas_df.iterrows():
        u = str(r.get("ERCOT_UnitCode", "")).strip()
        if not u:
            continue
        add_alias(u, u)  # the code itself
        for f in ["Name", "CDR Name", "ERCOT_INR Code"]:
            if f in gas_df.columns:
                val = r.get(f)
                if pd.isna(val):
                    continue
                add_alias(str(val), u)

    # Keep only 1:1 aliases (drop ambiguous)
    alias_map = {alias: list(units)[0] for alias, units in alias_to_units.items() if len(units) == 1}
    return alias_map, gas_units

# ====== OPTIONAL: constrained fuzzy (gas∩CDR-NG only) ======
def build_constrained_fuzzy_corpus(gas_df: pd.DataFrame, cdr_df: pd.DataFrame):
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))
    cdr_gas = cdr_df[cdr_df["CDR_UNIT_CODE"].isin(gas_units)].copy()
    corpus = []
    # Use CDR UNIT NAME + gas fields
    for _, r in cdr_gas.iterrows():
        unit = r["CDR_UNIT_CODE"]
        raw = r["CDR_UNIT_NAME"]
        corpus.append((unit, "CDR_UNIT_NAME", raw, normalize(raw)))
    for _, r in gas_df.iterrows():
        unit = str(r.get("ERCOT_UnitCode", "")).strip()
        for f in ["ERCOT_UnitCode", "Name", "CDR Name", "ERCOT_INR Code"]:
            if f in gas_df.columns:
                val = r.get(f)
                if pd.isna(val):
                    continue
                raw = str(val).strip()
                corpus.append((unit, f, raw, normalize(raw)))
    return corpus

def best_fuzzy_match(q_norm: str, corpus, k=5):
    if not q_norm:
        return []
    scored = []
    for unit, field, raw, norm in corpus:
        scored.append((token_score(q_norm, norm), unit, field, raw))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]

# ================= MATCHING =================
def map_resource_names(
    resource_series: pd.Series,
    gas_df: pd.DataFrame,
    cdr_df: pd.DataFrame,
    manual_map: Dict[str, str],
    high_thresh: int = 90,
    review_thresh: int = 70,
    enable_fuzzy: bool = True
):
    alias_map, gas_units = build_deterministic_alias_map(gas_df, cdr_df)
    exact_id_map = {u.upper(): u for u in gas_units}
    fuzzy_corpus = build_constrained_fuzzy_corpus(gas_df, cdr_df) if enable_fuzzy else []

    out_values = []
    tracker_rows = []
    uncertain_rows = []

    for original in resource_series.astype(str).tolist():
        # 0) Manual override
        if original in manual_map and manual_map[original]:
            chosen = str(manual_map[original]).strip()
            out_values.append(chosen)
            tracker_rows.append({
                "Resource Name": original, "Matched Name": chosen, "Was Replaced": True,
                "Method": "manual_crosswalk", "Normalized Query": normalize(original),
                "Best Score": None, "Best Field": "manual", "Best Raw": original,
                "Alternatives (json)": "[]"
            })
            continue

        q_up = original.upper()
        q_norm = normalize(original)

        chosen: Optional[str] = None
        method = None
        aux = {}

        # 1) exact ERCOT_UnitCode (gas-only)
        if q_up in exact_id_map:
            chosen = exact_id_map[q_up]
            method = "exact_id"
        else:
            # 2) deterministic alias exact (normalized)
            if q_norm in alias_map:
                chosen = alias_map[q_norm]
                method = "alias_exact"
            elif enable_fuzzy and q_norm:
                # 3) constrained fuzzy (gas∩CDR-NG)
                top = best_fuzzy_match(q_norm, fuzzy_corpus, k=5)
                if top:
                    best_score, best_unit, best_field, best_raw = top[0]
                    aux = {
                        "best_score": best_score,
                        "best_field": best_field,
                        "best_raw": best_raw,
                        "alternatives": [{"score": s, "unit": u, "field": f, "raw": r} for s, u, f, r in top]
                    }
                    if best_score >= high_thresh:
                        chosen = best_unit
                        method = "fuzzy_auto"
                    elif best_score >= review_thresh:
                        method = "fuzzy_review"

        if chosen:
            out_values.append(chosen)
            tracker_rows.append({
                "Resource Name": original, "Matched Name": chosen, "Was Replaced": True,
                "Method": method, "Normalized Query": q_norm,
                "Best Score": aux.get("best_score"), "Best Field": aux.get("best_field"),
                "Best Raw": aux.get("best_raw"), "Alternatives (json)": json.dumps(aux.get("alternatives", []))
            })
        else:
            out_values.append(original)
            row = {
                "Resource Name": original, "Matched Name": "", "Was Replaced": False,
                "Method": "fuzzy_review" if method == "fuzzy_review" else "no_match",
                "Normalized Query": q_norm, "Best Score": aux.get("best_score"),
                "Best Field": aux.get("best_field"), "Best Raw": aux.get("best_raw"),
                "Alternatives (json)": json.dumps(aux.get("alternatives", []))
            }
            tracker_rows.append(row)
            if row["Method"] in ("fuzzy_review", "no_match"):
                uncertain_rows.append(row)

    return pd.Series(out_values), pd.DataFrame(tracker_rows), pd.DataFrame(uncertain_rows)

# ================= COVERAGE =================
def coverage_report(gas_df: pd.DataFrame, tracker_df: pd.DataFrame) -> pd.DataFrame:
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))
    matched_units = set(tracker_df.loc[tracker_df["Was Replaced"] == True, "Matched Name"].astype(str))
    missing = sorted(gas_units - matched_units)
    summary = pd.DataFrame({
        "Total gas units": [len(gas_units)],
        "Covered (confident)": [len(matched_units)],
        "Missing (need review or mapping)": [len(missing)]
    })
    return summary, gas_df[gas_df["ERCOT_UnitCode"].astype(str).isin(missing)].copy()

# ================= I/O GLUE =================
def update_resource_names(
    agg_path: str,
    gas_df: pd.DataFrame,
    cdr_df: pd.DataFrame,
    output_path: str,
    tracker_path: str,
    manual_crosswalk_path: str,
    review_path: str,
    coverage_path: str,
    missing_gas_path: str,
    high_thresh: int,
    review_thresh: int,
    enable_fuzzy: bool
):
    df = pd.read_csv(agg_path)
    if "Resource Name" not in df.columns:
        raise ValueError("aggregated file missing 'Resource Name' column")

    manual_map = apply_manual_crosswalk(df["Resource Name"], manual_crosswalk_path)

    new_values, tracker_df, uncertain_df = map_resource_names(
        df["Resource Name"], gas_df, cdr_df, manual_map,
        high_thresh=high_thresh, review_thresh=review_thresh, enable_fuzzy=enable_fuzzy
    )

    # Replace only when matched to gas subset
    df["Resource Name"] = new_values

    cov_df, missing_df = coverage_report(gas_df, tracker_df)

    # Writes
    df.to_csv(output_path, index=False)
    tracker_df.to_csv(tracker_path, index=False)
    uncertain_df.to_csv(review_path, index=False)
    cov_df.to_csv(coverage_path, index=False)
    missing_df[["ERCOT_UnitCode","Name","CDR Name","ERCOT_INR Code"]].to_csv(missing_gas_path, index=False)

    print(f"✅ Updated aggregated file: {output_path}")
    print(f"📊 Tracker file: {tracker_path}")
    print(f"📝 Uncertain matches: {review_path} ({len(uncertain_df)} rows)")
    print(f"📈 Gas coverage: {coverage_path}")
    print(f"❗ Missing gas units (not matched confidently): {missing_gas_path} ({len(missing_df)} units)")

def main():
    cfg = load_env()
    print("🔄 Loading gas subset...")
    gas = load_gas_resources(cfg["gas_file"])
    print(f"   Gas units: {len(gas)}")

    print("🔄 Loading CDR list (Natural Gas only) and intersecting with gas subset...")
    cdr = load_cdr_list(cfg["cdr_file"])
    print(f"   CDR (NG) rows: {len(cdr)}")

    print("🔄 Deterministic matching (alias 1:1) + optional constrained fuzzy...")
    update_resource_names(
        agg_path=cfg["agg_file"],
        gas_df=gas,
        cdr_df=cdr,
        output_path=cfg["output_file"],
        tracker_path=cfg["tracker_file"],
        manual_crosswalk_path=cfg["manual_crosswalk_file"],
        review_path=cfg["review_file"],
        coverage_path=cfg["coverage_file"],
        missing_gas_path=cfg["missing_gas_file"],
        high_thresh=cfg["high_thresh"],
        review_thresh=cfg["review_thresh"],
        enable_fuzzy=cfg["enable_fuzzy"],
    )

if __name__ == "__main__":
    main()
