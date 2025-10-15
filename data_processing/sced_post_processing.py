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
        "enable_fuzzy": os.getenv("ENABLE_FUZZY", "true").lower() in ("1", "true", "yes", "y"),
        "other_fuel_veto_thresh": int(os.getenv("OTHER_FUEL_VETO_THRESH", "88")),
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
    gas["ERCOT_UnitCode"] = gas["ERCOT_UnitCode"].astype(str).str.strip()
    for f in GAS_FIELDS:
        if f in gas.columns:
            gas[f] = gas[f].astype(str)
    return gas

def load_cdr_list(cdr_path: str) -> pd.DataFrame:
    """
    Required columns (case/whitespace insensitive):
    - UNIT CODE  -> ERCOT Unit Code
    - UNIT NAME  -> CDR unit name
    - FUEL       -> fuel family
    This loader returns ONLY Natural Gas rows (strict NG universe for matching).
    """
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
    # Keep Natural Gas variants only (strict gas universe for matching)
    cdr = cdr[cdr["CDR_FUEL"].isin(["GAS", "NATURAL GAS"])].copy()
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

# -------- Build 1:1 alias maps (gas vs other fuels) --------
def build_alias_maps(gas_df: pd.DataFrame, cdr_df_all: pd.DataFrame, cdr_full: pd.DataFrame):
    """
    Returns:
      alias_map_gas:   normalized alias -> ERCOT_UnitCode (Natural Gas ∩ gas_resources, 1:1 only)
      alias_map_other: normalized alias -> ERCOT_UnitCode (all NON-gas units from the full CDR, 1:1 only)
      gas_units: set of ERCOT_UnitCode in gas_df
    """
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))

    # Build NON-gas from full CDR by excluding NG rows (for veto)
    cols = {c.upper().strip(): c for c in cdr_full.columns}
    if not {"UNIT CODE", "UNIT NAME", "FUEL"}.issubset(set(cols.keys())):
        raise ValueError("Full CDR (for veto) must have UNIT CODE/UNIT NAME/FUEL columns.")
    cdr_full_std = cdr_full.rename(columns={
        cols["UNIT CODE"]: "UNIT_CODE",
        cols["UNIT NAME"]: "UNIT_NAME",
        cols["FUEL"]: "FUEL",
    }).copy()
    cdr_full_std["FUEL"] = cdr_full_std["FUEL"].astype(str).str.strip().str.upper()
    cdr_other = cdr_full_std[~cdr_full_std["FUEL"].isin(["GAS", "NATURAL GAS"])].copy()
    cdr_other["UNIT_CODE"] = cdr_other["UNIT_CODE"].astype(str).str.strip()
    cdr_other["UNIT_NAME"] = cdr_other["UNIT_NAME"].astype(str).str.strip()

    # NG subset intersected with gas units (authoritative)
    cdr_gas = cdr_df_all[cdr_df_all["CDR_UNIT_CODE"].isin(gas_units)].copy()

    def collect_aliases(rows, code_col, name_col, extra_fields_df=None):
        alias_to_units = {}
        def add(alias_raw, unit):
            if not alias_raw:
                return
            k = normalize(alias_raw)
            if not k:
                return
            alias_to_units.setdefault(k, set()).add(unit)
        # Authoritative names
        for _, r in rows.iterrows():
            u = str(r[code_col]).strip()
            add(str(r[name_col]).strip(), u)
        # Optionally add fields from gas_df (for gas map)
        if extra_fields_df is not None:
            for _, r in extra_fields_df.iterrows():
                u = str(r.get("ERCOT_UnitCode", "")).strip()
                if not u:
                    continue
                add(u, u)  # code itself
                for f in ["Name", "CDR Name", "ERCOT_INR Code"]:
                    if f in extra_fields_df.columns:
                        val = r.get(f)
                        if pd.isna(val):
                            continue
                        add(str(val), u)
        # Keep only 1:1
        alias_map = {k: list(v)[0] for k, v in alias_to_units.items() if len(v) == 1}
        return alias_map

    alias_map_gas = collect_aliases(cdr_gas, "CDR_UNIT_CODE", "CDR_UNIT_NAME", extra_fields_df=gas_df)
    alias_map_other = collect_aliases(cdr_other, "UNIT_CODE", "UNIT_NAME", extra_fields_df=None)

    return alias_map_gas, alias_map_other, gas_units

# -------- Fuzzy corpora (gas vs other) --------
def build_fuzzy_corpora(gas_df: pd.DataFrame, cdr_df_all: pd.DataFrame, cdr_full: pd.DataFrame):
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))
    cdr_gas = cdr_df_all[cdr_df_all["CDR_UNIT_CODE"].isin(gas_units)].copy()

    cols = {c.upper().strip(): c for c in cdr_full.columns}
    cdr_full_std = cdr_full.rename(columns={
        cols["UNIT CODE"]: "UNIT_CODE",
        cols["UNIT NAME"]: "UNIT_NAME",
        cols["FUEL"]: "FUEL",
    }).copy()
    cdr_full_std["FUEL"] = cdr_full_std["FUEL"].astype(str).str.strip().str.upper()
    cdr_other = cdr_full_std[~cdr_full_std["FUEL"].isin(["GAS", "NATURAL GAS"])].copy()

    def corpus_from(cdr_part, code_col, name_col, include_gas_fields: bool):
        corpus = []
        for _, r in cdr_part.iterrows():
            unit = str(r[code_col]).strip()
            raw = str(r[name_col]).strip()
            corpus.append((unit, name_col, raw, normalize(raw)))
        if include_gas_fields:
            for _, r in gas_df.iterrows():
                unit = str(r.get("ERCOT_UnitCode", "")).strip()
                if not unit:
                    continue
                for f in ["ERCOT_UnitCode", "Name", "CDR Name", "ERCOT_INR Code"]:
                    if f in gas_df.columns:
                        val = r.get(f)
                        if pd.isna(val):
                            continue
                        raw = str(val).strip()
                        corpus.append((unit, f, raw, normalize(raw)))
        return corpus

    corpus_gas = corpus_from(cdr_gas, "CDR_UNIT_CODE", "CDR_UNIT_NAME", include_gas_fields=True)
    corpus_other = corpus_from(cdr_other, "UNIT_CODE", "UNIT_NAME", include_gas_fields=False)
    return corpus_gas, corpus_other

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
    cdr_df_all: pd.DataFrame,   # NG-only subset of CDR
    cdr_full: pd.DataFrame,     # full CDR (all fuels) for veto
    manual_map: Dict[str, str],
    high_thresh: int = 90,
    review_thresh: int = 70,
    enable_fuzzy: bool = True,
    other_fuel_veto_thresh: int = 88
):
    alias_map_gas, alias_map_other, gas_units = build_alias_maps(gas_df, cdr_df_all, cdr_full)
    exact_id_map = {str(u).upper(): str(u) for u in gas_units}
    corpus_gas, corpus_other = build_fuzzy_corpora(gas_df, cdr_df_all, cdr_full) if enable_fuzzy else ([], [])

    out_values, tracker_rows, uncertain_rows = [], [], []

    # Ensure safe string series
    resource_series = resource_series.astype(str).fillna("")
    for original in resource_series.tolist():
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

        q_up = str(original).upper()
        q_norm = normalize(original)

        chosen: Optional[str] = None
        method: Optional[str] = None
        aux = {}

        # 1) exact ERCOT_UnitCode (gas-only)
        if q_up in exact_id_map:
            chosen = exact_id_map[q_up]
            method = "exact_id"
        else:
            # 2) exact alias (gas-only)
            if q_norm in alias_map_gas:
                chosen = alias_map_gas[q_norm]
                method = "alias_exact"
            elif enable_fuzzy and q_norm:
                # -------- other-fuel VETO (exact alias) --------
                if q_norm in alias_map_other:
                    out_values.append(original)
                    row = {
                        "Resource Name": original, "Matched Name": "",
                        "Was Replaced": False, "Method": "blocked_by_other_fuel_exact",
                        "Normalized Query": q_norm, "Best Score": None,
                        "Best Field": None, "Best Raw": None,
                        "Alternatives (json)": "[]"
                    }
                    tracker_rows.append(row)
                    uncertain_rows.append(row)
                    continue

                # -------- other-fuel VETO (fuzzy) --------
                other_top = best_fuzzy_match(q_norm, corpus_other, k=1)
                other_best = other_top[0] if other_top else None
                if other_best and other_best[0] >= other_fuel_veto_thresh:
                    out_values.append(original)
                    row = {
                        "Resource Name": original, "Matched Name": "",
                        "Was Replaced": False, "Method": "blocked_by_other_fuel_fuzzy",
                        "Normalized Query": q_norm, "Best Score": int(other_best[0]),
                        "Best Field": other_best[2], "Best Raw": other_best[3],
                        "Alternatives (json)": "[]"
                    }
                    tracker_rows.append(row)
                    uncertain_rows.append(row)
                    continue

                # -------- gas fuzzy (since not vetoed) --------
                gas_top = best_fuzzy_match(q_norm, corpus_gas, k=5)
                if gas_top:
                    best_score, best_unit, best_field, best_raw = gas_top[0]
                    aux = {
                        "best_score": best_score,
                        "best_field": best_field,
                        "best_raw": best_raw,
                        "alternatives": [{"score": s, "unit": u, "field": f, "raw": r} for s, u, f, r in gas_top]
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
                "Method": "fuzzy_review" if method == "fuzzy_review" else (method or "no_match"),
                "Normalized Query": q_norm, "Best Score": aux.get("best_score"),
                "Best Field": aux.get("best_field"), "Best Raw": aux.get("best_raw"),
                "Alternatives (json)": json.dumps(aux.get("alternatives", []))
            }
            tracker_rows.append(row)
            if row["Method"] in ("fuzzy_review", "no_match", "blocked_by_other_fuel_exact", "blocked_by_other_fuel_fuzzy"):
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
    cdr_df_ng_only: pd.DataFrame,
    cdr_df_full: pd.DataFrame,
    output_path: str,
    tracker_path: str,
    manual_crosswalk_path: str,
    review_path: str,
    coverage_path: str,
    missing_gas_path: str,
    high_thresh: int,
    review_thresh: int,
    enable_fuzzy: bool,
    other_fuel_veto_thresh: int
):
    df = pd.read_csv(agg_path)
    if "Resource Name" not in df.columns:
        raise ValueError("aggregated file missing 'Resource Name' column")

    manual_map = apply_manual_crosswalk(df["Resource Name"], manual_crosswalk_path)

    new_values, tracker_df, uncertain_df = map_resource_names(
        df["Resource Name"],
        gas_df,
        cdr_df_ng_only,
        cdr_df_full,
        manual_map,
        high_thresh=high_thresh,
        review_thresh=review_thresh,
        enable_fuzzy=enable_fuzzy,
        other_fuel_veto_thresh=other_fuel_veto_thresh,
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

    print("🔄 Loading CDR list (Natural Gas only) for matching...")
    cdr_ng = load_cdr_list(cfg["cdr_file"])
    print(f"   CDR (NG) rows: {len(cdr_ng)}")

    print("🔄 Loading full CDR (all fuels) for other-fuel veto...")
    cdr_full = pd.read_csv(cfg["cdr_file"])  # same file; use all fuels for veto
    print(f"   CDR (full) rows: {len(cdr_full)}")

    print("🔄 Deterministic alias matching + optional constrained fuzzy with other-fuel veto...")
    update_resource_names(
        agg_path=cfg["agg_file"],
        gas_df=gas,
        cdr_df_ng_only=cdr_ng,
        cdr_df_full=cdr_full,
        output_path=cfg["output_file"],
        tracker_path=cfg["tracker_file"],
        manual_crosswalk_path=cfg["manual_crosswalk_file"],
        review_path=cfg["review_file"],
        coverage_path=cfg["coverage_file"],
        missing_gas_path=cfg["missing_gas_file"],
        high_thresh=cfg["high_thresh"],
        review_thresh=cfg["review_thresh"],
        enable_fuzzy=cfg["enable_fuzzy"],
        other_fuel_veto_thresh=cfg["other_fuel_veto_thresh"],
    )

if __name__ == "__main__":
    main()
