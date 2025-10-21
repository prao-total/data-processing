import os
import re
import json
import pandas as pd
from typing import Dict, Tuple, Optional, List

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
UNIT_PATTERNS = [r"\bUNIT\s*\d+\b", r"\bU\s*[-]?\s*\d+\b", r"\bBLOCK\s*\d+\b", r"\bSTG?\s*\d+\b", r"\bCT\s*\d+\b", r"\bST\s*\d+\b"]
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

ROMAN_MAP = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
ROMAN_RE = re.compile(r"\b(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\b", re.I)

def roman_to_int(s: str) -> Optional[int]:
    s = s.upper()
    if not s or not ROMAN_RE.fullmatch(s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = ROMAN_MAP.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total if total > 0 else None

def extract_unit_numbers(raw: str) -> List[int]:
    """
    Extract probable unit numbers from strings like:
    'U2', 'U-2', 'UNIT 2', 'CT2', 'ST 3', 'BLOCK 1', 'UNIT II'
    """
    if raw is None:
        return []
    s = str(raw)
    nums: List[int] = []

    # Arabic digit patterns
    for pat in [r"\bU\s*[-]?\s*(\d+)\b", r"\bUNIT\s+(\d+)\b", r"\bCT\s*[-]?\s*(\d+)\b",
                r"\bST\s*[-]?\s*(\d+)\b", r"\bBLOCK\s+(\d+)\b", r"\bSTG?\s*[-]?\s*(\d+)\b"]:
        for m in re.finditer(pat, s, flags=re.I):
            try:
                nums.append(int(m.group(1)))
            except:
                pass

    # Roman numerals after UNIT/Block/etc.
    for pat in [r"\bUNIT\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\b",
                r"\bU\s*[-]?\s*(I{1,3}|IV|V|VI{0,3}|IX|X)\b",
                r"\bCT\s*[-]?\s*(I{1,3}|IV|V|VI{0,3}|IX|X)\b",
                r"\bST\s*[-]?\s*(I{1,3}|IV|V|VI{0,3}|IX|X)\b",
                r"\bBLOCK\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\b"]:
        for m in re.finditer(pat, s, flags=re.I):
            val = roman_to_int(m.group(1))
            if val:
                nums.append(val)

    return sorted(set(nums))

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).upper()
    for k, v in ABBREV_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    for pat in GENERIC_SUFFIXES:
        s = re.sub(pat, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def plant_base(s: str) -> str:
    """Normalize then remove unit-number tokens to get a 'plant base' name."""
    n = normalize(s)
    # strip common unit tokens
    n = re.sub(r"\b(UNIT|U|CT|ST|BLOCK|STG)\s*[-]?\s*[0-9]+\b", " ", n)
    n = re.sub(r"\b(UNIT|U|CT|ST|BLOCK|STG)\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

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
    # Precompute helper columns
    gas["PLANT_BASE"] = gas["Name"].map(plant_base) if "Name" in gas.columns else ""
    gas["PLANT_BASE_CDR"] = gas["CDR Name"].map(plant_base) if "CDR Name" in gas.columns else ""
    return gas

def load_cdr_list(cdr_path: str) -> pd.DataFrame:
    """
    Must contain columns (case-insensitive): UNIT CODE, UNIT NAME, FUEL
    Returns NG-only subset, with normalized helpers.
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
    cdr["CDR_UNIT_NAME_N"] = cdr["CDR_UNIT_NAME"].map(normalize)
    cdr["CDR_PLANT_BASE"] = cdr["CDR_UNIT_NAME"].map(plant_base)
    # NG subset for matching universe
    cdr_ng = cdr[cdr["CDR_FUEL"].isin(["GAS", "NATURAL GAS"])].copy()
    return cdr_ng, cdr  # (ng_only, full_for_veto)

def apply_manual_crosswalk(names: pd.Series, manual_path: str) -> Dict[str, str]:
    try:
        if os.path.exists(manual_path):
            xw = pd.read_csv(manual_path)
            xw = xw.dropna(subset=["Resource Name", "ERCOT_UnitCode"])
            return dict(zip(xw["Resource Name"].astype(str), xw["ERCOT_UnitCode"].astype(str)))
    except Exception as e:
        print(f"⚠️ Manual crosswalk load failed: {e}")
    return {}

# ======== Alias & Corpus Builders (massive recall focus) ========
def generate_unit_aliases(plant_str: str, unit_nums: List[int]) -> List[str]:
    """
    Given a plant base and list of unit numbers, generate robust aliases:
    e.g., 'FOO' + [2] -> ['FOO U2','FOO UNIT 2','FOO CT2','FOO ST 2','FOO BLOCK 2','FOO STG2']
    """
    aliases = []
    base = plant_base(plant_str)
    for n in unit_nums:
        if n is None:
            continue
        nstr = str(n)
        for tpl in [
            "{b} U{n}", "{b} UNIT {n}", "{b} CT{n}", "{b} ST {n}", "{b} ST{n}",
            "{b} BLOCK {n}", "{b} STG{n}", "{b} STG {n}"
        ]:
            aliases.append(tpl.format(b=base, n=nstr))
    return list(dict.fromkeys(aliases))  # unique, preserve order

def build_unit_number_index(cdr_ng: pd.DataFrame) -> Dict[str, List[int]]:
    """Map ERCOT unit code -> list of unit numbers inferred from CDR name."""
    idx = {}
    for _, r in cdr_ng.iterrows():
        code = r["CDR_UNIT_CODE"]
        nums = extract_unit_numbers(r["CDR_UNIT_NAME"])
        if nums:
            idx[code] = nums
    return idx

def build_alias_maps_rich(gas_df: pd.DataFrame, cdr_ng: pd.DataFrame, cdr_full: pd.DataFrame):
    """
    Build two alias maps (gas vs other) with 1:1 enforcement, using:
    - CDR NG unit names
    - Gas fields (Name/CDR Name/INR code/code)
    - Generated aliases from plant base + unit numbers
    """
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))

    # Full CDR split for non-gas veto
    cols = {c.upper().strip(): c for c in cdr_full.columns}
    cdr_full_std = cdr_full.rename(columns={
        cols["UNIT CODE"]: "UNIT_CODE",
        cols["UNIT NAME"]: "UNIT_NAME",
        cols["FUEL"]: "FUEL",
    }).copy()
    cdr_full_std["FUEL"] = cdr_full_std["FUEL"].astype(str).str.strip().str.upper()
    cdr_other = cdr_full_std[~cdr_full_std["FUEL"].isin(["GAS", "NATURAL GAS"])].copy()
    cdr_other["UNIT_CODE"] = cdr_other["UNIT_CODE"].astype(str).str.strip()
    cdr_other["UNIT_NAME"] = cdr_other["UNIT_NAME"].astype(str).str.strip()

    # NG subset intersected with gas units
    cdr_gas = cdr_ng[cdr_ng["CDR_UNIT_CODE"].isin(gas_units)].copy()

    # Precompute plant base and unit numbers for CDR gas units
    unit_num_idx = build_unit_number_index(cdr_gas)
    cdr_gas["PLANT_BASE"] = cdr_gas["CDR_UNIT_NAME"].map(plant_base)

    def collect_aliases_gas() -> Dict[str, set]:
        alias_to_units = {}
        def add(alias_raw, unit):
            if not alias_raw:
                return
            k = normalize(alias_raw)
            if not k:
                return
            alias_to_units.setdefault(k, set()).add(unit)

        # Authoritative CDR names (gas)
        for _, r in cdr_gas.iterrows():
            u = r["CDR_UNIT_CODE"]
            add(r["CDR_UNIT_NAME"], u)

        # Gas dataframe fields
        for _, r in gas_df.iterrows():
            u = str(r.get("ERCOT_UnitCode", "")).strip()
            if not u:
                continue
            add(u, u)
            for f in ["Name", "CDR Name", "ERCOT_INR Code"]:
                if f in gas_df.columns:
                    val = r.get(f)
                    if pd.isna(val):
                        continue
                    add(str(val), u)

        # Generated aliases from plant base + unit numbers
        # First, build mapping unit -> base using whichever we have
        unit_to_base = {}
        # Prefer CDR base if present
        for _, r in cdr_gas.iterrows():
            unit_to_base[r["CDR_UNIT_CODE"]] = r["PLANT_BASE"]
        # Fall back to gas_df bases
        for _, r in gas_df.iterrows():
            u = str(r.get("ERCOT_UnitCode", "")).strip()
            if u and u not in unit_to_base:
                b = r.get("PLANT_BASE_CDR") or r.get("PLANT_BASE") or ""
                unit_to_base[u] = b

        for unit, base in unit_to_base.items():
            if not base:
                continue
            nums = unit_num_idx.get(unit, [])
            # If no unit number inferred, try heuristics: if code ends with digits
            m = re.search(r"(\d+)$", str(unit))
            if not nums and m:
                try:
                    nums = [int(m.group(1))]
                except:
                    pass
            for alias in generate_unit_aliases(base, nums or [1,2,3,4]):  # try a small range if unknown
                add(alias, unit)

        return alias_to_units

    def collect_aliases_other() -> Dict[str, set]:
        alias_to_units = {}
        def add(alias_raw, unit):
            if not alias_raw:
                return
            k = normalize(alias_raw)
            if not k:
                return
            alias_to_units.setdefault(k, set()).add(unit)

        for _, r in cdr_other.iterrows():
            u = r["UNIT_CODE"]
            add(r["UNIT_NAME"], u)
        return alias_to_units

    gas_aliases = collect_aliases_gas()
    other_aliases = collect_aliases_other()

    # Enforce 1:1
    alias_map_gas = {k: list(v)[0] for k, v in gas_aliases.items() if len(v) == 1}
    alias_map_other = {k: list(v)[0] for k, v in other_aliases.items() if len(v) == 1}
    return alias_map_gas, alias_map_other, gas_units

# -------- Fuzzy corpora (gas vs other) --------
def build_fuzzy_corpora(gas_df: pd.DataFrame, cdr_ng: pd.DataFrame, cdr_full: pd.DataFrame):
    gas_units = set(gas_df["ERCOT_UnitCode"].astype(str))
    cdr_gas = cdr_ng[cdr_ng["CDR_UNIT_CODE"].isin(gas_units)].copy()

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
    cdr_ng: pd.DataFrame,       # NG-only
    cdr_full: pd.DataFrame,     # all fuels (for veto)
    manual_map: Dict[str, str],
    high_thresh: int = 90,
    review_thresh: int = 70,
    enable_fuzzy: bool = True,
    other_fuel_veto_thresh: int = 88
):
    alias_map_gas, alias_map_other, gas_units = build_alias_maps_rich(gas_df, cdr_ng, cdr_full)
    exact_id_map = {str(u).upper(): str(u) for u in gas_units}
    corpus_gas, corpus_other = build_fuzzy_corpora(gas_df, cdr_ng, cdr_full) if enable_fuzzy else ([], [])

    out_values, tracker_rows, uncertain_rows = [], [], []

    resource_series = resource_series.astype(str).fillna("")
    for original in resource_series.tolist():
        q_up = str(original).upper()
        q_norm = normalize(original)
        q_base = plant_base(original)
        q_nums = extract_unit_numbers(original)

        # 0) Manual override
        if original in manual_map and manual_map[original]:
            chosen = str(manual_map[original]).strip()
            out_values.append(chosen)
            tracker_rows.append({
                "Resource Name": original, "Matched Name": chosen, "Was Replaced": True,
                "Method": "manual_crosswalk", "Normalized Query": q_norm,
                "Plant Base": q_base, "Derived Unit Nums": json.dumps(q_nums),
                "Best Score": None, "Best Field": "manual", "Best Raw": original,
                "Alternatives (json)": "[]"
            })
            continue

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
            else:
                # 3) plant-base + unit number prioritization (deterministic)
                #    try to find aliases that start with the same plant base and unit num variations
                #    by generating the same alias patterns we used for indexing
                if q_base and q_nums:
                    # generate the same forms and see if any hits exist in alias_map_gas
                    candidates = []
                    for alias in generate_unit_aliases(q_base, q_nums):
                        key = normalize(alias)
                        if key in alias_map_gas:
                            candidates.append(alias_map_gas[key])
                    candidates = list(dict.fromkeys(candidates))
                    if len(candidates) == 1:
                        chosen = candidates[0]
                        method = "plant_base+unitnum_exact"
                    elif len(candidates) > 1:
                        # still ambiguous; leave for fuzzy/review
                        pass

                # 4) (optional) constrained fuzzy with non-gas veto
                if chosen is None and enable_fuzzy and q_norm:
                    # other-fuel exact veto
                    if q_norm in alias_map_other:
                        out_values.append(original)
                        row = {
                            "Resource Name": original, "Matched Name": "",
                            "Was Replaced": False, "Method": "blocked_by_other_fuel_exact",
                            "Normalized Query": q_norm, "Plant Base": q_base,
                            "Derived Unit Nums": json.dumps(q_nums), "Best Score": None,
                            "Best Field": None, "Best Raw": None,
                            "Alternatives (json)": "[]"
                        }
                        tracker_rows.append(row); uncertain_rows.append(row)
                        continue
                    # other-fuel fuzzy veto
                    other_top = best_fuzzy_match(q_norm, corpus_other, k=1)
                    other_best = other_top[0] if other_top else None
                    if other_best and other_best[0] >= other_fuel_veto_thresh:
                        out_values.append(original)
                        row = {
                            "Resource Name": original, "Matched Name": "",
                            "Was Replaced": False, "Method": "blocked_by_other_fuel_fuzzy",
                            "Normalized Query": q_norm, "Plant Base": q_base,
                            "Derived Unit Nums": json.dumps(q_nums), "Best Score": int(other_best[0]),
                            "Best Field": other_best[2], "Best Raw": other_best[3],
                            "Alternatives (json)": "[]"
                        }
                        tracker_rows.append(row); uncertain_rows.append(row)
                        continue

                    # gas fuzzy (not vetoed)
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
                "Plant Base": q_base, "Derived Unit Nums": json.dumps(q_nums),
                "Best Score": aux.get("best_score"), "Best Field": aux.get("best_field"),
                "Best Raw": aux.get("best_raw"), "Alternatives (json)": json.dumps(aux.get("alternatives", []))
            })
        else:
            out_values.append(original)
            row = {
                "Resource Name": original, "Matched Name": "", "Was Replaced": False,
                "Method": "fuzzy_review" if method == "fuzzy_review" else (method or "no_match"),
                "Normalized Query": q_norm, "Plant Base": q_base,
                "Derived Unit Nums": json.dumps(q_nums), "Best Score": aux.get("best_score"),
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
    cdr_ng_only: pd.DataFrame,
    cdr_full: pd.DataFrame,
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
        cdr_ng_only,
        cdr_full,
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

    print("🔄 Loading CDR list...")
    cdr_ng, cdr_full = load_cdr_list(cfg["cdr_file"])
    print(f"   CDR NG rows: {len(cdr_ng)}, Full CDR rows: {len(cdr_full)}")

    print("🔄 Deterministic alias matching (rich) + optional constrained fuzzy with other-fuel veto...")
    update_resource_names(
        agg_path=cfg["agg_file"],
        gas_df=gas,
        cdr_ng_only=cdr_ng,
        cdr_full=cdr_full,
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
