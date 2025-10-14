import os
import re
import json
import pandas as pd
from typing import Dict, Tuple, Optional

# -------- Config / ENV --------
def load_env():
    """Load paths from environment variables (fall back to sensible defaults)."""
    from dotenv import load_dotenv
    load_dotenv()
    return {
        "gas_file": os.getenv("GAS_RESOURCE_FILE", "gas_resources.xlsx"),
        "agg_file": os.getenv("AGGREGATED_MATRIX_FILE", "aggregated_base_point_matrix.csv"),
        "output_file": os.getenv("OUTPUT_FILE", "aggregated_base_point_matrix_RELABELED.csv"),
        "tracker_file": os.getenv("TRACKER_FILE", "resource_name_tracker.csv"),
        # Optional: manual overrides applied FIRST, e.g. two columns: Resource Name,ERCOT_UnitCode
        "manual_crosswalk_file": os.getenv("MANUAL_CROSSWALK_FILE", "manual_crosswalk.csv"),
        # Optional: where to dump review/ borderline fuzzy candidates
        "review_file": os.getenv("REVIEW_FILE", "resource_name_review_candidates.csv"),
        # Fuzzy thresholds
        "high_thresh": int(os.getenv("FUZZY_HIGH_THRESH", "90")),
        "review_thresh": int(os.getenv("FUZZY_REVIEW_THRESH", "70")),
    }

# -------- Normalization / Fuzzy --------
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
    s = re.sub(r"[^A-Z0-9\s]", " ", s)  # strip punctuation
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
    return int(difflib.SequenceMatcher(a=a, b=b).ratio() * 100)

# -------- Gas corpus builders --------
GAS_FIELDS = ["ERCOT_UnitCode", "Name", "CDR Name", "ERCOT_INR Code"]

def load_gas_resources(excel_path: str) -> pd.DataFrame:
    """Load gas resources (subset of ERCOT units you want to map to)."""
    gas = pd.read_excel(excel_path, sheet_name=0)
    # Ensure required columns present
    missing = [c for c in ["ERCOT_UnitCode"] if c not in gas.columns]
    if missing:
        raise ValueError(f"gas_resources missing required columns: {missing}")
    return gas

def build_gas_corpus(gas_df: pd.DataFrame):
    """
    Returns a list of tuples:
    (ercot_unit, source_field, raw_value, normalized_value)
    """
    corpus = []
    for _, r in gas_df.iterrows():
        unit = str(r.get("ERCOT_UnitCode", "")).strip()
        if not unit:
            continue
        for f in GAS_FIELDS:
            if f not in gas_df.columns:
                continue
            val = r.get(f)
            if pd.isna(val):
                continue
            raw = str(val).strip()
            norm = normalize(raw)
            if norm:
                corpus.append((unit, f, raw, norm))
    return corpus

def build_fast_indexes(gas_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, set]]:
    """
    exact_id_map: exact (upper) ERCOT_UnitCode -> ERCOT_UnitCode
    normalized_to_units: normalized string -> set of ERCOT_UnitCodes
    """
    exact_id_map = {}
    normalized_to_units = {}

    for _, r in gas_df.iterrows():
        unit = str(r.get("ERCOT_UnitCode", "")).strip()
        if not unit:
            continue
        exact_id_map[unit.upper()] = unit
        for f in GAS_FIELDS:
            if f not in gas_df.columns:
                continue
            val = r.get(f)
            if pd.isna(val):
                continue
            norm = normalize(str(val))
            if not norm:
                continue
            normalized_to_units.setdefault(norm, set()).add(unit)

    return exact_id_map, normalized_to_units

def best_fuzzy_match(q_norm: str, corpus, k=5):
    if not q_norm:
        return []
    scored = []
    for unit, field, raw, norm in corpus:
        sc = token_score(q_norm, norm)
        scored.append((sc, unit, field, raw))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]

# -------- Mapping logic --------
def apply_manual_crosswalk(names: pd.Series, manual_path: str) -> Dict[str, str]:
    """
    Returns a dict mapping original Resource Name -> ERCOT_UnitCode based on manual CSV.
    CSV format: Resource Name,ERCOT_UnitCode
    """
    try:
        if os.path.exists(manual_path):
            xw = pd.read_csv(manual_path)
            xw = xw.dropna(subset=["Resource Name", "ERCOT_UnitCode"])
            # Use the *raw* Resource Name as key (no normalization) for deterministic overrides.
            return dict(zip(xw["Resource Name"].astype(str), xw["ERCOT_UnitCode"].astype(str)))
    except Exception as e:
        print(f"⚠️ Manual crosswalk load failed: {e}")
    return {}

def map_resource_names(
    resource_series: pd.Series,
    gas_df: pd.DataFrame,
    manual_map: Dict[str, str],
    high_thresh: int = 90,
    review_thresh: int = 70
):
    """
    Only relabel rows that map to gas resources (subset); otherwise leave untouched.
    Returns:
      - new_values: pd.Series with replacements applied where confident
      - tracker_df: audit log
      - review_df: borderline/unknown cases for manual review
    """
    corpus = build_gas_corpus(gas_df)
    exact_id_map, norm_index = build_fast_indexes(gas_df)

    out_values = []
    tracker_rows = []
    review_rows = []

    for original in resource_series.astype(str).tolist():
        # 0) Manual crosswalk first (explicit overrides)
        if original in manual_map and manual_map[original]:
            chosen = str(manual_map[original]).strip()
            out_values.append(chosen)
            tracker_rows.append({
                "Resource Name": original,
                "Matched Name": chosen,
                "Was Replaced": True,
                "Method": "manual_crosswalk",
                "Normalized Query": normalize(original),
                "Best Score": None,
                "Best Field": "manual",
                "Best Raw": original,
                "Alternatives (json)": "[]"
            })
            continue

        original_up = original.upper()
        norm_q = normalize(original)

        chosen: Optional[str] = None
        method = None
        aux = {}

        # 1) If original is already an ERCOT_UnitCode for a gas unit (subset), keep it as-is
        if original_up in exact_id_map:
            chosen = exact_id_map[original_up]
            method = "exact_id"  # subset by definition
        else:
            # 2) normalized exact -> unique gas unit
            if norm_q in norm_index and len(norm_index[norm_q]) == 1:
                chosen = next(iter(norm_index[norm_q]))
                method = "normalized_exact"
            else:
                # 3) fuzzy match against GAS CORPUS ONLY (subset)
                if norm_q:
                    top = best_fuzzy_match(norm_q, corpus, k=5)
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

        # Output + tracking
        if chosen:
            out_values.append(chosen)
            tracker_rows.append({
                "Resource Name": original,
                "Matched Name": chosen,
                "Was Replaced": True,
                "Method": method,
                "Normalized Query": norm_q,
                "Best Score": aux.get("best_score", None),
                "Best Field": aux.get("best_field", None),
                "Best Raw": aux.get("best_raw", None),
                "Alternatives (json)": json.dumps(aux.get("alternatives", []))
            })
        else:
            # Not a confident gas match: leave as original and, if fuzzy_review/no_match, queue to review
            out_values.append(original)
            row = {
                "Resource Name": original,
                "Matched Name": "",
                "Was Replaced": False,
                "Method": "fuzzy_review" if method == "fuzzy_review" else "no_match",
                "Normalized Query": norm_q,
                "Best Score": aux.get("best_score", None),
                "Best Field": aux.get("best_field", None),
                "Best Raw": aux.get("best_raw", None),
                "Alternatives (json)": json.dumps(aux.get("alternatives", []))
            }
            tracker_rows.append(row)
            if row["Method"] in ("fuzzy_review", "no_match"):
                review_rows.append(row)

    return pd.Series(out_values), pd.DataFrame(tracker_rows), pd.DataFrame(review_rows)

# -------- I/O glue --------
def update_resource_names(
    agg_path: str,
    gas_df: pd.DataFrame,
    output_path: str,
    tracker_path: str,
    manual_crosswalk_path: str,
    review_path: str,
    high_thresh: int,
    review_thresh: int
):
    df = pd.read_csv(agg_path)
    if "Resource Name" not in df.columns:
        raise ValueError("aggregated file missing 'Resource Name' column")

    manual_map = apply_manual_crosswalk(df["Resource Name"], manual_crosswalk_path)

    new_values, tracker_df, review_df = map_resource_names(
        df["Resource Name"], gas_df, manual_map,
        high_thresh=high_thresh, review_thresh=review_thresh
    )

    # Replace ONLY where a gas mapping was found
    df["Resource Name"] = new_values

    # Write outputs
    df.to_csv(output_path, index=False)
    tracker_df.to_csv(tracker_path, index=False)
    if not review_df.empty:
        review_df.to_csv(review_path, index=False)

    print(f"✅ Updated aggregated file: {output_path}")
    print(f"📊 Tracker file: {tracker_path}")
    if not review_df.empty:
        print(f"📝 Review candidates: {review_path} ({len(review_df)} rows)")
    else:
        print("📝 No review candidates — all confident or unchanged.")

# -------- Main --------
def main():
    cfg = load_env()
    print("🔄 Loading gas resources (subset)...")
    gas = load_gas_resources(cfg["gas_file"])
    print(f"   Gas subset count: {len(gas)}")

    print("🔄 Relabeling aggregated matrix where confidently matched to gas subset...")
    update_resource_names(
        agg_path=cfg["agg_file"],
        gas_df=gas,
        output_path=cfg["output_file"],
        tracker_path=cfg["tracker_file"],
        manual_crosswalk_path=cfg["manual_crosswalk_file"],
        review_path=cfg["review_file"],
        high_thresh=cfg["high_thresh"],
        review_thresh=cfg["review_thresh"],
    )

if __name__ == "__main__":
    main()
