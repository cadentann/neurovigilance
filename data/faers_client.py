"""
data/faers_client.py — openFDA FAERS API client.

Key fixes vs. v8:
  1. DEDUPLICATION: Every report is keyed by (primaryid, caseversion).
     Only the most-recent version of each case is retained — eliminates
     15–25% duplicate report contamination common in raw FAERS data.
  2. AGE UNIT: patientonsetageunit field is now respected (decade / year /
     month / week / day / hour) instead of the heuristic "divide by 365 if
     age > 130" which misclassified infant reports as adults.
  3. SERIOUSNESS: seriousness=1 means Serious; absence/0 means Non-serious.
     There is no value "2" — the previous map silently dropped most
     Non-serious reports to "Unknown", breaking the seriousness filter.
  4. primaryid is preserved throughout the pipeline so that td/to denominators
     in stats/prr.py can be computed over UNIQUE REPORTS, not exploded rows.
  5. quarter is derived from receivedate for temporal analysis.

Architecture note
-----------------
This client uses the openFDA REST API which has a hard skip limit of 25,000
and rate limits (240 req/min without API key; 1,000 with).  For more than
~5,000 reports per drug, switch to the FAERS quarterly bulk ASCII downloads:
  https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
and process with DuckDB (see README_bulk.md for instructions).
"""

from __future__ import annotations

import time
import streamlit as st
import pandas as pd
import requests

from config import (
    FDA_BASE_URL,
    FDA_PAGE_SIZE,
    TARGET_REPORTS_PER_DRUG,
    FAERS_CACHE_TTL,
    AGE_UNIT_TO_YEARS,
    SERIOUSNESS_MAP,
    DRUG_SYNONYMS,
)


def _get_api_key() -> str:
    """Load FDA API key from Streamlit secrets (optional but raises rate limits 4×)."""
    try:
        return st.secrets.get("FDA_API_KEY", "")
    except Exception:
        return ""


def _parse_report(rep: dict, drug: str) -> dict:
    """
    Parse a single FAERS JSON report into a flat record.

    Returns a dict with all fields needed for both report-level and
    reaction-level analysis. primaryid is the canonical case identifier.
    """
    p = rep.get("patient", {})

    # ── Reactions (list of MedDRA PTs) ───────────────────────────────────────
    # Word-level capitalize: str.split() on whitespace, then capitalize each word.
    # This correctly handles apostrophes ("ALZHEIMER'S" → "Alzheimer's") unlike
    # str.title() which capitalises after any non-letter including apostrophes.
    # For hyphenated MedDRA PTs ("DRUG-INDUCED LIVER INJURY"), split on whitespace
    # gives ["DRUG-INDUCED", ...] → capitalize gives "Drug-induced" (lowercase 'i').
    # Apply a second pass to capitalise after hyphens too.
    import re as _re3
    def _meddra_case(s: str) -> str:
        words = s.strip().split()
        capped = " ".join(w.capitalize() for w in words)
        # Also capitalise first letter after each hyphen
        return _re3.sub(r"-([a-z])", lambda m: "-" + m.group(1).upper(), capped)

    reactions = [
        _meddra_case(rx.get("reactionmeddrapt", ""))
        for rx in p.get("reaction", [])
        if rx.get("reactionmeddrapt")
    ]

    # ── Sex ───────────────────────────────────────────────────────────────────
    sx  = p.get("patientsex")
    sex = {1: "Male", "1": "Male", 2: "Female", "2": "Female"}.get(sx, "Unknown")

    # ── Age — FIX: respect patientonsetageunit field ──────────────────────────
    age_raw  = pd.to_numeric(p.get("patientonsetage"), errors="coerce")
    age_unit = str(p.get("patientonsetageunit", "801"))  # default: years
    factor   = AGE_UNIT_TO_YEARS.get(age_unit, 1.0)
    age      = float(age_raw * factor) if pd.notna(age_raw) else None

    # Sanity-clip: valid human age range 0–130 years
    if age is not None and (age < 0 or age > 130):
        age = None

    # ── Seriousness — FIX: field is boolean 1/absent, not 1/2 binary ─────────
    sr_raw  = rep.get("seriousness", 0)
    serious = SERIOUSNESS_MAP.get(str(sr_raw), SERIOUSNESS_MAP.get(sr_raw, "Non-serious"))

    # ── Date / quarter ────────────────────────────────────────────────────────
    ds      = rep.get("receivedate", "")
    year    = None
    quarter = None
    try:
        if len(ds) >= 4:
            year = int(ds[:4])
        if year and len(ds) >= 6:
            month   = int(ds[4:6])
            quarter = f"{year}-Q{(month - 1) // 3 + 1}"
    except (ValueError, TypeError):
        pass

    # ── Case identifiers for deduplication ────────────────────────────────────
    # In openFDA JSON, 'safetyreportid' is the 8-digit FDA case identifier.
    # This is the same field as 'primaryid' in the FAERS ASCII bulk files —
    # different naming conventions, same data. The variable is named 'primaryid'
    # throughout this codebase to match FAERS ASCII terminology and the bulk README.
    primaryid   = rep.get("safetyreportid", "")
    caseversion = int(pd.to_numeric(rep.get("safetyreportversion", 1), errors="coerce") or 1)

    return {
        "primaryid":   str(primaryid),
        "caseversion": caseversion,
        "drug":        drug,
        "sex":         sex,
        "age":         age,
        "serious":     serious,
        "year":        year,
        "quarter":     quarter,
        "reactions":   reactions,
    }


@st.cache_data(show_spinner=False, ttl=FAERS_CACHE_TTL)
def fetch_drug_reports(
    drug: str,
    target: int = TARGET_REPORTS_PER_DRUG,
    synonyms: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch up to `target` FAERS reports for a given drug via the openFDA API.

    Parameters
    ----------
    drug     : Canonical drug name (used as the primaryid tag in the DataFrame)
    target   : Maximum reports to fetch (hard-capped at openFDA skip ceiling)
    synonyms : Optional list of name variants for the search OR query.
               Falls back to DRUG_SYNONYMS[drug] then [drug.lower()] if None.

    Returns a raw DataFrame (deduplication happens in build_report_and_rxn_dfs).
    """
    api_key    = _get_api_key()
    rows: list = []
    target     = min(target, 25000)   # Hard openFDA skip ceiling
    name_terms = synonyms or DRUG_SYNONYMS.get(drug, [drug.lower()])

    # Pre-compute name matching function once outside the pagination loop.
    # _name_matches was previously redefined inside the loop body on every page
    # fetch (up to 250 iterations), creating a new closure object each time.
    # name_terms_lower and the regex split are invariant across pages.
    import re as _re
    name_terms_lower = {t.lower() for t in name_terms}

    def _name_matches(reported: str) -> bool:
        """True iff any query synonym is a subset of the reported name's word tokens.
        Note: combination products (e.g. METFORMIN/SITAGLIPTIN) will match single-drug
        queries (e.g. 'metformin') because the target token is present. This conservatively
        inflates td (underestimates PRR). The magnitude depends on combo-product reporting
        rates in FAERS for each drug, typically a few percent of entries.
        """
        rep_tokens = set(_re.split(r"[\s\-/]+", reported.lower()))
        for nt in name_terms_lower:
            nt_tokens = set(_re.split(r"[\s\-/]+", nt))
            if nt_tokens and nt_tokens.issubset(rep_tokens):
                return True
        return False

    _pagination_exhausted = False
    for skip in range(0, target, FDA_PAGE_SIZE):
        if _pagination_exhausted:
            break   # Data exhausted on a prior page — stop outer loop
        params: dict = {
            # Build OR query across all known synonyms, brand names, and salt forms.
            # Searching only the INN misses e.g. "GALANTAMINE HBR", "RAZADYNE", etc.,
            # systematically underestimating td (denominator) → inflated PRR.
            # drugcharacterization=1 = Primary Suspect only (consistent with bulk path).
            "search": (
                "patient.drug.medicinalproduct:("
                + " ".join(f'"{s}"' for s in name_terms)
                + ")"
                # Standard Lucene boolean AND (not +AND+ which encodes as %2B).
                + " AND patient.drug.drugcharacterization:\"1\""
            ),
            "limit":  FDA_PAGE_SIZE,
            "skip":   skip,
        }
        if api_key:
            params["api_key"] = api_key

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                r = requests.get(FDA_BASE_URL, params=params, timeout=15)
                if r.status_code == 404:
                    break
                if r.status_code == 429:
                    try:
                        import streamlit as _st
                        _st.toast("⚠️ FDA API rate limit hit — waiting 15s…", icon="⏳")
                    except Exception:
                        pass
                    time.sleep(15)
                    continue  # retry this page
                r.raise_for_status()
                results = r.json().get("results", [])
                if not results:
                    # Data exhausted: break BOTH the inner retry loop and the
                    # outer pagination loop. Without this, the outer loop
                    # continues advancing skip to target, making up to 91 wasted
                    # API calls per run (13 drugs × 7 empty pages each), pushing
                    # close to the 240-req/min rate limit and causing artificial
                    # 15-second delays.
                    _pagination_exhausted = True
                    break
                # Post-parse primary-suspect filter: verify the target drug
                # appears as a primary suspect in this specific report's drug array.
                # (_name_matches is defined before the pagination loop.)
                for rep in results:
                    drugs_in_rep = rep.get("patient", {}).get("drug", [])
                    if any(
                        str(d.get("drugcharacterization", "0")) == "1"
                        and _name_matches(d.get("medicinalproduct", ""))
                        for d in drugs_in_rep
                    ):
                        rows.append(_parse_report(rep, drug))
                time.sleep(0.30)   # 240 req/min limit → minimum 0.25s; 0.30s provides margin
                break  # success — exit retry loop
            except requests.exceptions.Timeout:
                if attempt < max_attempts - 1:
                    time.sleep(3 * (attempt + 1))  # back-off: 3s, 6s
                    continue
                # All retries exhausted for this page — skip it
                break
            except requests.exceptions.HTTPError:
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.RequestException):
                break
            except (ValueError, KeyError):
                break
        else:
            # 429 retry loop exhausted — stop pagination
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_report_and_rxn_dfs(
    raw_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From raw fetched data, produce two canonical DataFrames:

    report_df — one row per UNIQUE FAERS case (deduplicated by primaryid,
                keeping the highest caseversion). Used for correct td/to
                denominator computation.

    rxn_df    — report_df exploded on reactions (one row per report × reaction
                pair). Used for per-reaction signal counting.

    Deduplication is done here (not in fetch) so caching fetch_drug_reports()
    preserves the raw data while allowing re-deduplication parameters to vary.
    """
    if not raw_frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(raw_frames, ignore_index=True)

    if combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Deduplicate: keep highest caseversion per (primaryid, drug) ──────────
    # IMPORTANT: Dedup on (primaryid, drug) not just primaryid.
    # A FAERS case can appear in multiple drug fetches when multiple drugs
    # are co-primary suspects. Global dedup on primaryid alone would collapse
    # both drug rows into one, silently losing one drug's td denominator.
    # Deduplication on (primaryid, drug) preserves each drug's attribution
    # while still eliminating duplicate follow-up submissions per drug.
    combined  = combined.sort_values("caseversion", ascending=False)
    report_df = (
        combined
        .drop_duplicates(subset=["primaryid", "drug"], keep="first")
        .reset_index(drop=True)
    )

    # ── Explode reactions → rxn_df ────────────────────────────────────────────
    rxn_df = (
        report_df
        .explode("reactions")
        .rename(columns={"reactions": "reaction"})
        .pipe(lambda d: d[d["reaction"].notna() & (d["reaction"] != "")])
        .reset_index(drop=True)
    )

    return report_df, rxn_df
