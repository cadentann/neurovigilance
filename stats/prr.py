"""
stats/prr.py — Proportional Reporting Ratio and full disproportionality pipeline.

Key fixes applied across versions:
  1. td / to count UNIQUE REPORTS (via primaryid), not exploded reaction-rows.
  2. Reaction counts a, c use groupby().nunique() — vectorised, not O(R²) scans.
  3. Composite score uses ABSOLUTE reference maxima (config.COMPOSITE_REF_MAX).
  4. EBGM priors are fit to data via stats.ebgm.fit_priors, not hardcoded.
  5. is_labeled() requires the FULL MedDRA PT phrase verbatim.
  6. All early-exit paths return (DataFrame, bool) tuple — no bare DataFrame returns.
  7. weber_flag uses primaryid.nunique() for count consistency.
  8. TIER_THRESH: (0, "NONE") removed; fall-through return handles PRR < 2.

References
----------
Evans SJW et al. (2001). Pharmacoepidemiol Drug Saf 10:483–486.
DuMouchel W (1999). The American Statistician 53(3):177–190.
Benjamini Y, Hochberg Y (1995). J Royal Stat Soc B 57(1):289–300.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import false_discovery_control
from scipy.stats import fisher_exact
from scipy.stats import chi2 as _chi2dist

from config import (
    SIGNAL_GROUP_MAP, TIER_THRESH, APPROVAL_YEARS,
    COMPOSITE_REF_MAX, COMPOSITE_WEIGHTS,
)
from stats.bcpnn import bcpnn_ic
from stats.ebgm  import ebgm_row, fit_priors, Priors


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_tier(prr: float) -> str:
    """
    Map PRR to a signal tier using TIER_THRESH thresholds.
    PRR < 2 (below FDA signal threshold) → "NONE".
    """
    for threshold, label in TIER_THRESH:
        if prr >= threshold:
            return label
    return "NONE"   # PRR in [0, 2) — below FDA threshold


def is_labeled(rxn: str, label_text: str) -> bool:
    """
    Return True iff the MedDRA PT (case-insensitive) appears as a whole phrase
    in label_text, after normalising British→American medical spelling.

    Uses word-boundary regex (\\b) to prevent false positives from substring
    matches: "Pain" must not match "painless"; "Mania" must not match "Romania".
    Full-phrase matching prevents first-word false positives (e.g. "orthostatic"
    in dosing guidance wrongly matching "Orthostatic Hypotension").

    Most MedDRA PTs are multi-word, making the word-boundary guard critical only
    for single-word PTs (Pain, Rash, Mania, Death, Anxiety, etc.), but it is
    applied uniformly for consistency and robustness.
    """
    if not label_text:
        return True   # No label available → conservatively mark as labeled

    _BR_TO_US: list[tuple[str, str]] = [
        ("diarrhoea",       "diarrhea"),
        ("haemorrhage",     "hemorrhage"),
        ("haemorrhagic",    "hemorrhagic"),
        ("haemoglobin",     "hemoglobin"),
        ("haematuria",      "hematuria"),
        ("haematoma",       "hematoma"),
        ("oedema",          "edema"),
        ("oedematous",      "edematous"),
        ("hypokalaemia",    "hypokalemia"),
        ("hyperkalaemia",   "hyperkalemia"),
        ("hypocalcaemia",   "hypocalcemia"),
        ("hypercalcaemia",  "hypercalcemia"),
        ("hyponatraemia",   "hyponatremia"),
        ("hypernatraemia",  "hypernatremia"),
        ("anaemia",         "anemia"),
        ("anaesthetic",     "anesthetic"),
        ("leukaemia",       "leukemia"),
        ("foetal",          "fetal"),
        ("foetus",          "fetus"),
        ("tumour",          "tumor"),
        ("colour",          "color"),
        ("labelled",        "labeled"),
        ("behaviour",       "behavior"),
        ("paediatric",      "pediatric"),
        # Additional -(a)e- → -(e)- substitutions common in MedDRA PTs
        ("paraesthesia",    "paresthesia"),
        ("dysaesthesia",    "dysesthesia"),
        ("anaesthesia",     "anesthesia"),
        ("hypoalbuminaemia","hypoalbuminemia"),
        ("hyperbilirubinaemia", "hyperbilirubinemia"),
        ("hypoxaemia",      "hypoxemia"),
        ("haemolysis",      "hemolysis"),
        ("haemolytic",      "hemolytic"),
        ("oesophageal",     "esophageal"),
        ("oesophagitis",    "esophagitis"),
        ("aetiology",       "etiology"),
        ("gynaecological",  "gynecological"),
    ]

    rxn_lower = rxn.lower()
    for brit, us in _BR_TO_US:
        rxn_lower = rxn_lower.replace(brit, us)

    # MedDRA PTs often use formal/verbose terminology while FDA labels use common
    # clinical abbreviations. These aliases handle the most frequent mismatches
    # that cause false-novel classifications for clearly-labeled reactions.
    _MEDDRA_TO_FDA: dict[str, list[str]] = {
        "electrocardiogram qt prolonged": ["qt prolongation", "qtc prolongation",
                                            "prolonged qt", "qt interval prolongation"],
        "torsade de pointes":             ["torsades de pointes"],
        "bundle branch block left":       ["left bundle branch block", "lbbb"],
        "bundle branch block right":      ["right bundle branch block", "rbbb"],
        "alanine aminotransferase increased": ["alt elevation", "elevated alt",
                                               "alanine aminotransferase elevation"],
        "aspartate aminotransferase increased": ["ast elevation", "elevated ast"],
        # "blood pressure increased" → "hypertension" removed: these are distinct
        # MedDRA PTs. "Blood Pressure Increased" is a measurement finding;
        # "Hypertension" is a chronic condition diagnosis. Aliasing them causes
        # false "labeled" classifications for drugs with isolated pressor responses.
        "blood glucose increased":        ["hyperglycemia", "elevated blood glucose"],
        "weight decreased":               ["weight loss"],
        "weight increased":               ["weight gain"],
        # "sinus tachycardia" → "tachycardia" alias INTENTIONALLY OMITTED.
        # "Tachycardia" in an FDA label may refer to SVT, VT, reflex tachycardia,
        # or other non-sinus mechanisms — it does not imply sinus node origin.
        # Including this alias would cause any label mention of "tachycardia"
        # to suppress the sinus-specific MedDRA PT, conflating distinct cardiac
        # pathophysiologies. The alias was pharmacologically over-broad.
    }

    # Try the normalised MedDRA term first
    pattern = r'\b' + re.escape(rxn_lower) + r'\b'
    if re.search(pattern, label_text.lower()):
        return True

    # Then try any known FDA-label aliases for this MedDRA PT
    for alias in _MEDDRA_TO_FDA.get(rxn_lower, []):
        if re.search(r'\b' + re.escape(alias) + r'\b', label_text.lower()):
            return True

    return False


def weber_flag(
    rxn_df: pd.DataFrame,
    drug: str,
    rxn: str,
    min_n: int = 8,
    approval_years: dict[str, int] | None = None,
) -> tuple[bool, float | None]:
    """
    Weber effect flag: True if ≥ 60% of *unique* reports fall within the
    first 3 years after the drug's FDA approval date.

    Uses primaryid.nunique() for report counts — consistent with compute_prr
    and resistant to any residual duplication in rxn_df.

    Parameters
    ----------
    rxn_df        : Exploded reaction dataframe with primaryid, drug, reaction, year
    drug          : Target drug name
    rxn           : Target reaction (MedDRA PT)
    min_n         : Minimum unique reports required to compute (default 8)
    approval_years: Drug → approval year. Falls back to config.APPROVAL_YEARS.

    Returns
    -------
    (flagged: bool, pct: float | None)
    """
    years_map = approval_years or APPROVAL_YEARS
    approval  = years_map.get(drug)

    sub = (
        rxn_df[(rxn_df["drug"] == drug) & (rxn_df["reaction"] == rxn)]
        .dropna(subset=["year"])
    )

    total = sub["primaryid"].nunique()

    if total < min_n or approval is None:
        return False, None

    # Lower bound year >= approval required: FAERS contains pre-approval entries
    # (IND compassionate-use, foreign authorisations). Without the lower bound,
    # year <= approval+2 matches ALL pre-approval years, falsely counting them
    # as early post-marketing reports and triggering spurious Weber flags.
    # Example: 5 pre-approval + 5 post-2003 reports → 50% (false flag) vs 0% (correct).
    early = sub[
        (sub["year"] >= approval) & (sub["year"] <= approval + 2)
    ]["primaryid"].nunique()
    pct   = round(early / total * 100, 1)
    return pct >= 60.0, pct


# ── Rolling PRR ────────────────────────────────────────────────────────────────

def rolling_prr(
    rxn_df: pd.DataFrame,
    report_df: pd.DataFrame,
    drug: str,
    reactions: list[str],
    window: int = 4,
) -> pd.DataFrame:
    """
    Rolling-window PRR over calendar quarters.
    Uses unique-report denominators (td, to) per window.
    """
    dq  = rxn_df[rxn_df["drug"] == drug].copy()
    oq  = rxn_df[rxn_df["drug"] != drug].copy()
    dr  = report_df[report_df["drug"] == drug].copy()
    or_ = report_df[report_df["drug"] != drug].copy()

    quarters = sorted(dq["quarter"].dropna().unique())
    rows = []

    for i, q in enumerate(quarters):
        wqs = quarters[max(0, i - window + 1): i + 1]

        # Drug-reports in this window
        drug_ids_w = set(dr[dr["quarter"].isin(wqs)]["primaryid"])
        td = len(drug_ids_w)

        # Background: exclude any report that also contains the target drug
        # (polypharmacy patients), matching the compute_prr() denominator logic.
        # Without this, shared patients inflate 'to', producing systematically
        # lower rolling PRRs than the static analysis for the same reaction.
        bg_ids_w = set(or_[or_["quarter"].isin(wqs)]["primaryid"]) - drug_ids_w
        to = len(bg_ids_w)

        if td < 5 or to < 5:
            continue

        dw = dq[dq["primaryid"].isin(drug_ids_w)]
        ow = oq[oq["primaryid"].isin(bg_ids_w)]

        # Vectorised reaction counts — O(R) groupby, not O(R×N) per-reaction scans
        a_counts = dw.groupby("reaction")["primaryid"].nunique()
        c_counts = ow.groupby("reaction")["primaryid"].nunique()

        for rxn in reactions:
            a = int(a_counts.get(rxn, 0))
            c = int(c_counts.get(rxn, 0))
            if c < 1:
                # c=0 in this window: reaction appears only in target drug.
                # PRR is undefined. Absence from temporal chart does NOT mean
                # absence of signal — it may indicate a drug-specific novel reaction.
                continue
            # Use raw (uncorrected) point estimate to match compute_prr().
            # Previously used Haldane-corrected cells in the point estimate,
            # producing systematically lower PRRs (up to 8% at a=20,td=50)
            # than the forest-plot / signal-table values — an internal
            # inconsistency that made the temporal chart incomparable to the
            # main results table. Haldane correction belongs in CI calculation only.
            prr = (a / td) / (c / to) if c > 0 and to > 0 else float("nan")
            rows.append({"Quarter": q, "Reaction": rxn, "PRR": round(prr, 2)})

    return pd.DataFrame(rows)


# ── Main PRR computation ───────────────────────────────────────────────────────

def compute_prr(
    rxn_df: pd.DataFrame,
    report_df: pd.DataFrame,
    drug: str,
    confounders: set[str],
    serious_filter: str = "All",
    label_text: str = "",
    ebgm_priors: Priors | None = None,
    fit_ebgm: bool = True,
    min_n: int = 3,
) -> tuple[pd.DataFrame, bool]:
    """
    Compute full disproportionality table for a target drug vs. background.

    Parameters
    ----------
    rxn_df        : Exploded DataFrame (one row per report × reaction pair).
    report_df     : De-duplicated report-level DataFrame (one row per primaryid).
    drug          : Target drug name
    confounders   : MedDRA PTs to flag as confounders
    serious_filter: "All" | "Serious" | "Non-serious"
    label_text    : Full-text FDA label for novelty detection
    ebgm_priors   : Pre-fit priors; re-fit from data if None and fit_ebgm=True
    fit_ebgm      : Fit EBGM priors from this corpus (recommended)
    min_n         : Minimum case count for display. Rows with n < min_n are
                    hidden but still included in BH-FDR correction, which is
                    applied over the full a>=3 hypothesis family BEFORE filtering.
                    Applying BH only over the displayed subset inflates FDR above
                    the nominal level (fewer hypotheses → weaker penalty per rank).

    Returns
    -------
    (DataFrame sorted by Composite descending, ebgm_fit_ok: bool)
    """
    if rxn_df.empty or report_df.empty:
        return pd.DataFrame(), False

    ebgm_fit_ok = True

    # ── Seriousness filter ────────────────────────────────────────────────────
    if serious_filter != "All":
        valid_ids = set(report_df[report_df["serious"] == serious_filter]["primaryid"])
        report_df = report_df[report_df["primaryid"].isin(valid_ids)]
        rxn_df    = rxn_df[rxn_df["primaryid"].isin(valid_ids)]

    # ── Unique report denominators ────────────────────────────────────────────
    # Per EMA/WHO-UMC methodology: the background for drug X must EXCLUDE reports
    # that also contain drug X as a primary suspect.
    # After the (primaryid, drug) dedup fix, a patient on both Galantamine and
    # Metformin has two rows — the same primaryid appears in both drug_ids and
    # bg_ids. Including it in bg inflates 'to' with reports that are also in 'td'.
    # Correct: bg_ids = {reports containing background drugs} MINUS {reports
    # that also contain the target drug}.
    drug_ids = set(report_df[report_df["drug"] == drug]["primaryid"])
    bg_ids   = set(report_df[report_df["drug"] != drug]["primaryid"]) - drug_ids
    td = len(drug_ids)
    to = len(bg_ids)

    if td == 0 or to == 0:
        return pd.DataFrame(), False   # ← was bare pd.DataFrame() — crash bug fixed

    ddf = rxn_df[rxn_df["primaryid"].isin(drug_ids)]
    odf = rxn_df[rxn_df["primaryid"].isin(bg_ids)]

    # ── Vectorised reaction counts (O(R) groupby, not O(R²) per-row scans) ───
    a_counts = ddf.groupby("reaction")["primaryid"].nunique()
    c_counts = odf.groupby("reaction")["primaryid"].nunique()

    # ── Fit EBGM priors from this corpus ─────────────────────────────────────
    if fit_ebgm and ebgm_priors is None:
        common_rxns = a_counts.index.intersection(
            c_counts.index[c_counts > 0]
        )
        n_cells  = a_counts[common_rxns].tolist()
        mu_cells = (td * c_counts[common_rxns] / to).tolist()
        ebgm_priors, ebgm_fit_ok = fit_priors(n_cells, mu_cells)

    # ── Per-reaction computation ──────────────────────────────────────────────
    rows = []
    _c0_dropped: list[str] = []   # reactions with a>=3 but c==0 (PRR undefined)
    for rxn, a in a_counts.items():
        a = int(a)
        c = int(c_counts.get(rxn, 0))

        # Minimum drug-cell count: Evans (2001) criterion requires n ≥ 3 in the
        # target drug × reaction cell (a). No published guideline (Evans, EMA/CHMP,
        # WHO-UMC, DuMouchel GPS) requires c ≥ 3 in the background cell. The
        # previous `c < 3` filter silently dropped valid signals for rare but
        # drug-specific reactions (e.g. a=5, c=2 → PRR could be high), with no
        # visibility to the user. Removed: c=0 is handled by the PRR guard below.
        if a < 3:
            continue
        if c == 0:
            # Cannot compute PRR with zero background counts — formula undefined.
            # Track these for UI disclosure (high a, c=0 may represent drug-specific signals).
            _c0_dropped.append(rxn)
            continue

        b = td - a
        d = to - c

        ah, bh, ch, dh = a + 0.5, b + 0.5, c + 0.5, d + 0.5

        # Point estimates use raw counts (standard Evans/EMA formula).
        # Haldane correction belongs in the SE/CI derivation, not the
        # point estimate — applying it there introduces a systematic
        # downward bias of ~4.5% at n=10, ~1% at n=100.
        prr = (a / td) / (c / to) if c > 0 and to > 0 else float("nan")
        ror = (a * (to - c)) / (b * c) if b > 0 and c > 0 else float("nan")

        # ROR 95% CI (log-normal, delta method — uses Haldane-corrected cells
        # for SE stability when any raw cell approaches zero)
        se_ror    = np.sqrt(1/ah + 1/bh + 1/ch + 1/dh)
        ror_cilo  = float(np.exp(np.log(max(ror, 1e-9)) - 1.96 * se_ror))
        ror_cihi  = float(np.exp(np.log(max(ror, 1e-9)) + 1.96 * se_ror))

        E  = td * (c / to)
        # OE (observed/expected) is algebraically identical to PRR:
        # OE = a/E = a/(td·c/to) = (a/td)/(c/to) = PRR
        # It is not stored as a separate column to avoid implying distinct info.

        ic, ic025, ic975 = bcpnn_ic(a, b, c, d)
        em, e05          = ebgm_row(a, max(E, 1e-6), priors=ebgm_priors)

        _test_used = "chi2"  # default; overwritten inside try block
        try:
            # EMA/WHO-UMC guidance: Pearson χ² without Yates correction.
            # However, χ² requires min(expected) ≥ 1 (Cochran's rule) for the
            # asymptotic distribution to be valid. At typical API corpus sizes
            # (td≈1000, to≈10000), E[a] = td×(a+c)/(td+to) ≈ 0.55 when a=c=3
            # — below the validity threshold. In that regime, χ²≈12 while
            # Fisher's exact gives p≈0.012 (24× discrepancy), inflating Signal_Evans.
            # Fix: use Fisher's exact test when min(expected) < 5; χ² otherwise.
            # Single chi2_contingency call; switch to Fisher when expected < 5
            chi2v_raw, pv_raw, _, expected = chi2_contingency(
                [[a, b], [c, d]], correction=False
            )
            if expected.min() < 5:
                # One-sided test (alternative="greater"): pharmacovigilance asks
                # "is this reaction reported MORE often than expected?" — a directional
                # hypothesis. The Evans chi2≥4 threshold is also directional.
                # Note: For PRR=2.0 cases where Fisher fires (min(expected)<5),
                # one-sided Fisher chi2_equiv ≈ 4.1 correctly signals; for cases
                # where min(expected)≥5 (Pearson chi2 used instead), Pearson chi2
                # at PRR=2.0 with a=10,c=50 gives 4.19 — above the threshold.
                _, pv = fisher_exact([[a, b], [c, d]], alternative="greater")
                # Back-convert Fisher p to chi2-equivalent for display/threshold.
                # Guard: pv ≤ ~1e-17 underflows 1-pv → 1.0 → ppf = +inf.
                # When pv is extremely small (< ~1e-300), 1-pv=1.0 in float64,
                # causing ppf(1.0)=+inf. Clip to 999.9 as a display sentinel rather
                # than 1e6 — avoids confusing "Chi2: 1,000,000.0" in the output table.
                if pv < 1.0:
                    raw_chi2 = float(_chi2dist.ppf(min(1.0 - pv, 1.0 - 1e-15), df=1))
                else:
                    raw_chi2 = 0.0
                chi2v = float(np.clip(raw_chi2, 0.0, 999.9))
                _test_used = "fisher"
            else:
                chi2v, pv = chi2v_raw, pv_raw
                _test_used = "chi2"
        except Exception:
            chi2v, pv = 0.0, 1.0

        # PRR 95% CI: Haldane-corrected log-normal SE (Gart 1966).
        # Haldane correction adds 0.5 to each cell: ah=a+0.5, bh=b+0.5.
        # Therefore the corrected margin is ah+bh = td+1.0, NOT td+0.5.
        # td+0.5 is neither the raw formula nor the standard Haldane formula —
        # it is a non-standard hybrid that was introduced in a previous revision.
        se   = np.sqrt(
            1/(a + 0.5) - 1/(td + 1.0) +
            1/(c + 0.5) - 1/(to + 1.0)
        )
        cilo = float(np.exp(np.log(max(prr, 1e-9)) - 1.96 * se))
        cihi = float(np.exp(np.log(max(prr, 1e-9)) + 1.96 * se))

        rows.append({
            "Reaction":   rxn,
            "n":          a,
            "b":          b,
            "c_bg":       c,
            "d":          d,
            "td":         td,
            "to":         to,
            "PRR":        round(prr, 3),
            "ROR":        round(ror, 3),
            "ROR_lo":     round(ror_cilo, 3),
            "ROR_hi":     round(ror_cihi, 3),
            "IC":         ic,
            "IC025":      ic025,
            "IC975":      ic975,
            "EBGM":       em,
            "EB05":       e05,
            "CI_lo":      round(cilo, 3),
            "CI_hi":      round(cihi, 3),
            "Chi2":       round(chi2v, 2),
            "Test_used":  _test_used,    # "chi2" or "fisher" (for per-row audit trail)
            "p_raw":      round(float(pv), 6),
            "Signal_Group": SIGNAL_GROUP_MAP.get(rxn, "Other"),
            "Confound":   rxn in confounders,
            "Labeled":    is_labeled(rxn, label_text),
            "Signal_raw": (prr >= 2) and (chi2v >= 4) and (a >= 3),
        })

    if not rows:
        return pd.DataFrame(), False   # early-exit: no reactions passed a>=3/c>=3

    res = pd.DataFrame(rows)

    # ── BH FDR correction (applied over full a>=3 hypothesis set) ────────────
    # BH correction is applied BEFORE min_n filtering, over all reactions with
    # a >= 3 (the minimal inclusion criterion). This is the correct approach:
    # BH guarantees FDR ≤ α when applied over the full test family. Applying it
    # over a filtered subset (min_n > 3) increases the rejection threshold for
    # each rank (i·α/m grows as m shrinks), producing MORE rejections for the
    # displayed set and inflating the false discovery rate above the nominal level.
    # Empirically: BH over 500 hypotheses → 0 rejections in shown set vs.
    # BH over 50 → 50 rejections, for the same underlying p-values.
    # The subset approach is anti-conservative, not conservative as previously stated.
    if len(res) > 1:
        res["p_adj"] = false_discovery_control(res["p_raw"].values, method="bh")
    else:
        res["p_adj"] = res["p_raw"]

    # ── Apply min_n display filter AFTER BH-FDR ───────────────────────────────
    # Rows with n < min_n are hidden from the display table but were included
    # in the BH correction denominator above, which is methodologically correct.
    res = res[res["n"] >= min_n].copy()
    if res.empty:
        return pd.DataFrame(), False

    # ── Signal flags ──────────────────────────────────────────────────────────
    # Signal_Evans : published Evans (2001) / WHO-UMC criterion alone.
    #   PRR ≥ 2, χ² ≥ 4, n ≥ 3 — the industry-standard for spontaneous
    #   reporting disproportionality. No additional stacking.
    # Signal       : adds BH-FDR AND GPS (EB05 ≥ 2.0). More conservative.
    #   At API-constrained corpus sizes (~1,000 reports/drug), GPS posterior
    #   is dominated by the prior; Signal_Evans is preferred in that regime.
    res["Signal_Evans"] = res["Signal_raw"].copy()

    res["Signal"] = (
        res["Signal_raw"]
        & (res["p_adj"] < 0.05)
        & (res["EB05"] >= 2.0)
    )

    res["Tier"] = res["PRR"].apply(get_tier)

    # ── Composite score (absolute reference maxima) ───────────────────────────
    # IC and IC025 are allowed to contribute negatively when below zero.
    # Previous version used clip(lower=0), which silently zeroed negative IC
    # values — giving reactions with IC=−1.5 the same IC contribution as IC=0.
    # This removed the "downside-risk sensitivity" that IC025 was supposed to add,
    # and inflated composite scores for borderline reactions with negative Bayesian
    # evidence (e.g. PRR=2.1, χ²=4.1, IC=−0.5 got same IC component as IC=0).
    #
    # Fix: IC and IC025 use a signed normalisation centred at 0:
    #   positive IC → positive contribution (as before, normalised by ref max)
    #   negative IC → negative contribution (penalty, normalised by |ref min|)
    # PRR, Chi2, EBGM remain non-negative by construction, so clip is correct there.
    _IC_REF_MIN = -3.0    # plausible minimum IC in practice (strongly negative)

    def _ic_component(series: pd.Series, ref_max: float, ref_min: float = _IC_REF_MIN) -> pd.Series:
        """Signed log-scale normalisation: positive → [0,1], negative → [ref_min/ref_min, 0]."""
        pos = series.clip(lower=0)
        neg = series.clip(upper=0)
        return (
            np.log1p(pos)            / np.log1p(ref_max) +
            -np.log1p(neg.abs())     / np.log1p(abs(ref_min))
        )

    ref = COMPOSITE_REF_MAX
    wts = COMPOSITE_WEIGHTS
    res["Composite"] = (
        wts["PRR"]   * np.log1p(res["PRR"].clip(lower=0))   / np.log1p(ref["PRR"])  +
        wts["IC"]    * _ic_component(res["IC"],   ref["IC"])                         +
        wts["Chi2"]  * np.log1p(res["Chi2"].clip(lower=0, upper=1e6)) / np.log1p(ref["Chi2"]) +
        wts["EBGM"]  * np.log1p(res["EBGM"].clip(lower=0))  / np.log1p(ref["EBGM"]) +
        wts["IC025"] * _ic_component(res["IC025"], ref["IC025"])
    ).replace([np.inf, -np.inf], np.nan).clip(lower=-0.5, upper=1.0).round(3)
    # Clip to [-0.5, 1.0]: upper=1.0 prevents extreme values (PRR=200 would give >1);
    # lower=-0.5 prevents confusing deeply negative scores for the "ranking tool" framing.

    # ── Three-framework concordance ───────────────────────────────────────────
    # Each framework's published signal criterion:
    #   Signal_Evans — Evans (2001)/EMA: PRR≥2 AND χ²≥4 AND n≥3 (full criterion)
    #   EB05 ≥ 2.0  — FDA MGPS (DuMouchel 1999): 5th-percentile posterior lower bound
    #   IC025 > 0   — WHO-UMC (Bate et al. 1998): 2.5th-percentile credible lower bound
    #
    # Previous versions used EBGM≥2 for GPS — but EBGM is the geometric mean and
    # exceeds EB05 substantially at small n (n=3,mu=0.5: EBGM=2.19, EB05=0.29).
    # This produced "Full GPS concordance" for 218 tested (n,mu) pairs where the
    # Signal flag (EB05≥2) would correctly say no signal — an internal contradiction.
    # Replaced PRR≥2 alone with Signal_Evans to include the full Evans criterion.
    res["N_agree"] = (
        (res["Signal_Evans"]).astype(int) +         # Evans/EMA full criterion
        (res["EB05"]  >= 2.0).astype(int) +         # FDA MGPS: EB05 ≥ 2.0
        (res["IC025"] >  0).astype(int)             # WHO-UMC: IC025 > 0
    )
    res["Concordance"] = res["N_agree"].map(
        {3: "Full", 2: "Partial", 1: "Weak", 0: "None"}
    )

    res_sorted = res.sort_values("Composite", ascending=False).reset_index(drop=True)

    # Attach c=0 dropped count as DataFrame attribute for UI surfacing.
    # These reactions (a>=3 but c==0) cannot have PRR computed but may represent
    # drug-specific novel reactions. Users should be informed they exist.
    res_sorted.attrs["c0_dropped"] = len(_c0_dropped)
    res_sorted.attrs["c0_reactions"] = _c0_dropped[:10]  # sample for display

    return res_sorted, ebgm_fit_ok
