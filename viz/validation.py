"""
viz/validation.py — Known-Signal Validation Tab

Runs NeuroVigilance's signal detection pipeline against drug-reaction pairs
where the signal is established in the published pharmacovigilance literature.
Confirms the tool produces positive detections on gold-standard cases.

Validation pairs and their literature anchors:
────────────────────────────────────────────────────────────────
Donepezil × Bradycardia
  Mechanism: AChE inhibition → ↑ACh → M2 receptor activation → SA node
  suppression. A direct pharmacodynamic class effect of all ChEIs.
  Literature: Park-Wyllie et al. (2009) JAMA 302(22):2413–2420.
              van Noord et al. (2010) Eur Heart J 31(11):1383–1391.
  Note: Bradycardia is in the Donepezil confounder set because it IS a
  class effect — the tool is expected to flag it AND mark Confound=True.
  Both outcomes together are the correct signal: real, labeled, expected.

Quetiapine × QT Prolongation
  Mechanism: hERG K⁺ channel blockade (IC₅₀ ~1µM) → delayed ventricular
  repolarisation → QTc prolongation → Torsades de Pointes risk.
  Literature: Liperoti et al. (2005) Arch Intern Med 165(9):981–986.
              FDA MedWatch Safety Alert (2011): Quetiapine QT prolongation.
  Note: QT prolongation is intentionally NOT in the antipsychotic confounder
  set because drug-specific variation is clinically meaningful (Aripiprazole
  << Quetiapine << Clozapine). Signal_Evans=True + Confound=False = novel
  drug-specific signal. This is the stronger of the two validation cases.

Paroxetine × Hyponatraemia
  Mechanism: SERT → ↑5-HT → 5-HT1A/2C → ↑ADH release → SIADH.
  Paroxetine has substantially higher incidence than other SSRIs due to
  stronger SERT affinity and potent CYP2D6 inhibition.
  Literature: Wilkinson et al. (1999) BMJ 318:1087.
              Spigset & Hedenmalm (1997) Drug Saf 17(3):153–160.
  Note: Hyponatraemia is intentionally NOT in the SSRI confounder set
  (unlike Insomnia/Sexual Dysfunction) precisely because of drug-specific
  variation. A signal here confirms the class-level exclusion decision is
  correct for the right mechanistic reason.
────────────────────────────────────────────────────────────────

Usage (from app.py):
    from viz.validation import render_validation_tab
    render_validation_tab(report_df, rxn_df)
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from stats.prr import compute_prr


# ── Validation pair registry ──────────────────────────────────────────────────

VALIDATION_PAIRS: list[dict] = [
    {
        "drug":       "Donepezil",
        "reaction":   "Bradycardia",
        "drug_class": "Cholinesterase Inhibitors",
        "mechanism":  "AChE inhibition → ↑ACh → M2 receptor activation → SA node suppression.",
        "citations": [
            "Park-Wyllie et al. (2009). JAMA 302(22):2413–2420.",
            "van Noord et al. (2010). Eur Heart J 31(11):1383–1391.",
        ],
        "expected_signal":   True,
        "expected_confound": True,   # class effect — should be flagged Confound=True
        "expected_labeled":  True,   # on FDA label
        "notes": (
            "Bradycardia is a direct pharmacodynamic class effect of all ChEIs and is "
            "listed in the Donepezil confounder set. The validation passes when the tool "
            "detects Signal_Evans=True AND Confound=True — both outcomes together are "
            "the correct result for a known, labeled, mechanism-based ADR."
        ),
    },
    {
        "drug":       "Quetiapine",
        "reaction":   "Electrocardiogram Qt Prolonged",
        "drug_class": "Atypical Antipsychotics",
        "mechanism":  "hERG K⁺ channel blockade (IC₅₀ ~1µM) → delayed ventricular repolarisation → QTc prolongation.",
        "citations": [
            "Liperoti et al. (2005). Arch Intern Med 165(9):981–986.",
            "FDA MedWatch Safety Alert (2011): Quetiapine QT prolongation.",
        ],
        "expected_signal":   True,
        "expected_confound": False,  # intentionally NOT in antipsychotic confounder set
        "expected_labeled":  True,
        "notes": (
            "QT prolongation is intentionally excluded from the antipsychotic confounder set "
            "because drug-specific variation is clinically meaningful (Aripiprazole << "
            "Quetiapine << Clozapine). Validation passes when Signal_Evans=True AND "
            "Confound=False — this is the stronger case: a real, drug-specific, non-confounded "
            "signal that the tool should surface independently."
        ),
    },
    {
        "drug":       "Paroxetine",
        "reaction":   "Hyponatraemia",
        "drug_class": "SSRIs",
        "mechanism":  "SERT inhibition → ↑5-HT → 5-HT1A/2C → ↑ADH release → SIADH. Higher incidence than other SSRIs due to stronger SERT affinity + CYP2D6 inhibition.",
        "citations": [
            "Wilkinson et al. (1999). BMJ 318:1087.",
            "Spigset & Hedenmalm (1997). Drug Saf 17(3):153–160.",
        ],
        "expected_signal":   True,
        "expected_confound": False,  # intentionally excluded from SSRI confounder set
        "expected_labeled":  True,
        "notes": (
            "Hyponatraemia is intentionally excluded from the SSRI class confounder set "
            "because drug-specific variation (Paroxetine >> Escitalopram/Citalopram) is "
            "clinically meaningful. A signal here confirms both the detection pipeline "
            "and the confounder exclusion decision."
        ),
    },
]


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_validation_tab(
    report_df: pd.DataFrame,
    rxn_df:    pd.DataFrame,
    drug_classes: dict,
) -> None:
    """
    Render the Validation tab in the Streamlit app.

    Parameters
    ----------
    report_df    : De-duplicated report-level DataFrame (from build_report_and_rxn_dfs)
    rxn_df       : Exploded reaction DataFrame
    drug_classes : DRUG_CLASSES dict from config (for confounders + label lookup)
    """
    st.markdown(
        "<div class='method-card'>"
        "Runs signal detection against three drug-reaction pairs with established "
        "literature support. A valid implementation should detect all three. "
        "Pairs span both expected class effects (Donepezil × Bradycardia) and "
        "drug-specific novel signals (Quetiapine × QT prolongation, Paroxetine × Hyponatraemia). "
        "Green = pass, Red = fail, Yellow = insufficient data."
        "</div>",
        unsafe_allow_html=True,
    )

    passed = 0
    failed = 0
    insufficient = 0

    for pair in VALIDATION_PAIRS:
        drug     = pair["drug"]
        reaction = pair["reaction"]
        cls_name = pair["drug_class"]

        st.markdown(f"---")
        st.markdown(f"#### {drug} × {reaction}")
        st.markdown(
            f"<div class='method-card'>"
            f"<strong>Mechanism:</strong> {pair['mechanism']}<br>"
            f"<strong>References:</strong> {' · '.join(pair['citations'])}<br>"
            f"<strong>Expected:</strong> Signal_Evans={'✓' if pair['expected_signal'] else '✗'} · "
            f"Confound={'✓' if pair['expected_confound'] else '✗'} · "
            f"Labeled={'✓' if pair['expected_labeled'] else '✗'}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Check data availability
        drug_reports = report_df[report_df["drug"] == drug]
        drug_rxns    = rxn_df[(rxn_df["drug"] == drug) & (rxn_df["reaction"] == reaction)]
        n_drug       = drug_reports["primaryid"].nunique()
        n_rxn        = drug_rxns["primaryid"].nunique()

        if n_drug < 50:
            st.warning(
                f"⚠️ Insufficient data: only {n_drug} unique {drug} reports in current corpus. "
                f"Load 'Cholinesterase Inhibitors' or 'Atypical Antipsychotics' / 'SSRIs' "
                f"class first, or lower the minimum reports slider."
            )
            insufficient += 1
            continue

        if n_rxn < 3:
            st.warning(
                f"⚠️ Only {n_rxn} reports of {reaction} for {drug} in current corpus "
                f"(minimum 3 required for Evans criterion). "
                f"This may reflect the seriousness filter or a small corpus — try 'All'."
            )
            insufficient += 1
            continue

        # Run the pipeline
        cls_config    = drug_classes.get(cls_name, {})
        confounders   = cls_config.get("confounders", set())

        with st.spinner(f"Running pipeline: {drug} × {reaction}…"):
            res, _ = compute_prr(
                rxn_df=rxn_df,
                report_df=report_df,
                drug=drug,
                confounders=confounders,
                serious_filter="All",
                label_text="",   # skip label lookup for validation — test signal detection only
                fit_ebgm=True,
                min_n=3,
            )

        if res.empty:
            st.error(f"❌ Pipeline returned empty results for {drug}.")
            failed += 1
            continue

        # Find the target reaction row
        row = res[res["Reaction"] == reaction]

        if row.empty:
            st.warning(
                f"⚠️ '{reaction}' not found in results table for {drug}. "
                f"MedDRA PT spelling must match exactly. "
                f"Available reactions containing 'brad' or 'QT' or 'natr': "
                + ", ".join(
                    r for r in res["Reaction"].tolist()
                    if any(s in r.lower() for s in ["brad", "qt", "natr", "sodium"])
                )[:200]
            )
            insufficient += 1
            continue

        row = row.iloc[0]

        # Extract key values
        n_obs        = int(row.get("n", 0))
        prr          = row.get("PRR", float("nan"))
        prr_lo       = row.get("PRR_CI_lo", float("nan"))
        ic           = row.get("IC", float("nan"))
        ic025        = row.get("IC025", float("nan"))
        ebgm         = row.get("EBGM", float("nan"))
        e05          = row.get("EB05", float("nan"))
        signal_evans = bool(row.get("Signal_Evans", False))
        confound     = bool(row.get("Confound", False))
        labeled      = bool(row.get("Labeled", True))
        tier         = str(row.get("Tier", "NONE"))

        # Evaluate pass/fail
        signal_ok  = (signal_evans == pair["expected_signal"])
        confound_ok = (confound == pair["expected_confound"])

        overall_pass = signal_ok and confound_ok

        # Display result
        if overall_pass:
            passed += 1
            badge_color = "#1a7340"
            badge_bg    = "#d4edda"
            badge_text  = "✅ PASS"
        else:
            failed += 1
            badge_color = "#721c24"
            badge_bg    = "#f8d7da"
            badge_text  = "❌ FAIL"

        st.markdown(
            f"<div style='background:{badge_bg};border-radius:8px;padding:12px 16px;"
            f"margin-bottom:8px;'>"
            f"<span style='font-weight:600;color:{badge_color};font-size:14px;'>{badge_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Stats summary
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("n (reports)", n_obs)
        c2.metric("PRR", f"{prr:.2f}" if pd.notna(prr) else "—",
                  delta=f"CI_lo {prr_lo:.2f}" if pd.notna(prr_lo) else None)
        c3.metric("IC (BCPNN)", f"{ic:.3f}" if pd.notna(ic) else "—",
                  delta=f"IC025 {ic025:.3f}" if pd.notna(ic025) else None)
        c4.metric("EBGM", f"{ebgm:.2f}" if pd.notna(ebgm) else "—",
                  delta=f"EB05 {e05:.2f}" if pd.notna(e05) else None)
        c5.metric("Tier", tier)

        # Signal criteria breakdown
        checks = {
            f"Signal_Evans = {signal_evans}":    signal_ok,
            f"Confound = {confound}":             confound_ok,
            f"Labeled = {labeled}":               True,   # informational only
        }
        for label_text, ok in checks.items():
            icon = "✅" if ok else "❌"
            st.markdown(f"- {icon} {label_text}")

        if not overall_pass:
            st.markdown(
                f"<div class='warn-card'>"
                f"Expected Signal_Evans={pair['expected_signal']}, "
                f"Confound={pair['expected_confound']}. "
                f"Got Signal_Evans={signal_evans}, Confound={confound}. "
                f"Check: (1) corpus size, (2) seriousness filter, "
                f"(3) confounder set in config.py."
                f"</div>",
                unsafe_allow_html=True,
            )

        with st.expander("📖 Notes on this validation pair"):
            st.markdown(pair["notes"])

    # ── Summary scorecard ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Validation Summary")
    total_run = passed + failed
    total     = len(VALIDATION_PAIRS)

    if total_run == 0:
        st.info("No validation pairs had sufficient data to run. Load a full drug class first.")
    else:
        score_color = "#1a7340" if failed == 0 else ("#e67e22" if passed > 0 else "#721c24")
        st.markdown(
            f"<div style='background:#F6F4EF;border-radius:8px;padding:16px;'>"
            f"<span style='font-size:20px;font-weight:600;color:{score_color};'>"
            f"{passed}/{total_run} pairs passed"
            f"</span>"
            f"<span style='font-size:12px;color:#6B6760;margin-left:12px;'>"
            f"({insufficient} skipped — insufficient data)"
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if failed == 0 and total_run == total:
            st.success(
                "All validation pairs detected. "
                "The signal detection pipeline reproduces known pharmacovigilance findings."
            )
        elif failed > 0:
            st.warning(
                f"{failed} pair(s) failed. Check corpus size, seriousness filter, "
                "and confounder configuration."
            )
