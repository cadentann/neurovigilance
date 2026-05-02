"""
tests/test_data.py — Data-layer tests for FAERS client and deduplication logic.

Uses unittest.mock to avoid real network calls. Tests the most business-critical
data processing: deduplication (higher caseversion wins), age unit conversion,
seriousness mapping, and rxn_df structure.

Run with:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# faers_client imports streamlit at module level for @st.cache_data.
# Stub it out before importing so tests run without a Streamlit server.
from unittest.mock import MagicMock, patch
_st_stub = MagicMock()
_st_stub.cache_data = lambda *a, **kw: (lambda f: f)   # passthrough decorator
_st_stub.secrets.get = lambda k, d=None: d
sys.modules.setdefault("streamlit", _st_stub)

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from data.faers_client import build_report_and_rxn_dfs, _parse_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_report(
    primaryid: str,
    caseversion: int = 1,
    drug: str = "Galantamine",
    reactions: list[str] | None = None,
    age: float | None = 65.0,
    age_unit: str = "801",    # 801 = years
    sex: int = 1,
    seriousness: int = 1,
    receivedate: str = "20180601",
) -> dict:
    """Build a minimal FAERS-like JSON report dict."""
    return {
        "safetyreportid":      primaryid,
        "safetyreportversion": str(caseversion),
        "seriousness":         seriousness,
        "receivedate":         receivedate,
        "patient": {
            "patientonsetage":     str(age) if age is not None else None,
            "patientonsetageunit": age_unit,
            "patientsex":          sex,
            "reaction": [
                {"reactionmeddrapt": r}
                for r in (reactions or ["Orthostatic Hypotension"])
            ],
            "drug": [{"medicinalproduct": drug}],
        },
    }


# ── build_report_and_rxn_dfs ─────────────────────────────────────────────────

class TestBuildReportAndRxnDfs:
    def test_deduplication_keeps_highest_caseversion(self):
        """
        When the same (primaryid, drug) appears with two different caseversions,
        only the record with the highest caseversion should survive.

        Deduplication is on (primaryid, drug), NOT just primaryid, so that
        the same case appearing in two drug fetches (co-primary suspects)
        preserves both drug attributions.
        """
        raw = [
            pd.DataFrame([
                # Same case, same drug, two versions — only v3 should survive
                {"primaryid": "case-001", "caseversion": 1, "drug": "Galantamine",
                 "sex": "Male", "age": 65.0, "serious": "Serious",
                 "year": 2018, "quarter": "2018-Q2", "reactions": ["Nausea"]},
                {"primaryid": "case-001", "caseversion": 3, "drug": "Galantamine",
                 "sex": "Male", "age": 65.0, "serious": "Serious",
                 "year": 2018, "quarter": "2018-Q2", "reactions": ["Orthostatic Hypotension"]},
                # Same primaryid but DIFFERENT drug — MUST survive independently
                {"primaryid": "case-001", "caseversion": 1, "drug": "Metformin",
                 "sex": "Male", "age": 65.0, "serious": "Serious",
                 "year": 2018, "quarter": "2018-Q2", "reactions": ["Nausea"]},
                {"primaryid": "case-002", "caseversion": 1, "drug": "Galantamine",
                 "sex": "Female", "age": 72.0, "serious": "Non-serious",
                 "year": 2019, "quarter": "2019-Q1", "reactions": ["Vomiting"]},
            ])
        ]
        report_df, rxn_df = build_report_and_rxn_dfs(raw)

        # case-001 should appear TWICE: once for Galantamine (v3), once for Metformin (v1)
        # case-002 appears once for Galantamine
        assert len(report_df) == 3, (
            f"Expected 3 rows (case-001×Galantamine, case-001×Metformin, case-002×Galantamine); "
            f"got {len(report_df)}"
        )

        case001_gal = report_df[
            (report_df["primaryid"] == "case-001") & (report_df["drug"] == "Galantamine")
        ]
        assert len(case001_gal) == 1
        assert case001_gal.iloc[0]["caseversion"] == 3, "Higher caseversion (3) must win"

        # Metformin row for case-001 must also survive
        case001_met = report_df[
            (report_df["primaryid"] == "case-001") & (report_df["drug"] == "Metformin")
        ]
        assert len(case001_met) == 1, (
            "Metformin attribution of case-001 must survive dedup — "
            "global primaryid dedup (old bug) would have dropped it"
        )

    def test_deduplication_idempotent(self):
        """No duplicates in input → no change in output."""
        raw = [
            pd.DataFrame([
                {"primaryid": f"id-{i}", "caseversion": 1, "drug": "Galantamine",
                 "sex": "Male", "age": 60.0 + i, "serious": "Serious",
                 "year": 2018, "quarter": "2018-Q1",
                 "reactions": ["Nausea"]}
                for i in range(5)
            ])
        ]
        report_df, rxn_df = build_report_and_rxn_dfs(raw)
        assert report_df["primaryid"].nunique() == 5
        assert len(report_df) == 5

    def test_rxn_df_has_one_row_per_report_reaction_pair(self):
        """rxn_df must have exactly one row per (primaryid, reaction) pair."""
        raw = [
            pd.DataFrame([
                {"primaryid": "id-1", "caseversion": 1, "drug": "Galantamine",
                 "sex": "Male", "age": 70.0, "serious": "Serious",
                 "year": 2018, "quarter": "2018-Q2",
                 "reactions": ["Nausea", "Vomiting", "Orthostatic Hypotension"]},
            ])
        ]
        _, rxn_df = build_report_and_rxn_dfs(raw)
        # 3 reactions → 3 rows in rxn_df
        assert len(rxn_df) == 3
        assert set(rxn_df["reaction"]) == {"Nausea", "Vomiting", "Orthostatic Hypotension"}

    def test_cross_drug_dedup_preserves_both_attributions(self):
        """
        If case-001 appears in both DrugA and DrugB fetches (co-primary suspects),
        both attributions must survive after build_report_and_rxn_dfs.

        Old bug: dedup on primaryid alone dropped one drug's row, underestimating
        that drug's td denominator and inflating its PRR.
        """
        frame_a = pd.DataFrame([{
            "primaryid": "shared-001", "caseversion": 1, "drug": "DrugA",
            "sex": "Male", "age": 70.0, "serious": "Serious",
            "year": 2020, "quarter": "2020-Q1", "reactions": ["Nausea"],
        }])
        frame_b = pd.DataFrame([{
            "primaryid": "shared-001", "caseversion": 1, "drug": "DrugB",
            "sex": "Male", "age": 70.0, "serious": "Serious",
            "year": 2020, "quarter": "2020-Q1", "reactions": ["Nausea"],
        }])
        report_df, _ = build_report_and_rxn_dfs([frame_a, frame_b])

        # Both attributions must survive
        assert len(report_df) == 2, (
            f"Expected 2 rows (shared-001×DrugA, shared-001×DrugB); got {len(report_df)}"
        )
        assert set(report_df["drug"]) == {"DrugA", "DrugB"}

    def test_empty_input_returns_empty_dfs(self):
        """Empty raw_frames → two empty DataFrames, no crash."""
        report_df, rxn_df = build_report_and_rxn_dfs([])
        assert report_df.empty
        assert rxn_df.empty

    def test_multi_drug_concat(self):
        """Frames for multiple drugs are concatenated correctly."""
        frame_a = pd.DataFrame([{
            "primaryid": "ga-1", "caseversion": 1, "drug": "Galantamine",
            "sex": "Male", "age": 65.0, "serious": "Serious",
            "year": 2019, "quarter": "2019-Q1", "reactions": ["Syncope"],
        }])
        frame_b = pd.DataFrame([{
            "primaryid": "don-1", "caseversion": 1, "drug": "Donepezil",
            "sex": "Female", "age": 74.0, "serious": "Non-serious",
            "year": 2019, "quarter": "2019-Q1", "reactions": ["Nausea"],
        }])
        report_df, _ = build_report_and_rxn_dfs([frame_a, frame_b])
        assert report_df["primaryid"].nunique() == 2
        assert set(report_df["drug"]) == {"Galantamine", "Donepezil"}


# ── _parse_report ─────────────────────────────────────────────────────────────

class TestParseReport:
    def test_age_unit_year(self):
        """801 = years → age passes through as-is."""
        rep   = _minimal_report("p1", age=65.0, age_unit="801")
        row   = _parse_report(rep, "Galantamine")
        assert row["age"] == pytest.approx(65.0, abs=0.5)

    def test_age_unit_month(self):
        """802 = months → 6 months ≈ 0.5 years."""
        rep = _minimal_report("p1", age=6.0, age_unit="802")
        row = _parse_report(rep, "Galantamine")
        assert row["age"] == pytest.approx(6 / 12, abs=0.05)

    def test_age_unit_decade(self):
        """800 = decades → 6 decades = 60 years."""
        rep = _minimal_report("p1", age=6.0, age_unit="800")
        row = _parse_report(rep, "Galantamine")
        assert row["age"] == pytest.approx(60.0, abs=1.0)

    def test_age_unit_day(self):
        """804 = days → 365 days ≈ 1 year."""
        rep = _minimal_report("p1", age=365.0, age_unit="804")
        row = _parse_report(rep, "Galantamine")
        assert row["age"] == pytest.approx(1.0, abs=0.05)

    def test_age_out_of_range_clipped(self):
        """Ages > 130 or < 0 after unit conversion are set to None."""
        rep = _minimal_report("p1", age=9999.0, age_unit="801")
        row = _parse_report(rep, "Galantamine")
        assert row["age"] is None

    def test_seriousness_1_is_serious(self):
        """seriousness=1 → 'Serious'."""
        rep = _minimal_report("p1", seriousness=1)
        row = _parse_report(rep, "Galantamine")
        assert row["serious"] == "Serious"

    def test_seriousness_0_is_non_serious(self):
        """seriousness=0 → 'Non-serious'."""
        rep = _minimal_report("p1", seriousness=0)
        row = _parse_report(rep, "Galantamine")
        assert row["serious"] == "Non-serious"

    def test_seriousness_absent_is_non_serious(self):
        """Missing seriousness field (absent in FAERS) → 'Non-serious'."""
        rep = _minimal_report("p1")
        del rep["seriousness"]
        row = _parse_report(rep, "Galantamine")
        assert row["serious"] == "Non-serious"

    def test_primaryid_preserved(self):
        """safetyreportid and safetyreportversion must be preserved."""
        rep = _minimal_report("case-XYZ-999", caseversion=5)
        row = _parse_report(rep, "Galantamine")
        assert row["primaryid"]   == "case-XYZ-999"
        assert row["caseversion"] == 5

    def test_quarter_derived_from_receivedate(self):
        """receivedate YYYYMMDD → correct calendar quarter."""
        rep = _minimal_report("p1", receivedate="20220715")  # July → Q3
        row = _parse_report(rep, "Galantamine")
        assert row["quarter"] == "2022-Q3"
        assert row["year"]    == 2022

    def test_reactions_title_cased(self):
        """MedDRA PTs are normalised to Title Case."""
        rep = _minimal_report("p1", reactions=["orthostatic hypotension", "NAUSEA"])
        row = _parse_report(rep, "Galantamine")
        assert "Orthostatic Hypotension" in row["reactions"]
        assert "Nausea" in row["reactions"]

    def test_sex_mapping(self):
        """patientsex 1 → Male, 2 → Female, other → Unknown."""
        for sx, expected in [(1, "Male"), (2, "Female"), (9, "Unknown")]:
            rep = _minimal_report("p1", sex=sx)
            row = _parse_report(rep, "Galantamine")
            assert row["sex"] == expected, f"sex={sx} → expected {expected}, got {row['sex']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
