"""
data/label_client.py — FDA drug label fetcher via openFDA label endpoint.

Retrieves the adverse reactions, warnings, precautions, and boxed warning
sections from the FDA prescribing information for label novelty detection.

Used by stats/prr.py::is_labeled() which now requires the FULL MedDRA PT
phrase to appear verbatim — so returning full lowercased prose is correct.
"""

from __future__ import annotations

import streamlit as st
import requests

from config import FDA_LABEL_URL, FAERS_CACHE_TTL


def _get_api_key() -> str:
    try:
        return st.secrets.get("FDA_API_KEY", "")
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=86400)  # Cache labels for 24h
def fetch_label(drug: str, brands: dict[str, str]) -> str:
    """
    Fetch the full-text FDA label for a drug and return it as a lowercased
    string (for is_labeled() full-phrase matching in stats/prr.py).

    Tries brand name first, then generic name. Returns empty string on failure.

    Parameters
    ----------
    drug   : Generic drug name (e.g. "Galantamine")
    brands : Dict mapping generic name → brand name slug (e.g. {"Galantamine": "razadyne"})

    Returns
    -------
    str : Concatenated label sections, lowercased. Empty string on failure.
    """
    api_key = _get_api_key()
    brand   = brands.get(drug, drug.lower())

    queries = [
        f'openfda.brand_name:"{brand}"',
        f'openfda.generic_name:"{drug.lower()}"',
    ]

    for q in queries:
        params: dict = {"search": q, "limit": 1}
        if api_key:
            params["api_key"] = api_key

        try:
            r = requests.get(FDA_LABEL_URL, params=params, timeout=10)
            if r.status_code != 200:
                continue
            results = r.json().get("results", [])
            if not results:
                continue

            lbl   = results[0]
            # clinical_pharmacology excluded: describes mechanism/pharmacodynamics,
            # not formal ADR listings. Including it marks mechanistically predicted
            # reactions as "labeled" (e.g. "bradycardia can be expected" for ChEIs),
            # suppressing Novel flags even when absent from adverse_reactions section.
            parts = (
                lbl.get("adverse_reactions",     []) +
                lbl.get("warnings",              []) +
                lbl.get("precautions",           []) +
                lbl.get("boxed_warning",         []) +
                lbl.get("warnings_and_cautions", []) +
                lbl.get("drug_interactions",     []) +
                lbl.get("contraindications",     [])
            )
            return " ".join(parts).lower()

        except Exception:
            continue

    return ""
