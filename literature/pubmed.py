"""
literature/pubmed.py — PubMed E-utilities client for per-signal literature context.

Fix vs. v8:
  - Single consistent NCBI email sourced from config.NCBI_EMAIL.
    v8 used two different emails ("cadentan2029@" in esearch, "neurovigilance@"
    in esummary) — inconsistent identity is a violation of NCBI terms of service
    and can trigger throttling of the tool ID.
  - Tool name sourced from config.NCBI_TOOL.
"""

from __future__ import annotations

import time
import streamlit as st
import requests

from config import NCBI_EMAIL, NCBI_TOOL, PUBMED_CACHE_TTL

_ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Single identity block — used in every NCBI call
_NCBI_PARAMS = {"tool": NCBI_TOOL, "email": NCBI_EMAIL}


@st.cache_data(show_spinner=False, ttl=PUBMED_CACHE_TTL)
def fetch_pubmed_signal(drug: str, reaction: str, max_results: int = 5) -> list[dict]:
    """
    Query PubMed for pharmacovigilance literature on a drug-reaction pair.

    Returns a list of article dicts:
      {pmid, title, authors, journal, year, volume, issue, pages, url, art_type}

    Uses NCBI E-utilities esearch (relevance-ranked) + esummary.
    Falls back to a broader query if the targeted search returns nothing.
    Results are cached for PUBMED_CACHE_TTL seconds (24h by default).
    """
    # ── Primary targeted query ─────────────────────────────────────────────────
    query = (
        f'"{drug}"[Title/Abstract] AND "{reaction}"[Title/Abstract] '
        f'AND (pharmacovigilance OR "adverse drug reaction" OR "adverse event" '
        f'OR "case report" OR "spontaneous report")'
    )
    ids = _esearch(query, max_results)

    # ── Fallback: broader query without field restriction ──────────────────────
    if not ids:
        ids = _esearch(f"{drug} {reaction} adverse drug reaction", max_results)

    if not ids:
        return []

    return _esummary(ids)


# NCBI E-utilities rate limit: 3 requests/second without an API key.
# A Streamlit app with 5 drug tabs fires esearch + esummary per tab — potentially
# 10 requests in rapid succession, exceeding the limit and causing HTTP 429 or
# silent IP blocking. Sleep 0.35s before each request (≤ 3/sec).
_NCBI_MIN_INTERVAL = 0.35


def _esearch(query: str, max_results: int) -> list[str]:
    """Run an esearch and return a list of PubMed IDs."""
    params = {
        **_NCBI_PARAMS,
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "sort":    "relevance",
        "retmode": "json",
    }
    try:
        time.sleep(_NCBI_MIN_INTERVAL)
        r = requests.get(_ESEARCH, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def _esummary(ids: list[str]) -> list[dict]:
    """Fetch document summaries for a list of PubMed IDs."""
    params = {
        **_NCBI_PARAMS,
        "db":      "pubmed",
        "id":      ",".join(ids),
        "retmode": "json",
    }
    try:
        time.sleep(_NCBI_MIN_INTERVAL)
        s = requests.get(_ESUMMARY, params=params, timeout=10)
        s.raise_for_status()
        results = s.json().get("result", {})
    except Exception:
        return []

    out = []
    for uid in ids:
        doc = results.get(uid, {})
        if not doc or doc.get("error"):
            continue

        title = doc.get("title", "").rstrip(".")
        if not title:
            continue

        authors_raw = doc.get("authors", [])
        names       = [a.get("name", "") for a in authors_raw[:3]]
        author_str  = ", ".join(n for n in names if n)
        if len(authors_raw) > 3:
            author_str += " et al."

        pub_date = doc.get("pubdate", "")
        year     = pub_date[:4] if pub_date else ""

        pub_types = [pt.get("value", "") for pt in doc.get("pubtype", [])]
        if any("Randomized Controlled Trial" in pt for pt in pub_types):
            art_type = "RCT"
        elif any("Review" in pt for pt in pub_types):
            art_type = "Review"
        elif any("Case Reports" in pt for pt in pub_types):
            art_type = "Case Report"
        else:
            art_type = "Original Article"

        out.append({
            "pmid":     uid,
            "title":    title,
            "authors":  author_str,
            "journal":  doc.get("fulljournalname", doc.get("source", "")),
            "year":     year,
            "volume":   doc.get("volume", ""),
            "issue":    doc.get("issue", ""),
            "pages":    doc.get("pages", ""),
            "url":      f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            "art_type": art_type,
        })

    return out


def render_pubmed_section(drug: str, reaction: str) -> None:
    """Render the PubMed literature panel in Streamlit."""
    st.markdown(
        "<div class='section-label'>Literature Context — PubMed (NCBI E-utilities)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class='method-card'>Real-time literature search for
        <strong>{drug} × {reaction}</strong>. Queries PubMed for pharmacovigilance,
        adverse event, and case report literature. Results cached 24h.</div>""",
        unsafe_allow_html=True,
    )

    with st.spinner(f"Querying PubMed for {drug} × {reaction}…"):
        pubs = fetch_pubmed_signal(drug, reaction, max_results=5)

    if not pubs:
        st.markdown(
            "<div class='warn-card'>No PubMed results found for this drug–reaction pair. "
            "This may represent a literature gap for novel signals.</div>",
            unsafe_allow_html=True,
        )
        return

    type_colors = {
        "RCT":              ("#EAF3DE", "#27500A"),
        "Review":           ("#EEEDFE", "#3C3489"),
        "Case Report":      ("#FAEEDA", "#633806"),
        "Original Article": ("#F6F4EF", "#6B6760"),
    }

    for pub in pubs:
        bg, fg = type_colors.get(pub["art_type"], ("#F6F4EF", "#6B6760"))
        citation_parts = []
        if pub["journal"]:
            citation_parts.append(f"<em>{pub['journal']}</em>")
        if pub["year"]:
            citation_parts.append(pub["year"])
        vol_str = pub["volume"]
        if pub["issue"]:
            vol_str += f"({pub['issue']})"
        if vol_str:
            citation_parts.append(vol_str)
        if pub["pages"]:
            citation_parts.append(f"pp. {pub['pages']}")
        citation = " · ".join(citation_parts)

        st.markdown(f"""
        <div class='pub-card'>
          <div style='display:flex;align-items:flex-start;gap:10px;'>
            <span style='display:inline-block;background:{bg};color:{fg};border-radius:20px;
                         padding:2px 8px;font-size:9px;font-weight:500;font-family:"DM Sans",sans-serif;
                         letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;margin-top:2px;'>
              {pub["art_type"]}
            </span>
            <div>
              <div class='pub-title'>
                <a href='{pub["url"]}' target='_blank'>{pub["title"]}</a>
              </div>
              <div class='pub-meta'>
                {pub["authors"]}{"  ·  " if pub["authors"] else ""}{citation}
                &nbsp;&nbsp;
                <a href='{pub["url"]}' target='_blank'
                   style='color:#7F77DD;font-size:10px;text-decoration:none;'>
                  PMID {pub["pmid"]} ↗
                </a>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    pubmed_search_url = (
        f"https://pubmed.ncbi.nlm.nih.gov/?term="
        f"{requests.utils.quote(drug)}+{requests.utils.quote(reaction)}"
        f"+adverse+drug+reaction&sort=relevance"
    )
    st.markdown(
        f"<div style='text-align:right;margin-top:8px;'>"
        f"<a href='{pubmed_search_url}' target='_blank' "
        f"style='font-family:\"DM Sans\",sans-serif;font-size:11px;color:#7F77DD;text-decoration:none;'>"
        f"View all results in PubMed ↗</a></div>",
        unsafe_allow_html=True,
    )
