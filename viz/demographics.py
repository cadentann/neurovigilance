"""
viz/demographics.py — Demographic breakdown for a target drug-reaction pair.

Compares age distribution and sex breakdown between:
  - Reports of the drug WITH the target reaction (cases)
  - Reports of the drug WITHOUT the target reaction (controls)

This helps assess whether a signal is driven by a specific demographic subgroup,
which is relevant for pharmacovigilance case causality assessment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def demographic_charts(
    report_df: pd.DataFrame,
    rxn_df: pd.DataFrame,
    drug: str,
    reaction: str,
) -> go.Figure | None:
    """
    Produce a two-panel demographic figure for a drug-reaction pair.

    Left panel  : Age histogram — cases (drug+reaction) vs controls (drug, no reaction)
    Right panel : Sex breakdown — pie / bar comparison

    Parameters
    ----------
    report_df : De-duplicated report-level DataFrame with 'primaryid', 'drug', 'age', 'sex'
    rxn_df    : Exploded reaction DataFrame with 'primaryid', 'drug', 'reaction'
    drug      : Target drug name
    reaction  : Target MedDRA PT

    Returns
    -------
    plotly Figure with two subplots, or None if insufficient data
    """
    drug_reports = set(report_df[report_df["drug"] == drug]["primaryid"])
    rxn_reports  = set(
        rxn_df[(rxn_df["drug"] == drug) & (rxn_df["reaction"] == reaction)]["primaryid"]
    )

    cases    = report_df[report_df["primaryid"].isin(rxn_reports)].copy()
    controls = report_df[
        report_df["primaryid"].isin(drug_reports - rxn_reports)
    ].copy()

    if len(cases) < 3:
        return None

    # ── Age data ──────────────────────────────────────────────────────────────
    ages_case = cases["age"].dropna()
    ages_ctrl = controls["age"].dropna()

    has_age = len(ages_case) >= 3

    # ── Sex data ──────────────────────────────────────────────────────────────
    def sex_counts(df: pd.DataFrame) -> dict[str, int]:
        vc = df["sex"].value_counts()
        return {
            "Male":    int(vc.get("Male",    0)),
            "Female":  int(vc.get("Female",  0)),
            "Unknown": int(vc.get("Unknown", 0)),
        }

    case_sex = sex_counts(cases)
    ctrl_sex = sex_counts(controls)
    has_sex  = sum(case_sex.values()) >= 3

    if not has_age and not has_sex:
        return None

    # Build subplots
    specs  = [[{"type": "xy"}, {"type": "xy"}]] if has_age else [[{"type": "xy"}]]
    titles = []
    if has_age:
        titles.append("Age Distribution")
    titles.append("Sex Breakdown")

    n_cols = 2 if has_age else 1
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles)

    col_idx = 1

    # ── Age histogram ─────────────────────────────────────────────────────────
    if has_age:
        bins = list(range(0, 110, 10))

        fig.add_trace(go.Histogram(
            x=ages_case,
            xbins=dict(start=0, end=110, size=10),
            name=f"With {reaction}",
            marker_color="#791F1F",
            opacity=0.65,
            histnorm="percent",
        ), row=1, col=col_idx)

        fig.add_trace(go.Histogram(
            x=ages_ctrl,
            xbins=dict(start=0, end=110, size=10),
            name="Without reaction",
            marker_color="#3C3489",
            opacity=0.65,
            histnorm="percent",
        ), row=1, col=col_idx)

        fig.update_xaxes(title_text="Age (years)", row=1, col=col_idx)
        fig.update_yaxes(title_text="% of reports", row=1, col=col_idx)

        # Median annotation
        if len(ages_case) > 0:
            med = float(np.median(ages_case))
            fig.add_vline(
                x=med, line_dash="dash", line_color="#791F1F",
                annotation_text=f"Median {med:.0f}y",
                annotation_font_size=9,
                row=1, col=col_idx,
            )

        col_idx += 1

    # ── Sex bar chart ─────────────────────────────────────────────────────────
    sex_cats = ["Male", "Female", "Unknown"]
    sex_colors = {"Male": "#3C3489", "Female": "#791F1F", "Unknown": "#AAAAAA"}

    case_pct = _to_pct(case_sex)
    ctrl_pct = _to_pct(ctrl_sex)

    fig.add_trace(go.Bar(
        name=f"With {reaction}",
        x=sex_cats,
        y=[case_pct[s] for s in sex_cats],
        marker_color=[sex_colors[s] for s in sex_cats],
        opacity=0.85,
        showlegend=False,
        customdata=[[case_sex[s]] for s in sex_cats],
        hovertemplate="%{x}: %{y:.1f}% (n=%{customdata[0]})<extra>Cases</extra>",
    ), row=1, col=col_idx)

    fig.add_trace(go.Bar(
        name="Without reaction",
        x=sex_cats,
        y=[ctrl_pct[s] for s in sex_cats],
        marker_color=[sex_colors[s] for s in sex_cats],
        opacity=0.35,
        showlegend=False,
        customdata=[[ctrl_sex[s]] for s in sex_cats],
        hovertemplate="%{x}: %{y:.1f}% (n=%{customdata[0]})<extra>Controls</extra>",
    ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Sex", row=1, col=col_idx)
    fig.update_yaxes(title_text="% of reports", row=1, col=col_idx)

    fig.update_layout(
        title=dict(
            text=f"Demographic Profile — {drug} × {reaction}",
            font=dict(size=14, color="#3C3A35"), x=0.0, xanchor="left",
        ),
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#3C3A35"),
        height=360,
        margin=dict(l=20, r=20, t=70, b=50),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        annotations=[
            *fig.layout.annotations,
            dict(
                text=(
                    f"Cases: n={len(cases)} | "
                    f"Controls: n={len(controls)}"
                ),
                xref="paper", yref="paper",
                x=0.0, y=-0.08, showarrow=False,
                font=dict(size=10, color="#6B6760"),
            ),
        ],
    )
    return fig


def _to_pct(d: dict[str, int]) -> dict[str, float]:
    total = sum(d.values())
    if total == 0:
        return {k: 0.0 for k in d}
    return {k: 100.0 * v / total for k, v in d.items()}
