"""
viz/temporal.py — Temporal stability chart: rolling PRR over calendar quarters.

Shows whether signals are persistent or artefactual (Weber-spike then decay).
A stable pharmacological signal should show sustained PRR elevation.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config import DRUG_COLORS, TIER_COLORS


def rolling_prr_chart(
    roll_df: pd.DataFrame,
    drug: str,
    top_n: int = 5,
    window: int = 4,
) -> go.Figure | None:
    """
    Line chart of rolling-window PRR over calendar quarters.

    Parameters
    ----------
    roll_df : DataFrame returned by stats.prr.rolling_prr()
              Columns: Quarter, Reaction, PRR
    drug    : Drug name (used in title)
    top_n   : Number of top reactions to display

    Returns
    -------
    plotly Figure, or None if roll_df is empty / has no data to show
    """
    if roll_df.empty:
        return None

    # Pick the top_n reactions by their maximum PRR across all quarters
    top_rxns = (
        roll_df.groupby("Reaction")["PRR"]
        .max()
        .nlargest(top_n)
        .index.tolist()
    )
    df = roll_df[roll_df["Reaction"].isin(top_rxns)].copy()

    if df.empty:
        return None

    quarters_sorted = sorted(df["Quarter"].unique())
    if len(quarters_sorted) < 2:
        return None

    # Colour palette — cycle through a fixed set
    palette = [
        "#791F1F", "#3C3489", "#27500A", "#633806", "#7F77DD",
        "#4A9E6B", "#C05A1F", "#6B3A8A", "#2A7A8A", "#8A5A2A",
    ]

    fig = go.Figure()

    for i, rxn in enumerate(top_rxns):
        sub = df[df["Reaction"] == rxn].sort_values("Quarter")
        color = palette[i % len(palette)]

        # Determine which quarters have full vs. partial windows
        # A partial window is when fewer than `window` quarters of data precede it
        all_qs = sorted(df["Quarter"].unique())
        sub_q  = sub["Quarter"].tolist()

        # Full window starts at index (window-1) in the sorted quarter list
        if len(all_qs) > 0:
            first_full_q = all_qs[window - 1] if len(all_qs) >= window else all_qs[-1]
        else:
            first_full_q = None

        partial = sub[sub["Quarter"] < first_full_q] if first_full_q else sub.iloc[:0]
        full    = sub[sub["Quarter"] >= first_full_q] if first_full_q else sub

        # Full-window portion: solid line
        if not full.empty:
            fig.add_trace(go.Scatter(
                x=full["Quarter"],
                y=full["PRR"],
                mode="lines+markers",
                name=rxn,
                legendgroup=rxn,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6),
                showlegend=True,
                hovertemplate=(
                    f"<b>{rxn}</b><br>"
                    "Quarter: %{x}<br>"
                    "PRR: %{y:.2f} (full 4-quarter window)"
                    "<extra></extra>"
                ),
            ))

        # Partial-window portion: dashed line with hollow markers + tooltip warning
        if not partial.empty:
            fig.add_trace(go.Scatter(
                x=partial["Quarter"],
                y=partial["PRR"],
                mode="lines+markers",
                name=f"{rxn} (partial window)",
                legendgroup=rxn,
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(color="white", size=7,
                            line=dict(color=color, width=2)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{rxn}</b><br>"
                    "Quarter: %{x}<br>"
                    "PRR: %{y:.2f}<br>"
                    "<i>⚠ Partial window — fewer than 4 quarters of data.<br>"
                    "Estimates are noisier; early spikes may not indicate Weber effect.</i>"
                    "<extra></extra>"
                ),
            ))

    # FDA threshold reference
    fig.add_hline(
        y=2.0, line_width=1.5, line_dash="dash", line_color="#791F1F",
        annotation_text="PRR=2", annotation_font_size=9,
        annotation_position="right",
    )

    fig.update_layout(
        title=dict(
            text=f"Rolling PRR — {drug} (up to {window}-quarter window · dashed = partial window)",
            font=dict(size=15, color="#3C3A35"), x=0.0, xanchor="left",
        ),
        xaxis=dict(
            title="Calendar Quarter",
            tickangle=-45,
            showgrid=True, gridcolor="#F0EFEB",
        ),
        yaxis=dict(
            title="Proportional Reporting Ratio (PRR)",
            showgrid=True, gridcolor="#F0EFEB",
            rangemode="tozero",
        ),
        legend=dict(
            title="Reaction",
            orientation="v",
            x=1.01, y=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#3C3A35"),
        height=400,
        margin=dict(l=20, r=20, t=60, b=60),
        hovermode="x unified",
    )
    return fig
