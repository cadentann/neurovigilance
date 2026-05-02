"""
viz/concordance.py — Three-framework concordance scatter: PRR × EBGM × IC.

DuMouchel's recommendation: a signal is robust when all three frameworks
(PRR, EBGM/GPS, BCPNN/IC) agree. This chart makes that agreement visible.

X-axis : PRR (Proportional Reporting Ratio)
Y-axis : EBGM (Empirical Bayes Geometric Mean)
Color  : IC point estimate (Information Component)
Size   : n (report count)
Shape  : Concordance (Full / Partial / Weak)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import TIER_COLORS


_CONCORD_SYMBOLS = {
    "Full":    "circle",
    "Partial": "square",
    "Weak":    "triangle-up",
    "None":    "x",
}


def concordance_scatter(res: pd.DataFrame, drug: str) -> go.Figure:
    """
    Three-framework concordance scatter plot.

    Parameters
    ----------
    res  : compute_prr() result DataFrame
    drug : Drug name (used in title)

    Returns
    -------
    plotly Figure
    """
    df = res.copy()
    df = df[(df["PRR"] > 0) & (df["EBGM"] > 0)].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data.", showarrow=False,
                           font=dict(size=14, color="#6B6760"))
        return fig

    df["marker_sz"]  = (np.sqrt(df["n"].clip(lower=1)) * 3.5).clip(2, 35)
    df["IC_clipped"] = df["IC"].clip(lower=-3, upper=6)

    # Iterate over all four concordance levels; skip empty subsets
    fig = go.Figure()

    for concord in ["Full", "Partial", "Weak", "None"]:
        if "Concordance" not in df.columns:
            # Concordance column always produced by compute_prr; this path
            # should never be reached. Treat all rows as "Full" would be wrong
            # (all reactions would be labeled Full concordance). Skip gracefully.
            continue
        sub = df[df["Concordance"] == concord]
        if sub.empty:
            continue

        # IC as colour is handled by Plotly's continuous colorscale (color=sub["IC_clipped"])
        # below — no manual per-point color computation needed here.

        symbol = _CONCORD_SYMBOLS.get(concord, "circle")
        signal_flags = sub.get("Signal_Evans", pd.Series(False, index=sub.index))

        fig.add_trace(go.Scatter(
            x=sub["PRR"],
            y=sub["EBGM"],
            mode="markers",
            name=f"{concord} concordance",
            marker=dict(
                symbol=symbol,
                size=sub["marker_sz"],
                color=sub["IC_clipped"],
                colorscale="RdBu",
                cmin=-2, cmax=5,
                opacity=0.8,
                line=dict(
                    color=["#222222" if s else "rgba(0,0,0,0.2)"
                           for s in signal_flags],
                    width=[2 if s else 0.5 for s in signal_flags],
                ),
                showscale=(concord == "Full"),
                colorbar=dict(
                    title=dict(text="IC", side="right"),
                    len=0.6, y=0.5,
                    tickfont=dict(size=9),
                ) if concord == "Full" else None,
            ),
            customdata=np.column_stack([
                sub["Reaction"],
                sub["PRR"].round(2),
                sub["EBGM"].round(2),
                sub.get("EB05", pd.Series(float("nan"), index=sub.index)).round(2),
                sub["IC"].round(2),
                sub["n"].astype(int),
                sub.get("Tier", pd.Series("?", index=sub.index)),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PRR: %{customdata[1]}  EBGM: %{customdata[2]}<br>"
                "EB05: %{customdata[3]} (FDA criterion ≥2.0)<br>"
                "IC: %{customdata[4]}  n=%{customdata[5]}<br>"
                "Tier: %{customdata[6]}"
                "<extra></extra>"
            ),
        ))

    # Diagonal reference: EBGM ≈ PRR (perfect agreement line)
    max_val = float(max(df["PRR"].max(), df["EBGM"].max(), 5.0))
    fig.add_trace(go.Scatter(
        x=[1, max_val], y=[1, max_val],
        mode="lines",
        name="Perfect agreement (EBGM≈PRR at large n)",
        line=dict(color="#AAAAAA", width=1, dash="dot"),
        showlegend=True,
        hoverinfo="skip",
    ))

    # Reference lines (PRR=2 = Evans/EMA; EBGM=2 = reference; FDA criterion is EB05≥2)
    fig.add_vline(x=2.0, line_width=1.2, line_dash="dash",
                  line_color="#791F1F", annotation_text="PRR=2 (Evans/EMA)",
                  annotation_font_size=9)
    fig.add_hline(y=2.0, line_width=1.2, line_dash="dash",
                  line_color="#3C3489", annotation_text="EBGM=2 (ref — FDA criterion is EB05≥2.0, shown in table)",
                  annotation_font_size=9, annotation_position="right")

    fig.update_layout(
        title=dict(
            text=f"Three-Framework Concordance — {drug}",
            font=dict(size=15, color="#3C3A35"), x=0.0, xanchor="left",
        ),
        xaxis=dict(
            title="PRR (Proportional Reporting Ratio)",
            type="log",
            showgrid=True, gridcolor="#F0EFEB",
        ),
        yaxis=dict(
            title="EBGM (Empirical Bayes Geometric Mean)",
            type="log",
            showgrid=True, gridcolor="#F0EFEB",
        ),
        legend=dict(
            orientation="v", x=1.12, y=1.0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#3C3A35"),
        height=480,
        margin=dict(l=20, r=100, t=60, b=50),
        hovermode="closest",
        annotations=[
            dict(
                text="Dot outline = Signal | Color = IC value",
                xref="paper", yref="paper",
                x=0.0, y=-0.08, showarrow=False,
                font=dict(size=10, color="#6B6760"),
            ),
        ],
    )
    return fig
