"""
viz/volcano.py — Volcano plot: log₂(PRR) vs −log₁₀(BH-adjusted p-value).

Classic pharmacovigilance two-dimensional signal overview.
Upper-right quadrant = strong, significant signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import TIER_COLORS


def volcano_plot(res: pd.DataFrame, drug: str) -> go.Figure:
    """
    Volcano plot for a drug's disproportionality result table.

    X-axis : log₂(PRR)  — effect size
    Y-axis : −log₁₀(p_adj BH-FDR) — statistical evidence
    Color  : Tier (STRONG / MODERATE / WATCH / NONE)
    Size   : sqrt(n) — proportional to report count
    Labels : Top-10 signals by composite score

    Returns
    -------
    plotly Figure
    """
    df = res.copy()
    df = df[df["PRR"] > 0].copy()
    df["log2_PRR"]  = np.log2(df["PRR"].clip(lower=1e-3))
    df["neg_log_p"] = -np.log10(df["p_adj"].clip(lower=1e-12))
    df["marker_sz"] = (np.sqrt(df["n"].clip(lower=1)) * 2.5).clip(upper=30)

    fig = go.Figure()

    # Plot by tier so each gets its own legend entry
    for tier, color in TIER_COLORS.items():
        sub = df[df["Tier"] == tier]
        if sub.empty:
            continue

        fig.add_trace(go.Scatter(
            x=sub["log2_PRR"],
            y=sub["neg_log_p"],
            mode="markers",
            name=tier,
            marker=dict(
                color=color,
                size=sub["marker_sz"],
                opacity=0.75,
                line=dict(color="white", width=0.5),
            ),
            customdata=np.column_stack([
                sub["Reaction"],
                sub["PRR"].round(2),
                sub["p_adj"].round(5),
                sub["n"].astype(int),
                sub["EBGM"].round(2),
                sub["IC"].round(2),
                sub.get("Labeled", pd.Series(True, index=sub.index)).astype(str),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PRR: %{customdata[1]}×  |  p_adj: %{customdata[2]}<br>"
                "n=%{customdata[3]}  EBGM=%{customdata[4]}  IC=%{customdata[5]}<br>"
                "Labeled: %{customdata[6]}"
                "<extra></extra>"
            ),
        ))

    # Label top 10 signals — use Signal_Evans (Evans/EMA criterion) to match
    # the app's primary output recommendation. Using the conservative GPS+FDR
    # Signal column would label fewer reactions than the main table suggests
    # are signals, creating a visual inconsistency.
    signal_col = "Signal_Evans" if "Signal_Evans" in df.columns else "Signal"
    top10 = df[df[signal_col]].nlargest(10, "Composite") if signal_col in df.columns else df.nlargest(10, "Composite")
    for _, row in top10.iterrows():
        fig.add_annotation(
            x=row["log2_PRR"],
            y=row["neg_log_p"],
            text=row["Reaction"],
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor="#AAAAAA",
            font=dict(size=9, color="#3C3A35"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            ax=20, ay=-20,
        )

    # Vertical reference: PRR=2 threshold (log₂(2)=1)
    fig.add_vline(x=1.0, line_width=1.5, line_dash="dash",
                  line_color="#791F1F",
                  annotation_text="PRR=2", annotation_font_size=9,
                  annotation_position="top right")

    # Horizontal reference: BH significance threshold
    sig_y = -np.log10(0.05)
    fig.add_hline(y=sig_y, line_width=1.5, line_dash="dash",
                  line_color="#3C3489",
                  annotation_text="p_adj=0.05", annotation_font_size=9,
                  annotation_position="right")

    fig.update_layout(
        title=dict(text=f"Volcano Plot — {drug}",
                   font=dict(size=15, color="#3C3A35"), x=0.0, xanchor="left"),
        xaxis=dict(
            title="log₂(PRR)",
            showgrid=True, gridcolor="#F0EFEB",
            zeroline=True, zerolinecolor="#CCCCCC",
        ),
        yaxis=dict(
            title="−log₁₀(BH-adjusted p-value)",
            showgrid=True, gridcolor="#F0EFEB",
        ),
        legend=dict(
            title="Tier", orientation="v",
            x=1.01, y=1.0, bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#3C3A35"),
        height=480,
        margin=dict(l=20, r=20, t=60, b=50),
        hovermode="closest",
    )
    return fig
