"""
viz/forest.py — Forest plot: PRR + 95% CI for top drug-reaction signals.

Shows Haldane-corrected PRR with lognormal CIs, coloured by tier.
Reference lines at PRR=1 (null) and PRR=2 (Evans/EMA criterion — not the FDA criterion; FDA uses EB05≥2.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import TIER_COLORS


def forest_plot(
    res: pd.DataFrame,
    drug: str,
    max_reactions: int = 25,
    show_only_signals: bool = False,
) -> go.Figure:
    """
    Forest plot of PRR + 95% CI for the top reactions for a drug.

    Parameters
    ----------
    res              : compute_prr() result DataFrame
    drug             : Drug name (used in title)
    max_reactions    : Max reactions to display (sorted by PRR descending)
    show_only_signals: If True, restrict to rows where Signal=True

    Returns
    -------
    plotly Figure
    """
    df = res.copy()
    if show_only_signals:
        # Filter on Signal_Evans (Evans/EMA criterion), matching app guidance
        # that Signal_Evans is the recommended primary output. Using the more
        # conservative GPS+FDR Signal column would produce an empty or sparse
        # forest plot at API corpus sizes where EB05 is prior-dominated.
        sig_col = "Signal_Evans" if "Signal_Evans" in df.columns else "Signal" if "Signal" in df.columns else None
        if sig_col:
            df = df[df[sig_col]]

    df = df.nlargest(max_reactions, "PRR").sort_values("PRR")

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No reactions to display.", showarrow=False,
                           font=dict(size=14, color="#6B6760"))
        fig.update_layout(_base_layout(f"Forest Plot — {drug}"))
        return fig

    fig = go.Figure()

    for _, row in df.iterrows():
        rxn    = row["Reaction"]
        prr    = row["PRR"]
        ci_lo  = row["CI_lo"]
        ci_hi  = row["CI_hi"]
        tier   = row.get("Tier", "NONE")
        color  = TIER_COLORS.get(tier, "#6B6760")
        n      = int(row["n"])
        signal = bool(row.get("Signal_Evans", row.get("Signal", False)))
        novel  = signal and not bool(row.get("Labeled", True))

        label = rxn
        if novel:
            label = f"🆕 {rxn}"
        elif signal:
            label = f"● {rxn}"

        # Error bar line
        fig.add_trace(go.Scatter(
            x=[ci_lo, prr, ci_hi],
            y=[label, label, label],
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Point estimate diamond
        fig.add_trace(go.Scatter(
            x=[prr],
            y=[label],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=10,
                color=color,
                line=dict(color="white", width=1),
            ),
            name=tier,
            legendgroup=tier,
            showlegend=False,
            hovertemplate=(
                f"<b>{rxn}</b><br>"
                f"PRR: {prr:.2f} [{ci_lo:.2f}–{ci_hi:.2f}]<br>"
                f"n = {n}<br>"
                f"Tier: {tier}<br>"
                f"Signal: {'Yes' if signal else 'No'}"
                "<extra></extra>"
            ),
        ))

        # CI whisker caps
        for x_cap in [ci_lo, ci_hi]:
            fig.add_trace(go.Scatter(
                x=[x_cap, x_cap],
                y=[label, label],
                mode="markers",
                marker=dict(symbol="line-ns", size=8, color=color,
                            line=dict(color=color, width=1.5)),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Null line (PRR = 1)
    fig.add_vline(x=1.0, line_width=1, line_dash="solid",
                  line_color="#AAAAAA", annotation_text="PRR=1",
                  annotation_font_size=10, annotation_position="top")

    # PRR = 2 reference (Evans 2001 / EMA — NOT the FDA criterion; FDA uses EB05 ≥ 2.0)
    fig.add_vline(x=2.0, line_width=1.5, line_dash="dash",
                  line_color="#791F1F", annotation_text="PRR=2 (Evans/EMA)",
                  annotation_font_size=10, annotation_position="top")

    # Legend items per tier
    for tier, color in TIER_COLORS.items():
        if tier == "NONE":
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color=color),
            name=tier,
            legendgroup=tier,
        ))

    max_x = min(float(df["CI_hi"].max()) * 1.15, 200.0)

    layout = _base_layout(f"Forest Plot — {drug}")
    layout.update(dict(
        xaxis=dict(
            title="Proportional Reporting Ratio (PRR)",
            type="log",
            range=[np.log10(0.5), np.log10(max_x)],
            showgrid=True, gridcolor="#F0EFEB",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11),
            showgrid=False,
        ),
        height=max(350, 28 * len(df) + 80),
        legend=dict(
            title="Tier", orientation="v",
            x=1.01, y=1.0, bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=20, r=20, t=60, b=40),
    ))
    fig.update_layout(layout)
    return fig


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#3C3A35"), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#3C3A35"),
        hovermode="closest",
    )
