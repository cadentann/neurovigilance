import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import requests
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroVigilance | PRR Signal Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background-color: #0a0e1a; color: #e0e6f0; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    [data-testid="stSidebar"] {
        background-color: #0d1120;
        border-right: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] label {
        color: #5577aa !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: #334466;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border-bottom: 1px solid #131d30;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
        margin-top: 2rem;
    }

    .signal-row {
        display: grid;
        grid-template-columns: 2.2fr 0.6fr 0.7fr 1fr 0.7fr 0.7fr 0.8fr;
        gap: 0;
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #0d1322;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        align-items: center;
    }
    .signal-row:hover { background: #0d1322; }

    .signal-header {
        background: #060a12;
        color: #334466;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-top: 1px solid #1e2d4a;
        border-bottom: 2px solid #1e2d4a;
    }

    .badge-strong  { display:inline-block; background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.3);  border-radius:2px; padding:1px 6px; font-size:0.68rem; font-weight:600; }
    .badge-moderate{ display:inline-block; background:rgba(234,179,8,0.12);  color:#eab308; border:1px solid rgba(234,179,8,0.25);  border-radius:2px; padding:1px 6px; font-size:0.68rem; font-weight:600; }
    .badge-watch   { display:inline-block; background:rgba(59,130,246,0.12); color:#60a5fa; border:1px solid rgba(59,130,246,0.25); border-radius:2px; padding:1px 6px; font-size:0.68rem; font-weight:600; }

    .disclaimer {
        background:#0a1020; border:1px solid #1a2540; border-left:3px solid #334466;
        padding:0.8rem 1rem; font-size:0.71rem; color:#445577;
        font-family:'IBM Plex Mono',monospace; line-height:1.7; margin-top:2rem;
    }

    .warning-box {
        background:rgba(234,179,8,0.06); border:1px solid rgba(234,179,8,0.2);
        border-left:3px solid #eab308; padding:0.6rem 1rem;
        font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#92781a;
        margin-bottom:1rem;
    }

    [data-testid="stMetricValue"]  { font-family:'IBM Plex Mono',monospace !important; color:#e0e6f0 !important; }
    [data-testid="stMetricLabel"]  { color:#5577aa !important; font-size:0.7rem !important; text-transform:uppercase; letter-spacing:0.08em; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    hr { border-color:#131d30; margin:1.5rem 0; }
    div[data-testid="column"] { padding: 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DRUGS = ["Donepezil", "Rivastigmine", "Galantamine"]

SERIOUSNESS_MAP = {
    '1': 'Serious',   1: 'Serious',
    '2': 'Non-serious', 2: 'Non-serious',
}

PLOT_BG     = '#0a0e1a'
PLOT_PAPER  = '#0a0e1a'
GRID_COLOR  = '#131d30'
TEXT_COLOR  = '#8899bb'
FONT        = 'IBM Plex Mono'

# ─────────────────────────────────────────────
# DATA LAYER  (single fetch per drug, cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_drug(drug_name: str) -> pd.DataFrame:
    """Fetch up to 500 reports for one drug from openFDA. Returns report-level df."""
    url = "https://api.fda.gov/drug/event.json"
    params = {
        'search': f'patient.drug.medicinalproduct:("{drug_name}")',
        'limit': 500
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get('results', [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for rep in results:
        patient  = rep.get('patient', {})
        reactions = [rx.get('reactionmeddrapt', '').strip().title()
                     for rx in patient.get('reaction', [])
                     if rx.get('reactionmeddrapt')]

        sex_raw = patient.get('patientsex')
        sex = {1:'Male','1':'Male',2:'Female','2':'Female'}.get(sex_raw, 'Unknown')

        age_raw = pd.to_numeric(patient.get('patientonsetage'), errors='coerce')
        if pd.notna(age_raw) and age_raw > 130:
            age_raw = age_raw / 365.0

        serious_raw = rep.get('seriousness')
        serious = SERIOUSNESS_MAP.get(serious_raw, 'Unknown')

        date_str = rep.get('receivedate', '')
        try:
            year = int(date_str[:4]) if len(date_str) >= 4 else None
        except Exception:
            year = None

        rows.append({
            'drug':      drug_name,
            'sex':       sex,
            'age':       age_raw,
            'serious':   serious,
            'year':      year,
            'reactions': reactions,   # list — exploded later
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_corpus() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns:
        report_df  — one row per report (for demographics)
        rxn_df     — one row per (report × reaction)
        missing    — drugs that returned empty
    """
    report_frames, missing = [], []
    for drug in DRUGS:
        df = fetch_drug(drug)
        if df.empty:
            missing.append(drug)
        else:
            report_frames.append(df)

    if not report_frames:
        return pd.DataFrame(), pd.DataFrame(), missing

    report_df = pd.concat(report_frames, ignore_index=True)

    # Explode reactions
    rxn_df = report_df.copy()
    rxn_df = rxn_df.explode('reactions').rename(columns={'reactions': 'reaction'})
    rxn_df = rxn_df[rxn_df['reaction'].notna() & (rxn_df['reaction'] != '')]
    rxn_df = rxn_df.reset_index(drop=True)

    return report_df, rxn_df, missing


# ─────────────────────────────────────────────
# PRR ENGINE
# ─────────────────────────────────────────────

def compute_prr(rxn_df: pd.DataFrame, drug: str,
                serious_filter: str = 'All') -> pd.DataFrame:
    """
    Compute PRR for every reaction reported for `drug` vs. all other drugs.
    serious_filter: 'All' | 'Serious' | 'Non-serious'
    Returns full PRR table sorted by PRR descending.
    """
    if rxn_df.empty:
        return pd.DataFrame()

    df = rxn_df.copy()
    if serious_filter != 'All':
        df = df[df['serious'] == serious_filter]
        if df.empty:
            return pd.DataFrame()

    drug_df  = df[df['drug'] == drug]
    other_df = df[df['drug'] != drug]

    total_drug  = len(drug_df)
    total_other = len(other_df)

    if total_drug == 0 or total_other == 0:
        return pd.DataFrame()

    reactions = drug_df['reaction'].unique()
    rows = []

    for rxn in reactions:
        a = int((drug_df['reaction']  == rxn).sum())
        c = int((other_df['reaction'] == rxn).sum())
        b = total_drug  - a
        d = total_other - c

        if a == 0:
            continue

        prop_drug  = a / total_drug
        prop_other = c / total_other if total_other > 0 else 0

        prr = prop_drug / prop_other if prop_other > 0 else np.inf

        # Chi-squared (Yates correction)
        try:
            chi2, p_val, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)
        except Exception:
            chi2, p_val = 0.0, 1.0

        # 95% CI on log(PRR)
        ci_lo = ci_hi = np.nan
        if a > 0 and b > 0 and c > 0 and d > 0 and prr != np.inf:
            se = np.sqrt(1/a - 1/total_drug + 1/c - 1/total_other)
            ci_lo = np.exp(np.log(prr) - 1.96 * se)
            ci_hi = np.exp(np.log(prr) + 1.96 * se)

        signal = (prr >= 2) and (chi2 >= 4) and (a >= 3)

        rows.append({
            'Reaction':   rxn,
            'n':          a,
            'PRR':        round(float(prr), 2) if prr != np.inf else 999.0,
            'Chi2':       round(chi2, 2),
            'p_value':    round(p_val, 4),
            'CI_lo':      round(ci_lo, 2) if not np.isnan(ci_lo) else None,
            'CI_hi':      round(ci_hi, 2) if not np.isnan(ci_hi) else None,
            'Signal':     signal,
        })

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values('PRR', ascending=False)
              .reset_index(drop=True))


def tier(prr: float) -> tuple[str, str]:
    if prr >= 10:  return 'STRONG',   'badge-strong'
    if prr >= 5:   return 'MODERATE', 'badge-moderate'
    return 'WATCH', 'badge-watch'


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem 0;'>
      <div style='font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#2563eb;
                  letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.3rem;'>
        NeuroVigilance</div>
      <div style='font-family:IBM Plex Mono,monospace;font-size:1rem;
                  color:#c0cce0;font-weight:600;'>PRR Signal Detection</div>
      <div style='font-family:IBM Plex Mono,monospace;font-size:0.63rem;
                  color:#334466;margin-top:0.2rem;'>FDA FAERS · Cholinesterase Inhibitors</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Target Drug**")
    selected_drug = st.selectbox("Drug", DRUGS, label_visibility="collapsed")

    st.markdown("**Seriousness Filter**")
    serious_filter = st.selectbox(
        "Seriousness",
        ["All", "Serious", "Non-serious"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Signal Thresholds**")
    min_prr  = st.slider("Min PRR",  1.0, 10.0, 2.0, 0.5)
    min_chi2 = st.slider("Min Chi²", 1.0, 20.0, 4.0, 0.5)
    min_n    = st.slider("Min n",    1,   20,    3,   1)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;
                color:#334466;line-height:2;'>
      PRR = [a/(a+b)] / [c/(c+d)]<br>
      Signal: PRR≥2 ∧ χ²≥4 ∧ n≥3<br>
      Evans et al. 2001
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

with st.spinner("Querying FDA FAERS..."):
    report_df, rxn_df, missing_drugs = build_corpus()

if rxn_df.empty:
    st.error("openFDA returned no data. API may be rate-limited — wait 60s and refresh.")
    st.stop()

if missing_drugs:
    st.markdown(f"""
    <div class='warning-box'>
      ⚠ Data unavailable for: {', '.join(missing_drugs)}. 
      Cross-drug comparisons may be incomplete.
    </div>
    """, unsafe_allow_html=True)

# Compute PRR for selected drug under selected seriousness filter
prr_df = compute_prr(rxn_df, selected_drug, serious_filter)

# Apply user thresholds
if not prr_df.empty:
    signals_df = prr_df[
        (prr_df['PRR']  >= min_prr)  &
        (prr_df['Chi2'] >= min_chi2) &
        (prr_df['n']    >= min_n)    &
        (prr_df['Signal'])
    ].copy()
else:
    signals_df = pd.DataFrame()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <span style='font-family:IBM Plex Mono,monospace;font-size:1.4rem;
               font-weight:600;color:#e0e6f0;'>NeuroVigilance</span>
  <span style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
               color:#334466;margin-left:1rem;letter-spacing:0.08em;'>
    PHARMACOVIGILANCE PRR SIGNAL DETECTION · FDA FAERS · EVANS 2001
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS  (single source of truth: report_df)
# ─────────────────────────────────────────────

drug_reports   = report_df[report_df['drug'] == selected_drug]
n_reports      = len(drug_reports)
n_ae_screened  = len(prr_df) if not prr_df.empty else 0
n_signals      = len(signals_df)
top_prr_val    = signals_df['PRR'].max() if n_signals > 0 else 0.0
serious_pct    = (
    (drug_reports['serious'] == 'Serious').sum() / n_reports * 100
    if n_reports > 0 else 0.0
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Target Drug",       selected_drug)
c2.metric("FAERS Reports",     f"{n_reports:,}")
c3.metric("AEs Screened",      f"{n_ae_screened:,}")
c4.metric("Signals Detected",  f"{n_signals:,}",
          delta="Evans criteria met" if n_signals > 0 else None)
c5.metric("Highest PRR",
          f"{top_prr_val:.1f}×" if top_prr_val > 0 else "—")

st.markdown("---")

# ─────────────────────────────────────────────
# SIGNAL TABLE
# ─────────────────────────────────────────────

st.markdown(
    f"<div class='section-header'>"
    f"Signal Detection Output — {selected_drug} vs. Background Corpus"
    f"{' · ' + serious_filter if serious_filter != 'All' else ''}"
    f"</div>",
    unsafe_allow_html=True
)

if signals_df.empty:
    st.warning("No signals meet current thresholds. Try lowering PRR / Chi² / n in the sidebar.")
else:
    # ── CSV export ──────────────────────────────
    export_df = signals_df.copy()
    export_df.columns = ['Reaction','n','PRR','Chi2','p-value','CI_lo','CI_hi','Signal']
    csv_buf = io.StringIO()
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇ Export signals as CSV",
        data=csv_buf.getvalue(),
        file_name=f"prr_signals_{selected_drug.lower()}_{serious_filter.lower().replace('-','_')}.csv",
        mime="text/csv",
    )

    # ── Rendered table ──────────────────────────
    st.markdown("""
    <div class='signal-row signal-header'>
      <span>Adverse Event (MedDRA PT)</span>
      <span>n</span>
      <span>PRR</span>
      <span>95% CI</span>
      <span>Chi²</span>
      <span>p</span>
      <span>Tier</span>
    </div>
    """, unsafe_allow_html=True)

    for _, row in signals_df.head(40).iterrows():
        tier_label, tier_cls = tier(row['PRR'])
        ci_str  = (f"{row['CI_lo']:.2f}–{row['CI_hi']:.2f}"
                   if row['CI_lo'] is not None else "—")
        prr_str = f"{row['PRR']:.2f}" if row['PRR'] < 999 else ">999"
        st.markdown(f"""
        <div class='signal-row'>
          <span style='color:#c0cce0;font-weight:500;'>{row['Reaction']}</span>
          <span style='color:#7799bb;'>{int(row['n'])}</span>
          <span style='color:#e0e6f0;font-weight:600;'>{prr_str}</span>
          <span style='color:#445577;font-size:0.73rem;'>{ci_str}</span>
          <span style='color:#7799bb;'>{row['Chi2']:.1f}</span>
          <span style='color:#445577;'>{row['p_value']:.4f}</span>
          <span><span class='{tier_cls}'>{tier_label}</span></span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

st.markdown("<div class='section-header'>Signal Landscape</div>",
            unsafe_allow_html=True)

v1, v2 = st.columns([3, 2])

# ── Bar: Top PRR signals ─────────────────────
with v1:
    if not signals_df.empty:
        top20 = signals_df.head(20).copy()
        top20['color'] = top20['PRR'].apply(
            lambda x: '#ef4444' if x >= 10 else ('#eab308' if x >= 5 else '#3b82f6')
        )
        fig_bar = go.Figure(go.Bar(
            x=top20['PRR'],
            y=top20['Reaction'],
            orientation='h',
            marker_color=top20['color'],
            marker_line_width=0,
            error_x=dict(
                type='data',
                symmetric=False,
                array=(top20['CI_hi'] - top20['PRR']).fillna(0).tolist(),
                arrayminus=(top20['PRR'] - top20['CI_lo']).fillna(0).tolist(),
                color='#334466',
                thickness=1.2,
                width=3,
            ),
            text=top20['PRR'].apply(lambda x: f"{x:.1f}×"),
            textposition='outside',
            textfont=dict(family=FONT, size=9, color='#445577'),
        ))
        fig_bar.add_vline(x=2, line_dash="dot", line_color="#334466", line_width=1,
                          annotation_text="threshold", annotation_font_size=8,
                          annotation_font_color="#334466")
        fig_bar.update_layout(
            title=dict(text=f"Top Signals — {selected_drug}",
                       font=dict(family=FONT, size=11, color=TEXT_COLOR)),
            xaxis=dict(title="PRR", gridcolor=GRID_COLOR, color=TEXT_COLOR,
                       tickfont=dict(family=FONT, size=9)),
            yaxis=dict(autorange='reversed', gridcolor=GRID_COLOR, color=TEXT_COLOR,
                       tickfont=dict(family=FONT, size=9), categoryorder='total ascending'),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
            margin=dict(l=10, r=70, t=40, b=20), height=430, showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Scatter: full PRR × Chi² space ───────────
with v2:
    if not prr_df.empty:
        scatter_df = prr_df[prr_df['PRR'] < 200].copy()   # show all, cap inf
        scatter_df['is_signal'] = scatter_df['Signal'].map(
            {True: 'Signal', False: 'Sub-threshold'}
        )
        fig_sc = px.scatter(
            scatter_df,
            x='PRR', y='Chi2',
            size='n',
            color='is_signal',
            hover_name='Reaction',
            hover_data={'PRR': ':.2f', 'Chi2': ':.1f', 'n': True, 'is_signal': False},
            color_discrete_map={'Signal': '#ef4444', 'Sub-threshold': '#1e3a5f'},
            size_max=20,
        )
        fig_sc.add_hline(y=4, line_dash="dot", line_color="#334466", line_width=1)
        fig_sc.add_vline(x=2, line_dash="dot", line_color="#334466", line_width=1)
        fig_sc.update_layout(
            title=dict(text="PRR × Chi² Signal Space",
                       font=dict(family=FONT, size=11, color=TEXT_COLOR)),
            xaxis=dict(title="PRR", gridcolor=GRID_COLOR, color=TEXT_COLOR,
                       tickfont=dict(family=FONT, size=9)),
            yaxis=dict(title="Chi²", gridcolor=GRID_COLOR, color=TEXT_COLOR,
                       tickfont=dict(family=FONT, size=9)),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
            margin=dict(l=10, r=10, t=40, b=20), height=430,
            legend=dict(font=dict(family=FONT, size=9, color=TEXT_COLOR),
                        bgcolor='rgba(0,0,0,0)', bordercolor='#131d30')
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────
# SERIOUSNESS STRATIFICATION
# ─────────────────────────────────────────────

st.markdown("<div class='section-header'>Seriousness Stratification</div>",
            unsafe_allow_html=True)

s1, s2 = st.columns(2)

with s1:
    # Pie: serious vs non-serious for selected drug
    drug_rep = report_df[report_df['drug'] == selected_drug]
    serious_counts = drug_rep['serious'].value_counts().reset_index()
    serious_counts.columns = ['Category', 'Count']
    fig_pie = px.pie(
        serious_counts,
        names='Category', values='Count',
        hole=0.45,
        color_discrete_sequence=['#ef4444', '#3b82f6', '#445577'],
        title=f"Report Seriousness — {selected_drug}"
    )
    fig_pie.update_traces(textfont_family=FONT, textfont_size=10)
    fig_pie.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        font=dict(family=FONT, color=TEXT_COLOR),
        title_font=dict(family=FONT, size=11, color=TEXT_COLOR),
        legend=dict(font=dict(family=FONT, size=9, color=TEXT_COLOR),
                    bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with s2:
    # PRR comparison: Serious-only vs All for top signals
    if not signals_df.empty:
        top10_rxns = signals_df.head(10)['Reaction'].tolist()

        prr_all     = compute_prr(rxn_df, selected_drug, 'All')
        prr_serious = compute_prr(rxn_df, selected_drug, 'Serious')

        if not prr_all.empty and not prr_serious.empty:
            all_filt  = prr_all[prr_all['Reaction'].isin(top10_rxns)][['Reaction','PRR']].assign(Stratum='All reports')
            ser_filt  = prr_serious[prr_serious['Reaction'].isin(top10_rxns)][['Reaction','PRR']].assign(Stratum='Serious only')
            comp_df   = pd.concat([all_filt, ser_filt], ignore_index=True)

            fig_strat = px.bar(
                comp_df, x='PRR', y='Reaction', color='Stratum',
                orientation='h', barmode='group',
                color_discrete_map={'All reports':'#3b82f6', 'Serious only':'#ef4444'},
                title="PRR: All vs Serious Reports"
            )
            fig_strat.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                font=dict(family=FONT, color=TEXT_COLOR),
                title_font=dict(family=FONT, size=11, color=TEXT_COLOR),
                xaxis=dict(title="PRR", gridcolor=GRID_COLOR, color=TEXT_COLOR,
                           tickfont=dict(family=FONT, size=9)),
                yaxis=dict(autorange='reversed', gridcolor=GRID_COLOR, color=TEXT_COLOR,
                           tickfont=dict(family=FONT, size=9), categoryorder='total ascending'),
                legend=dict(font=dict(family=FONT, size=9, color=TEXT_COLOR),
                            bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=40, b=10), height=280
            )
            st.plotly_chart(fig_strat, use_container_width=True)
        else:
            st.info("Insufficient serious-only reports for stratification.")
    else:
        st.info("No signals to stratify.")


# ─────────────────────────────────────────────
# CROSS-DRUG PRR HEATMAP
# ─────────────────────────────────────────────

st.markdown("<div class='section-header'>Cross-Drug PRR Heatmap — Cholinesterase Inhibitor Class</div>",
            unsafe_allow_html=True)

with st.spinner("Computing PRR for all three drugs..."):
    drug_prr_maps: dict[str, dict[str, float]] = {}
    for drug in DRUGS:
        df_d = compute_prr(rxn_df, drug, 'All')
        if not df_d.empty:
            drug_prr_maps[drug] = dict(zip(df_d['Reaction'], df_d['PRR']))

# Union of all signals from any drug
all_signal_rxns: set[str] = set()
for drug in DRUGS:
    df_d = compute_prr(rxn_df, drug, 'All')
    if not df_d.empty:
        sigs = df_d[df_d['Signal']]['Reaction'].tolist()
        all_signal_rxns.update(sigs)

if all_signal_rxns and drug_prr_maps:
    rxn_list = sorted(all_signal_rxns)

    heatmap_data = []
    for drug in DRUGS:
        row = []
        for rxn in rxn_list:
            val = drug_prr_maps.get(drug, {}).get(rxn, 0.0)
            row.append(min(val, 30.0))   # cap for colour scale legibility
        heatmap_data.append(row)

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data,
        x=rxn_list,
        y=DRUGS,
        colorscale=[
            [0.0,  '#0a0e1a'],
            [0.05, '#0d1a35'],
            [0.15, '#1e3a5f'],
            [0.35, '#1d4ed8'],
            [0.65, '#eab308'],
            [1.0,  '#ef4444'],
        ],
        zmin=0, zmax=30,
        colorbar=dict(
            title=dict(text='PRR', font=dict(family=FONT, size=10, color=TEXT_COLOR)),
            tickfont=dict(family=FONT, size=9, color=TEXT_COLOR),
            bgcolor='rgba(0,0,0,0)',
            outlinecolor='#131d30',
        ),
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>%{y}: PRR = %{z:.1f}<extra></extra>',
    ))
    fig_heat.update_layout(
        title=dict(text="PRR Heatmap — Signal Reactions × Drug",
                   font=dict(family=FONT, size=11, color=TEXT_COLOR)),
        xaxis=dict(tickangle=-45, tickfont=dict(family=FONT, size=8, color=TEXT_COLOR),
                   gridcolor=GRID_COLOR, color=TEXT_COLOR),
        yaxis=dict(tickfont=dict(family=FONT, size=10, color=TEXT_COLOR),
                   color=TEXT_COLOR),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        margin=dict(l=10, r=10, t=50, b=120),
        height=280,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Class-level signal callout
    class_signals = [
        rxn for rxn in all_signal_rxns
        if sum(
            1 for drug in DRUGS
            if compute_prr(rxn_df, drug, 'All') is not None
            and rxn in drug_prr_maps.get(drug, {})
            and drug_prr_maps[drug].get(rxn, 0) >= 2
        ) >= 2
    ]
    if class_signals:
        st.markdown(f"""
        <div style='background:#0a0e1a;border:1px solid #1a2540;border-left:3px solid #2563eb;
                    padding:0.7rem 1rem;font-family:IBM Plex Mono,monospace;
                    font-size:0.73rem;color:#5577aa;margin-top:0.5rem;'>
          <span style='color:#3b82f6;font-weight:600;'>CLASS-LEVEL SIGNALS</span>
          (PRR≥2 in ≥2 drugs): {' · '.join(sorted(class_signals)[:15])}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Insufficient data to render cross-drug heatmap.")


# ─────────────────────────────────────────────
# FULL PRR TABLE (expandable + searchable)
# ─────────────────────────────────────────────

with st.expander("Full PRR Table — All Reactions Screened"):
    if not prr_df.empty:
        display = prr_df.copy()
        display['95% CI'] = display.apply(
            lambda r: f"{r['CI_lo']:.2f}–{r['CI_hi']:.2f}"
                      if r['CI_lo'] is not None else "—",
            axis=1
        )
        display = display.rename(columns={
            'n': 'Cases (n)', 'Chi2': 'Chi²', 'p_value': 'p-value'
        })
        st.dataframe(
            display[['Reaction','Cases (n)','PRR','95% CI','Chi²','p-value','Signal']],
            use_container_width=True,
            height=400
        )

        # Full table export
        full_csv = io.StringIO()
        display.to_csv(full_csv, index=False)
        st.download_button(
            label="⬇ Export full PRR table as CSV",
            data=full_csv.getvalue(),
            file_name=f"prr_full_{selected_drug.lower()}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
# METHODOLOGY + LIMITATIONS
# ─────────────────────────────────────────────

st.markdown("""
<div class='disclaimer'>
  <strong style='color:#5577aa;'>METHODOLOGY</strong><br>
  PRR computed per Evans SJ et al. (2001) <em>Pharmacoepidemiology and Drug Safety</em> 10:483–486.
  PRR = [a/(a+b)] / [c/(c+d)]. Signal threshold: PRR≥2 ∧ χ²(Yates)≥4 ∧ n≥3.
  95% CI calculated on log scale: SE = √(1/a − 1/(a+b) + 1/c − 1/(c+d)).
  Corpus: donepezil, rivastigmine, galantamine (up to 500 FAERS reports each, live API).<br><br>
  <strong style='color:#5577aa;'>LIMITATIONS</strong><br>
  FAERS is a voluntary spontaneous reporting database. PRR measures disproportionate reporting frequency,
  not absolute incidence or population risk. Data subject to under-reporting, Weber effect inflation,
  and stimulated reporting bias. No exposure denominator available. Confounding by indication is probable
  in elderly dementia populations (falls, cardiac events, cognitive symptoms reflect underlying disease).
  PRR and χ² are measures of association, not causality. This tool is for research purposes only
  and does not constitute medical or regulatory advice.
</div>
""", unsafe_allow_html=True)
