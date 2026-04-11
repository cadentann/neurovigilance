import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, false_discovery_control
import requests
import io

st.set_page_config(page_title="NeuroVigilance | PRR Signal Detection", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #fff; color: #1a2332; }
.stApp { background: #fafbfc; }
.main .block-container { padding-top: 2rem; max-width: 1400px; }

[data-testid="stSidebar"] { background: #f8f9fb; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] label { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 500; }
[data-testid="stSidebar"] .stMarkdown p { color: #64748b; font-size: 0.78rem; }

.app-header { border-bottom: 2px solid #1a2332; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.app-title { font-family: 'Merriweather', serif; font-size: 1.6rem; font-weight: 700; color: #1a2332; letter-spacing: -0.02em; }
.app-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #94a3b8; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.3rem; }

.section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #94a3b8; letter-spacing: 0.14em; text-transform: uppercase; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.4rem; margin: 1.8rem 0 1rem 0; }

.method-box { background: #f1f5f9; border: 1px solid #e2e8f0; border-left: 3px solid #1a2332; padding: 1rem 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #475569; line-height: 1.8; border-radius: 0 4px 4px 0; margin: 1rem 0; }
.warning-box { background: #fefce8; border: 1px solid #fde68a; border-left: 3px solid #f59e0b; padding: 0.6rem 1rem; font-size: 0.75rem; color: #92400e; margin-bottom: 1rem; border-radius: 0 4px 4px 0; }
.confound-tag { display:inline-block; background:#fef3c7; color:#92400e; border:1px solid #fde68a; border-radius:3px; padding:1px 5px; font-size:0.65rem; font-weight:600; margin-left:4px; }
.ddi-tag { display:inline-block; background:#f0fdf4; color:#166534; border:1px solid #bbf7d0; border-radius:3px; padding:1px 5px; font-size:0.65rem; font-weight:600; margin-left:4px; }

[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #1a2332 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 500; }
[data-testid="stMetricDelta"] { font-size: 0.7rem !important; }

#MainMenu { visibility: hidden; } footer { visibility: hidden; }
hr { border-color: #e2e8f0; margin: 1.5rem 0; }
div[data-testid="column"] { padding: 0 0.5rem; }
.stDownloadButton button { background: #1a2332; color: #fff; border: none; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; padding: 0.4rem 1rem; border-radius: 3px; }
.stDownloadButton button:hover { background: #2d3f55; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DRUGS = ["Donepezil", "Rivastigmine", "Galantamine"]

# Confounding by indication — dementia drug population
CONFOUNDERS = {
    "Memory Impairment", "Dementia", "Dementia Alzheimer'S Type",
    "Cognitive Disorder", "Alzheimer'S Disease", "Agitation",
    "Confusional State", "Disorientation", "Fall", "Gait Disturbance",
    "Drug Ineffective", "Disease Progression", "Loss Of Personal Independence In Daily Activities",
}

# SMQ-style groupings (simplified cardiac + neuro)
SMQ_MAP = {
    "Bradycardia": "SMQ: Cardiac Conduction",
    "Sinus Bradycardia": "SMQ: Cardiac Conduction",
    "Heart Rate Decreased": "SMQ: Cardiac Conduction",
    "Atrioventricular Block": "SMQ: Cardiac Conduction",
    "Electrocardiogram Qt Prolonged": "SMQ: Cardiac Conduction",
    "Torsade De Pointes": "SMQ: Cardiac Conduction",
    "Syncope": "SMQ: Cardiac Conduction",
    "Seizure": "SMQ: Seizure/Convulsion",
    "Convulsion": "SMQ: Seizure/Convulsion",
    "Epilepsy": "SMQ: Seizure/Convulsion",
    "Generalised Tonic-Clonic Seizure": "SMQ: Seizure/Convulsion",
    "Nausea": "SMQ: Cholinergic GI",
    "Vomiting": "SMQ: Cholinergic GI",
    "Diarrhoea": "SMQ: Cholinergic GI",
    "Abdominal Pain": "SMQ: Cholinergic GI",
}

SERIOUSNESS_WEIGHTS = {
    "Death": 5, "Life-Threatening": 4, "Hospitalisation": 3,
    "Disability": 2, "Other": 1, "Unknown": 1, "Non-serious": 1, "Serious": 3,
}

PLOT_THEME = dict(
    plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
    font=dict(family='Inter', color='#475569'),
    xaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0', tickfont=dict(size=10)),
    yaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0', tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=40, b=20),
)

# ─── DATA LAYER ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_drug(drug_name: str) -> pd.DataFrame:
    url = "https://api.fda.gov/drug/event.json"
    params = {'search': f'patient.drug.medicinalproduct:("{drug_name}")', 'limit': 500}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get('results', [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for rep in results:
        patient = rep.get('patient', {})
        reactions = [rx.get('reactionmeddrapt', '').strip().title()
                     for rx in patient.get('reaction', []) if rx.get('reactionmeddrapt')]
        sex_raw = patient.get('patientsex')
        sex = {1:'Male','1':'Male',2:'Female','2':'Female'}.get(sex_raw, 'Unknown')
        age_raw = pd.to_numeric(patient.get('patientonsetage'), errors='coerce')
        if pd.notna(age_raw) and age_raw > 130: age_raw = age_raw / 365.0
        serious_raw = str(rep.get('seriousness', ''))
        serious = {'1':'Serious','2':'Non-serious'}.get(serious_raw, 'Unknown')
        date_str = rep.get('receivedate', '')
        try: year = int(date_str[:4]) if len(date_str) >= 4 else None
        except: year = None
        quarter = None
        if year and len(date_str) >= 6:
            try:
                month = int(date_str[4:6])
                quarter = f"{year}-Q{(month-1)//3+1}"
            except: pass

        rows.append({
            'drug': drug_name, 'sex': sex, 'age': age_raw,
            'serious': serious, 'year': year, 'quarter': quarter,
            'reactions': reactions,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_corpus():
    report_frames, missing = [], []
    for drug in DRUGS:
        df = fetch_drug(drug)
        if df.empty: missing.append(drug)
        else: report_frames.append(df)
    if not report_frames:
        return pd.DataFrame(), pd.DataFrame(), missing
    report_df = pd.concat(report_frames, ignore_index=True)
    rxn_df = (report_df.explode('reactions')
                       .rename(columns={'reactions': 'reaction'})
                       .pipe(lambda d: d[d['reaction'].notna() & (d['reaction'] != '')])
                       .reset_index(drop=True))
    return report_df, rxn_df, missing

# ─── PRR ENGINE ───────────────────────────────────────────────────────────────

def compute_prr(rxn_df: pd.DataFrame, drug: str, serious_filter: str = 'All') -> pd.DataFrame:
    if rxn_df.empty: return pd.DataFrame()
    df = rxn_df.copy()
    if serious_filter != 'All':
        df = df[df['serious'] == serious_filter]
        if df.empty: return pd.DataFrame()

    drug_df  = df[df['drug'] == drug]
    other_df = df[df['drug'] != drug]
    total_drug  = len(drug_df)
    total_other = len(other_df)
    if total_drug == 0 or total_other == 0: return pd.DataFrame()

    rows = []
    for rxn in drug_df['reaction'].unique():
        a = int((drug_df['reaction']  == rxn).sum())
        c = int((other_df['reaction'] == rxn).sum())

        # ── Minimum background gate (improvement #5) ──
        if c < 3: continue

        b = total_drug  - a
        d = total_other - c

        # ── Haldane-Anscombe correction (improvement #1) ──
        ah, bh, ch, dh = a+0.5, b+0.5, c+0.5, d+0.5

        prr = (ah / (ah + bh)) / (ch / (ch + dh))

        # ROR (improvement #2)
        ror = (ah * dh) / (bh * ch)

        # IC / BCPNN (improvement #3)
        O = a
        E = total_drug * (c / total_other) if total_other > 0 else 0
        ic = np.log2((O + 0.5) / (E + 0.5)) if (E + 0.5) > 0 else 0.0

        # Chi-squared with Yates
        try:
            chi2_val, p_val, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)
        except:
            chi2_val, p_val = 0.0, 1.0

        # 95% CI (log scale, corrected cells)
        se = np.sqrt(1/ah - 1/(ah+bh) + 1/ch - 1/(ch+dh))
        ci_lo = np.exp(np.log(prr) - 1.96 * se)
        ci_hi = np.exp(np.log(prr) + 1.96 * se)

        # Severity-weighted score (improvement #14)
        sev_weight = SERIOUSNESS_WEIGHTS.get(serious_filter, 1)
        sev_score  = round(prr * sev_weight, 2)

        rows.append({
            'Reaction': rxn, 'n': a, 'c_background': c,
            'PRR': round(prr, 3), 'ROR': round(ror, 3),
            'IC': round(ic, 3),
            'CI_lo': round(ci_lo, 3), 'CI_hi': round(ci_hi, 3),
            'Chi2': round(chi2_val, 2), 'p_raw': round(p_val, 6),
            'SevScore': sev_score,
            'SMQ': SMQ_MAP.get(rxn, ''),
            'Confound': rxn in CONFOUNDERS,
            'Signal_raw': (prr >= 2) and (chi2_val >= 4) and (a >= 3),
        })

    if not rows: return pd.DataFrame()
    result = pd.DataFrame(rows)

    # ── FDR correction (improvement #4) ──
    if len(result) > 1:
        result['p_adj'] = false_discovery_control(result['p_raw'].values, method='bh')
    else:
        result['p_adj'] = result['p_raw']

    result['Signal'] = (
        result['Signal_raw'] &
        (result['p_adj'] < 0.05)
    )
    return result.sort_values('PRR', ascending=False).reset_index(drop=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem;'>
      <div style='font-family:Merriweather,serif;font-size:1.05rem;font-weight:700;color:#1a2332;'>NeuroVigilance</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#94a3b8;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.2rem;'>PRR Signal Detection</div>
    </div>
    """, unsafe_allow_html=True)

    selected_drug   = st.selectbox("Target Drug", DRUGS)
    serious_filter  = st.selectbox("Seriousness Filter", ["All","Serious","Non-serious"])
    show_confounders = st.toggle("Show confounding-flagged AEs", value=False)
    st.markdown("---")
    st.markdown("**Signal Thresholds**")
    min_prr  = st.slider("Min PRR",  1.0, 10.0, 2.0, 0.5)
    min_chi2 = st.slider("Min Chi²", 1.0, 20.0, 4.0, 0.5)
    min_n    = st.slider("Min n",    1,   20,    3,   1)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#94a3b8;line-height:1.9;'>
    PRR = [a+½/(a+c+1)] / [b+½/(b+d+1)]<br>
    Haldane correction applied<br>
    FDR: Benjamini-Hochberg<br>
    Background gate: c ≥ 3<br>
    Evans et al. 2001
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD ─────────────────────────────────────────────────────────────────────

with st.spinner("Querying FDA FAERS..."):
    report_df, rxn_df, missing_drugs = build_corpus()

if rxn_df.empty:
    st.error("openFDA returned no data. API may be rate-limited — wait 60s and refresh.")
    st.stop()

if missing_drugs:
    st.markdown(f"<div class='warning-box'>⚠ Data unavailable for: {', '.join(missing_drugs)}. Cross-drug comparisons may be incomplete.</div>", unsafe_allow_html=True)

prr_df = compute_prr(rxn_df, selected_drug, serious_filter)

if prr_df.empty:
    st.error("No reactions computed. Try a different drug or filter.")
    st.stop()

# Apply filters
signals_df = prr_df[
    (prr_df['PRR']  >= min_prr)  &
    (prr_df['Chi2'] >= min_chi2) &
    (prr_df['n']    >= min_n)    &
    (prr_df['Signal'])
].copy()

if not show_confounders:
    signals_df = signals_df[~signals_df['Confound']]

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class='app-header'>
  <div class='app-title'>NeuroVigilance</div>
  <div class='app-subtitle'>Pharmacovigilance Signal Detection · FDA FAERS · Evans 2001 · Haldane Correction · BH-FDR</div>
</div>
""", unsafe_allow_html=True)

# ─── METRICS ──────────────────────────────────────────────────────────────────

drug_reports = report_df[report_df['drug'] == selected_drug]
n_reports    = len(drug_reports)
n_screened   = len(prr_df)
n_signals    = len(signals_df)
top_prr      = signals_df['PRR'].max() if n_signals > 0 else 0.0
n_bg         = len(rxn_df[rxn_df['drug'] != selected_drug])

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Target Drug",    selected_drug)
c2.metric("FAERS Reports",  f"{n_reports:,}")
c3.metric("AEs Screened",   f"{n_screened:,}")
c4.metric("Signals (FDR)",  f"{n_signals:,}", delta="BH-adjusted p<0.05")
c5.metric("Top PRR",        f"{top_prr:.2f}×" if top_prr > 0 else "—")

# ─── DYNAMIC METHODOLOGY BOX (improvement #17) ────────────────────────────────

st.markdown(f"""
<div class='method-box'>
  Analysis conducted on <strong>{n_reports:,}</strong> reports for <em>{selected_drug}</em> against a background corpus 
  of <strong>{n_bg:,}</strong> reports ({', '.join([d for d in DRUGS if d != selected_drug])}).
  Signals defined as PRR ≥ {min_prr}, χ²(Yates) ≥ {min_chi2}, n ≥ {min_n}, and BH-adjusted p &lt; 0.05.
  Haldane-Anscombe correction (add 0.5) applied to all contingency cells.
  Background gate: minimum 3 background reports required per reaction.
  Reactions flagged for confounding by indication are {'shown' if show_confounders else 'hidden'}.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── SIGNAL TABLE (st.dataframe with column_config) (improvement #16) ─────────

st.markdown("<div class='section-label'>Signal Detection Output</div>", unsafe_allow_html=True)

if signals_df.empty:
    st.warning("No signals meet current thresholds. Try lowering filters in the sidebar.")
else:
    display = signals_df[[
        'Reaction','n','PRR','ROR','IC','CI_lo','CI_hi','Chi2','p_raw','p_adj','SevScore','SMQ','Confound'
    ]].copy()
    display['95% CI'] = display.apply(lambda r: f"{r['CI_lo']:.2f} – {r['CI_hi']:.2f}", axis=1)
    display['Confound'] = display['Confound'].map({True:'⚠ Confound', False:''})
    display['SMQ'] = display['SMQ'].fillna('')

    export_cols = ['Reaction','n','PRR','ROR','IC','95% CI','Chi2','p_raw','p_adj','SevScore','SMQ','Confound']
    final_display = display[export_cols].rename(columns={
        'n':'Cases', 'p_raw':'p (raw)', 'p_adj':'p (BH-adj)', 'SevScore':'Sev. Score'
    })

    csv_buf = io.StringIO()
    final_display.to_csv(csv_buf, index=False)
    st.download_button("⬇ Export signals CSV", csv_buf.getvalue(),
                       f"signals_{selected_drug.lower()}.csv", "text/csv")

    st.dataframe(
        final_display,
        use_container_width=True,
        height=420,
        column_config={
            'PRR': st.column_config.ProgressColumn(
                'PRR', format="%.2f", min_value=0, max_value=float(signals_df['PRR'].max() or 10)),
            'ROR': st.column_config.NumberColumn('ROR', format="%.2f"),
            'IC':  st.column_config.NumberColumn('IC', format="%.3f"),
            'p (raw)':    st.column_config.NumberColumn(format="%.4f"),
            'p (BH-adj)': st.column_config.NumberColumn(format="%.4f"),
            'Sev. Score': st.column_config.NumberColumn(format="%.1f"),
        }
    )

# ─── VOLCANO PLOT (improvement #11) ───────────────────────────────────────────

st.markdown("<div class='section-label'>Volcano Plot — log₂(PRR) × −log₁₀(p adj)</div>", unsafe_allow_html=True)

vplot_df = prr_df[prr_df['p_adj'] > 0].copy()
vplot_df['log2_prr'] = np.log2(vplot_df['PRR'].clip(lower=0.01))
vplot_df['neg_logp']  = -np.log10(vplot_df['p_adj'].clip(lower=1e-10))
vplot_df['category']  = 'Sub-threshold'
vplot_df.loc[vplot_df['Signal'] & ~vplot_df['Confound'], 'category'] = 'Signal'
vplot_df.loc[vplot_df['Signal'] &  vplot_df['Confound'], 'category'] = 'Signal (confound)'

color_map = {
    'Signal': '#dc2626',
    'Signal (confound)': '#f59e0b',
    'Sub-threshold': '#cbd5e1'
}

fig_v = px.scatter(
    vplot_df, x='log2_prr', y='neg_logp',
    color='category', hover_name='Reaction',
    color_discrete_map=color_map,
    hover_data={'PRR':':.2f','p_adj':':.4f','n':True,'category':False,'log2_prr':False,'neg_logp':False},
    size='n', size_max=16,
    labels={'log2_prr':'log₂(PRR)', 'neg_logp':'−log₁₀(p adj)'},
)
fig_v.add_vline(x=1,  line_dash="dash", line_color="#94a3b8", line_width=1,
                annotation_text="PRR=2", annotation_font_size=9, annotation_font_color="#94a3b8")
fig_v.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#94a3b8", line_width=1,
                annotation_text="p=0.05", annotation_font_size=9, annotation_font_color="#94a3b8",
                annotation_position="right")
fig_v.update_layout(
    **PLOT_THEME, height=420,
    legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e8f0', borderwidth=1),
    title=dict(text=f"Volcano Plot — {selected_drug}", font=dict(family='Merriweather', size=13, color='#1a2332')),
)
st.plotly_chart(fig_v, use_container_width=True)

# ─── TEMPORAL TREND (improvement #12) ─────────────────────────────────────────

st.markdown("<div class='section-label'>Temporal Signal Emergence — Top 5 Signals by Quarter</div>",
            unsafe_allow_html=True)

if not signals_df.empty:
    top5_rxns = signals_df.sort_values('PRR', ascending=False).head(5)['Reaction'].tolist()
    drug_rxn_q = rxn_df[rxn_df['drug'] == selected_drug].copy()
    other_rxn   = rxn_df[rxn_df['drug'] != selected_drug]

    quarters = sorted(drug_rxn_q['quarter'].dropna().unique())
    trend_rows = []
    for q in quarters:
        dq = drug_rxn_q[drug_rxn_q['quarter'] == q]
        od = other_rxn[other_rxn['quarter'] == q]
        td, to = len(dq), len(od)
        if td < 5 or to < 5: continue
        for rxn in top5_rxns:
            a = int((dq['reaction'] == rxn).sum())
            c = int((od['reaction'] == rxn).sum())
            if c < 1: continue
            ah,bh,ch,dh = a+0.5, td-a+0.5, c+0.5, to-c+0.5
            prr_q = (ah/(ah+bh)) / (ch/(ch+dh))
            trend_rows.append({'Quarter': q, 'Reaction': rxn, 'PRR': round(prr_q, 2)})

    if trend_rows:
        trend_df = pd.DataFrame(trend_rows)
        fig_t = px.line(
            trend_df, x='Quarter', y='PRR', color='Reaction',
            markers=True,
            color_discrete_sequence=['#dc2626','#2563eb','#16a34a','#9333ea','#ea580c'],
        )
        fig_t.add_hline(y=2, line_dash="dot", line_color="#cbd5e1", line_width=1,
                        annotation_text="PRR=2", annotation_font_size=8)
        fig_t.update_layout(
            **PLOT_THEME, height=360,
            title=dict(text=f"PRR Over Time — {selected_drug}", font=dict(family='Merriweather', size=13, color='#1a2332')),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            legend=dict(font=dict(size=9), bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e8f0'),
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Insufficient temporal data for trend analysis.")
else:
    st.info("No signals to trend.")

# ─── SUNBURST (improvement #13) ───────────────────────────────────────────────

st.markdown("<div class='section-label'>Demographic Sunburst — Drug → Sex → Age Group → Seriousness</div>",
            unsafe_allow_html=True)

sb_col1, sb_col2 = st.columns([2,1])

with sb_col1:
    drug_rep = report_df[report_df['drug'] == selected_drug].copy()
    drug_rep['age_group'] = pd.cut(
        drug_rep['age'], bins=[0,40,60,75,90,200],
        labels=['<40','40-60','60-75','75-90','90+']
    ).astype(str).replace('nan','Unknown')

    sb_df = (drug_rep.groupby(['sex','age_group','serious'])
                     .size().reset_index(name='count'))
    sb_df = sb_df[sb_df['count'] > 0]

    if not sb_df.empty:
        fig_sb = px.sunburst(
            sb_df, path=['sex','age_group','serious'], values='count',
            color='serious',
            color_discrete_map={'Serious':'#dc2626','Non-serious':'#2563eb','Unknown':'#cbd5e1'},
            title=f"Demographics — {selected_drug}"
        )
        fig_sb.update_traces(textfont_size=10)
        fig_sb.update_layout(
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            title=dict(font=dict(family='Merriweather', size=13, color='#1a2332')),
            margin=dict(l=0,r=0,t=40,b=0), height=380,
        )
        st.plotly_chart(fig_sb, use_container_width=True)
    else:
        st.info("Insufficient demographic data.")

with sb_col2:
    # Age distribution
    age_data = drug_rep['age'].dropna()
    if not age_data.empty:
        fig_age = px.histogram(
            drug_rep, x='age', nbins=20, color='sex',
            color_discrete_map={'Male':'#2563eb','Female':'#dc2626','Unknown':'#cbd5e1'},
            barmode='overlay', opacity=0.75,
        )
        fig_age.update_layout(
            **PLOT_THEME, height=180,
            title=dict(text="Age Distribution", font=dict(size=11, color='#1a2332')),
            showlegend=True,
            legend=dict(font=dict(size=9)),
            xaxis_title="Age (years)", yaxis_title="Count",
            margin=dict(l=10,r=10,t=35,b=20),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Seriousness bar
    ser_counts = drug_rep['serious'].value_counts().reset_index()
    ser_counts.columns = ['Category','Count']
    fig_ser = px.bar(
        ser_counts, x='Category', y='Count',
        color='Category',
        color_discrete_map={'Serious':'#dc2626','Non-serious':'#2563eb','Unknown':'#cbd5e1'},
    )
    fig_ser.update_layout(
        **PLOT_THEME, height=165,
        title=dict(text="Seriousness", font=dict(size=11, color='#1a2332')),
        showlegend=False, xaxis_title='', yaxis_title='Reports',
        margin=dict(l=10,r=10,t=35,b=10),
    )
    st.plotly_chart(fig_ser, use_container_width=True)

# ─── CROSS-DRUG HEATMAP ───────────────────────────────────────────────────────

st.markdown("<div class='section-label'>Cross-Drug PRR Heatmap — Cholinesterase Inhibitor Class</div>",
            unsafe_allow_html=True)

with st.spinner("Computing cross-drug PRR..."):
    all_prr: dict[str, dict[str,float]] = {}
    all_sig_rxns: set[str] = set()
    for drug in DRUGS:
        df_d = compute_prr(rxn_df, drug, 'All')
        if not df_d.empty:
            all_prr[drug] = dict(zip(df_d['Reaction'], df_d['PRR']))
            all_sig_rxns.update(df_d[df_d['Signal']]['Reaction'].tolist())

if all_sig_rxns and all_prr:
    rxn_list = sorted(all_sig_rxns)
    z = [[min(all_prr.get(drug, {}).get(rxn, 0.0), 25.0) for rxn in rxn_list]
         for drug in DRUGS]

    fig_h = go.Figure(go.Heatmap(
        z=z, x=rxn_list, y=DRUGS,
        colorscale=[[0,'#f8fafc'],[0.1,'#dbeafe'],[0.4,'#3b82f6'],[0.7,'#dc2626'],[1,'#7f1d1d']],
        zmin=0, zmax=25,
        colorbar=dict(title=dict(text='PRR', font=dict(size=10)),
                      tickfont=dict(size=9), len=0.8),
        hovertemplate='<b>%{x}</b><br>%{y}: PRR = %{z:.2f}<extra></extra>',
    ))
    fig_h.update_layout(
        **PLOT_THEME, height=220,
        title=dict(text="PRR Heatmap — Shared Signal Reactions × Drug",
                   font=dict(family='Merriweather', size=13, color='#1a2332')),
        xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=120),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    class_sigs = [rxn for rxn in all_sig_rxns
                  if sum(1 for d in DRUGS if all_prr.get(d,{}).get(rxn,0) >= 2) >= 2]
    if class_sigs:
        st.markdown(f"""
        <div class='method-box'>
          <strong>Class-level signals</strong> (PRR≥2 in ≥2 drugs):
          {' · '.join(sorted(class_sigs)[:20])}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Insufficient data to render heatmap.")

# ─── FULL PRR TABLE ───────────────────────────────────────────────────────────

with st.expander("Full PRR Table — All Reactions Screened"):
    if not prr_df.empty:
        full_display = prr_df[['Reaction','n','c_background','PRR','ROR','IC','CI_lo','CI_hi',
                                'Chi2','p_raw','p_adj','Signal','Confound','SMQ']].copy()
        full_display['Confound'] = full_display['Confound'].map({True:'Yes',False:'No'})
        st.dataframe(full_display, use_container_width=True, height=400,
                     column_config={
                         'PRR': st.column_config.NumberColumn(format="%.3f"),
                         'p_adj': st.column_config.NumberColumn("p (BH-adj)", format="%.4f"),
                     })
        full_csv = io.StringIO()
        full_display.to_csv(full_csv, index=False)
        st.download_button("⬇ Export full PRR table", full_csv.getvalue(),
                           f"full_prr_{selected_drug.lower()}.csv", "text/csv")

# ─── LIMITATIONS ──────────────────────────────────────────────────────────────

st.markdown(f"""
<div class='method-box' style='margin-top:2rem;'>
  <strong>METHODOLOGY</strong><br>
  PRR computed per Evans SJ et al. (2001) Pharmacoepidemiology and Drug Safety 10:483–486.
  Haldane-Anscombe correction: 0.5 added to all contingency cells before computation.
  Multiple testing corrected via Benjamini-Hochberg FDR procedure (scipy.stats.false_discovery_control).
  Background exposure gate: reactions with &lt;3 background reports excluded.
  ROR = (a×d)/(b×c). IC = log₂((O+0.5)/(E+0.5)) per WHO BCPNN framework.<br><br>
  <strong>LIMITATIONS</strong><br>
  FAERS is a voluntary spontaneous reporting database.
  PRR/ROR measure disproportionate reporting frequency, not absolute incidence or causality.
  Subject to under-reporting, Weber effect inflation, and stimulated reporting bias.
  No exposure denominator available. Confounding by indication is probable in this elderly dementia population.
  Corpus limited to 500 reports per drug due to API constraints — signals may be unstable in small strata.
</div>
""", unsafe_allow_html=True)
