"""
NeuroVigilance v8 — Neurological Pharmacovigilance Signal Detection
FDA FAERS · PRR · EBGM · BCPNN/IC · FDR · Haldane · Weber

Drug classes (neuro-narrative):
  Cholinesterase Inhibitors  — Alzheimer's disease
  SSRIs                      — MDD / anxiety
  Atypical Antipsychotics    — Schizophrenia / bipolar / TRD

Statistical framework:
  Evans et al. 2001           — PRR
  DuMouchel 1999              — EBGM / GPS
  Bate 1998 / Noren 2006      — BCPNN IC / IC025 / IC975
  Benjamini & Hochberg 1995   — FDR (BH)
  Haldane 1940; Anscombe 1956 — +0.5 continuity correction
  Weber 1984                  — temporal reporting flag
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, false_discovery_control
from scipy.stats import gamma as gamma_dist
from scipy.stats import nbinom as scipy_nbinom
import requests
import io
import time

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

st.set_page_config(
    page_title="NeuroVigilance | Pharmacovigilance",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  font-weight: 400;
  font-size: 15px;
  color: #1E1C1A;
  background: #FDFCF9 !important;
}
.stApp { background: #FDFCF9 !important; }
.main .block-container { padding-top: 2rem; max-width: 1380px; background: #FDFCF9; }

[data-testid="stSidebar"] { background: #F6F4EF !important; border-right: 0.5px solid #D8D4CB !important; }
[data-testid="stSidebar"] * { color: #6B6760 !important; }
[data-testid="stSidebar"] label {
  font-family: 'DM Sans', sans-serif !important; font-size: 10px !important;
  font-weight: 500 !important; text-transform: uppercase !important;
  letter-spacing: 0.13em !important; color: #A09B94 !important;
}

[data-testid="metric-container"] {
  background: #fff !important; border: 0.5px solid #D8D4CB !important;
  border-radius: 14px !important; padding: 18px 20px !important;
  transition: border-color 0.15s ease;
}
[data-testid="metric-container"]:hover { border-color: #C4BEB3 !important; }
[data-testid="stMetricValue"] {
  font-family: 'EB Garamond', serif !important; font-size: 28px !important;
  font-weight: 400 !important; color: #1E1C1A !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'DM Sans', sans-serif !important; font-size: 10px !important;
  font-weight: 500 !important; text-transform: uppercase !important;
  letter-spacing: 0.13em !important; color: #A09B94 !important;
}

[data-testid="stTabs"] [role="tablist"] { border-bottom: 0.5px solid #D8D4CB; gap: 0; }
[data-testid="stTabs"] button[role="tab"] {
  font-family: 'DM Sans', sans-serif !important; font-size: 12px !important;
  font-weight: 400 !important; color: #6B6760 !important;
  border-bottom: 2px solid transparent !important; padding: 8px 16px !important;
  background: transparent !important; transition: color 0.15s ease;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #3C3489 !important; border-bottom: 2px solid #3C3489 !important; font-weight: 500 !important;
}

[data-testid="stDataFrame"] { border: 0.5px solid #D8D4CB !important; border-radius: 14px !important; overflow: hidden; }
[data-testid="stExpander"] { background: #fff !important; border: 0.5px solid #D8D4CB !important; border-radius: 14px !important; }

.stDownloadButton button {
  background: transparent !important; color: #3C3489 !important;
  border: 0.5px solid #7F77DD !important; border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
  font-weight: 500 !important; padding: 9px 20px !important; transition: all 0.15s ease !important;
}
.stDownloadButton button:hover { background: #F4F3FE !important; }
.stButton button {
  background: transparent !important; color: #6B6760 !important;
  border: 0.5px solid #D8D4CB !important; border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
  padding: 9px 20px !important; transition: all 0.15s ease !important;
}
.stButton button:hover { background: #F4F3FE !important; color: #3C3489 !important; border-color: #7F77DD !important; }

[data-baseweb="select"] > div { background: #fff !important; border: 0.5px solid #D8D4CB !important; border-radius: 10px !important; }
[data-baseweb="select"] span { color: #1E1C1A !important; font-family: 'DM Sans', sans-serif !important; }
[data-baseweb="popover"] { background: #fff !important; border: 0.5px solid #D8D4CB !important; }
[data-baseweb="menu"] li { color: #1E1C1A !important; font-size: 13px !important; }
[data-baseweb="menu"] li:hover { background: #F4F3FE !important; }

.nv-header {
  border-bottom: 0.5px solid #D8D4CB; padding-bottom: 24px; margin-bottom: 32px;
  display: flex; align-items: flex-end; justify-content: space-between;
}
.nv-wordmark { font-family: 'EB Garamond', serif; font-size: 34px; font-weight: 400; color: #1E1C1A; letter-spacing: -0.01em; line-height: 1; }
.nv-wordmark span { color: #3C3489; }
.nv-tagline { font-family: 'DM Sans', sans-serif; font-size: 11px; color: #B8B3AC; letter-spacing: 0.05em; margin-top: 8px; }
.nv-drug-badge { font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500; color: #3C3489; background: #EEEDFE; border-radius: 20px; padding: 3px 12px; letter-spacing: 0.05em; text-transform: uppercase; }

.section-label {
  font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500;
  color: #A09B94; letter-spacing: 0.13em; text-transform: uppercase;
  border-bottom: 0.5px solid #D8D4CB; padding-bottom: 8px; margin: 32px 0 16px 0;
}

.method-card {
  background: #fff; border: 0.5px solid #D8D4CB; border-left: 3px solid #7F77DD;
  border-radius: 0 14px 14px 0; padding: 16px 20px;
  font-family: 'DM Sans', sans-serif; font-size: 13px; color: #6B6760; line-height: 1.65; margin: 12px 0;
}
.method-card strong { color: #1E1C1A; font-weight: 500; }

.warn-card {
  background: #FAEEDA; border: 0.5px solid #D8D4CB; border-left: 3px solid #633806;
  border-radius: 0 14px 14px 0; padding: 12px 16px;
  font-family: 'DM Sans', sans-serif; font-size: 13px; color: #633806; margin-bottom: 16px;
}

.stat-card { background: #fff; border: 0.5px solid #D8D4CB; border-radius: 14px; padding: 18px 20px; transition: border-color 0.15s ease; }
.stat-card:hover { border-color: #C4BEB3; }
.stat-num { font-family: 'EB Garamond', serif; font-size: 28px; font-weight: 400; color: #1E1C1A; line-height: 1; }
.stat-label { font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500; color: #A09B94; letter-spacing: 0.13em; text-transform: uppercase; margin-top: 4px; }

.pill-purple { display:inline-block; background:#EEEDFE; color:#3C3489; border-radius:20px; padding:3px 9px; font-size:10px; font-weight:500; font-family:'DM Sans',sans-serif; }
.pill-green  { display:inline-block; background:#EAF3DE; color:#27500A; border-radius:20px; padding:3px 9px; font-size:10px; font-family:'DM Sans',sans-serif; }
.pill-amber  { display:inline-block; background:#FAEEDA; color:#633806; border-radius:20px; padding:3px 9px; font-size:10px; font-family:'DM Sans',sans-serif; }
.pill-red    { display:inline-block; background:#FCEBEB; color:#791F1F; border-radius:20px; padding:3px 9px; font-size:10px; font-family:'DM Sans',sans-serif; }
.pill-neutral{ display:inline-block; background:#F6F4EF; color:#6B6760; border-radius:20px; padding:3px 9px; font-size:10px; font-family:'DM Sans',sans-serif; }

.ct-table { font-family:'DM Sans',sans-serif; font-size:13px; border-collapse:collapse; width:100%; }
.ct-table th { background:#F6F4EF; color:#6B6760; font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:0.08em; padding:8px 12px; border:0.5px solid #D8D4CB; text-align:center; }
.ct-table td { padding:8px 12px; border:0.5px solid #D8D4CB; text-align:center; color:#1E1C1A; }
.ct-table td.hl { color:#3C3489; font-weight:500; }

#MainMenu { visibility:hidden; } footer { visibility:hidden; }
div[data-testid="column"] { padding: 0 6px; }
hr { border-color: #D8D4CB; border-width: 0.5px; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DRUG_CLASSES = {
    "Cholinesterase Inhibitors": {
        "drugs":  ["Donepezil","Rivastigmine","Galantamine"],
        "brands": {"Donepezil":"aricept","Rivastigmine":"exelon","Galantamine":"razadyne"},
        "indication": "Alzheimer's Disease / Dementia",
        "confounders": {
            "Memory Impairment","Dementia","Dementia Alzheimer'S Type","Cognitive Disorder",
            "Alzheimer'S Disease","Agitation","Confusional State","Disorientation","Fall",
            "Gait Disturbance","Drug Ineffective","Disease Progression",
            "Loss Of Personal Independence In Daily Activities",
        },
    },
    "SSRIs": {
        "drugs":  ["Fluoxetine","Sertraline","Escitalopram"],
        "brands": {"Fluoxetine":"prozac","Sertraline":"zoloft","Escitalopram":"lexapro"},
        "indication": "Major Depressive Disorder / Anxiety",
        "confounders": {
            "Depression","Major Depressive Disorder","Anxiety Disorder","Suicidal Ideation",
            "Insomnia","Drug Ineffective","Anxiety","Panic Attack","Depressed Mood","Mood Altered",
        },
    },
    "Atypical Antipsychotics": {
        "drugs":  ["Quetiapine","Olanzapine","Risperidone"],
        "brands": {"Quetiapine":"seroquel","Olanzapine":"zyprexa","Risperidone":"risperdal"},
        "indication": "Schizophrenia / Bipolar / Treatment-Resistant Depression",
        "confounders": {
            "Schizophrenia","Bipolar Disorder","Psychosis","Hallucination",
            "Drug Ineffective","Agitation","Aggression","Delusion",
        },
    },
}

SMQ_MAP = {
    "Bradycardia":"Cardiac","Sinus Bradycardia":"Cardiac","Heart Rate Decreased":"Cardiac",
    "Electrocardiogram Qt Prolonged":"Cardiac","Torsade De Pointes":"Cardiac","Syncope":"Cardiac",
    "Atrial Fibrillation":"Cardiac","Tachycardia":"Cardiac",
    "Seizure":"Seizure/CNS","Convulsion":"Seizure/CNS","Epilepsy":"Seizure/CNS",
    "Tremor":"Seizure/CNS","Dystonia":"Seizure/CNS","Extrapyramidal Disorder":"Seizure/CNS",
    "Tardive Dyskinesia":"Seizure/CNS","Akathisia":"Seizure/CNS",
    "Nausea":"GI/Cholinergic","Vomiting":"GI/Cholinergic","Diarrhoea":"GI/Cholinergic",
    "Abdominal Pain":"GI/Cholinergic","Decreased Appetite":"GI/Cholinergic","Constipation":"GI/Cholinergic",
    "Insomnia":"Neuropsychiatric","Depression":"Neuropsychiatric","Hallucination":"Neuropsychiatric",
    "Anxiety":"Neuropsychiatric","Mania":"Neuropsychiatric","Suicidal Ideation":"Neuropsychiatric",
    "Aggression":"Neuropsychiatric",
    "Weight Increased":"Metabolic","Weight Decreased":"Metabolic","Hyperglycaemia":"Metabolic",
    "Hyperlipidaemia":"Metabolic","Hyponatraemia":"Metabolic","Hyperprolactinaemia":"Metabolic",
    "Serotonin Syndrome":"Serotonin Synd.","Hyperthermia":"Serotonin Synd.","Myoclonus":"Serotonin Synd.",
    "Rash":"Dermatological","Pruritus":"Dermatological","Stevens-Johnson Syndrome":"Dermatological",
    "Death":"Fatal","Sudden Death":"Fatal","Cardiac Arrest":"Fatal",
}

TIER_COLORS = {"STRONG":"#791F1F","MODERATE":"#633806","WATCH":"#3C3489"}
TIER_BG     = {"STRONG":"#FCEBEB","MODERATE":"#FAEEDA","WATCH":"#EEEDFE"}
TIER_THRESH = [(8,"STRONG"),(3,"MODERATE"),(2,"WATCH")]

_BASE_LAYOUT = dict(
    plot_bgcolor="#FDFCF9", paper_bgcolor="#FDFCF9",
    font=dict(family="DM Sans",color="#6B6760",size=11),
    margin=dict(l=10,r=10,t=44,b=20),
)
_BASE_AXIS = dict(gridcolor="#EEECEA",linecolor="#D8D4CB",tickfont=dict(size=9,color="#A09B94"))

def _layout(**overrides):
    """Merge base layout with per-call overrides without duplicate kwarg errors."""
    merged = dict(_BASE_LAYOUT)
    # Deep-merge xaxis/yaxis so callers can extend without conflict
    if "xaxis" in overrides:
        merged["xaxis"] = {**_BASE_AXIS, **overrides.pop("xaxis")}
    else:
        merged["xaxis"] = dict(_BASE_AXIS)
    if "yaxis" in overrides:
        merged["yaxis"] = {**_BASE_AXIS, **overrides.pop("yaxis")}
    else:
        merged["yaxis"] = dict(_BASE_AXIS)
    merged.update(overrides)
    return merged

# Keep PLOT_THEME as alias for code that doesn't override axes
PLOT_THEME = _layout()

EBGM_A1,EBGM_B1 = 0.20,0.06
EBGM_A2,EBGM_B2 = 1.40,1.80
EBGM_W1 = 0.10

DRUG_COLORS = {
    "Donepezil":"#3C3489","Rivastigmine":"#7F77DD","Galantamine":"#27500A",
    "Fluoxetine":"#3C3489","Sertraline":"#7F77DD","Escitalopram":"#27500A",
    "Quetiapine":"#3C3489","Olanzapine":"#7F77DD","Risperidone":"#27500A",
}

def get_tier(prr):
    for t,l in TIER_THRESH:
        if prr>=t: return l
    return "WATCH"

def is_labeled(rxn, txt):
    if not txt: return True
    r=rxn.lower(); w=r.split()
    return r in txt or (w[0] in txt if w else False)

# ─── EBGM ─────────────────────────────────────────────────────────────────────
def _nbinom_lpmf(n,a,b,mu):
    p=b/(mu+b+1e-300); return float(scipy_nbinom.logpmf(int(n),a,p))

def ebgm_row(n,mu):
    eps=1e-300
    l1=_nbinom_lpmf(n,EBGM_A1,EBGM_B1,mu); l2=_nbinom_lpmf(n,EBGM_A2,EBGM_B2,mu)
    lw1=np.log(EBGM_W1+eps); lw2=np.log(1-EBGM_W1+eps)
    ld=np.logaddexp(lw1+l1,lw2+l2)
    w1=float(np.exp(lw1+l1-ld)); w2=1-w1
    pm1=(EBGM_A1+n)/(EBGM_B1+mu+eps); pm2=(EBGM_A2+n)/(EBGM_B2+mu+eps)
    em=float(np.exp(w1*np.log(pm1+eps)+w2*np.log(pm2+eps)))
    pmean=w1*pm1+w2*pm2
    v1=(EBGM_A1+n)/(EBGM_B1+mu+eps)**2; v2=(EBGM_A2+n)/(EBGM_B2+mu+eps)**2
    pv=max(w1*(v1+pm1**2)+w2*(v2+pm2**2)-pmean**2,1e-10)
    ga=pmean**2/pv; gb=pmean/pv
    try: e05=float(gamma_dist.ppf(0.05,ga,scale=1/gb))
    except: e05=em*0.5
    return round(em,3),round(e05,3)

# ─── BCPNN ────────────────────────────────────────────────────────────────────
def bcpnn_ic(a,b,c,d):
    N=a+b+c+d+1e-300
    p11=(a+.5)/(N+2); p1x=(a+b+1)/(N+2); px1=(a+c+1)/(N+2)
    ic=float(np.log2(p11/(p1x*px1+1e-300)))
    ls=np.log(2)**2
    var=((1-p11)/(p11*N*ls+1e-300)+(1-p1x)/(p1x*N*ls+1e-300)+(1-px1)/(px1*N*ls+1e-300))
    se=np.sqrt(max(var,1e-10))
    return round(ic,3),round(ic-1.96*se,3),round(ic+1.96*se,3)

# ─── DATA LAYER ───────────────────────────────────────────────────────────────
def _parse(rep,drug):
    p=rep.get("patient",{})
    rxns=[rx.get("reactionmeddrapt","").strip().title() for rx in p.get("reaction",[]) if rx.get("reactionmeddrapt")]
    sx=p.get("patientsex"); sex={1:"Male","1":"Male",2:"Female","2":"Female"}.get(sx,"Unknown")
    age=pd.to_numeric(p.get("patientonsetage"),errors="coerce")
    if pd.notna(age) and age>130: age=age/365.0
    sr=str(rep.get("seriousness","")); serious={"1":"Serious","2":"Non-serious"}.get(sr,"Unknown")
    ds=rep.get("receivedate",""); year=quarter=None
    try:
        if len(ds)>=4: year=int(ds[:4])
        if year and len(ds)>=6: month=int(ds[4:6]); quarter=f"{year}-Q{(month-1)//3+1}"
    except: pass
    return {"drug":drug,"sex":sex,"age":age,"serious":serious,"year":year,"quarter":quarter,"reactions":rxns}

@st.cache_data(show_spinner=False,ttl=3600)
def fetch_drug(drug,target=1000):
    url="https://api.fda.gov/drug/event.json"; rows=[]
    for skip in range(0,target,100):
        params={"search":f'patient.drug.medicinalproduct:("{drug}")',"limit":100,"skip":skip}
        try:
            r=requests.get(url,params=params,timeout=15)
            if r.status_code==404: break
            r.raise_for_status()
            res=r.json().get("results",[])
            if not res: break
            rows.extend(_parse(rep,drug) for rep in res); time.sleep(0.06)
        except: break
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False,ttl=86400)
def fetch_label(drug,brands):
    brand=brands.get(drug,drug.lower())
    for q in [f'openfda.brand_name:"{brand}"',f'openfda.generic_name:"{drug.lower()}"']:
        try:
            r=requests.get("https://api.fda.gov/drug/label.json",params={"search":q,"limit":1},timeout=10)
            if r.status_code==200:
                res=r.json().get("results",[])
                if res:
                    lbl=res[0]
                    parts=(lbl.get("adverse_reactions",[])+lbl.get("warnings",[])+lbl.get("precautions",[])+lbl.get("boxed_warning",[]))
                    return " ".join(parts).lower()
        except: continue
    return ""

# ─── PRR ENGINE ───────────────────────────────────────────────────────────────
def compute_prr(rxn_df,drug,confounders,serious_filter="All",label_text=""):
    if rxn_df.empty: return pd.DataFrame()
    df=rxn_df.copy()
    if serious_filter!="All":
        df=df[df["serious"]==serious_filter]
        if df.empty: return pd.DataFrame()
    ddf=df[df["drug"]==drug]; odf=df[df["drug"]!=drug]
    td,to=len(ddf),len(odf)
    if td==0 or to==0: return pd.DataFrame()
    rows=[]
    for rxn in ddf["reaction"].unique():
        a=int((ddf["reaction"]==rxn).sum()); c=int((odf["reaction"]==rxn).sum())
        if c<3: continue
        b=td-a; d=to-c
        ah,bh,ch,dh=a+.5,b+.5,c+.5,d+.5
        prr=(ah/(ah+bh))/(ch/(ch+dh)); ror=(ah*dh)/(bh*ch)
        E=td*(c/to) if to>0 else 0.0
        ic,ic025,ic975=bcpnn_ic(a,b,c,d)
        em,e05=ebgm_row(a,max(E,1e-6))
        try: chi2v,pv,_,_=chi2_contingency([[a,b],[c,d]],correction=True)
        except: chi2v,pv=0.0,1.0
        se=np.sqrt(1/ah-1/(ah+bh)+1/ch-1/(ch+dh))
        cilo=np.exp(np.log(prr)-1.96*se); cihi=np.exp(np.log(prr)+1.96*se)
        rows.append({"Reaction":rxn,"n":a,"b":b,"c_bg":c,"d":d,"td":td,"to":to,
                     "PRR":round(prr,3),"ROR":round(ror,3),
                     "IC":ic,"IC025":ic025,"IC975":ic975,"EBGM":em,"EB05":e05,
                     "CI_lo":round(cilo,3),"CI_hi":round(cihi,3),
                     "Chi2":round(chi2v,2),"p_raw":round(pv,6),
                     "SMQ":SMQ_MAP.get(rxn,"Other"),
                     "Confound":rxn in confounders,
                     "Labeled":is_labeled(rxn,label_text),
                     "Signal_raw":(prr>=2)and(chi2v>=4)and(a>=3)})
    if not rows: return pd.DataFrame()
    res=pd.DataFrame(rows)
    res["p_adj"]=(false_discovery_control(res["p_raw"].values,method="bh") if len(res)>1 else res["p_raw"])
    res["Signal"]=res["Signal_raw"]&(res["p_adj"]<0.05)&(res["EB05"]>=1.0)
    res["Tier"]=res["PRR"].apply(get_tier)
    def _ln(s): mx=s.max(); return np.log1p(s)/(np.log1p(mx)+1e-9) if mx>0 else s*0.0
    res["Composite"]=(0.30*_ln(res["PRR"])+0.20*_ln(res["IC"].clip(lower=0))+0.20*_ln(res["Chi2"])+0.15*_ln(res["EBGM"])+0.15*_ln(res["IC025"].clip(lower=0))).round(3)
    res["N_agree"]=((res["PRR"]>=2).astype(int)+(res["EBGM"]>=2).astype(int)+(res["IC"]>0).astype(int))
    res["Concordance"]=res["N_agree"].map({3:"Full",2:"Partial",1:"Weak",0:"None"})
    return res.sort_values("Composite",ascending=False).reset_index(drop=True)

def rolling_prr(rxn_df,drug,reactions,window=4):
    dq=rxn_df[rxn_df["drug"]==drug].copy(); oq=rxn_df[rxn_df["drug"]!=drug].copy()
    quarters=sorted(dq["quarter"].dropna().unique()); rows=[]
    for i,q in enumerate(quarters):
        wqs=quarters[max(0,i-window+1):i+1]
        dw=dq[dq["quarter"].isin(wqs)]; ow=oq[oq["quarter"].isin(wqs)]
        td,to=len(dw),len(ow)
        if td<5 or to<5: continue
        for rxn in reactions:
            a=int((dw["reaction"]==rxn).sum()); c=int((ow["reaction"]==rxn).sum())
            if c<1: continue
            ah=a+.5;bh=td-a+.5;ch=c+.5;dh=to-c+.5
            rows.append({"Quarter":q,"Reaction":rxn,"PRR":round((ah/(ah+bh))/(ch/(ch+dh)),2)})
    return pd.DataFrame(rows)

def weber_flag(rxn_df,drug,rxn):
    sub=rxn_df[(rxn_df["drug"]==drug)&(rxn_df["reaction"]==rxn)].dropna(subset=["year"])
    if len(sub)<8: return False,None
    mn=int(sub["year"].min()); pct=round(int((sub["year"]<=mn+2).sum())/len(sub)*100,1)
    return pct>=60.0,pct

# ─── VISUALIZATION BUILDERS ───────────────────────────────────────────────────
def forest_fig(sigs,drug):
    df=sigs.head(15).sort_values("PRR",ascending=True).copy()
    fig=go.Figure()
    for _,row in df.iterrows():
        color=TIER_COLORS.get(row["Tier"],"#3C3489"); rxn=row["Reaction"]
        fig.add_trace(go.Scatter(
            x=[row["CI_lo"],row["PRR"],row["CI_hi"]],y=[rxn]*3,
            mode="lines+markers",
            marker=dict(size=[0,10,0],color=[color]*3,symbol="diamond"),
            line=dict(color=color,width=2),showlegend=False,
            hovertemplate=(f"<b>{rxn}</b><br>PRR:{row['PRR']:.2f} [{row['CI_lo']:.2f}–{row['CI_hi']:.2f}]<br>"
                           f"EBGM:{row['EBGM']:.2f} EB05:{row['EB05']:.2f}<br>"
                           f"IC:{row['IC']:.2f} [{row['IC025']:.2f},{row['IC975']:.2f}]<br>"
                           f"n={int(row['n'])} Tier:{row['Tier']}<extra></extra>")))
    fig.add_vline(x=2,line_dash="dash",line_color="#D8D4CB",line_width=1.5,
                  annotation_text="PRR = 2",annotation_font_size=9,annotation_font_color="#A09B94")
    fig.add_vline(x=1,line_dash="dot",line_color="#EEECEA",line_width=1)
    fig.update_layout(**_layout(
        height=max(300,32*len(df)+80),
        title=dict(text=f"Forest Plot — {drug}  ·  PRR with 95% CI",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
        xaxis=dict(title="PRR (log scale)",type="log"),
        yaxis=dict(tickfont=dict(size=9,color="#6B6760")),
        margin=dict(l=10,r=30,t=50,b=30)))
    return fig

def volcano_fig(prr_df,drug):
    df=prr_df[prr_df["p_adj"]>0].copy()
    df["log2_prr"]=np.log2(df["PRR"].clip(lower=0.01))
    df["neg_logp"]=-np.log10(df["p_adj"].clip(lower=1e-10))
    df["cat"]="Sub-threshold"
    df.loc[df["Signal"]&~df["Confound"],"cat"]="Signal"
    df.loc[df["Signal"]& df["Confound"],"cat"]="Signal (confound)"
    fig=px.scatter(df,x="log2_prr",y="neg_logp",color="cat",hover_name="Reaction",
        color_discrete_map={"Signal":"#3C3489","Signal (confound)":"#633806","Sub-threshold":"#D8D4CB"},
        hover_data={"PRR":":.2f","p_adj":":.4f","EBGM":":.3f","n":True,"cat":False,"log2_prr":False,"neg_logp":False},
        size="n",size_max=14,labels={"log2_prr":"log₂(PRR)","neg_logp":"−log₁₀(p adj)"})
    fig.add_vline(x=1,line_dash="dash",line_color="#D8D4CB",line_width=1,annotation_text="PRR=2",annotation_font_size=9,annotation_font_color="#A09B94")
    fig.add_hline(y=-np.log10(0.05),line_dash="dash",line_color="#D8D4CB",line_width=1,annotation_text="p=0.05",annotation_font_size=9,annotation_font_color="#A09B94",annotation_position="right")
    fig.update_layout(**_layout(height=380,
        title=dict(text=f"Volcano — {drug}",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
        legend=dict(font=dict(size=10),bgcolor="rgba(253,252,249,0.95)",bordercolor="#D8D4CB",borderwidth=1)))
    return fig

def sankey_fig(sigs,drug):
    if sigs.empty: return None
    top=sigs.head(20).copy()
    smq_list=[s for s in top["SMQ"].unique() if s and s!="Other"]
    rxn_list=top["Reaction"].tolist()
    nodes=[drug]+smq_list+rxn_list; ni={n:i for i,n in enumerate(nodes)}
    src,tgt,val,col=[],[],[],[]
    for smq,total in top.groupby("SMQ")["n"].sum().items():
        if smq not in ni: continue
        src.append(ni[drug]); tgt.append(ni[smq]); val.append(int(total)); col.append("rgba(127,119,221,0.35)")
    for _,row in top.iterrows():
        smq=row["SMQ"]
        if smq not in ni or row["Reaction"] not in ni: continue
        src.append(ni[smq]); tgt.append(ni[row["Reaction"]]); val.append(int(row["n"]))
        col.append({"STRONG":"rgba(121,31,31,0.25)","MODERATE":"rgba(99,56,6,0.25)"}.get(row["Tier"],"rgba(60,52,137,0.2)"))
    nc=["#3C3489"]+["#7F77DD"]*len(smq_list)+[TIER_COLORS.get(top.loc[top["Reaction"]==r,"Tier"].iloc[0] if r in top["Reaction"].values else "WATCH","#A09B94") for r in rxn_list]
    fig=go.Figure(go.Sankey(
        node=dict(pad=14,thickness=14,line=dict(color="#D8D4CB",width=0.5),label=nodes,color=nc),
        link=dict(source=src,target=tgt,value=val,color=col)))
    fig.update_layout(**_layout(height=480,
        title=dict(text=f"Signal Flow — {drug} → System Organ Class → Reaction",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
        font=dict(size=9,family="DM Sans")))
    return fig

def threed_fig(all_dfs,drugs):
    rows=[]
    for drug in drugs:
        df=all_dfs.get(drug,pd.DataFrame())
        if df.empty: continue
        for _,row in df[df["Signal"]].iterrows():
            rows.append({"Drug":drug,"Reaction":row["Reaction"],"PRR":min(float(row["PRR"]),30.0),"EBGM":min(float(row["EBGM"]),20.0),"IC":float(row["IC"]),"n":int(row["n"]),"Tier":row["Tier"]})
    if not rows: return None
    df3=pd.DataFrame(rows); fig=go.Figure()
    for drug in drugs:
        sub=df3[df3["Drug"]==drug]
        if sub.empty: continue
        fig.add_trace(go.Scatter3d(
            x=sub["PRR"],y=sub["EBGM"],z=sub["IC"],mode="markers",name=drug,
            marker=dict(size=sub["n"].apply(lambda x:max(3,min(14,x**0.5))).tolist(),color=DRUG_COLORS.get(drug,"#6B6760"),opacity=0.82,line=dict(width=0.4,color="#FDFCF9")),
            text=sub["Reaction"],hovertemplate=f"<b>%{{text}}</b><br>{drug}<br>PRR:%{{x:.2f}} EBGM:%{{y:.2f}} IC:%{{z:.2f}}<extra></extra>"))
    fig.update_layout(paper_bgcolor="#FDFCF9",height=540,
        title=dict(text="3D Signal Space — PRR × EBGM × IC  (size ∝ √n)",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
        scene=dict(
            xaxis=dict(title="PRR",backgroundcolor="#F6F4EF",gridcolor="#D8D4CB",showbackground=True,tickfont=dict(size=8,color="#A09B94")),
            yaxis=dict(title="EBGM",backgroundcolor="#F6F4EF",gridcolor="#D8D4CB",showbackground=True,tickfont=dict(size=8,color="#A09B94")),
            zaxis=dict(title="IC",backgroundcolor="#F6F4EF",gridcolor="#D8D4CB",showbackground=True,tickfont=dict(size=8,color="#A09B94")),
            bgcolor="#FDFCF9",camera=dict(eye=dict(x=1.5,y=1.5,z=0.9))),
        legend=dict(font=dict(size=10,color="#6B6760"),bgcolor="rgba(253,252,249,0.95)",bordercolor="#D8D4CB",borderwidth=1),
        margin=dict(l=0,r=0,t=50,b=0))
    return fig

def network_fig(sigs,drug):
    if not NX_AVAILABLE or sigs.empty: return None
    top=sigs.head(25); G=nx.Graph(); G.add_node(drug,node_type="drug")
    for _,row in top.iterrows():
        rxn=row["Reaction"]
        G.add_node(rxn,node_type="reaction",prr=row["PRR"],tier=row["Tier"],n=row["n"],ebgm=row["EBGM"],labeled=row.get("Labeled",True))
        G.add_edge(drug,rxn,weight=float(row["PRR"]))
    pos=nx.spring_layout(G,seed=42,k=2.8)
    edge_traces=[]
    for u,v,data in G.edges(data=True):
        x0,y0=pos[u]; x1,y1=pos[v]; w=data["weight"]
        edge_traces.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
            line=dict(width=max(0.5,min(5,w/3)),color=f"rgba(127,119,221,{min(0.6,0.1+w/30)})"),hoverinfo="none",showlegend=False))
    nx_,ny_,nc_,ns_,nt_,nh_=[],[],[],[],[],[]
    for node,data in G.nodes(data=True):
        x,y=pos[node]; nx_.append(x); ny_.append(y)
        if data.get("node_type")=="drug":
            nc_.append("#3C3489"); ns_.append(28); nt_.append(f"<b>{node}</b>"); nh_.append(f"<b>{node}</b>")
        else:
            prr_=data.get("prr",1.0); t_=data.get("tier","WATCH"); n_=data.get("n",0); eb_=data.get("ebgm",0.0)
            nc_.append(TIER_COLORS.get(t_,"#A09B94")); ns_.append(max(8,min(22,6+n_**0.5)))
            nt_.append(node if len(node)<=14 else ""); lab_="✓" if data.get("labeled",True) else "🆕"
            nh_.append(f"<b>{node}</b><br>PRR={prr_:.2f} EBGM={eb_:.2f}<br>n={n_} {lab_}")
    node_t=go.Scatter(x=nx_,y=ny_,mode="markers+text",
        marker=dict(size=ns_,color=nc_,line=dict(width=1,color="#D8D4CB")),
        text=nt_,textposition="top center",textfont=dict(size=8,color="#6B6760",family="DM Sans"),
        hovertext=nh_,hoverinfo="text",showlegend=False)
    fig=go.Figure(data=edge_traces+[node_t])
    fig.update_layout(**_layout(height=500,
        title=dict(text=f"Signal Network — {drug}  ·  AE Bipartite Graph",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        margin=dict(l=0,r=0,t=50,b=0)))
    return fig

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px;border-bottom:0.5px solid #D8D4CB;margin-bottom:16px;'>
      <div style='font-family:"EB Garamond",serif;font-size:22px;font-weight:400;color:#1E1C1A;'>
        Neuro<span style="color:#3C3489;">Vigilance</span></div>
      <div style='font-family:"DM Sans",sans-serif;font-size:10px;font-weight:500;color:#A09B94;
                  letter-spacing:0.13em;text-transform:uppercase;margin-top:6px;'>
        FDA FAERS · PRR · EBGM · IC</div>
    </div>
    """,unsafe_allow_html=True)

    sel_class=st.selectbox("Drug Class",list(DRUG_CLASSES.keys()))
    ci=DRUG_CLASSES[sel_class]
    DRUGS=ci["drugs"]; CONFOUNDERS=ci["confounders"]; BRANDS=ci["brands"]

    sel_drug=st.selectbox("Target Drug",DRUGS)
    serious_flt=st.selectbox("Seriousness",["All","Serious","Non-serious"])
    show_conf=st.toggle("Show confounders",value=False)
    novel_only=st.toggle("Novel signals only",value=False)

    st.markdown("---")
    st.markdown("<span style='font-family:DM Sans,sans-serif;font-size:10px;font-weight:500;color:#A09B94;text-transform:uppercase;letter-spacing:0.13em;'>Thresholds</span>",unsafe_allow_html=True)
    min_prr =st.slider("Min PRR", 1.0,10.0,2.0,0.5)
    min_chi2=st.slider("Min χ²",  1.0,20.0,4.0,0.5)
    min_n   =st.slider("Min n",   1,  20,  3,  1)
    min_eb05=st.slider("Min EB05",0.0,5.0, 1.0,0.5)

    st.markdown("---")
    st.markdown("""<div style='font-family:DM Sans,sans-serif;font-size:12px;color:#A09B94;line-height:2;'>
    PRR · Evans 2001<br>EBGM · DuMouchel 1999<br>IC · Bate 1998 / Noren 2006<br>
    FDR · Benjamini-Hochberg<br>Haldane +0.5 correction<br>Background gate: c ≥ 3<br>
    Corpus: 1,000 reports/drug</div>""",unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
prog=st.progress(0); stat=st.empty()
report_frames,missing=[],[]
for i,drug in enumerate(DRUGS):
    stat.markdown(f"<span style='font-family:DM Sans,sans-serif;font-size:13px;color:#6B6760;'>Querying FAERS — {drug}…</span>",unsafe_allow_html=True)
    df=fetch_drug(drug)
    if df.empty: missing.append(drug)
    else: report_frames.append(df)
    prog.progress((i+1)/(len(DRUGS)+1))

stat.markdown("<span style='font-family:DM Sans,sans-serif;font-size:13px;color:#6B6760;'>Fetching FDA label…</span>",unsafe_allow_html=True)
label_text=fetch_label(sel_drug,BRANDS); label_ok=len(label_text)>0
prog.progress(1.0); stat.empty(); prog.empty()

if not report_frames:
    st.error("openFDA returned no data. API may be rate-limited — wait 60s and refresh."); st.stop()
if missing:
    st.markdown(f"<div class='warn-card'>⚠ Data unavailable for: {', '.join(missing)}.</div>",unsafe_allow_html=True)

report_df=pd.concat(report_frames,ignore_index=True)
rxn_df=(report_df.explode("reactions").rename(columns={"reactions":"reaction"})
        .pipe(lambda d:d[d["reaction"].notna()&(d["reaction"]!="")]).reset_index(drop=True))

prr_df=compute_prr(rxn_df,sel_drug,CONFOUNDERS,serious_flt,label_text)
if prr_df.empty: st.error("No reactions computed."); st.stop()

sigs=prr_df[(prr_df["PRR"]>=min_prr)&(prr_df["Chi2"]>=min_chi2)&(prr_df["n"]>=min_n)&(prr_df["EB05"]>=min_eb05)&prr_df["Signal"]].copy()
if not show_conf: sigs=sigs[~sigs["Confound"]]
if novel_only and label_ok: sigs=sigs[~sigs["Labeled"]]

if not sigs.empty:
    wf,wp=[],[]
    for rxn in sigs["Reaction"]:
        f,p=weber_flag(rxn_df,sel_drug,rxn); wf.append(f); wp.append(p)
    sigs["Weber"]=wf; sigs["WeberPct"]=wp

# ─── HEADER ───────────────────────────────────────────────────────────────────
dr=report_df[report_df["drug"]==sel_drug]
n_reports=len(dr); n_bg=len(rxn_df[rxn_df["drug"]!=sel_drug])
n_sigs=len(sigs); n_strong=int((sigs["Tier"]=="STRONG").sum()) if n_sigs else 0
n_novel=int((~sigs["Labeled"]).sum()) if (n_sigs and label_ok) else 0
top_eb05=float(sigs["EB05"].max()) if n_sigs else 0.0
n_full=int((sigs["N_agree"]==3).sum()) if n_sigs else 0

st.markdown(f"""
<div class='nv-header'>
  <div>
    <div class='nv-wordmark'>Neuro<span>Vigilance</span></div>
    <div class='nv-tagline'>PRR · EBGM · BCPNN/IC · FDR · Haldane · Weber · FDA FAERS · v8</div>
  </div>
  <div class='nv-drug-badge'>{sel_drug} · {sel_class}</div>
</div>
""",unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6,c7=st.columns(7)
c1.metric("FAERS Reports",f"{n_reports:,}")
c2.metric("AEs Screened",f"{len(prr_df):,}")
c3.metric("Signals (FDR)",f"{n_sigs:,}",delta="BH p<0.05")
c4.metric("Strong  PRR≥8",f"{n_strong:,}")
c5.metric("Novel AEs",f"{n_novel:,}" if label_ok else "—")
c6.metric("Full Concordance",f"{n_full:,}",delta="PRR+EBGM+IC")
c7.metric("Top EB05",f"{top_eb05:.1f}×" if top_eb05 else "—",delta="≥2 = FDA criterion")

st.markdown(f"""<div class='method-card'>
  <strong>{n_reports:,}</strong> FAERS reports for <em>{sel_drug}</em> vs.
  <strong>{n_bg:,}</strong> background ({', '.join(d for d in DRUGS if d!=sel_drug)}).
  Thresholds: PRR≥{min_prr} · χ²≥{min_chi2} · n≥{min_n} · EB05≥{min_eb05} · BH p&lt;0.05.
  Haldane +0.5. Background gate c≥3. Label novelty: {'active' if label_ok else 'unavailable'}. Confounders: {'shown' if show_conf else 'hidden'}.
</div>""",unsafe_allow_html=True)
st.markdown("<hr>",unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
t_signals,t_landscape,t_3d,t_network,t_temporal,t_demo,t_crossdrug,t_full=st.tabs([
    "Signals","Landscape","3D Signal Space","Network","Temporal","Demographics","Cross-Drug","Full Table"])

# ══ TAB 1 — SIGNALS ══════════════════════════════════════════════════════════
with t_signals:
    st.markdown("<div class='section-label'>Signal Detection Output — PRR · EBGM · BCPNN IC · Concordance</div>",unsafe_allow_html=True)
    if sigs.empty:
        st.warning("No signals meet current thresholds. Adjust sidebar filters.")
    else:
        exp=sigs[["Reaction","Tier","Concordance","n","PRR","ROR","IC","IC025","IC975","EBGM","EB05","CI_lo","CI_hi","Chi2","p_raw","p_adj","Composite","SMQ","Confound","Labeled"]].copy()
        exp["95% CI"]=exp.apply(lambda r:f"{r['CI_lo']:.2f}–{r['CI_hi']:.2f}",axis=1)
        exp["Confound"]=exp["Confound"].map({True:"⚠",False:""})
        exp["Novelty"]=exp["Labeled"].map({True:"",False:"🆕"}) if label_ok else ""

        dl_col,_,nv_col=st.columns([1,4,1])
        with dl_col:
            buf=io.StringIO(); exp.to_csv(buf,index=False)
            st.download_button("⬇ Export CSV",buf.getvalue(),f"signals_{sel_drug.lower()}_v8.csv","text/csv")
        with nv_col:
            if n_novel>0 and label_ok:
                st.markdown(f"<span class='pill-green'>🆕 {n_novel} unlabeled</span>",unsafe_allow_html=True)

        show_cols=["Reaction","Tier","Concordance"]+( ["Novelty"] if label_ok else [])+["n","PRR","EBGM","EB05","IC","95% CI","Chi2","p_adj","Composite","SMQ","Confound"]
        disp=exp[[c for c in show_cols if c in exp.columns]].rename(columns={"n":"Cases","p_adj":"p (BH)","Composite":"Score"})
        st.dataframe(disp,use_container_width=True,height=420,
            column_config={
                "PRR":st.column_config.ProgressColumn("PRR",format="%.2f",min_value=0,max_value=float(sigs["PRR"].max() or 10)),
                "EBGM":st.column_config.NumberColumn("EBGM",format="%.3f",help="Empirical Bayes Geometric Mean (DuMouchel 1999)"),
                "EB05":st.column_config.NumberColumn("EB05",format="%.3f",help="5th-percentile credible lower bound. FDA criterion: EB05≥2"),
                "IC":st.column_config.NumberColumn("IC",format="%.3f",help="BCPNN Information Component (Bate 1998/Noren 2006)"),
                "Score":st.column_config.ProgressColumn("Score",format="%.3f",min_value=0,max_value=1.0),
                "p (BH)":st.column_config.NumberColumn(format="%.4f"),
                "Concordance":st.column_config.TextColumn(help="Full=PRR≥2∧EBGM≥2∧IC>0"),
            })

        st.markdown("<div class='section-label'>Metric Concordance</div>",unsafe_allow_html=True)
        cc1,cc2,cc3,cc4=st.columns(4)
        conc_c=sigs["Concordance"].value_counts()
        for col,(lbl,badge) in zip([cc1,cc2,cc3,cc4],[("Full","pill-green"),("Partial","pill-amber"),("Weak","pill-red"),("None","pill-neutral")]):
            col.markdown(f"<div class='stat-card'><div class='stat-num'>{conc_c.get(lbl,0)}</div><div class='stat-label'>{lbl} Concordance</div></div>",unsafe_allow_html=True)

        st.markdown("<div class='section-label'>2×2 Contingency Table</div>",unsafe_allow_html=True)
        sel_rxn=st.selectbox("Reaction",sigs["Reaction"].tolist(),key="ct",label_visibility="collapsed")
        if sel_rxn:
            row=sigs[sigs["Reaction"]==sel_rxn].iloc[0]
            a_=int(row["n"]); b_=int(row["b"]); c_=int(row["c_bg"]); d_=int(row["d"])
            td_=int(row["td"]); to_=int(row["to"]); E_=td_*(c_/to_) if to_>0 else 0; oe_=a_/E_ if E_>0 else float("nan")
            tc_=TIER_COLORS.get(row["Tier"],"#3C3489")
            nov_h="<span class='pill-green'>🆕 Unlabeled</span>" if (label_ok and not row["Labeled"]) else ""
            web_h=f"<span class='pill-amber'>⏱ Weber {row.get('WeberPct',''):.0f}%</span>" if row.get("Weber") else ""
            ct1,ct2=st.columns([1,1])
            with ct1:
                st.markdown(f"""<div style='font-family:DM Sans,sans-serif;font-size:13px;color:#3C3489;font-weight:500;margin-bottom:8px;'>{sel_rxn} {nov_h} {web_h}</div>
                <table class='ct-table'>
                  <tr><th></th><th>{sel_rxn[:20]}</th><th>All Other AEs</th><th>Total</th></tr>
                  <tr><td style='text-align:left;font-weight:500;color:#3C3489;'>{sel_drug}</td><td class='hl'>{a_:,}</td><td>{b_:,}</td><td>{a_+b_:,}</td></tr>
                  <tr><td style='text-align:left;color:#6B6760;'>Other drugs</td><td>{c_:,}</td><td>{d_:,}</td><td>{c_+d_:,}</td></tr>
                  <tr><td style='text-align:left;font-weight:500;'>Total</td><td>{a_+c_:,}</td><td>{b_+d_:,}</td><td>{a_+b_+c_+d_:,}</td></tr>
                </table>""",unsafe_allow_html=True)
            with ct2:
                st.markdown(f"""<div class='stat-card'>
                  <div style='font-family:DM Sans,sans-serif;font-size:13px;color:#6B6760;line-height:2.2;'>
                    <span style='color:#3C3489;font-weight:500;'>PRR</span> {row['PRR']:.3f} · <span style='color:#3C3489;font-weight:500;'>95% CI</span> [{row['CI_lo']:.3f}–{row['CI_hi']:.3f}]<br>
                    <span style='color:#3C3489;font-weight:500;'>EBGM</span> {row['EBGM']:.3f} · <span style='color:#3C3489;font-weight:500;'>EB05</span> {row['EB05']:.3f}<br>
                    <span style='color:#3C3489;font-weight:500;'>IC</span> {row['IC']:.3f} [{row['IC025']:.3f}, {row['IC975']:.3f}]<br>
                    <span style='color:#3C3489;font-weight:500;'>χ²</span> {row['Chi2']:.2f} · <span style='color:#3C3489;font-weight:500;'>p(BH)</span> {row['p_adj']:.4f}<br>
                    <span style='color:#3C3489;font-weight:500;'>O/E</span> {oe_:.2f}× · <span style='color:#3C3489;font-weight:500;'>ROR</span> {row['ROR']:.3f}<br>
                    <span style='color:#3C3489;font-weight:500;'>Tier</span> <span style='color:{tc_};font-weight:500;'>{row['Tier']}</span>
                  </div></div>""",unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Forest Plot — PRR 95% CI</div>",unsafe_allow_html=True)
        st.plotly_chart(forest_fig(sigs,sel_drug),use_container_width=True)

# ══ TAB 2 — LANDSCAPE ════════════════════════════════════════════════════════
with t_landscape:
    l1,l2=st.columns([3,2])
    with l1:
        st.markdown("<div class='section-label'>Volcano Plot</div>",unsafe_allow_html=True)
        st.plotly_chart(volcano_fig(prr_df,sel_drug),use_container_width=True)
        if not sigs.empty:
            st.markdown("<div class='section-label'>Observed vs. Expected — Top 8</div>",unsafe_allow_html=True)
            eo=sigs.head(8).copy(); eo["Expected"]=(eo["td"]*(eo["c_bg"]/eo["to"])).round(1)
            fig_eo=go.Figure()
            fig_eo.add_bar(x=eo["Reaction"],y=eo["n"],name="Observed",marker_color="#3C3489",opacity=0.9)
            fig_eo.add_bar(x=eo["Reaction"],y=eo["Expected"],name="Expected",marker_color="#D8D4CB",opacity=0.9)
            fig_eo.update_layout(**_layout(height=260,barmode="group",
                title=dict(text="Observed vs. Expected",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
                xaxis=dict(tickangle=-35),legend=dict(font=dict(size=10)),margin=dict(l=10,r=10,t=44,b=80)))
            st.plotly_chart(fig_eo,use_container_width=True)
    with l2:
        if not sigs.empty:
            st.markdown("<div class='section-label'>Signal Burden by SMQ</div>",unsafe_allow_html=True)
            smq_c=(sigs.groupby("SMQ").agg(signals=("Reaction","count"),mean_prr=("PRR","mean")).reset_index().sort_values("signals",ascending=True))
            smq_c=smq_c[smq_c["SMQ"]!=""]
            if not smq_c.empty:
                fig_smq=go.Figure(go.Bar(x=smq_c["signals"],y=smq_c["SMQ"],orientation="h",
                    marker=dict(color=smq_c["mean_prr"],colorscale=[[0,"#EEEDFE"],[0.5,"#7F77DD"],[1,"#3C3489"]],
                                colorbar=dict(title=dict(text="Avg PRR",font=dict(size=9,color="#A09B94")),tickfont=dict(size=8,color="#A09B94"),len=0.7)),
                    text=smq_c["signals"],textposition="outside",textfont=dict(size=9,color="#A09B94")))
                fig_smq.update_layout(**_layout(height=360,
                    title=dict(text="SMQ Signal Density",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
                    margin=dict(l=20,r=60,t=44,b=30)))
                st.plotly_chart(fig_smq,use_container_width=True)

            st.markdown("<div class='section-label'>EBGM vs PRR — Bayesian Shrinkage</div>",unsafe_allow_html=True)
            if len(sigs)>=3:
                mx=max(sigs["PRR"].max(),sigs["EBGM"].max())*1.05
                fig_ep=px.scatter(sigs,x="PRR",y="EBGM",color="Tier",size="n",hover_name="Reaction",
                    color_discrete_map={"STRONG":"#791F1F","MODERATE":"#633806","WATCH":"#3C3489"},
                    hover_data={"EB05":":.3f","Concordance":True,"n":True,"Tier":False},size_max=16)
                fig_ep.add_shape(type="line",x0=0,y0=0,x1=mx,y1=mx,line=dict(color="#D8D4CB",width=1,dash="dot"))
                fig_ep.add_vline(x=2,line_dash="dash",line_color="#D8D4CB",line_width=1)
                fig_ep.add_hline(y=2,line_dash="dash",line_color="#D8D4CB",line_width=1)
                fig_ep.update_layout(**_layout(height=300,
                    title=dict(text="EBGM shrinkage",font=dict(size=13,color="#1E1C1A")),
                    legend=dict(font=dict(size=9),bgcolor="rgba(253,252,249,0.95)"),margin=dict(l=10,r=10,t=40,b=30)))
                st.plotly_chart(fig_ep,use_container_width=True)

# ══ TAB 3 — 3D ═══════════════════════════════════════════════════════════════
with t_3d:
    st.markdown("<div class='section-label'>3D Signal Space — PRR × EBGM × IC Across Drug Class</div>",unsafe_allow_html=True)
    st.markdown("""<div class='method-card'>Each confirmed signal positioned in three-dimensional metric space: PRR (frequentist), EBGM (Bayesian), IC (information-theoretic). All drugs overlaid. Marker size ∝ √n. Rotate and hover for full readout.</div>""",unsafe_allow_html=True)
    with st.spinner("Computing class-level signals…"):
        all3d={drug:compute_prr(rxn_df,drug,CONFOUNDERS,"All","") for drug in DRUGS}
        all3d={k:v for k,v in all3d.items() if not v.empty}
    fig3d=threed_fig(all3d,DRUGS)
    if fig3d:
        st.plotly_chart(fig3d,use_container_width=True)
        rows3d=[{"Drug":d,"Signals":len(df[df["Signal"]]),"Mean PRR":round(df[df["Signal"]]["PRR"].mean(),2) if len(df[df["Signal"]])>0 else 0,"Mean EBGM":round(df[df["Signal"]]["EBGM"].mean(),2) if len(df[df["Signal"]])>0 else 0} for d,df in all3d.items()]
        if rows3d: st.dataframe(pd.DataFrame(rows3d),use_container_width=True,height=140)
    else: st.info("Insufficient confirmed signals. Lower thresholds.")

# ══ TAB 4 — NETWORK ══════════════════════════════════════════════════════════
with t_network:
    st.markdown("<div class='section-label'>Drug–AE Bipartite Signal Network</div>",unsafe_allow_html=True)
    if not NX_AVAILABLE: st.info("pip install networkx to enable.")
    elif sigs.empty: st.info("No signals to display.")
    else:
        fig_net=network_fig(sigs,sel_drug)
        if fig_net: st.plotly_chart(fig_net,use_container_width=True)
        st.markdown("""<div class='method-card'>Central purple node = target drug. Reaction nodes sized by n, coloured by tier (dark red=STRONG, amber=MODERATE, purple=WATCH). Edge width ∝ log(PRR). 🆕 = not in FDA label.</div>""",unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Signal Flow — Sankey Diagram</div>",unsafe_allow_html=True)
        fig_sk=sankey_fig(sigs,sel_drug)
        if fig_sk: st.plotly_chart(fig_sk,use_container_width=True)

# ══ TAB 5 — TEMPORAL ═════════════════════════════════════════════════════════
with t_temporal:
    top5=sigs.sort_values("PRR",ascending=False).head(5)["Reaction"].tolist() if not sigs.empty else []
    _,w_col=st.columns([5,1])
    with w_col: win=st.select_slider("Window (Qs)",options=[1,2,4,6,8],value=4,key="rwin")
    st.markdown("<div class='section-label'>Rolling-Window PRR Evolution — Top 5 Signals</div>",unsafe_allow_html=True)
    if top5:
        tr=rolling_prr(rxn_df,sel_drug,top5,window=win)
        if not tr.empty:
            fig_t=px.line(tr,x="Quarter",y="PRR",color="Reaction",markers=True,line_shape="spline",
                color_discrete_sequence=["#3C3489","#7F77DD","#633806","#27500A","#A09B94"])
            fig_t.add_hline(y=2,line_dash="dot",line_color="#D8D4CB",line_width=1,
                            annotation_text="PRR=2",annotation_font_size=9,annotation_font_color="#A09B94")
            fig_t.update_layout(**_layout(height=380,
                title=dict(text=f"{win}-Quarter Rolling PRR — {sel_drug}",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
                xaxis=dict(tickangle=-45,tickfont=dict(size=8)),
                legend=dict(font=dict(size=10),bgcolor="rgba(253,252,249,0.95)",bordercolor="#D8D4CB",borderwidth=1)))
            st.plotly_chart(fig_t,use_container_width=True)
            delta_rows=[]
            for rxn in top5:
                sub=tr[tr["Reaction"]==rxn].sort_values("Quarter")
                if len(sub)>=2:
                    d=sub["PRR"].iloc[-1]-sub["PRR"].iloc[0]
                    delta_rows.append({"Reaction":rxn,"First Q":sub["Quarter"].iloc[0],"Last Q":sub["Quarter"].iloc[-1],"PRR (first)":sub["PRR"].iloc[0],"PRR (last)":sub["PRR"].iloc[-1],"Δ PRR":round(d,2),"Δ %":round(d/sub["PRR"].iloc[0]*100,1) if sub["PRR"].iloc[0] else 0})
            if delta_rows:
                st.markdown("<div class='section-label'>Signal Trajectory</div>",unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(delta_rows),use_container_width=True,height=min(220,55+len(delta_rows)*38),
                    column_config={"Δ PRR":st.column_config.NumberColumn(format="%+.2f"),"Δ %":st.column_config.NumberColumn(format="%+.1f%%")})
        else: st.info("Insufficient temporal data (need ≥5 reports/quarter).")

    st.markdown("<div class='section-label'>Weber Effect Flags</div>",unsafe_allow_html=True)
    st.markdown("""<div class='method-card'><strong>Weber Effect (Weber 1984):</strong> Reporting peaks ~2 years post-launch. Signals with ≥60% of reports in first 3 years flagged ⏱ (minimum n=8).</div>""",unsafe_allow_html=True)
    if "Weber" in sigs.columns:
        wdf=sigs[sigs["Weber"]==True][["Reaction","Tier","n","PRR","EBGM","WeberPct","SMQ","Concordance"]].copy()
        wdf["WeberPct"]=wdf["WeberPct"].apply(lambda x:f"{x:.1f}%" if pd.notna(x) else "—")
        if not wdf.empty: st.dataframe(wdf,use_container_width=True,height=min(260,55+len(wdf)*38))
        else: st.info("No Weber flags in current signal set.")

# ══ TAB 6 — DEMOGRAPHICS ═════════════════════════════════════════════════════
with t_demo:
    dr2=dr.copy()
    dr2["age_group"]=pd.cut(dr2["age"],bins=[0,40,60,75,90,200],labels=["<40","40–60","60–75","75–90","90+"]).astype(str).replace("nan","Unknown")
    d1,d2=st.columns([3,2])
    with d1:
        st.markdown("<div class='section-label'>Sunburst — Sex → Age → Seriousness</div>",unsafe_allow_html=True)
        sb=dr2.groupby(["sex","age_group","serious"]).size().reset_index(name="count"); sb=sb[sb["count"]>0]
        if not sb.empty:
            fig_sb=px.sunburst(sb,path=["sex","age_group","serious"],values="count",color="serious",
                color_discrete_map={"Serious":"#791F1F","Non-serious":"#3C3489","Unknown":"#D8D4CB"})
            fig_sb.update_layout(paper_bgcolor="#FDFCF9",plot_bgcolor="#FDFCF9",
                title=dict(text=f"Demographics — {sel_drug}",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
                margin=dict(l=0,r=0,t=44,b=0),height=380)
            st.plotly_chart(fig_sb,use_container_width=True)
    with d2:
        if not dr2["age"].dropna().empty:
            st.markdown("<div class='section-label'>Age Distribution</div>",unsafe_allow_html=True)
            fig_age=px.histogram(dr2,x="age",nbins=22,color="sex",
                color_discrete_map={"Male":"#3C3489","Female":"#791F1F","Unknown":"#D8D4CB"},barmode="overlay",opacity=0.75)
            fig_age.update_layout(**_layout(height=190,title=dict(text="Age at Onset",font=dict(size=13,color="#1E1C1A")),
                xaxis_title="Age (years)",yaxis_title="Count",legend=dict(font=dict(size=9)),margin=dict(l=10,r=10,t=40,b=30)))
            st.plotly_chart(fig_age,use_container_width=True)
        st.markdown("<div class='section-label'>Seriousness</div>",unsafe_allow_html=True)
        sc=dr2["serious"].value_counts().reset_index(); sc.columns=["Category","Count"]
        fig_sc=px.bar(sc,x="Category",y="Count",color="Category",color_discrete_map={"Serious":"#791F1F","Non-serious":"#3C3489","Unknown":"#D8D4CB"})
        fig_sc.update_layout(**_layout(height=175,showlegend=False,title=dict(text="Seriousness",font=dict(size=13,color="#1E1C1A")),margin=dict(l=10,r=10,t=40,b=10)))
        st.plotly_chart(fig_sc,use_container_width=True)
        st.markdown("<div class='section-label'>Reports per Year</div>",unsafe_allow_html=True)
        yc=dr2["year"].dropna().astype(int).value_counts().reset_index(); yc.columns=["Year","Count"]; yc=yc.sort_values("Year")
        fig_yc=px.bar(yc,x="Year",y="Count",color_discrete_sequence=["#7F77DD"])
        fig_yc.update_layout(**_layout(height=170,showlegend=False,title=dict(text="Report Volume by Year",font=dict(size=13,color="#1E1C1A")),margin=dict(l=10,r=10,t=40,b=10)))
        st.plotly_chart(fig_yc,use_container_width=True)

# ══ TAB 7 — CROSS-DRUG ═══════════════════════════════════════════════════════
with t_crossdrug:
    st.markdown(f"<div class='section-label'>Cross-Drug PRR Heatmap — {sel_class}</div>",unsafe_allow_html=True)
    with st.spinner("Computing cross-drug PRR…"):
        all_raw={}; all_sig=set(); all_dfs={}
        for drug in DRUGS:
            df_d=compute_prr(rxn_df,drug,CONFOUNDERS,"All","")
            if not df_d.empty:
                all_dfs[drug]=df_d; all_raw[drug]=dict(zip(df_d["Reaction"],df_d["PRR"]))
                all_sig.update(df_d[df_d["Signal"]]["Reaction"].tolist())
    if all_sig and all_raw:
        rxn_list=sorted(all_sig)
        z=[[min(all_raw.get(d,{}).get(r,0.0),25.0) for r in rxn_list] for d in DRUGS]
        fig_h=go.Figure(go.Heatmap(z=z,x=rxn_list,y=DRUGS,
            colorscale=[[0,"#FDFCF9"],[0.1,"#EEEDFE"],[0.35,"#7F77DD"],[0.65,"#3C3489"],[0.85,"#791F1F"],[1,"#3d0a0a"]],
            zmin=0,zmax=25,colorbar=dict(title=dict(text="PRR",font=dict(size=10,color="#A09B94")),tickfont=dict(size=9,color="#A09B94"),len=0.8),
            hovertemplate="<b>%{x}</b><br>%{y}: PRR = %{z:.2f}<extra></extra>"))
        fig_h.update_layout(**_layout(height=230,
            title=dict(text="Class-Level PRR Heatmap",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
            xaxis=dict(tickangle=-40,tickfont=dict(size=8,color="#A09B94")),
            yaxis=dict(tickfont=dict(size=11,color="#6B6760")),margin=dict(l=10,r=10,t=50,b=130)))
        st.plotly_chart(fig_h,use_container_width=True)

        class_sigs=[r for r in all_sig if sum(1 for d in DRUGS if all_raw.get(d,{}).get(r,0)>=2)>=2]
        unique_sigs={drug:[r for r in all_sig if all_raw.get(drug,{}).get(r,0)>=2 and all(all_raw.get(o,{}).get(r,0)<2 for o in DRUGS if o!=drug)] for drug in DRUGS}
        cd1,cd2=st.columns(2)
        with cd1:
            if class_sigs: st.markdown(f"""<div class='method-card'><strong>Class-level signals</strong> (PRR≥2 in ≥2 drugs · n={len(class_sigs)}):<br>{' · '.join(sorted(class_sigs)[:20])}</div>""",unsafe_allow_html=True)
        with cd2:
            for drug in DRUGS:
                us=unique_sigs.get(drug,[])
                if us: st.markdown(f"""<div class='method-card'><strong>{drug}-unique</strong> (n={len(us)}):<br>{' · '.join(sorted(us)[:10])}</div>""",unsafe_allow_html=True)
        if class_sigs:
            top_c=sorted(class_sigs,key=lambda r:max(all_raw.get(d,{}).get(r,0) for d in DRUGS),reverse=True)[:8]
            bar_rows=[{"Drug":d,"Reaction":r,"PRR":round(all_raw.get(d,{}).get(r,0),2)} for d in DRUGS for r in top_c]
            fig_cb=px.bar(pd.DataFrame(bar_rows),x="Reaction",y="PRR",color="Drug",barmode="group",color_discrete_map=DRUG_COLORS)
            fig_cb.add_hline(y=2,line_dash="dash",line_color="#D8D4CB",line_width=1)
            fig_cb.update_layout(**_layout(height=320,
                title=dict(text="Top Shared Signals — PRR by Drug",font=dict(family="EB Garamond",size=16,color="#1E1C1A")),
                xaxis=dict(tickangle=-40,tickfont=dict(size=8)),
                legend=dict(font=dict(size=10),bgcolor="rgba(253,252,249,0.95)",bordercolor="#D8D4CB",borderwidth=1),margin=dict(l=10,r=10,t=44,b=100)))
            st.plotly_chart(fig_cb,use_container_width=True)
    else: st.info("Insufficient data. Lower thresholds.")

# ══ TAB 8 — FULL TABLE ═══════════════════════════════════════════════════════
with t_full:
    st.markdown("<div class='section-label'>Full Disproportionality Table — All Reactions Screened</div>",unsafe_allow_html=True)
    if not prr_df.empty:
        fd=prr_df[["Reaction","Tier","n","c_bg","PRR","ROR","IC","IC025","IC975","EBGM","EB05","CI_lo","CI_hi","Chi2","p_raw","p_adj","Composite","Signal","Confound","Labeled","SMQ"]].copy()
        fd["Signal"]=fd["Signal"].map({True:"✓",False:""})
        fd["Confound"]=fd["Confound"].map({True:"⚠",False:""})
        fd["Labeled"]=fd["Labeled"].map({True:"✓",False:"🆕"}) if label_ok else "?"
        st.dataframe(fd,use_container_width=True,height=480,
            column_config={"PRR":st.column_config.NumberColumn(format="%.3f"),"EBGM":st.column_config.NumberColumn(format="%.3f"),"EB05":st.column_config.NumberColumn(format="%.3f"),
                           "p_adj":st.column_config.NumberColumn("p (BH-adj)",format="%.4f"),
                           "Composite":st.column_config.ProgressColumn("Composite",format="%.3f",min_value=0,max_value=1),
                           "Labeled":st.column_config.TextColumn("In Label",help="✓=labeled · 🆕=unlabeled")})
        buf2=io.StringIO(); fd.to_csv(buf2,index=False)
        st.download_button("⬇ Export full PRR table",buf2.getvalue(),f"full_prr_{sel_drug.lower()}_v8.csv","text/csv")

# ─── METHODOLOGY FOOTER ───────────────────────────────────────────────────────
st.markdown("<hr>",unsafe_allow_html=True)
st.markdown(f"""<div class='method-card'>
  <strong>Statistical Methodology</strong><br>
  PRR per Evans SJ et al. (2001) <em>Pharmacoepidemiol Drug Saf</em> 10:483–486.
  Haldane-Anscombe correction (+0.5 to all 2×2 cells).
  χ² with Yates continuity correction. FDR: Benjamini-Hochberg (scipy).
  ROR = (a·d)/(b·c). IC = log₂(p₁₁/(p₁ₓ·pₓ₁)); 95% bounds per Norén et al. (2006).
  EBGM: Gamma-Poisson mixture (DuMouchel 1999; α₁=0.20, β₁=0.06, w₁=0.10, α₂=1.40, β₂=1.80).
  EB05 = 5th-percentile credible lower bound. Signal: PRR≥{min_prr} ∧ χ²≥{min_chi2} ∧ n≥{min_n} ∧ EB05≥{min_eb05} ∧ BH p&lt;0.05.
  Weber flag: ≥60% reports in first 3 years (n≥8).<br><br>
  <strong>Limitations</strong><br>
  FAERS is voluntary spontaneous reporting. PRR/EBGM measure disproportionate reporting frequency, not incidence or causality.
  Subject to under-reporting, Weber effect, notoriety bias, and confounding by indication. No exposure denominator.
  Corpus: up to 1,000 reports/drug. Label novelty detection uses approximate substring matching on prose text.
</div>""",unsafe_allow_html=True)
