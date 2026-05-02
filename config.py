"""
NeuroVigilance v9 — Central configuration and constants.
All thresholds, drug metadata, prior parameters, and reference values live here.
"""

# ── Drug class definitions ────────────────────────────────────────────────────
DRUG_CLASSES = {
    "Cholinesterase Inhibitors": {
        "drugs":     ["Donepezil", "Rivastigmine", "Galantamine"],
        "brands":    {"Donepezil": "aricept", "Rivastigmine": "exelon", "Galantamine": "razadyne"},
        "indication": "Alzheimer's Disease / Dementia",
        "confounders": {
            # Disease-related confounders (indication-driven)
            "Memory Impairment", "Dementia", "Dementia Alzheimer'S Type", "Cognitive Disorder",
            "Alzheimer'S Disease", "Agitation", "Confusional State", "Disorientation", "Fall",
            "Gait Disturbance", "Drug Ineffective", "Disease Progression",
            "Loss Of Personal Independence In Daily Activities",
            # Mechanism-based pharmacodynamic effects (cholinergic class effect —
            # direct consequence of AChE inhibition, not ADRs in the conventional sense).
            # GI effects: parasympathomimetic activation of enteric M3 receptors.
            # ChEIs INCREASE GI motility (pro-kinetic) → diarrhea/nausea/cramps.
            "Nausea", "Vomiting", "Diarrhoea", "Abdominal Pain", "Decreased Appetite",
            "Anorexia", "Weight Decreased",
            # Cardiovascular cholinergic effects: M2 receptor downstream on SA node.
            "Bradycardia", "Sinus Bradycardia", "Heart Rate Decreased",
            "Syncope", "Presyncope",
            # Constipation: included as a DISEASE-RELATED confounder (not a ChEI
            # pharmacodynamic effect). ChEIs are pro-kinetic (M3-mediated) and
            # pharmacologically oppose constipation. Constipation in dementia
            # patients is driven by opioid co-prescribing, reduced mobility, and
            # low fiber intake — not the drug. It is flagged here to prevent false
            # signals driven by the patient population rather than the drug.
            "Constipation",
        },
    },
    "SSRIs": {
        "drugs":     ["Fluoxetine", "Sertraline", "Escitalopram", "Citalopram", "Paroxetine"],
        "brands":    {
            "Fluoxetine": "prozac", "Sertraline": "zoloft", "Escitalopram": "lexapro",
            "Citalopram": "celexa", "Paroxetine": "paxil",
        },
        "indication": "Major Depressive Disorder / Anxiety",
        "confounders": {
            "Depression", "Major Depressive Disorder", "Anxiety Disorder", "Suicidal Ideation",
            "Drug Ineffective", "Anxiety", "Panic Attack", "Depressed Mood", "Mood Altered",
            # SSRI class effects: SERT-dependent pharmacodynamic reactions.
            # NOTE: "Insomnia" moved here from the disease-confounder set above.
            # Insomnia is a well-characterized pharmacodynamic ADR of SSRIs mediated
            # by 5-HT2A receptor stimulation — dose-dependent, onset-coincident with
            # initiation, and resolves with discontinuation. It is NOT a disease
            # manifestation and should NOT be classified as a disease confounder.
            # Classifying it alongside "Depression" incorrectly suppresses a clinically
            # NOTE: "Insomnia" is listed here so Confound=True appears in the table
            # as an advisory note for users. It does NOT suppress Signal_Evans or
            # Signal detection — reactions with Confound=True still appear in the
            # signal table and can still be flagged as signals. The flag advises
            # analysts that drug-specific variation within SSRIs (Fluoxetine/Paroxetine
            # >> Citalopram/Escitalopram) may be meaningful and the class-effect label
            # should be interpreted cautiously in drug-level analyses.
            "Insomnia",
            "Sexual Dysfunction", "Decreased Libido", "Ejaculation Disorder",
            "Orgasm Abnormal", "Erectile Dysfunction",
            # NOTE: "Hyponatraemia" and "SIADH" intentionally NOT listed as class confounders.
            # Drug-specific variation within SSRIs is clinically meaningful:
            # Paroxetine causes hyponatremia at substantially higher rates than fluoxetine
            # or sertraline. The mechanism is serotonergic: SSRI-induced 5-HT receptor
            # stimulation (particularly 5-HT1A/2C) promotes ADH release → SIADH.
            # Paroxetine's higher incidence is better attributed to its stronger SERT
            # affinity, norepinephrine reuptake inhibition, and potent CYP2D6 inhibition
            # (affecting co-medication levels) — NOT to muscarinic anticholinergic
            # activity, which if anything would mildly oppose ADH release by reducing
            # cholinergic tone. The code decision is correct; citing muscarinic
            # mechanisms as the cause is pharmacologically wrong.
            # Citalopram/escitalopram are intermediate risk.
        },
    },
    "Atypical Antipsychotics": {
        "drugs":     ["Quetiapine", "Olanzapine", "Risperidone", "Aripiprazole", "Clozapine"],
        "brands":    {
            "Quetiapine":   "seroquel",
            "Olanzapine":   "zyprexa",
            "Risperidone":  "risperdal",
            "Aripiprazole": "abilify",
            "Clozapine":    "clozaril",
        },
        "indication": "Schizophrenia / Bipolar / Treatment-Resistant Depression",
        "confounders": {
            "Schizophrenia", "Bipolar Disorder", "Psychosis", "Hallucination",
            "Drug Ineffective", "Agitation", "Aggression", "Delusion",
            # NOTE: QT prolongation (Electrocardiogram Qt Prolonged, Torsade De Pointes),
            # metabolic effects (Weight Increased, Hyperglycaemia, etc.), hyperprolactinaemia,
            # EPS, and tardive dyskinesia are intentionally NOT listed as class confounders.
            #
            # The tool's own principle — do not suppress as a class effect when drug-specific
            # variation is clinically meaningful — applies equally to all of these:
            #   QT (clinical QTc prolongation, not in vitro hERG IC50):
            #     Aripiprazole (minimal) << Risperidone ≤ Quetiapine << Clozapine (~14–15ms)
            #     Note: quetiapine has lower in vitro hERG IC50 (~1µM) than clozapine (~3–6µM),
            #     but clinical QTc prolongation data reverses this (Clozapine > Quetiapine),
            #     likely due to differences in protein binding and active metabolites.
            #   Metabolic: Aripiprazole (near-neutral) << Quetiapine << Clozapine/Olanzapine
            #   Prolactin: Aripiprazole lowers prolactin; Risperidone raises it substantially
            #   EPS: Atypicals are defined by REDUCED EPS vs. conventionals — drug-specific
            #        signals (Risperidone EPS, Aripiprazole akathisia) are clinically important
            #
            # These should be detectable as drug-specific signals, not suppressed as class effects.
        },
    },
}

# ── Drug approval dates (for Weber effect) ────────────────────────────────────
# SOURCE: FDA approval records. Used as reference year for Weber flag.
APPROVAL_YEARS = {
    "Donepezil":    1996,
    "Rivastigmine": 2000,
    "Galantamine":  2001,
    "Fluoxetine":   1987,
    "Sertraline":   1991,
    "Escitalopram": 2002,
    "Citalopram":   1998,
    "Paroxetine":   1992,
    "Quetiapine":   1997,
    "Olanzapine":   1996,
    "Risperidone":  1993,
    "Aripiprazole": 2002,
    "Clozapine":    1989,
    # Background drugs
    "Metformin":      1994,
    "Lisinopril":     1987,
    "Atorvastatin":   1996,
    "Warfarin":       1954,
    "Amoxicillin":    1972,
    "Omeprazole":     1989,
    "Metoprolol":     1978,
    "Levothyroxine":  1950,
    "Prednisone":     1955,
    "Amlodipine":     1992,
}

# ── Drug name synonyms for FAERS search ──────────────────────────────────────
# FAERS uses free-text drug names with poor standardization. Searching only
# the INN (International Nonproprietary Name) misses brand names, salt forms,
# and common misspellings — systematically underestimating td (denominator).
# Each list should include: INN, salt forms, brand names, common variants.
# All entries are searched as an OR query to maximize FAERS recall.
DRUG_SYNONYMS: dict[str, list[str]] = {
    "Donepezil":    ["donepezil", "donepezil hcl", "donepezil hydrochloride", "aricept"],
    "Rivastigmine": ["rivastigmine", "rivastigmine tartrate", "exelon", "rivastigmine patch"],
    "Galantamine":  ["galantamine", "galantamine hbr", "galantamine hydrobromide",
                     "razadyne", "reminyl", "galantamine er",
                     # Historical British/botanical spelling (from Galanthus snowdrop).
                     # Appears in early FAERS entries (late 1990s–early 2000s) and
                     # foreign market reports. Absent from synonyms would undercount td.
                     "galanthamine", "galanthamine hbr", "galanthamine hydrobromide"],
    "Fluoxetine":   ["fluoxetine", "fluoxetine hcl", "fluoxetine hydrochloride",
                     "prozac", "sarafem"],
    "Sertraline":   ["sertraline", "sertraline hcl", "sertraline hydrochloride", "zoloft"],
    "Escitalopram": ["escitalopram", "escitalopram oxalate", "lexapro", "cipralex"],
    "Citalopram":   ["citalopram", "citalopram hbr", "citalopram hydrobromide", "celexa"],
    "Paroxetine":   ["paroxetine", "paroxetine hcl", "paroxetine hydrochloride",
                     "paxil", "paxil cr", "pexeva", "brisdelle"],
    "Quetiapine":   ["quetiapine", "quetiapine fumarate", "seroquel", "seroquel xr"],
    "Olanzapine":   ["olanzapine", "zyprexa", "zyprexa zydis", "olanzapine pamoate"],
    "Risperidone":  ["risperidone", "risperdal", "risperdal consta", "perseris"],
    "Aripiprazole": ["aripiprazole", "abilify", "abilify maintena", "aristada"],
    "Clozapine":    ["clozapine", "clozaril", "versacloz", "fazaclo"],
}


SIGNAL_GROUP_MAP = {
    "Bradycardia": "Cardiac", "Sinus Bradycardia": "Cardiac",
    "Heart Rate Decreased": "Cardiac", "Electrocardiogram Qt Prolonged": "Cardiac",
    "Torsade De Pointes": "Cardiac", "Syncope": "Cardiac",
    "Atrial Fibrillation": "Cardiac", "Tachycardia": "Cardiac",
    "Orthostatic Hypotension": "Cardiac", "Hypotension": "Cardiac",
    "Seizure": "Seizure/CNS", "Convulsion": "Seizure/CNS", "Epilepsy": "Seizure/CNS",
    "Tremor": "Movement Disorder", "Dystonia": "Movement Disorder",
    # EPS/TD are basal-ganglia movement disorders (dopaminergic D2 receptor blockade/
    # supersensitivity in the striatum). They are mechanistically and clinically distinct
    # from seizures (cortical hyperexcitability). Grouping them with "Seizure/CNS" would
    # mislead analysts reviewing antipsychotic signal tables.
    "Extrapyramidal Disorder": "Movement Disorder",
    "Tardive Dyskinesia": "Movement Disorder",
    "Akathisia": "Movement Disorder",
    "Nausea": "GI/Cholinergic", "Vomiting": "GI/Cholinergic", "Diarrhoea": "GI/Cholinergic",
    "Abdominal Pain": "GI/Cholinergic", "Decreased Appetite": "GI/Cholinergic",
    # Constipation signal group depends on drug class:
    # - For ChEIs: "GI/Disease-Related" (ChEIs are pro-kinetic; constipation is
    #   opioid/immobility-driven in dementia patients — NOT a ChEI pharmacodynamic effect)
    # - For Atypical Antipsychotics: "GI/Anticholinergic" (clozapine, olanzapine,
    #   quetiapine have strong anticholinergic activity; constipation can progress to
    #   paralytic ileus — a WHO-UMC priority signal with documented fatalities)
    # This global map defaults to the ChEI classification; app.py applies a per-drug
    # override for antipsychotics when displaying the Signal_Group column.
    "Constipation": "GI/Disease-Related",
    "Insomnia": "Neuropsychiatric", "Depression": "Neuropsychiatric",
    "Hallucination": "Neuropsychiatric", "Anxiety": "Neuropsychiatric",
    "Mania": "Neuropsychiatric", "Suicidal Ideation": "Neuropsychiatric",
    "Aggression": "Neuropsychiatric",
    "Weight Increased": "Metabolic", "Weight Decreased": "Metabolic",
    "Hyperglycaemia": "Metabolic", "Hyperlipidaemia": "Metabolic",
    "Hyponatraemia": "Metabolic", "Hyperprolactinaemia": "Metabolic",
    "Serotonin Syndrome": "Serotonin Synd.", "Hyperthermia": "Serotonin Synd.",
    "Myoclonus": "Serotonin Synd.",
    "Rash": "Dermatological", "Pruritus": "Dermatological",
    "Stevens-Johnson Syndrome": "Dermatological",
    "Death": "Fatal", "Sudden Death": "Fatal", "Cardiac Arrest": "Fatal",
}

# ── Background drug panel ─────────────────────────────────────────────────────
# When the analysis is run against only within-class drugs, every class-wide
# adverse effect produces PRR ≈ 1.0 (all drugs share it equally) — a systematic
# false null that suppresses the signals most relevant to the class.
#
# This panel provides ~300 background reports from diverse therapeutic classes
# (cardiovascular, analgesic, anticoagulant, anti-infective, endocrine) to
# serve as a FAERS reference population. These drugs are fetched at app start
# and included in the background denominator for all disproportionality metrics.
#
# Note: The openFDA API caps at 1,000 reports/drug and ~25,000 skip; for
# production use, replace with the full FAERS bulk corpus (README_bulk.md).
BACKGROUND_DRUGS: list[str] = [
    "Metformin",       # antidiabetic — high report volume, diverse reactions
    "Lisinopril",      # ACE inhibitor — cardiovascular reference
    "Atorvastatin",    # statin — diverse metabolic and musculoskeletal signals
    "Warfarin",        # anticoagulant — high spontaneous reporting rate
    "Amoxicillin",     # antibiotic — broad-spectrum, many users
    "Omeprazole",      # PPI — diverse GI and systemic reactions
    "Metoprolol",      # beta-blocker — cardiovascular and CNS overlap
    "Levothyroxine",   # thyroid — endocrine and systemic reactions
    "Prednisone",      # corticosteroid — broad systemic reactions
    "Amlodipine",      # calcium channel blocker — cardiovascular
]

BACKGROUND_DRUG_SYNONYMS: dict[str, list[str]] = {
    "Metformin":      ["metformin", "metformin hcl", "glucophage"],
    "Lisinopril":     ["lisinopril", "zestril", "prinivil"],
    "Atorvastatin":   ["atorvastatin", "atorvastatin calcium", "lipitor"],
    "Warfarin":       ["warfarin", "warfarin sodium", "coumadin", "jantoven"],
    "Amoxicillin":    ["amoxicillin", "amoxil", "trimox"],
    "Omeprazole":     ["omeprazole", "prilosec", "zegerid"],
    "Metoprolol":     ["metoprolol", "metoprolol tartrate", "metoprolol succinate",
                       "lopressor", "toprol"],
    "Levothyroxine":  ["levothyroxine", "levothyroxine sodium", "synthroid",
                       "levoxyl", "unithroid"],
    "Prednisone":     ["prednisone", "deltasone", "rayos"],
    "Amlodipine":     ["amlodipine", "amlodipine besylate", "norvasc"],
}


# SOURCE: FDA FAERS data element definitions
AGE_UNIT_TO_YEARS = {
    "800": 10.0,       # decade
    "801": 1.0,        # year
    "802": 1/12,       # month
    "803": 1/52.1775,  # week
    "804": 1/365.25,   # day
    "805": 1/8766,     # hour
}

# ── FAERS seriousness field ───────────────────────────────────────────────────
# seriousness=1 means the report is flagged as serious.
# Absence or 0 means non-serious. There is no value "2".
SERIOUSNESS_MAP = {
    "1": "Serious",
    1:   "Serious",
    "0": "Non-serious",
    0:   "Non-serious",
}

# ── Signal tier thresholds ────────────────────────────────────────────────────
# Evans (2001) and EMA define a signal at PRR ≥ 2, χ² ≥ 4, n ≥ 3.
# NOTE: WHO-UMC uses the IC framework (IC − 2·SD > 0), not PRR.
# FDA MGPS uses EBGM/EB05 ≥ 2.0. The PRR criterion is Evans/EMA only.
# The WATCH / MODERATE / STRONG tiers below are custom operational categories
# for prioritisation within detected signals — they have no regulatory basis
# and are not derived from published pharmacovigilance guidance.
# Tier thresholds: WATCH = PRR ≥ 2 (signal threshold), MODERATE = PRR ≥ 3,
# STRONG = PRR ≥ 8. PRR ≥ 8 is a high threshold; most clinically important
# reactions (e.g. levodopa → dyskinesia) report PRR ≈ 6–10 in FAERS.
TIER_THRESH = [(8, "STRONG"), (3, "MODERATE"), (2, "WATCH")]
# PRR < 2 falls through to "NONE" in get_tier() — no entry needed here.

# ── Default EBGM GPS mixture priors (DuMouchel 1999, Table 3) ─────────────────
# These are calibrated to the full FAERS corpus; re-fit at runtime when possible.
EBGM_DEFAULT_PRIORS = (0.20, 0.06, 1.40, 1.80, 0.10)
#                       a1    b1    a2    b2    w1

# ── Composite score absolute reference maxima ─────────────────────────────────
# Normalise against fixed reference values so scores are comparable across runs.
# NOTE: This composite score is a heuristic ranking tool, not a statistical test.
#   - PRR and EBGM are correlated (both measure disproportionality) — including
#     both means disproportionality evidence has combined weight 0.45 vs 0.40 for
#     statistical significance (IC + Chi²). This is intentional.
#   - IC025 is the lower CI bound of IC and is correlated with IC itself — its
#     inclusion adds downside-risk sensitivity (rewards robustness at small n).
#   - Chi² conflates effect size and sample size; large-n reactions with modest
#     PRR get inflated Chi². Users should not interpret Composite as a p-value.
#   - Weights (PRR 0.30, IC 0.20, Chi² 0.20, EBGM 0.15, IC025 0.15) are not
#     empirically validated. The score is for priority ranking only.
COMPOSITE_REF_MAX = {
    "PRR":   50.0,
    "IC":     5.0,
    "Chi2": 200.0,
    "EBGM":  30.0,
    "IC025":  4.0,
}
COMPOSITE_WEIGHTS = {"PRR": 0.30, "IC": 0.20, "Chi2": 0.20, "EBGM": 0.15, "IC025": 0.15}

# ── Visualisation palette ─────────────────────────────────────────────────────
TIER_COLORS = {"STRONG": "#791F1F", "MODERATE": "#633806", "WATCH": "#3C3489", "NONE": "#6B6760"}
TIER_BG     = {"STRONG": "#FCEBEB", "MODERATE": "#FAEEDA", "WATCH": "#EEEDFE", "NONE": "#F6F4EF"}

DRUG_COLORS = {
    "Donepezil":    "#3C3489",
    "Rivastigmine": "#7F77DD",
    "Galantamine":  "#27500A",
    "Fluoxetine":   "#3C3489",
    "Sertraline":   "#7F77DD",
    "Escitalopram": "#27500A",
    "Citalopram":   "#1B6B8A",
    "Paroxetine":   "#8B3A8B",
    "Quetiapine":   "#3C3489",
    "Olanzapine":   "#7F77DD",
    "Risperidone":  "#27500A",
    "Aripiprazole": "#1B6B8A",
    "Clozapine":    "#8B3A8B",
}

# ── PubMed / NCBI E-utilities ─────────────────────────────────────────────────
import os as _os

def _get_ncbi_email() -> str:
    """Return NCBI E-utilities identity email.
    For production: set NCBI_EMAIL env var or Streamlit secret.
    Avoids the privacy concern of hardcoding a personal address in source code.
    """
    from_env = _os.environ.get("NCBI_EMAIL", "")
    if from_env:
        return from_env
    try:
        import streamlit as _st
        from_secret = _st.secrets.get("NCBI_EMAIL", "")
        if from_secret:
            return from_secret
    except Exception:
        pass
    return "cadentan2029@northwestern.edu"

NCBI_EMAIL = _get_ncbi_email()
NCBI_TOOL  = "NeuroVigilance"
PUBMED_CACHE_TTL = 86400  # 24 hours

# ── openFDA API ───────────────────────────────────────────────────────────────
FDA_BASE_URL  = "https://api.fda.gov/drug/event.json"
FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
FDA_PAGE_SIZE = 100
FDA_MAX_SKIP  = 25000  # Hard API limit. Use bulk files for larger corpora.
TARGET_REPORTS_PER_DRUG = 1000  # Current default; raise with bulk FAERS files.
FAERS_CACHE_TTL = 3600  # 1 hour
