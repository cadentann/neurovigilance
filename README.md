# NeuroVigilance
**Open-source pharmacovigilance signal detection for neurological drug classes** FDA FAERS · PRR · EBGM · BCPNN/IC · FDR · PubMed Literature Integration

**ORCID:** [0009-0004-8611-8436](https://orcid.org/0009-0004-8611-8436)  
**DOI:** [FILL IN DOI AFTER MINTING]

---

## What It Does
NeuroVigilance queries FDA's Adverse Event Reporting System (FAERS) in real time to detect disproportionate drug safety signals for CNS drug classes using three independent statistical frameworks. For each detected signal, it surfaces relevant pharmacovigilance literature from PubMed automatically.

## Ethics & Data Statement
NeuroVigilance queries the FDA Adverse Event Reporting System (FAERS), a publicly available, fully de-identified dataset maintained by the U.S. Food and Drug Administration. FAERS data contains no personally identifiable information and is released under FDA's public access policy.

This project constitutes secondary analysis of publicly available, de-identified data and is exempt from IRB review under 45 CFR 46.104(d)(4)(ii), which exempts research involving the collection or study of existing data that is publicly available. No patient contact, no private identifiable information, and no experimental intervention are involved.

## Installation
```bash
git clone [https://github.com/cadentan2029-png/neurovigilance.git](https://github.com/cadentan2029-png/neurovigilance.git)
cd neurovigilance
pip install -r requirements.txt
streamlit run app.py
