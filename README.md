# Pharma Sales Force Recommender
### Physician Targeting Model · GLP-1 Drug Class · California Medicare 2023

Built using real CMS Medicare Part D data to replicate the commercial analytics
work done by firms like ZS Associates for pharmaceutical clients.

---

## What This Project Does

A sales rep promoting Tirzepatide (Mounjaro/Zepbound) in California has 15,000+
physicians to potentially call on. They have capacity for maybe 300 calls a week.
This model ranks every physician by their likelihood to adopt Tirzepatide —
so the rep spends time on the right doctors.

**Live dashboard → [your-streamlit-url-here]**

---

## Results

| Metric | Value |
|--------|-------|
| Physicians scored | 13,027 |
| Absolute lift over random targeting | 82 percentage points |
| Relative lift | 455% |
| Top decile lift (D1) | 8.6x |
| Z-statistic | 26.4 |
| P-value | < 0.000001 |
| Estimated revenue impact (CA) | $76M |

---

## Data Sources

All data is free and publicly available:

| Dataset | Source | What it provides |
|---------|--------|-----------------|
| Medicare Part D Prescribers 2023 | [CMS](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers) | Prescriptions by physician × drug |
| NPPES NPI Registry | [CMS](https://npiregistry.cms.hhs.gov) | Physician name, specialty, location |
| CMS Open Payments | [CMS](https://openpaymentsdata.cms.gov) | Pharma payments to physicians |

Filtered to California GLP-1 prescribers:
- **27,781** prescription records
- **15,289** unique physicians
- **6** GLP-1 drug generics
- **$1.5B** in total drug costs

---

## Methodology

### 1. Data Pipeline
Downloaded and filtered the CMS Part D national file (~3.6GB) to
California GLP-1 prescribers. Cleaned, standardized, and saved as
parquet. Built a physician × drug interaction matrix using
log-transformed claim counts as implicit feedback signals.

### 2. Targeting Model
**Volume-weighted cosine similarity:**
- Identified the top 25% of existing Tirzepatide prescribers by claim volume (n=570)
- Computed a centroid vector representing the ideal high-value adopter profile in drug-feature space
- Scored all 13,027 non-prescribing physicians by cosine similarity to this centroid
- Combined similarity score (50%) with normalized volume score (50%) for final affinity score

This mirrors the **blended affinity scoring** approach used in commercial
pharmaceutical targeting models.

### 3. Validation — Retrospective A/B Test
Simulated a sales force A/B test:
- **Treatment group:** top 500 physicians ranked by model
- **Control group:** 500 randomly sampled from same pool
- **Adoption threshold:** top-quartile GLP-1 prescribing (≥77 claims)
- **Test:** two-proportion z-test, one-sided alternative

Treatment adoption rate: **100%**
Control adoption rate: **18%**
Result: statistically significant at p < 0.000001

### 4. Limitations
- Medicare data only — excludes commercial insurance (~40% of market)
- ~18 month data lag from CMS publication schedule
- Single-year snapshot — cannot capture physician switching behavior over time
- Would benefit from multi-year longitudinal data and commercial claims integration

---

## Project Structure
```
pharma-sales-recommender/
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── data/
│   └── figures/              # EDA and validation charts
├── notebooks/
│   ├── phase1_data.ipynb     # Data acquisition & cleaning
│   ├── phase2_model.ipynb    # EDA & scoring model
│   └── phase3_validation.ipynb # A/B test & lift analysis
├── requirements.txt
└── README.md
```
---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| pandas / numpy | Data processing |
| scipy / statsmodels | Hypothesis testing & power analysis |
| implicit | ALS collaborative filtering |
| scikit-learn | Feature scaling |
| matplotlib | Visualizations |
| Streamlit | Interactive dashboard |
| CMS Open Data | Real Medicare prescribing data |

---

## Running Locally

```bash
git clone https://github.com/your-username/pharma-sales-recommender
cd pharma-sales-recommender
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Note: You will need to download the CMS Part D data and run the
notebooks in order to generate the processed data files.

---

## Key Findings

**Semaglutide dominates the market** (634k claims vs 79k for Tirzepatide)
— representing a significant conversion opportunity for Eli Lilly reps.

**Family Practice and Internal Medicine write more GLP-1 scripts than
Endocrinology** — a counterintuitive finding that traditional sales force
models based on specialty targeting alone would miss entirely.

**53% of physicians prescribe only one GLP-1 drug** — this sparse
interaction pattern is precisely what collaborative filtering is designed
to exploit.

---

## About

Built to demonstrate commercial analytics skills relevant to pharmaceutical
consulting. Methodology mirrors physician targeting models used in practice
at firms including ZS Associates, IQVIA, and Beghou Consulting.

Data: CMS Medicare Part D 2023 | Market: California | Drug: Tirzepatide
