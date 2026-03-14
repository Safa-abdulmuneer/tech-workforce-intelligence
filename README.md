# Tech Workforce Intelligence
## Layoff & Hiring Trend Analysis (2020–2024)
### MS Elevate × AICTE Internship — Capstone Project

---

## Project Structure

```
workforce_intel/
│
├── data/
│   └── 01_clean_data.py          ← RUN THIS FIRST
│
├── ml/
│   └── 02_ml_risk_model.py       ← RUN THIS SECOND
│
└── powerbi/
    └── 03_powerbi_guide.md       ← FOLLOW THIS to build the dashboard
```

---

## How to Run (Full Flow)

### Step 1 — Get the dataset
1. Go to: https://www.kaggle.com/datasets/swaptr/layoffs-2022
2. Download `layoffs.csv`
3. Rename it `layoffs_raw.csv`
4. Place it inside the `data/` folder

### Step 2 — Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### Step 3 — Clean the data
```bash
cd data/
python 01_clean_data.py
# Output: layoffs_clean.csv
```

### Step 4 — Run the ML model
```bash
cd ../ml/
cp ../data/layoffs_clean.csv .
python 02_ml_risk_model.py
# Output: layoffs_with_risk.csv
#         feature_importance.png  ← use in your PPT
#         confusion_matrix.png    ← use in your PPT
```

### Step 5 — Build the Power BI dashboard
- Open `powerbi/03_powerbi_guide.md`
- Load `layoffs_with_risk.csv` into Power BI
- Follow page-by-page instructions (5 pages total)

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python / Pandas | Data cleaning & feature engineering |
| Scikit-learn | Random Forest classifier for risk scoring |
| Matplotlib / Seaborn | Model evaluation charts |
| Power BI Desktop | Interactive dashboard (5 pages) |
| DAX | KPI measures & time intelligence |

---

## Dataset
- **Source:** layoffs.fyi via Kaggle
- **Records:** 3,000+ layoff events
- **Period:** 2020–2024
- **Fields:** Company, Industry, Country, Funding Stage, Layoff Count, % Laid Off, Funds Raised

---

## Key Outputs
- `layoffs_with_risk.csv` — cleaned + ML-scored dataset
- `feature_importance.png` — which features drive layoff risk
- `confusion_matrix.png` — model accuracy visualization
- Power BI `.pbix` dashboard — 5 interactive pages

