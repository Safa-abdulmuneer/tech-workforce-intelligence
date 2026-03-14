"""
===============================================================
  Tech Workforce Intelligence — Step 1: Data Cleaning
===============================================================
  Dataset : layoffs.fyi from Kaggle
  URL     : https://www.kaggle.com/datasets/swaptr/layoffs-2022
  
  HOW TO RUN:
    1. Download the CSV from Kaggle, rename it  →  layoffs_raw.csv
    2. Place layoffs_raw.csv in the SAME folder as this script
    3. pip install pandas numpy openpyxl
    4. python 01_clean_data.py
    
  OUTPUT:
    layoffs_clean.csv   ← load this into Power BI
    layoffs_clean.xlsx  ← backup Excel version
===============================================================
"""

import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Tech Workforce Intelligence — Data Cleaner")
print("=" * 55)

df = pd.read_csv("layoffs_raw.csv")
print(f"\n📂 Raw data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}\n")

# ── 1. Standardize column names ───────────────────────────────────────────────
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

rename_map = {
    "company":               "company",
    "location":              "city",
    "industry":              "industry",
    "total_laid_off":        "laid_off_count",
    "percentage_laid_off":   "pct_laid_off",
    "date":                  "date",
    "stage":                 "funding_stage",
    "country":               "country",
    "funds_raised_millions": "funds_raised_m",
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# ── 2. Parse dates ─────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"]       = df["date"].dt.year
df["month_num"]  = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b %Y")
df["quarter"]    = df["date"].dt.year.astype(str) + " Q" + df["date"].dt.quarter.astype(str)

# ── 3. Numeric columns ────────────────────────────────────────────────────────
for col in ["laid_off_count", "pct_laid_off", "funds_raised_m"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 4. Clean & standardize text columns ──────────────────────────────────────
for col in ["company", "industry", "funding_stage", "country", "city"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        df[col] = df[col].replace("Nan", np.nan).fillna("Unknown")

# ── 5. Group funding stages (messy raw values → clean buckets) ────────────────
stage_map = {
    "Series A": "Early Stage",  "Series B": "Early Stage",
    "Series C": "Growth Stage", "Series D": "Growth Stage",
    "Series E": "Late Stage",   "Series F": "Late Stage",
    "Series G": "Late Stage",   "Series H": "Late Stage",
    "Series I": "Late Stage",   "Series J": "Late Stage",
    "Ipo":      "Public",       "Post-Ipo": "Public",
    "Acquired": "Acquired",     "Private Equity": "Private Equity",
    "Seed":     "Seed",         "Unknown":  "Unknown",
}
df["stage_group"] = df["funding_stage"].map(stage_map).fillna("Other")

# ── 6. Layoff severity label ──────────────────────────────────────────────────
def get_severity(row):
    n = row.get("laid_off_count", np.nan)
    p = row.get("pct_laid_off",   np.nan)
    if pd.isna(n) and pd.isna(p):
        return "Unknown"
    pct = p if not pd.isna(p) else 0
    cnt = n if not pd.isna(n) else 0
    if pct >= 0.8 or cnt >= 5000:  return "Critical"
    elif pct >= 0.4 or cnt >= 1000: return "High"
    elif pct >= 0.2 or cnt >= 200:  return "Medium"
    else:                            return "Low"

df["severity"] = df.apply(get_severity, axis=1)

# ── 7. Filter to 2020–2024 ────────────────────────────────────────────────────
df = df.dropna(subset=["company", "date"])
df = df[df["year"].between(2020, 2024)].copy()

# ── 8. Export ──────────────────────────────────────────────────────────────────
df.to_csv("layoffs_clean.csv",  index=False)
df.to_excel("layoffs_clean.xlsx", index=False)

print(f"✅ Cleaned file saved!")
print(f"   Rows      : {len(df):,}")
print(f"   Columns   : {df.shape[1]}")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Countries : {df['country'].nunique()}")
print(f"   Industries: {df['industry'].nunique()}")
print(f"\n📋 Final columns:")
for c in df.columns:
    sample = df[c].dropna().iloc[0] if not df[c].dropna().empty else "—"
    print(f"   {c:<22} e.g. → {sample}")
print("\n✅ Load  layoffs_clean.csv  into Power BI!")
