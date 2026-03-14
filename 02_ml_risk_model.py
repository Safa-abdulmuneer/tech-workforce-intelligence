"""
===============================================================
  Tech Workforce Intelligence — Step 2: ML Risk Score Model
===============================================================
  INPUT  : layoffs_clean.csv  (output from 01_clean_data.py)
  OUTPUT : layoffs_with_risk.csv  ← final file for Power BI

  WHAT IT DOES:
    - Trains a Random Forest classifier to predict layoff
      severity (Low / Medium / High / Critical)
    - Adds a 0–100 "risk_score" column for each company event
    - Prints model accuracy + feature importances
    - Saves the enriched CSV for Power BI

  HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python 02_ml_risk_model.py
===============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  Tech Workforce Intelligence — ML Risk Model")
print("=" * 55)

# ── Load cleaned data ─────────────────────────────────────────
df = pd.read_csv("layoffs_clean.csv")
print(f"\n📂 Loaded: {len(df):,} rows")

# ── Feature Engineering ───────────────────────────────────────

# 1. Encode categorical features
le_stage    = LabelEncoder()
le_industry = LabelEncoder()
le_country  = LabelEncoder()

df["stage_enc"]    = le_stage.fit_transform(df["stage_group"].fillna("Unknown"))
df["industry_enc"] = le_industry.fit_transform(df["industry"].fillna("Unknown"))
df["country_enc"]  = le_country.fit_transform(df["country"].fillna("Unknown"))

# 2. Fill numeric features
df["funds_raised"]  = df["funds_raised"].fillna(df["funds_raised"].median())
df["pct_laid_off"]    = df["pct_laid_off"].fillna(0)
df["laid_off_count"]  = df["laid_off_count"].fillna(0)

# 3. Derived features
df["log_funds"]       = np.log1p(df["funds_raised"])
df["log_laid_off"]    = np.log1p(df["laid_off_count"])

# Industry-level stress (avg pct laid off per industry)
industry_stress = df.groupby("industry")["pct_laid_off"].mean().rename("industry_stress")
df = df.join(industry_stress, on="industry")

# Year-level trend weight (later years had more layoffs)
year_weight = {2020: 1.0, 2021: 0.8, 2022: 1.5, 2023: 2.0, 2024: 1.8}
df["year_weight"] = df["year"].map(year_weight).fillna(1.0)

# ── Feature list ──────────────────────────────────────────────
FEATURES = [
    "stage_enc",
    "industry_enc",
    "country_enc",
    "log_funds",
    "log_laid_off",
    "pct_laid_off",
    "industry_stress",
    "year_weight",
    "month_num",
]
# Only keep features that exist
FEATURES = [f for f in FEATURES if f in df.columns]

TARGET = "severity"

# ── Remove rows where target is Unknown (can't train on it) ──
df_model = df[df[TARGET] != "Unknown"].copy()
print(f"   Training rows (severity known): {len(df_model):,}")
print(f"   Class distribution:\n{df_model[TARGET].value_counts().to_string()}\n")

# ── Train / Test Split ────────────────────────────────────────
X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model: Random Forest ──────────────────────────────────────
print("🤖 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Test set report
y_pred = rf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ── Feature Importance Plot ───────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
importances.plot(kind="barh", color="#0891B2", ax=ax)
ax.set_title("Feature Importances — Layoff Risk Model", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\n📈 Saved: feature_importance.png")

# ── Confusion Matrix Plot ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=["Low","Medium","High","Critical"])
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low","Medium","High","Critical"],
            yticklabels=["Low","Medium","High","Critical"], ax=ax2)
ax2.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax2.set_ylabel("Actual"); ax2.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("📈 Saved: confusion_matrix.png")

# ── Generate risk_score (0–100) for ALL rows ─────────────────
# Uses predicted class probability to calculate a continuous score
# P(Critical)*100 + P(High)*65 + P(Medium)*35 + P(Low)*10

X_all = df[FEATURES]
proba = rf.predict_proba(X_all)
classes = list(rf.classes_)

score_weights = {"Critical": 100, "High": 65, "Medium": 35, "Low": 10}
risk_score = np.zeros(len(df))
for i, cls in enumerate(classes):
    w = score_weights.get(cls, 20)
    risk_score += proba[:, i] * w

df["risk_score"]       = risk_score.round(1)
df["risk_label"]       = rf.predict(X_all)
df["risk_probability"] = proba.max(axis=1).round(3)

# Risk tier for easy Power BI slicer
def tier(score):
    if score >= 75: return "🔴 High Risk"
    elif score >= 50: return "🟡 Medium Risk"
    else: return "🟢 Low Risk"

df["risk_tier"] = df["risk_score"].apply(tier)

# ── Save final file ───────────────────────────────────────────
out_cols = [c for c in df.columns if c not in
            ["stage_enc","industry_enc","country_enc",
             "log_funds","log_laid_off","year_weight","industry_stress"]]
df[out_cols].to_csv("layoffs_with_risk.csv", index=False)

print(f"\n✅ Final file saved: layoffs_with_risk.csv")
print(f"   New columns added: risk_score, risk_label, risk_tier, risk_probability")
print(f"\n   Risk tier breakdown:")
print(df["risk_tier"].value_counts().to_string())
print("\n✅ Load  layoffs_with_risk.csv  into Power BI!")
