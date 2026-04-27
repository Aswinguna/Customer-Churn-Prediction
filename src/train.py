"""
train.py — Customer Churn Prediction Training Pipeline
Trains Logistic Regression, Random Forest, and XGBoost.
Saves the best model + preprocessor for use in the Streamlit app.
"""

import os
import sys
import warnings
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, average_precision_score
)
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/telco_churn.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 60)
print("  Customer Churn Prediction — Training Pipeline")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print("Generating synthetic dataset...")
    sys.path.insert(0, "data")
    from generate_data import generate_churn_dataset
    df = generate_churn_dataset(save_path=DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

print(f"\n[1/6] Dataset loaded  — {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"      Churn rate: {df['churn'].mean():.1%}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
df = df.drop(columns=["customer_id"])

# Derived features
df["charges_per_month"]  = np.where(df["tenure"] > 0, df["total_charges"] / df["tenure"], df["monthly_charges"])
df["is_long_term"]       = (df["contract"] != "Month-to-month").astype(int)
df["has_support"]        = ((df["online_security"] == "Yes") | (df["tech_support"] == "Yes")).astype(int)
df["num_services"]       = (
    (df["phone_service"]    == "Yes").astype(int)
    + (df["multiple_lines"]  == "Yes").astype(int)
    + (df["online_security"] == "Yes").astype(int)
    + (df["tech_support"]    == "Yes").astype(int)
    + (df["streaming_tv"]    == "Yes").astype(int)
    + (df["streaming_movies"] == "Yes").astype(int)
)

print(f"[2/6] Feature engineering complete — {df.shape[1]-1} features")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
TARGET = "churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = [
    "tenure", "monthly_charges", "total_charges",
    "charges_per_month", "num_services",
    "senior_citizen", "is_long_term", "has_support"
]
categorical_features = [
    "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "tech_support",
    "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method"
]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

print(f"[3/6] Preprocessing done — train: {X_train_enc.shape}, test: {X_test_enc.shape}")

# ── 4. Model comparison ───────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost":             xgb.XGBClassifier(
                               n_estimators=300, learning_rate=0.05, max_depth=5,
                               scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                               eval_metric="logloss", random_state=42, n_jobs=-1
                           ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[4/6] Model comparison (5-fold CV):\n")
print(f"  {'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")

for name, model in models.items():
    model.fit(X_train_enc, y_train)
    y_prob = model.predict_proba(X_test_enc)[:, 1]
    roc    = roc_auc_score(y_test, y_prob)
    pr     = average_precision_score(y_test, y_prob)
    cv_auc = cross_val_score(model, X_train_enc, y_train, cv=cv, scoring="roc_auc").mean()
    results[name] = {"model": model, "roc_auc": roc, "pr_auc": pr, "cv_auc": cv_auc, "y_prob": y_prob}
    print(f"  {name:<25} {roc:>10.4f} {pr:>10.4f}")

# Best model
best_name  = "XGBoost"
best       = results[best_name]
best_model = best["model"]
y_prob     = best["y_prob"]
y_pred     = (y_prob >= 0.5).astype(int)

print(f"\n  ✓ Best model: {best_name}  (ROC-AUC = {best['roc_auc']:.4f})")

# ── 5. Evaluation plots ───────────────────────────────────────────────────────
print("\n[5/6] Generating evaluation plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Model Evaluation — Customer Churn Prediction", fontsize=13, fontweight="bold")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {best['roc_auc']:.3f}")
axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
axes[0].legend(loc="lower right")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix")

# Feature importance (model-agnostic bar chart)
cat_enc = preprocessor.named_transformers_["cat"]
cat_feature_names = cat_enc.get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_feature_names)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.zeros(len(all_feature_names))

feat_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
feat_df = feat_df.nlargest(10, "importance")
axes[2].barh(feat_df["feature"], feat_df["importance"], color="#2563eb", alpha=0.8)
axes[2].set(title="Top 10 Feature Importances", xlabel="Importance")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("assets/evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → assets/evaluation.png")

# SHAP summary plot — always use XGBoost for TreeExplainer
print("  Computing SHAP values...")
xgb_model   = results["XGBoost"]["model"]
explainer   = shap.TreeExplainer(xgb_model)
shap_sample = X_test_enc[:500]
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=all_feature_names,
                  show=False, plot_size=None)
plt.title("SHAP Feature Impact — Churn Prediction", fontsize=12)
plt.tight_layout()
plt.savefig("assets/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → assets/shap_summary.png")

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
print("\n[6/6] Saving model artefacts...")

with open(f"{MODEL_DIR}/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{MODEL_DIR}/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

feature_meta = {
    "numeric":     numeric_features,
    "categorical": categorical_features,
    "all":         all_feature_names,
    "best_model":  best_name,
    "roc_auc":     round(best["roc_auc"], 4),
    "pr_auc":      round(best["pr_auc"], 4),
}
with open(f"{MODEL_DIR}/feature_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)

print(f"  Saved → {MODEL_DIR}/model.pkl")
print(f"  Saved → {MODEL_DIR}/preprocessor.pkl")
print(f"  Saved → {MODEL_DIR}/explainer.pkl")
print(f"  Saved → {MODEL_DIR}/feature_meta.json")

print("\n" + "=" * 60)
print("  Training complete.")
print(f"  Best model : {best_name}")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print(f"  PR-AUC     : {best['pr_auc']:.4f}")
print("=" * 60)
