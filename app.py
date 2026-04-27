"""
app.py — Customer Churn Prediction Dashboard
Run: streamlit run app.py
"""

import os
import pickle
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_artefacts():
    model_dir = "models"
    if not os.path.exists(f"{model_dir}/model.pkl"):
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()
    with open(f"{model_dir}/model.pkl",        "rb") as f: model        = pickle.load(f)
    with open(f"{model_dir}/preprocessor.pkl", "rb") as f: preprocessor = pickle.load(f)
    with open(f"{model_dir}/explainer.pkl",    "rb") as f: explainer    = pickle.load(f)
    with open(f"{model_dir}/feature_meta.json","r")  as f: meta         = json.load(f)
    return model, preprocessor, explainer, meta

model, preprocessor, explainer, meta = load_artefacts()

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; }
.risk-high   { color: #dc2626; font-weight: 700; font-size: 1.4rem; }
.risk-medium { color: #d97706; font-weight: 700; font-size: 1.4rem; }
.risk-low    { color: #16a34a; font-weight: 700; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

st.title("📉 Customer Churn Prediction System")
st.markdown(
    f"**Model:** {meta['best_model']}  ·  "
    f"**ROC-AUC:** {meta['roc_auc']}  ·  "
    f"**PR-AUC:** {meta['pr_auc']}"
)
st.divider()

st.sidebar.header("Customer Profile")
st.sidebar.markdown("Fill in the customer details to predict churn risk.")

with st.sidebar:
    st.markdown("##### Account")
    tenure            = st.slider("Tenure (months)", 1, 72, 12)
    contract          = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method    = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    st.markdown("##### Charges")
    monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total charges ($)", min_value=0.0,
                                      value=float(monthly_charges * tenure), step=10.0)
    st.markdown("##### Demographics")
    senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner        = st.selectbox("Has partner",    ["Yes", "No"])
    dependents     = st.selectbox("Has dependents", ["Yes", "No"])
    st.markdown("##### Services")
    phone_service    = st.selectbox("Phone service",    ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple lines",   ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    online_security  = st.selectbox("Online security",  ["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Tech support",     ["Yes", "No", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

num_services = sum([
    phone_service    == "Yes",
    multiple_lines   == "Yes",
    online_security  == "Yes",
    tech_support     == "Yes",
    streaming_tv     == "Yes",
    streaming_movies == "Yes",
])
charges_per_month = total_charges / tenure if tenure > 0 else monthly_charges
is_long_term      = int(contract != "Month-to-month")
has_support       = int(online_security == "Yes" or tech_support == "Yes")

input_data = pd.DataFrame([{
    "senior_citizen":    senior_citizen,
    "partner":           partner,
    "dependents":        dependents,
    "tenure":            tenure,
    "phone_service":     phone_service,
    "multiple_lines":    multiple_lines,
    "internet_service":  internet_service,
    "online_security":   online_security,
    "tech_support":      tech_support,
    "streaming_tv":      streaming_tv,
    "streaming_movies":  streaming_movies,
    "contract":          contract,
    "paperless_billing": paperless_billing,
    "payment_method":    payment_method,
    "monthly_charges":   monthly_charges,
    "total_charges":     total_charges,
    "charges_per_month": charges_per_month,
    "is_long_term":      is_long_term,
    "has_support":       has_support,
    "num_services":      num_services,
}])

X_enc      = preprocessor.transform(input_data)
churn_prob = model.predict_proba(X_enc)[0][1]
churn_pct  = churn_prob * 100

if churn_prob >= 0.70:
    risk_label = "🔴 HIGH RISK"
    risk_class = "risk-high"
    action     = "Immediate retention intervention required."
elif churn_prob >= 0.40:
    risk_label = "🟡 MEDIUM RISK"
    risk_class = "risk-medium"
    action     = "Monitor closely. Consider a loyalty offer."
else:
    risk_label = "🟢 LOW RISK"
    risk_class = "risk-low"
    action     = "Customer appears stable. No action needed."

col1, col2, col3, col4 = st.columns(4)
col1.metric("Churn Probability", f"{churn_pct:.1f}%")
col2.metric("Tenure",            f"{tenure} months")
col3.metric("Monthly Charges",   f"${monthly_charges:.2f}")
col4.metric("Services Active",   str(num_services))

st.markdown(f'<p class="{risk_class}">{risk_label}</p>', unsafe_allow_html=True)
st.caption(action)
st.divider()

col_shap, col_eval = st.columns([1, 1])

with col_shap:
    st.subheader("Why this prediction?")
    st.caption("SHAP values show which features pushed the risk up (red) or down (blue).")

    try:
        raw = explainer.shap_values(X_enc)
        if isinstance(raw, list):
            sv = raw[1][0]
        elif hasattr(raw, 'ndim') and raw.ndim == 2:
            sv = raw[0]
        else:
            sv = raw

        feat_names = meta["all"]
        shap_df = pd.DataFrame({"feature": feat_names, "value": sv})
        shap_df = shap_df.reindex(shap_df["value"].abs().sort_values(ascending=True).index)
        top = shap_df.tail(12)
        colors = ["#dc2626" if v > 0 else "#2563eb" for v in top["value"]]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(top["feature"], top["value"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on churn probability)")
        ax.set_title("Feature contributions for this customer")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP chart unavailable: {e}")

with col_eval:
    st.subheader("Model Performance")
    st.caption("Evaluation on held-out test set (20% of data).")
    if os.path.exists("assets/evaluation.png"):
        st.image("assets/evaluation.png", use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate evaluation plots.")
    st.divider()
    st.subheader("SHAP Summary — Full Dataset")
    st.caption("Global feature importance across all customers.")
    if os.path.exists("assets/shap_summary.png"):
        st.image("assets/shap_summary.png", use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate SHAP summary.")

st.divider()
st.subheader("Customer Risk Profile")

profile = {
    "Contract Type":     contract,
    "Internet Service":  internet_service,
    "Payment Method":    payment_method,
    "Online Security":   online_security,
    "Tech Support":      tech_support,
    "Paperless Billing": paperless_billing,
    "Total Services":    str(num_services),
    "Estimated CLV ($)": f"{monthly_charges * 24:.2f}",
}
st.dataframe(
    pd.DataFrame(profile.items(), columns=["Attribute", "Value"]),
    use_container_width=True,
    hide_index=True,
)