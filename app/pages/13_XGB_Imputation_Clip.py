"""Streamlit page: XGBoost – Imputation & Outlier Clipping (Notebook #13)."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from shared.data_loading import load_training_data

TARGET = "meat_type"
FEATURES = [
    "fat_content_pct", "protein_pct", "marbling_score", "animal_age_months",
    "storage_days", "organic", "cut_quality", "price_eur_per_kg",
]
CLASS_LABELS = [1, 2, 3, 4, 5]


@st.cache_data
def get_df():
    return load_training_data()


@st.cache_resource
def train():
    df = get_df()
    X = df[FEATURES]
    y = df[TARGET].astype(int) - 1   # 1-5 -> 0-4
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr)
    train_min = X_tr.min()
    train_max = X_tr.max()
    train_medians = X_tr.median()
    return model, X_te, y_te + 1, train_min, train_max, train_medians


# ------------------------------------------------------------

st.title("⚡ XGBoost – Imputation & Outlier Clipping")
st.markdown(
    "Tests how different preprocessing strategies affect **stress-test** "
    "accuracy: native NaN, global median, KNN, and IterativeImputer — "
    "each with and without outlier clipping to training min/max. "
    "The model itself is identical to #10 but with `max_depth=3`. "
    "This page shows internal test performance; full strategy comparison is "
    "in `classification/xgb_classification_imputation_clip.py`."
)

with st.spinner("Training model …"):
    model, X_test, y_test, train_min, train_max, train_medians = train()
    df_train = get_df()

# Sidebar
st.sidebar.header("Filter the data")
meat_opts = sorted(df_train[TARGET].unique().tolist())
sel_meat = st.sidebar.multiselect("Meat type", meat_opts, default=meat_opts)
organic_opt = st.sidebar.radio("Organic", ["All", "Only organic", "Only non-organic"])
n_rows = st.sidebar.slider("Rows to show", 5, 50, 10, step=5)

df_f = df_train[df_train[TARGET].isin(sel_meat)]
if organic_opt == "Only organic":
    df_f = df_f[df_f["organic"] == 1]
elif organic_opt == "Only non-organic":
    df_f = df_f[df_f["organic"] == 0]

# Key metrics
y_pred_raw = model.predict(X_test) + 1
acc = accuracy_score(y_test, y_pred_raw)
st.header("Key metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", len(df_f))
c2.metric("Classes in filter", df_f[TARGET].nunique())
c3.metric("max_depth", 3)
c4.metric("Test accuracy", f"{acc:.2%}")

# Data explorer
st.header("Data explorer")
st.dataframe(df_f.head(n_rows), use_container_width=True, hide_index=True)

# Model performance
st.header("Model performance")
prec = precision_score(y_test, y_pred_raw, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred_raw, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred_raw, average="macro", zero_division=0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("Precision (macro)", f"{prec:.2%}")
c3.metric("Recall (macro)", f"{rec:.2%}")
c4.metric("F1 (macro)", f"{f1:.2%}")

st.subheader("Per-class metrics")
report_dict = classification_report(
    y_test, y_pred_raw, labels=CLASS_LABELS, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T.loc[[str(c) for c in CLASS_LABELS]].round(3)
st.dataframe(report_df, use_container_width=True)

st.subheader("Confusion matrix")
cm = confusion_matrix(y_test, y_pred_raw, labels=CLASS_LABELS)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Feature importance
st.header("Feature importance")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
st.bar_chart(importances)

# Strategy overview
st.header("Imputation strategies (from notebook)")
st.markdown(
    """
    | Strategy | Clipping | Notes |
    |---|---|---|
    | Native NaN | No / Yes | XGBoost default missing direction |
    | Global median | No / Yes | Training-set medians, no leakage |
    | KNN (k=5) | No / Yes | StandardScaler + KNNImputer on training data |
    | Iterative | No / Yes | IterativeImputer, 10 rounds, rounded for integer features |
    | **LEAKY** group median | — | Uses true meat_type label — invalid in production |

    See the standalone script for full results on 91 stress-test rows.
    """
)

# Predict meat type (with optional clipping)
st.header("Predict meat type")
apply_clip = st.checkbox("Apply outlier clipping to training bounds", value=True)
c_a, c_b = st.columns(2)
with c_a:
    in_fat = st.slider("Fat content (%)", 2.0, 35.0, 18.0, 0.1)
    in_protein = st.slider("Protein (%)", 15.0, 26.0, 20.5, 0.1)
    in_marbling = st.slider("Marbling score", 0, 12, 5)
    in_age = st.slider("Animal age (months)", 2, 100, 36)
with c_b:
    in_storage = st.slider("Storage days", 0, 30, 10)
    in_organic = st.radio("Organic", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    in_quality = st.slider("Cut quality", 1, 5, 3)
    in_price = st.slider("Price (EUR/kg)", 2.5, 65.0, 23.0, 0.5)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "fat_content_pct": in_fat, "protein_pct": in_protein,
        "marbling_score": in_marbling, "animal_age_months": in_age,
        "storage_days": in_storage, "organic": in_organic,
        "cut_quality": in_quality, "price_eur_per_kg": in_price,
    }])[FEATURES]
    if apply_clip:
        row = row.clip(lower=train_min, upper=train_max, axis=1)
    pred_raw = int(model.predict(row)[0])
    pred_class = pred_raw + 1
    proba = model.predict_proba(row)[0]
    st.success(f"### Predicted meat type: **{pred_class}**")
    proba_df = pd.Series(proba, index=[f"Type {c}" for c in CLASS_LABELS], name="Probability")
    st.bar_chart(proba_df)
