"""Streamlit page: Decision Tree – Meat Type Classification (Notebook #1)."""

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
from sklearn.tree import DecisionTreeClassifier

from shared.data_loading import load_training_data

TARGET = "meat_type"
FEATURES_BASE = [
    "fat_content_pct", "protein_pct", "marbling_score", "animal_age_months",
    "storage_days", "organic", "cut_quality", "price_eur_per_kg",
]
FEATURES_ENG = ["price_per_protein", "fat_to_protein_ratio", "price_x_marbling"]
FEATURES = FEATURES_BASE + FEATURES_ENG
CLASS_LABELS = [1, 2, 3, 4, 5]


@st.cache_data
def get_df():
    """Load and engineer features; cache the result."""
    df = load_training_data()
    df["price_per_protein"] = df["price_eur_per_kg"] / df["protein_pct"]
    df["fat_to_protein_ratio"] = df["fat_content_pct"] / df["protein_pct"]
    df["price_x_marbling"] = df["price_eur_per_kg"] * df["marbling_score"]
    return df


@st.cache_resource
def train():
    """Train a Decision Tree classifier and return (model, X_test, y_test)."""
    df = get_df()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------

st.title("🌳 Decision Tree – Meat Type Classification")
st.markdown(
    """
    Predicts **meat type (1–5)** from 11 features: 8 raw columns plus
    three price-derived features (`price_per_protein`, `fat_to_protein_ratio`,
    `price_x_marbling`). No label leakage — the target is meat type, not price.
    """
)

with st.spinner("Training model …"):
    model, X_test, y_test = train()
    df_train = get_df()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Filter the data")
meat_types = sorted(df_train[TARGET].unique().tolist())
sel_meat = st.sidebar.multiselect("Meat type", meat_types, default=meat_types)
organic_opt = st.sidebar.radio("Organic", ["All", "Only organic", "Only non-organic"])
n_rows = st.sidebar.slider("Rows to show", 5, 50, 10, step=5)

df_f = df_train[df_train[TARGET].isin(sel_meat)]
if organic_opt == "Only organic":
    df_f = df_f[df_f["organic"] == 1]
elif organic_opt == "Only non-organic":
    df_f = df_f[df_f["organic"] == 0]

# ------------------------------------------------------------
# Key metrics
# ------------------------------------------------------------

st.header("Key metrics")
y_pred_all = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_all)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", len(df_f))
c2.metric("Classes in filter", df_f[TARGET].nunique())
c3.metric("Most common class", int(df_f[TARGET].mode()[0]) if len(df_f) else "—")
c4.metric("Test accuracy", f"{acc:.2%}")

# ------------------------------------------------------------
# Data explorer
# ------------------------------------------------------------

st.header("Data explorer")
st.dataframe(df_f.head(n_rows), width="stretch", hide_index=True)

# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------

st.header("Model performance")
prec = precision_score(y_test, y_pred_all, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred_all, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred_all, average="macro", zero_division=0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("Precision (macro)", f"{prec:.2%}")
c3.metric("Recall (macro)", f"{rec:.2%}")
c4.metric("F1 (macro)", f"{f1:.2%}")

# Classification report per class
st.subheader("Per-class metrics")
report_dict = classification_report(
    y_test, y_pred_all, labels=CLASS_LABELS, output_dict=True, zero_division=0
)
report_df = (
    pd.DataFrame(report_dict)
    .T.loc[[str(c) for c in CLASS_LABELS]]
    .rename(columns={"support": "support"})
    .round(3)
)
st.dataframe(report_df, width="stretch")

# Confusion matrix
st.subheader("Confusion matrix")
cm = confusion_matrix(y_test, y_pred_all, labels=CLASS_LABELS)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
ax.set_xlabel("Predicted meat type")
ax.set_ylabel("True meat type")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

st.header("Feature importance")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
st.bar_chart(importances)

# ------------------------------------------------------------
# Predict meat type
# ------------------------------------------------------------

st.header("Predict meat type")
st.markdown("Enter feature values and click **Predict** to see the model's classification.")

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
        "price_per_protein": in_price / in_protein,
        "fat_to_protein_ratio": in_fat / in_protein,
        "price_x_marbling": in_price * in_marbling,
    }])[FEATURES]
    pred_class = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    st.success(f"### Predicted meat type: **{pred_class}**")
    proba_df = pd.Series(proba, index=[f"Type {c}" for c in CLASS_LABELS], name="Probability")
    st.bar_chart(proba_df)
