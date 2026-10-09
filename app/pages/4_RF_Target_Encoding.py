"""Streamlit page: Random Forest – Target Encoding (Notebook #9)."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from shared.data_loading import load_training_data

TARGET = "meat_type"
FEATURES = [
    "fat_content_pct", "protein_pct", "marbling_score", "animal_age_months",
    "storage_days", "organic", "cut_quality", "price_eur_per_kg",
]
ENCODE_COLS = ["cut_quality", "organic", "marbling_score"]
N_PRICE_BINS = 5
CLASS_LABELS = [1, 2, 3, 4, 5]


def build_encoding_maps(X_train, y_train):
    """Compute target-encoding maps and price bin edges from training data."""
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    enc = {}
    for col in ENCODE_COLS:
        enc[col] = train_df.groupby(col)[TARGET].mean()
    price_bins = pd.qcut(train_df["price_eur_per_kg"], q=N_PRICE_BINS, labels=False)
    train_df["price_bin"] = price_bins
    enc["price_bin"] = train_df.groupby("price_bin")[TARGET].mean()
    _, bin_edges = pd.qcut(
        train_df["price_eur_per_kg"], q=N_PRICE_BINS, retbins=True
    )
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return enc, bin_edges


def apply_target_encoding(X_data, enc, bin_edges):
    """Add four _te columns to X_data using pre-computed maps."""
    X_enc = X_data.copy()
    global_means = {col: enc[col].mean() for col in ENCODE_COLS + ["price_bin"]}
    for col in ENCODE_COLS:
        X_enc[f"{col}_te"] = X_data[col].map(enc[col]).fillna(global_means[col])
    price_bin = pd.cut(X_data["price_eur_per_kg"], bins=bin_edges, labels=False)
    X_enc["price_bin_te"] = price_bin.map(enc["price_bin"]).fillna(global_means["price_bin"])
    return X_enc


@st.cache_data
def get_df():
    return load_training_data()


@st.cache_resource
def train():
    df = get_df()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    enc, bin_edges = build_encoding_maps(X_tr, y_tr)
    X_tr_enc = apply_target_encoding(X_tr, enc, bin_edges)
    X_te_enc = apply_target_encoding(X_te, enc, bin_edges)
    all_cols = list(X_tr_enc.columns)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=7, n_jobs=-1, random_state=42
    )
    model.fit(X_tr_enc, y_tr)
    return model, X_te_enc, y_te, enc, bin_edges, all_cols


# ------------------------------------------------------------

st.title("🌲 Random Forest – Target Encoding")
st.markdown(
    "Extends the RF classifier with 4 target-encoded features: "
    "`cut_quality_te`, `organic_te`, `marbling_score_te`, `price_bin_te`. "
    "Encoding maps are built from training data only (no leakage). "
    "12 features total (8 raw + 4 encoded)."
)

with st.spinner("Training model …"):
    model, X_test, y_test, enc, bin_edges, all_cols = train()
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
y_pred_all = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_all)
st.header("Key metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", len(df_f))
c2.metric("Classes in filter", df_f[TARGET].nunique())
c3.metric("Most common class", int(df_f[TARGET].mode()[0]) if len(df_f) else "—")
c4.metric("Test accuracy", f"{acc:.2%}")

# Data explorer
st.header("Data explorer")
st.dataframe(df_f.head(n_rows), width="stretch", hide_index=True)

# Model performance
st.header("Model performance")
prec = precision_score(y_test, y_pred_all, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred_all, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred_all, average="macro", zero_division=0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("Precision (macro)", f"{prec:.2%}")
c3.metric("Recall (macro)", f"{rec:.2%}")
c4.metric("F1 (macro)", f"{f1:.2%}")

st.subheader("Per-class metrics")
report_dict = classification_report(
    y_test, y_pred_all, labels=CLASS_LABELS, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T.loc[[str(c) for c in CLASS_LABELS]].round(3)
st.dataframe(report_df, width="stretch")

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

# Feature importance
st.header("Feature importance")
importances = pd.Series(model.feature_importances_, index=all_cols).sort_values(ascending=True)
st.bar_chart(importances)

# Encoding maps
with st.expander("Target encoding maps"):
    for col in ENCODE_COLS:
        st.write(f"**{col}**")
        st.dataframe(enc[col].rename("mean_meat_type").reset_index(), width="stretch")
    st.write("**price_bin** (5 quantile bins)")
    st.dataframe(enc["price_bin"].rename("mean_meat_type").reset_index(), width="stretch")

# Predict meat type
st.header("Predict meat type")
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
    row_enc = apply_target_encoding(row, enc, bin_edges)
    pred_class = int(model.predict(row_enc)[0])
    proba = model.predict_proba(row_enc)[0]
    st.success(f"### Predicted meat type: **{pred_class}**")
    proba_df = pd.Series(proba, index=[f"Type {c}" for c in CLASS_LABELS], name="Probability")
    st.bar_chart(proba_df)
