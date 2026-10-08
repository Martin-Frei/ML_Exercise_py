"""Streamlit page: XGBoost – Per-Class Target Encoding (Notebook #12)."""

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from shared.data_loading import load_training_data

TARGET = "meat_type"
FEATURES = [
    "fat_content_pct", "protein_pct", "marbling_score", "animal_age_months",
    "storage_days", "organic", "cut_quality", "price_eur_per_kg",
]
TE_FEATURES = ["cut_quality", "organic", "marbling_score", "price_bin"]
N_BINS = 5
N_CLASSES = 5
SMOOTHING = 10
CLASS_LABELS = [1, 2, 3, 4, 5]


def make_bin_edges(X_train):
    inner = X_train["price_eur_per_kg"].quantile(
        np.linspace(0, 1, N_BINS + 1)[1:-1]
    ).values
    return np.concatenate([[-np.inf], inner, [np.inf]])


def price_to_bin(price_series, bin_edges):
    return pd.cut(price_series, bins=bin_edges, labels=False)


def fit_per_class_encoding(values, y_shifted, m=SMOOTHING):
    """Per-class smoothed share: 5 encodings (one per class) for each feature value."""
    n = len(y_shifted)
    enc = {}
    global_priors = {}
    for k in range(N_CLASSES):
        is_k = (y_shifted == k).astype(float)
        prior = is_k.mean()
        stats = is_k.groupby(values).agg(["mean", "count"])
        col_enc = (stats["count"] * stats["mean"] + m * prior) / (stats["count"] + m)
        enc[k] = col_enc
        global_priors[k] = prior
    return enc, global_priors


def apply_per_class_encoding(values, enc, global_priors):
    """Return DataFrame with N_CLASSES columns for one feature."""
    cols = {}
    for k in range(N_CLASSES):
        out = values.map(enc[k])
        out[values.notna() & out.isna()] = global_priors[k]
        cols[k] = out.astype(float).values
    return cols


def build_te_features(X_tr, X_others, y_shifted, bin_edges):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def add_bin(X):
        X = X.copy()
        X["price_bin"] = price_to_bin(X["price_eur_per_kg"], bin_edges)
        return X

    X_tr = add_bin(X_tr)
    X_others = [add_bin(X) for X in X_others]

    # Full maps for test/prediction
    full_enc = {}
    full_priors = {}
    for feat in TE_FEATURES:
        enc, priors = fit_per_class_encoding(X_tr[feat], y_shifted)
        full_enc[feat] = enc
        full_priors[feat] = priors

    # OOF encode training rows (28 new columns: 4 features × 5 classes)
    te_cols = {f"{feat}_te_c{k+1}": np.full(len(X_tr), np.nan) for feat in TE_FEATURES for k in range(N_CLASSES)}
    idx_arr = np.arange(len(X_tr))
    for fit_idx, enc_idx in skf.split(X_tr, y_shifted):
        for feat in TE_FEATURES:
            enc, priors = fit_per_class_encoding(
                X_tr[feat].iloc[fit_idx], y_shifted.iloc[fit_idx]
            )
            col_vals = apply_per_class_encoding(X_tr[feat].iloc[enc_idx], enc, priors)
            for k in range(N_CLASSES):
                arr = te_cols[f"{feat}_te_c{k+1}"]
                arr[enc_idx] = col_vals[k]

    te_tr_df = pd.DataFrame(te_cols, index=X_tr.index)
    X_tr_enc = pd.concat([X_tr[FEATURES], te_tr_df], axis=1)

    result_others = []
    for X_o in X_others:
        te_rows = {}
        for feat in TE_FEATURES:
            col_vals = apply_per_class_encoding(X_o[feat], full_enc[feat], full_priors[feat])
            for k in range(N_CLASSES):
                te_rows[f"{feat}_te_c{k+1}"] = col_vals[k]
        te_o_df = pd.DataFrame(te_rows, index=X_o.index)
        result_others.append(pd.concat([X_o[FEATURES], te_o_df], axis=1))

    return X_tr_enc, result_others, full_enc, full_priors


@st.cache_data
def get_df():
    return load_training_data()


@st.cache_resource
def train():
    df = get_df()
    X = df[FEATURES]
    y_shifted = df[TARGET] - 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_shifted, test_size=0.2, random_state=42, stratify=y_shifted
    )
    bin_edges = make_bin_edges(X_tr)
    X_tr_enc, (X_te_enc,), full_enc, full_priors = build_te_features(
        X_tr, [X_te], y_tr, bin_edges
    )
    all_cols = list(X_tr_enc.columns)
    model = XGBClassifier(
        objective="multi:softprob", n_estimators=150, learning_rate=0.08,
        max_depth=3, random_state=42, eval_metric="mlogloss",
    )
    model.fit(X_tr_enc, y_tr)
    return model, X_te_enc, y_te + 1, all_cols, bin_edges, full_enc, full_priors


# ------------------------------------------------------------

st.title("⚡ XGBoost – Per-Class Target Encoding")
st.markdown(
    "Encodes each feature with **5 per-class probability columns** (instead "
    "of 1 mean column in #11). Result: 8 + 4 × 5 = **28 features**. "
    "Each `{feat}_te_c{k}` is the smoothed share of class k for that feature value. "
    "OOF 5-fold, smoothing m=10, `max_depth=3`."
)

with st.spinner("Training model (per-class OOF encoding) …"):
    model, X_test, y_test, all_cols, bin_edges, full_enc, full_priors = train()
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
c2.metric("Total features", len(all_cols))
c3.metric("TE feature cols", 4 * N_CLASSES)
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

# Feature importance (grouped by original feature)
st.header("Feature importance")
st.subheader("Grouped by original feature")
imp_series = pd.Series(model.feature_importances_, index=all_cols)
groups = {}
for feat in FEATURES:
    # Original feature column importance
    if feat in imp_series.index:
        groups[feat] = imp_series[feat]
    # Sum of per-class TE columns
    te_cols_feat = [c for c in all_cols if c.startswith(f"{feat}_te_")]
    if te_cols_feat:
        groups[f"{feat}_te (sum)"] = imp_series[te_cols_feat].sum()
grouped = pd.Series(groups).sort_values(ascending=True)
st.bar_chart(grouped)

with st.expander("All 28 feature importances"):
    st.bar_chart(imp_series.sort_values(ascending=True))

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
    row_base = pd.DataFrame([{
        "fat_content_pct": in_fat, "protein_pct": in_protein,
        "marbling_score": in_marbling, "animal_age_months": in_age,
        "storage_days": in_storage, "organic": in_organic,
        "cut_quality": in_quality, "price_eur_per_kg": in_price,
    }])
    row_base["price_bin"] = price_to_bin(row_base["price_eur_per_kg"], bin_edges)
    te_row = {}
    for feat in TE_FEATURES:
        col_vals = apply_per_class_encoding(row_base[feat], full_enc[feat], full_priors[feat])
        for k in range(N_CLASSES):
            te_row[f"{feat}_te_c{k+1}"] = col_vals[k][0]
    row_enc = pd.concat([row_base[FEATURES], pd.DataFrame([te_row])], axis=1)
    pred_raw = int(model.predict(row_enc)[0])
    pred_class = pred_raw + 1
    proba = model.predict_proba(row_enc)[0]
    st.success(f"### Predicted meat type: **{pred_class}**")
    proba_df = pd.Series(proba, index=[f"Type {c}" for c in CLASS_LABELS], name="Probability")
    st.bar_chart(proba_df)
