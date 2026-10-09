"""Streamlit page: XGBoost Regression – Combined Dataset (Notebook #17)."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from shared.data_loading import load_stress_data, load_training_data

TARGET = "price_eur_per_kg"
FEATURES = [
    "meat_type", "fat_content_pct", "protein_pct", "marbling_score",
    "animal_age_months", "storage_days", "organic", "cut_quality",
    "protein_fat_ratio",
]


def add_protein_fat_ratio(df):
    df = df.copy()
    df["protein_fat_ratio"] = df["protein_pct"] / (df["fat_content_pct"] + 1e-5)
    return df


def group_wise_impute(df, group_col="meat_type"):
    """Fill NaN with the subgroup median (by meat_type) within df."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median())
            )
    return df


@st.cache_data
def get_df():
    df_clean = load_training_data()
    df_stress_raw = load_stress_data()

    # Drop stress rows with missing meat_type or price
    df_stress = df_stress_raw.dropna(subset=["meat_type", TARGET]).copy()
    df_stress = group_wise_impute(df_stress)

    df_clean = add_protein_fat_ratio(df_clean)
    df_stress = add_protein_fat_ratio(df_stress)

    df_clean["is_stress"] = 0
    df_stress["is_stress"] = 1
    combined = pd.concat([df_clean, df_stress], ignore_index=True)
    return combined


@st.cache_resource
def train():
    combined = get_df()
    X = combined[FEATURES]
    y = combined[TARGET]
    is_stress = combined["is_stress"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=is_stress
    )
    is_stress_te = combined.loc[y_te.index, "is_stress"]
    model = XGBRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
    )
    model.fit(X_tr, y_tr)
    return model, X_te, y_te, is_stress_te


# ------------------------------------------------------------

st.title("⚡ XGBoost Regression – Combined Dataset")
st.markdown(
    "Combines clean (1200) and stress (91 valid-price rows) into a single "
    "training set. NaN in stress data are filled via **group-wise imputation** "
    "(per-`meat_type` subgroup median). An `is_stress` flag stratifies the split. "
    "9 features (8 raw + `protein_fat_ratio`)."
)

with st.spinner("Training model …"):
    model, X_test, y_test, is_stress_test = train()
    df_combined = get_df()

# Sidebar
st.sidebar.header("Filter the data")
source_opt = st.sidebar.radio("Source", ["All", "Clean only", "Stress only"])
meat_opts = sorted(df_combined["meat_type"].unique().tolist())
sel_meat = st.sidebar.multiselect("Meat type", meat_opts, default=meat_opts)
organic_opt = st.sidebar.radio("Organic", ["All", "Only organic", "Only non-organic"])
price_range = st.sidebar.slider(
    "Price range (EUR/kg)", float(df_combined[TARGET].min()),
    float(df_combined[TARGET].max()),
    (float(df_combined[TARGET].min()), float(df_combined[TARGET].max()))
)
n_rows = st.sidebar.slider("Rows to show", 5, 50, 10, step=5)

df_f = df_combined[
    df_combined["meat_type"].isin(sel_meat) &
    df_combined[TARGET].between(price_range[0], price_range[1])
]
if source_opt == "Clean only":
    df_f = df_f[df_f["is_stress"] == 0]
elif source_opt == "Stress only":
    df_f = df_f[df_f["is_stress"] == 1]
if organic_opt == "Only organic":
    df_f = df_f[df_f["organic"] == 1]
elif organic_opt == "Only non-organic":
    df_f = df_f[df_f["organic"] == 0]

# Key metrics
y_pred = model.predict(X_test)
st.header("Key metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", len(df_f))
c2.metric("Combined rows", len(df_combined))
c3.metric("Clean rows", int((df_combined["is_stress"] == 0).sum()))
c4.metric("Stress rows", int((df_combined["is_stress"] == 1).sum()))

# Data explorer
st.header("Data explorer")
st.dataframe(df_f.head(n_rows), width="stretch", hide_index=True)

# Model performance
st.header("Model performance")
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.4f} EUR")
c2.metric("RMSE", f"{rmse:.4f} EUR")
c3.metric("R²", f"{r2:.4f}")

# Performance by source
st.subheader("Performance by source")
clean_mask = is_stress_test == 0
stress_mask = is_stress_test == 1
perf_data = []
for label, mask in [("Clean", clean_mask), ("Stress", stress_mask)]:
    if mask.sum() > 0:
        yt, yp = y_test[mask], y_pred[mask.values]
        perf_data.append({
            "Source": label, "N": int(mask.sum()),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "MAE": round(mean_absolute_error(yt, yp), 4),
            "R²": round(r2_score(yt, yp), 4),
        })
st.dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
st.scatter_chart(scatter_df, x="Actual", y="Predicted")

# Feature importance
st.header("Feature importance")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
st.bar_chart(importances)

# Predict price
st.header("Predict price")
c_a, c_b = st.columns(2)
with c_a:
    in_mt = st.selectbox("Meat type", [1, 2, 3, 4, 5])
    in_fat = st.slider("Fat content (%)", 2.0, 35.0, 18.0, 0.1)
    in_protein = st.slider("Protein (%)", 15.0, 26.0, 20.5, 0.1)
    in_marbling = st.slider("Marbling score", 0, 12, 5)
with c_b:
    in_age = st.slider("Animal age (months)", 2, 100, 36)
    in_storage = st.slider("Storage days", 0, 30, 10)
    in_organic = st.radio("Organic", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    in_quality = st.slider("Cut quality", 1, 5, 3)

if st.button("Predict", type="primary"):
    pfr = in_protein / (in_fat + 1e-5)
    row = pd.DataFrame([{
        "meat_type": in_mt, "fat_content_pct": in_fat, "protein_pct": in_protein,
        "marbling_score": in_marbling, "animal_age_months": in_age,
        "storage_days": in_storage, "organic": in_organic,
        "cut_quality": in_quality, "protein_fat_ratio": pfr,
    }])[FEATURES]
    pred = float(model.predict(row)[0])
    st.success(f"### Predicted price: **{pred:.2f} EUR/kg**")
    st.caption(f"protein_fat_ratio = {pfr:.4f}")
