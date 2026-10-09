"""Streamlit page: Random Forest – Outlier Clipping & Imputation (Notebook #8)."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shared.data_loading import load_training_data

TARGET = "price_eur_per_kg"
FEATURES = [
    "meat_type", "fat_content_pct", "protein_pct", "marbling_score",
    "animal_age_months", "storage_days", "organic", "cut_quality",
]
CLIP_LOWER_PCTILE = 0.01
CLIP_UPPER_PCTILE = 0.99


@st.cache_data
def get_df():
    return load_training_data()


@st.cache_resource
def train():
    df = get_df()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=7, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    # Store KNN imputer and clip bounds for the prediction widget
    knn = KNNImputer(n_neighbors=5)
    knn.fit(X_tr)
    clip_lo = X_tr.quantile(CLIP_LOWER_PCTILE)
    clip_hi = X_tr.quantile(CLIP_UPPER_PCTILE)
    return model, X_te, y_te, knn, clip_lo, clip_hi


# ------------------------------------------------------------

st.title("🌲 Random Forest – Outlier Clipping & KNN Imputation")
st.markdown(
    "Compares three strategies for handling stress-test NaN values: "
    "median fill, KNN imputation (5 neighbours), and KNN + outlier clipping "
    "(1st–99th percentile). The page shows internal test performance; "
    "the standalone script reports all three strategies on the stress set."
)

with st.spinner("Training model …"):
    model, X_test, y_test, knn, clip_lo, clip_hi = train()
    df_train = get_df()

# Sidebar
st.sidebar.header("Filter the data")
meat_opts = sorted(df_train["meat_type"].unique().tolist())
sel_meat = st.sidebar.multiselect("Meat type", meat_opts, default=meat_opts)
organic_opt = st.sidebar.radio("Organic", ["All", "Only organic", "Only non-organic"])
price_range = st.sidebar.slider(
    "Price range (EUR/kg)", float(df_train[TARGET].min()),
    float(df_train[TARGET].max()),
    (float(df_train[TARGET].min()), float(df_train[TARGET].max()))
)
n_rows = st.sidebar.slider("Rows to show", 5, 50, 10, step=5)

df_f = df_train[
    df_train["meat_type"].isin(sel_meat) &
    df_train[TARGET].between(price_range[0], price_range[1])
]
if organic_opt == "Only organic":
    df_f = df_f[df_f["organic"] == 1]
elif organic_opt == "Only non-organic":
    df_f = df_f[df_f["organic"] == 0]

# Key metrics
y_pred = model.predict(X_test)
st.header("Key metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", len(df_f))
c2.metric("Mean price", f"{df_f[TARGET].mean():.2f} EUR" if len(df_f) else "—")
c3.metric("Min price", f"{df_f[TARGET].min():.2f} EUR" if len(df_f) else "—")
c4.metric("Max price", f"{df_f[TARGET].max():.2f} EUR" if len(df_f) else "—")

# Data explorer
st.header("Data explorer")
st.dataframe(df_f.head(n_rows), width="stretch", hide_index=True)

# Model performance
st.header("Model performance (internal test set)")
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.4f} EUR")
c2.metric("RMSE", f"{rmse:.4f} EUR")
c3.metric("R²", f"{r2:.4f}")

st.caption(
    "Stress-test comparison (all 3 strategies) is in the standalone script "
    "`regression/random_forest_regression_outlier_imputation.py`."
)

scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
st.scatter_chart(scatter_df, x="Actual", y="Predicted")

# Feature importance
st.header("Feature importance")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
st.bar_chart(importances)

# Predict price
st.header("Predict price")
st.markdown(
    "Input values are processed through **KNN imputation + outlier clipping** "
    "before prediction — matching the best strategy from the notebook."
)
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
    row = pd.DataFrame([{
        "meat_type": in_mt, "fat_content_pct": in_fat, "protein_pct": in_protein,
        "marbling_score": in_marbling, "animal_age_months": in_age,
        "storage_days": in_storage, "organic": in_organic, "cut_quality": in_quality,
    }])[FEATURES]
    # Apply KNN imputation + clip (mirrors best strategy for stress test)
    row_imp = pd.DataFrame(knn.transform(row), columns=FEATURES)
    row_clip = row_imp.clip(lower=clip_lo, upper=clip_hi, axis=1)
    pred = float(model.predict(row_clip)[0])
    st.success(f"### Predicted price: **{pred:.2f} EUR/kg**")
