"""Streamlit page: Random Forest – Combined Dataset (Notebook #4)."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shared.data_loading import STRESS_FILE, TRAIN_FILE

TARGET = "price_eur_per_kg"
FEATURES = [
    "meat_type", "fat_content_pct", "protein_pct", "marbling_score",
    "animal_age_months", "storage_days", "organic", "cut_quality",
]
MODE_COLS = ["meat_type", "organic"]
MEDIAN_COLS = ["fat_content_pct", "protein_pct", "marbling_score",
               "animal_age_months", "storage_days", "cut_quality", TARGET]


@st.cache_data
def get_df():
    df_main = pd.read_csv(TRAIN_FILE)
    df_stress = pd.read_csv(STRESS_FILE)
    df_main["source"] = "main"
    df_stress["source"] = "stress"
    df = pd.concat([df_main, df_stress], ignore_index=True)
    for col in MODE_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in MEDIAN_COLS:
        df[col] = df[col].fillna(df[col].median())
    return df


@st.cache_resource
def train():
    df = get_df()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=7, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    source_te = df.loc[y_te.index, "source"]
    return model, X_te, y_te, source_te


# ------------------------------------------------------------

st.title("🌲 Random Forest – Combined Dataset")
st.markdown(
    "Trains on **1300 rows** (1200 clean + 100 stress) after imputing missing "
    "values (mode for meat_type/organic, median for all others including price). "
    "An 80/20 split is applied on the combined data."
)

with st.spinner("Training model …"):
    model, X_test, y_test, source_test = train()
    df_train = get_df()

# Sidebar
st.sidebar.header("Filter the data")
source_opts = ["All", "Main only", "Stress only"]
sel_source = st.sidebar.radio("Data source", source_opts)
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
if sel_source == "Main only":
    df_f = df_f[df_f["source"] == "main"]
elif sel_source == "Stress only":
    df_f = df_f[df_f["source"] == "stress"]
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
c3.metric("Main rows", int((df_train["source"] == "main").sum()))
c4.metric("Stress rows", int((df_train["source"] == "stress").sum()))

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

# Breakdown by source
st.subheader("Performance by source")
main_mask = source_test == "main"
stress_mask = source_test == "stress"
perf_data = []
for label, mask in [("Main (clean)", main_mask), ("Stress", stress_mask)]:
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
    row = pd.DataFrame([{
        "meat_type": in_mt, "fat_content_pct": in_fat, "protein_pct": in_protein,
        "marbling_score": in_marbling, "animal_age_months": in_age,
        "storage_days": in_storage, "organic": in_organic, "cut_quality": in_quality,
    }])[FEATURES]
    pred = float(model.predict(row)[0])
    st.success(f"### Predicted price: **{pred:.2f} EUR/kg**")
