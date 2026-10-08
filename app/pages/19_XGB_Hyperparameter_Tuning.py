"""Streamlit page: XGBoost Hyperparameter Tuning (Notebook #14)."""

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

from shared.data_loading import load_stress_data, load_training_data
from shared.modeling import load_model, model_exists

TARGET = "price_eur_per_kg"
# Protein_fat_ratio was accidentally excluded from the notebook's feature set.
# This 8-feature set faithfully replicates that behaviour.
FEATURES = [
    "meat_type", "fat_content_pct", "protein_pct", "marbling_score",
    "animal_age_months", "storage_days", "organic", "cut_quality",
]
MODEL_NAME = "xgb_hyperparameter_tuning"


@st.cache_data
def get_data():
    """Load and prepare clean training data + stress data."""
    df = load_training_data()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_medians = X_tr.median()

    df_stress = load_stress_data()
    valid_mask = df_stress[TARGET].notna()
    df_stress_filled = df_stress.copy()
    for col in FEATURES:
        df_stress_filled[col] = df_stress_filled[col].fillna(train_medians[col])

    X_stress = df_stress_filled.loc[valid_mask, FEATURES]
    y_stress = df_stress.loc[valid_mask, TARGET]
    return X_te, y_te, X_stress, y_stress, df


@st.cache_resource
def load_tuned_model():
    return load_model(MODEL_NAME)


# ------------------------------------------------------------

st.title("⚡ XGBoost – Hyperparameter Tuning")

st.warning(
    "**Model not trained on-the-fly.** This page requires a pre-saved model. "
    "Run the tuning script with `--save-model` (takes 8–25 minutes):\n\n"
    "```\n"
    "python -m regression.xgb_hyperparameter_tuning --save-model\n"
    "```\n\n"
    f"This writes both `regression/saved_models/xgb_hyperparameter_tuning.pkl` "
    f"and `models/{MODEL_NAME}.joblib`. Once the .joblib file exists, reload this page."
)

if not model_exists(MODEL_NAME):
    st.error(
        f"Model file `models/{MODEL_NAME}.joblib` not found. "
        "Run the tuning script with `--save-model` and save the model via "
        "`shared.modeling.save_model(best_model, \"xgb_hyperparameter_tuning\")`. "
        "See the warning box above."
    )
    st.stop()

st.markdown(
    "**RandomizedSearchCV** (100 iterations broad + 50 iterations fine) "
    "over 11 XGBoost hyperparameters. "
    "Trained on clean + stress combined data, stratified by `is_stress` flag. "
    "⚠️ `protein_fat_ratio` was excluded from features due to a notebook bug — "
    "faithfully replicated here."
)

with st.spinner("Loading pre-trained model …"):
    model = load_tuned_model()
    X_test, y_test, X_stress, y_stress, df_train = get_data()

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
y_pred_te = model.predict(X_test)
y_pred_st = model.predict(X_stress)
r2_te = r2_score(y_test, y_pred_te)
r2_st = r2_score(y_stress, y_pred_st)
st.header("Key metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", len(df_f))
c2.metric("Mean price", f"{df_f[TARGET].mean():.2f} EUR" if len(df_f) else "—")
c3.metric("Test R²", f"{r2_te:.4f}")
c4.metric("Stress R²", f"{r2_st:.4f}")

# Data explorer
st.header("Data explorer")
st.dataframe(df_f.head(n_rows), use_container_width=True, hide_index=True)

# Model performance
st.header("Model performance")
mae_te = mean_absolute_error(y_test, y_pred_te)
rmse_te = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
mae_st = mean_absolute_error(y_stress, y_pred_st)
rmse_st = float(np.sqrt(mean_squared_error(y_stress, y_pred_st)))

st.subheader("Internal test set")
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae_te:.4f} EUR")
c2.metric("RMSE", f"{rmse_te:.4f} EUR")
c3.metric("R²", f"{r2_te:.4f}")

st.subheader("Stress test (91 rows)")
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae_st:.4f} EUR")
c2.metric("RMSE", f"{rmse_st:.4f} EUR")
c3.metric("R²", f"{r2_st:.4f}")

scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_te})
st.scatter_chart(scatter_df, x="Actual", y="Predicted")

# Best hyperparameters
st.header("Tuned hyperparameters")
params = model.get_params()
param_cols = ["n_estimators", "learning_rate", "max_depth", "subsample",
              "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"]
param_df = pd.DataFrame([
    {"Parameter": p, "Value": params.get(p, "—")} for p in param_cols if p in params
])
st.dataframe(param_df, use_container_width=True, hide_index=True)

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
