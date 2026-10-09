"""
Streamlit page: Random Forest Regression with GridSearchCV.

Purpose
-------
Interactive demo of the pre-trained Random Forest Regressor.

Features
--------
1. Sidebar filters: meat type, organic, price range
2. Data explorer: view filtered data
3. Prediction demo: enter custom values, get price prediction
4. Actual vs Predicted chart: see model performance live
5. Feature importance chart

The model is NOT retrained here - it is loaded from
models/random_forest_grid_search.joblib.
"""

import sys
from pathlib import Path

# Make sure the project root is on sys.path so we can import `shared`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shared.data_loading import load_stress_data, load_training_data
from shared.modeling import impute_with_train_medians, load_model, model_exists


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET = "price_eur_per_kg"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_NAME = "random_forest_grid_search"


# ------------------------------------------------------------
# Cached data + model loaders
# ------------------------------------------------------------

@st.cache_data
def get_data():
    """Load and prepare all datasets once and cache them."""
    df_train = load_training_data()
    df_stress = load_stress_data()

    feature_cols = [c for c in df_train.columns if c != TARGET]

    X = df_train[feature_cols]
    y = df_train[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    df_stress = impute_with_train_medians(df_train, df_stress, feature_cols)
    df_stress = df_stress.dropna(subset=[TARGET])

    return df_train, X_test, y_test, df_stress, feature_cols


@st.cache_resource
def get_model():
    """Load the pre-trained model once and cache it."""
    return load_model(MODEL_NAME)


# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------

st.title("🌳 Random Forest Regression – Grid Search")
st.markdown(
    """
    Interactive demo of a **Random Forest Regressor** tuned with
    GridSearchCV on the meat price dataset.

    Use the **filters in the sidebar** to explore the data,
    and the **prediction section below** to try custom inputs.
    """
)


# ------------------------------------------------------------
# Check that the model exists
# ------------------------------------------------------------

if not model_exists(MODEL_NAME):
    st.error(
        "**Model file not found.**\n\n"
        "Please train it first by running:\n\n"
        "```bash\n"
        "python -m regression.random_forest_grid_search --save-model\n"
        "```"
    )
    st.stop()


# ------------------------------------------------------------
# Load data + model
# ------------------------------------------------------------

with st.spinner("Loading model and data ..."):
    model = get_model()
    df_train, X_test, y_test, df_stress, feature_cols = get_data()


# ------------------------------------------------------------
# SIDEBAR: Filters
# ------------------------------------------------------------

st.sidebar.header("🔍 Filter the data")

# Meat type filter (multi-select)
meat_types = sorted(df_train["meat_type"].unique().tolist())
selected_meat_types = st.sidebar.multiselect(
    "Meat type",
    options=meat_types,
    default=meat_types,
    help="Select one or more meat types to include.",
)

# Organic filter (radio)
organic_option = st.sidebar.radio(
    "Organic",
    options=["All", "Only organic", "Only non-organic"],
    index=0,
)

# Price range slider
price_min = float(df_train[TARGET].min())
price_max = float(df_train[TARGET].max())
price_range = st.sidebar.slider(
    "Price range (EUR/kg)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=0.5,
)

# Rows to display
n_rows = st.sidebar.slider(
    "Rows to show in data explorer",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
)


# ------------------------------------------------------------
# Apply filters to the training data
# ------------------------------------------------------------

df_filtered = df_train[df_train["meat_type"].isin(selected_meat_types)]

if organic_option == "Only organic":
    df_filtered = df_filtered[df_filtered["organic"] == 1]
elif organic_option == "Only non-organic":
    df_filtered = df_filtered[df_filtered["organic"] == 0]

df_filtered = df_filtered[
    (df_filtered[TARGET] >= price_range[0])
    & (df_filtered[TARGET] <= price_range[1])
]


# ------------------------------------------------------------
# Top row: key metrics
# ------------------------------------------------------------

st.header("📊 Key metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Filtered rows", len(df_filtered))
with col2:
    st.metric("Mean price", f"{df_filtered[TARGET].mean():.2f} EUR")
with col3:
    st.metric("Min price", f"{df_filtered[TARGET].min():.2f} EUR")
with col4:
    st.metric("Max price", f"{df_filtered[TARGET].max():.2f} EUR")


# ------------------------------------------------------------
# Data explorer
# ------------------------------------------------------------

st.header("📋 Data explorer")
st.markdown(f"Showing the first **{n_rows}** rows after filtering:")

st.dataframe(
    df_filtered.head(n_rows),
    width="stretch",
    hide_index=True,
)


# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------

st.header("📈 Model performance")

# Evaluate on the full internal test set (not filtered, since the
# test set is fixed and used for evaluation)
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
r2 = r2_score(y_test, y_pred_test)

col1, col2, col3 = st.columns(3)
col1.metric("MAE (EUR)", f"{mae:.2f}")
col2.metric("RMSE (EUR)", f"{rmse:.2f}")
col3.metric("R²", f"{r2:.4f}")


# ------------------------------------------------------------
# Actual vs Predicted chart
# ------------------------------------------------------------

st.subheader("Actual vs Predicted (test set)")

# Build a small DataFrame for the scatter plot
df_scatter = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_test,
})

st.scatter_chart(
    df_scatter,
    x="Actual",
    y="Predicted",
    height=400,
)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

st.header("🎯 Feature importance")

importances = pd.Series(
    model.feature_importances_,
    index=feature_cols,
).sort_values(ascending=True)

st.bar_chart(importances)


# ------------------------------------------------------------
# Prediction demo
# ------------------------------------------------------------

st.header("🔮 Predict a custom price")
st.markdown(
    "Adjust the sliders below to see what the model predicts "
    "for a hypothetical piece of meat."
)

# Two-column layout for the inputs
col_a, col_b = st.columns(2)

with col_a:
    input_meat_type = st.slider("Meat type", 1, 5, 3)
    input_fat = st.slider("Fat content (%)", 2.0, 35.0, 18.0, step=0.1)
    input_protein = st.slider("Protein (%)", 15.0, 26.0, 20.5, step=0.1)
    input_marbling = st.slider("Marbling score", 0, 12, 5)

with col_b:
    input_age = st.slider("Animal age (months)", 2, 100, 36)
    input_storage = st.slider("Storage days", 0, 30, 10)
    input_organic = st.radio("Organic", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    input_quality = st.slider("Cut quality", 1, 5, 3)

# Build the input row for the model
if st.button("🔮 Predict", type="primary"):
    # Compute derived features the same way as in engineer_features()
    price_per_protein = 0.0  # will be overridden below, placeholder
    fat_to_protein_ratio = input_fat / input_protein

    # We don't know the price yet, so we cannot compute
    # price_per_protein and price_x_marbling exactly.
    # For the prediction we use a two-step approach:
    # 1. Use a reasonable guess for price (mean of filtered data)
    # 2. Compute the derived features
    # 3. Predict the price
    # 4. Recompute the derived features with the predicted price
    # 5. Predict again
    guess_price = float(df_filtered[TARGET].mean())

    input_row = pd.DataFrame([{
        "meat_type": input_meat_type,
        "fat_content_pct": input_fat,
        "protein_pct": input_protein,
        "marbling_score": input_marbling,
        "animal_age_months": input_age,
        "storage_days": input_storage,
        "organic": input_organic,
        "cut_quality": input_quality,
        "price_per_protein": guess_price / input_protein,
        "fat_to_protein_ratio": fat_to_protein_ratio,
        "price_x_marbling": guess_price * input_marbling,
    }])[feature_cols]

    # First prediction
    first_pred = float(model.predict(input_row)[0])

    # Recompute derived features with first prediction and predict again
    input_row["price_per_protein"] = first_pred / input_protein
    input_row["price_x_marbling"] = first_pred * input_marbling
    final_pred = float(model.predict(input_row)[0])

    st.success(f"### Predicted price: **{final_pred:.2f} EUR/kg**")
    st.caption(
        "Note: `price_per_protein` and `price_x_marbling` depend on the price "
        "itself, so we use a two-step iterative prediction."
    )