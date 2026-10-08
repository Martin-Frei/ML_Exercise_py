"""
Random Forest Regression – Grid Search Optimization
====================================================

Purpose
-------
Find the best Random Forest hyperparameters for predicting
`price_eur_per_kg` (meat price per kilogram) using GridSearchCV.
Then evaluate the best model on both the internal test split and
the stress-test dataset.

This script is the direct port of the notebook
`random_forest_regression_grid_search.ipynb`.

Step-by-step explanation
------------------------
1. Load the training data and apply the three engineered features
   via `shared.data_loading.load_training_data()`.
2. Split the training data into 80% train / 20% test.
3. Load and impute the stress-test data using training medians.
4. Run a GridSearchCV over four key hyperparameters.
5. Evaluate the best model on the internal test and the stress set.
6. Save the best model to `models/random_forest_grid_search.joblib`
   when run with `--save-model`.

Runtime
-------
Full training takes about 150 seconds on a modern laptop.
The Streamlit app loads the saved .joblib file instead of re-running.

Usage
-----
    python regression/random_forest_grid_search.py
    python regression/random_forest_grid_search.py --save-model
"""

import argparse
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from shared.data_loading import load_stress_data, load_training_data
from shared.modeling import impute_with_train_medians, save_model


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

# Column we want to predict
TARGET = "price_eur_per_kg"

# Fixed random seed for reproducibility
RANDOM_STATE = 42

# Test split fraction
TEST_SIZE = 0.20

# Name used when saving the trained model
MODEL_NAME = "random_forest_grid_search"


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def prepare_data():
    """
    Load and split the data, and prepare the stress-test features.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress, feature_cols)
    """
    # Load the training data (already has engineered features)
    df_train = load_training_data()

    # All columns except the target are features
    feature_cols = [c for c in df_train.columns if c != TARGET]

    # Split into features and target
    X = df_train[feature_cols]
    y = df_train[TARGET]

    # Standard train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Load and impute the stress set using training medians
    df_stress = load_stress_data()
    df_stress = impute_with_train_medians(df_train, df_stress, feature_cols)

    # Keep only rows where the target exists (we need it for scoring)
    df_stress = df_stress.dropna(subset=[TARGET])

    X_stress = df_stress[feature_cols]
    y_stress = df_stress[TARGET]

    return X_train, X_test, y_train, y_test, X_stress, y_stress, feature_cols


# ------------------------------------------------------------
# Grid search
# ------------------------------------------------------------

def run_grid_search(X_train, y_train):
    """
    Run the main GridSearchCV (the slow part).

    Parameters
    ----------
    X_train, y_train : arrays
        Training features and target.

    Returns
    -------
    GridSearchCV
        Fitted grid search object with .best_estimator_ available.
    """
    # The grid from the original notebook
    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [3, 5, 7, 9, 11, None],
        "min_samples_leaf":  [1, 3, 5, 10, 15],
        "max_features":      [None, "sqrt", "log2", 0.5],
    }

    # Grid search with 5-fold cross-validation
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\nStarting GridSearchCV (this takes about 2.5 minutes)...")

    # Time the search
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    duration = time.perf_counter() - t0

    print(f"\nGridSearchCV finished in {duration:.1f} seconds.")
    print(f"Best cross-validated MAE: {-grid.best_score_:.2f} EUR")
    print(f"Best parameters:          {grid.best_params_}")

    return grid


# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------

def evaluate(model, X, y, label):
    """
    Evaluate a regression model and print the metrics.

    Parameters
    ----------
    model : fitted estimator
    X, y : arrays
        Features and target to evaluate on.
    label : str
        Name shown in the printed output.

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = r2_score(y, y_pred)

    print(f"\n=== {label} ===")
    print(f"MAE:  {mae:.2f} EUR")
    print(f"RMSE: {rmse:.2f} EUR")
    print(f"R²:   {r2:.4f}")
    print(f"Relative error: {mae / y.mean() * 100:.1f}%")

    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main(save=False):
    """
    Run the full pipeline: prepare, train, evaluate, save.

    Parameters
    ----------
    save : bool
        If True, the best model is saved to models/<MODEL_NAME>.joblib
    """
    print("=" * 60)
    print("Random Forest Regression – Grid Search Optimization")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1/4] Loading and preparing data ...")
    (X_train, X_test, y_train, y_test,
     X_stress, y_stress, feature_cols) = prepare_data()

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"| Stress: {len(X_stress)} rows")
    print(f"Features: {feature_cols}")

    # 2. Run grid search
    print("\n[2/4] Running GridSearchCV ...")
    grid = run_grid_search(X_train, y_train)
    best_model = grid.best_estimator_

    # 3. Evaluate
    print("\n[3/4] Evaluating best model ...")
    evaluate(best_model, X_test, y_test, "Internal Test Set")
    evaluate(best_model, X_stress, y_stress, "Stress-Test Set")

    # 4. Save model (optional)
    print("\n[4/4] Model persistence ...")
    if save:
        path = save_model(best_model, MODEL_NAME)
        print(f"Model saved to: {path}")
    else:
        print("Model not saved (use --save-model to save it).")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest with GridSearchCV."
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model to models/ folder.",
    )
    args = parser.parse_args()

    main(save=args.save_model)