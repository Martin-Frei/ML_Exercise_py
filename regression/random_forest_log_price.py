"""
Random Forest Regression – Log Price Transformation
====================================================

Purpose
-------
Train a Random Forest on log(price) instead of raw price to improve
predictions on skewed price distributions and stress-test data. Direct
port of `random_forest_log_price.ipynb`.

Step-by-step explanation
------------------------
1. Load the training data and select the eight raw feature columns.
2. Visualise the raw vs log-transformed price distribution to confirm
   that log reduces skewness.
3. Apply np.log1p() to the target and split 80/20.
4. Train a RandomForestRegressor (200 trees, max_depth=7) on log(price).
5. Convert predictions back to EUR with np.expm1() and evaluate.
6. Also report R² and MAE in log space for reference.
7. Load the stress-test dataset, impute with KNN (fitted on training
   data), and evaluate on the stress test in EUR.
8. Compare with all previous experiments.

Variables glossary
------------------
- TARGET : str
    Raw target column ('price_eur_per_kg').
- FEATURE_COLS : list of str
    Eight raw input columns; engineered features are excluded to
    prevent label leakage.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- TEST_SIZE : float
    Fraction held out for the internal test split (0.20).

Runtime
-------
Medium. Full run takes under 20 seconds (KNN imputation adds ~5 s).

Usage
-----
    python -m regression.random_forest_log_price
    python -m regression.random_forest_log_price --no-plots
    python -m regression.random_forest_log_price --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shared.data_loading import load_stress_data, load_training_data


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TARGET = "price_eur_per_kg"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLS = [
    "meat_type",
    "fat_content_pct",
    "protein_pct",
    "marbling_score",
    "animal_age_months",
    "storage_days",
    "organic",
    "cut_quality",
]


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def prepare_data():
    """
    Load, transform, split, and prepare stress-test data.

    Returns
    -------
    tuple
        (X_train, X_test, y_train_log, y_test_log, y_test_raw,
         X_stress, y_stress_raw, knn_imputer)
    """
    df = load_training_data()

    X = df[FEATURE_COLS]
    y_raw = df[TARGET]
    y_log = np.log1p(y_raw)  # log(1 + price) to avoid log(0) edge case

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    y_test_raw = np.expm1(y_test_log)  # keep raw prices for EUR evaluation

    # Fit KNN imputer on training features (no data leakage)
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(X_train)

    df_stress = load_stress_data()
    X_stress_raw = df_stress[FEATURE_COLS].copy()
    y_stress_raw = df_stress[TARGET].copy()

    X_stress_filled = pd.DataFrame(
        knn_imputer.transform(X_stress_raw),
        columns=FEATURE_COLS,
        index=X_stress_raw.index,
    )

    # Drop rows where the target itself is NaN
    valid_mask = y_stress_raw.notna()
    X_stress = X_stress_filled[valid_mask]
    y_stress = y_stress_raw[valid_mask]

    return (X_train, X_test, y_train_log, y_test_log, y_test_raw,
            X_stress, y_stress, knn_imputer)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(X_train, y_train_log):
    """
    Train a Random Forest Regressor on log-transformed price.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train_log : pd.Series
        Log-transformed target.

    Returns
    -------
    RandomForestRegressor
        Fitted regressor.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)
    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_in_eur(model, X, y_raw, label):
    """
    Predict in log space and report metrics in EUR.

    Parameters
    ----------
    model : fitted regressor (trained on log-price)
    X : pd.DataFrame
    y_raw : pd.Series
        True prices in EUR.
    label : str

    Returns
    -------
    tuple
        (mae: float, rmse: float, r2: float, y_pred_eur: np.ndarray)
    """
    y_pred_log = model.predict(X)
    y_pred_eur = np.expm1(y_pred_log)

    mae  = mean_absolute_error(y_raw, y_pred_eur)
    rmse = float(np.sqrt(mean_squared_error(y_raw, y_pred_eur)))
    r2   = r2_score(y_raw, y_pred_eur)

    print(f"\n=== {label} ===")
    print(f"MAE  : {mae:.2f} EUR")
    print(f"RMSE : {rmse:.2f} EUR")
    print(f"R²   : {r2:.4f}")
    print(f"Mean price: {y_raw.mean():.2f} EUR")
    print(f"Relative error: {mae / y_raw.mean() * 100:.1f}%")

    return mae, rmse, r2, y_pred_eur


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def _save_or_show(fig, name, save_dir):
    """Save figure to disk or display it, then close."""
    if save_dir is not None:
        path = Path(save_dir) / f"{name}.png"
        fig.savefig(path)
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_price_distributions(df, save_dir=None):
    """
    Plot raw and log-transformed price distributions side by side.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame containing TARGET column.
    save_dir : str or None
    """
    prices = df[TARGET]
    log_prices = np.log1p(prices)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(prices, bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title("Raw Price Distribution")
    axes[0].set_xlabel("Price (EUR/kg)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(prices.mean(), color="red", linestyle="--", label="Mean")
    axes[0].axvline(prices.median(), color="green", linestyle="--", label="Median")
    axes[0].legend()

    axes[1].hist(log_prices, bins=30, color="teal", edgecolor="black")
    axes[1].set_title("log(1 + Price) Distribution")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(log_prices.mean(), color="red", linestyle="--", label="Mean")
    axes[1].axvline(log_prices.median(), color="green", linestyle="--", label="Median")
    axes[1].legend()

    plt.tight_layout()
    _save_or_show(fig, "price_distributions", save_dir)

    print(f"Raw price   — Mean: {prices.mean():.2f}  Median: {prices.median():.2f}  "
          f"Skew: {prices.skew():.3f}")
    print(f"Log price   — Mean: {log_prices.mean():.2f}  Median: {log_prices.median():.2f}  "
          f"Skew: {log_prices.skew():.3f}")


def plot_actual_vs_predicted(y_raw, y_pred_eur, label, color, save_dir=None):
    """
    Actual vs predicted (EUR) and residual plot.

    Parameters
    ----------
    y_raw : pd.Series
    y_pred_eur : np.ndarray
    label : str
    color : str
    save_dir : str or None
    """
    residuals = y_raw - y_pred_eur
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_raw, y_pred_eur, alpha=0.4, s=20, color=color)
    axes[0].plot(
        [y_raw.min(), y_raw.max()],
        [y_raw.min(), y_raw.max()],
        color="red", linestyle="--", linewidth=2, label="Perfect prediction",
    )
    axes[0].set_xlabel("Actual Price (EUR)")
    axes[0].set_ylabel("Predicted Price (EUR)")
    axes[0].set_title(f"Actual vs Predicted – {label}")
    axes[0].legend()

    axes[1].scatter(y_pred_eur, residuals, alpha=0.4, s=20, color="coral")
    axes[1].axhline(y=0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted Price (EUR)")
    axes[1].set_ylabel("Residual (EUR)")
    axes[1].set_title(f"Residual Plot – {label}")

    plt.tight_layout()
    _save_or_show(fig, f"actual_vs_predicted_{label.replace(' ', '_').lower()}", save_dir)

    print(f"Mean residual: {residuals.mean():.2f} EUR")
    print(f"Std  residual: {residuals.std():.2f} EUR")


def plot_feature_importance(model, save_dir=None):
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : RandomForestRegressor
    save_dir : str or None
    """
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="teal", edgecolor="black")
    ax.set_title("Feature Importance – RF with log(price)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance", save_dir)

    print("\nImportance values:")
    print(importances.round(4))


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, transform, train, evaluate, compare.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("Random Forest Regression – Log Price Transformation")
    print("=" * 60)

    print("\n[1/4] Loading and preparing data ...")
    (X_train, X_test, y_train_log, y_test_log, y_test_raw,
     X_stress, y_stress, _) = prepare_data()

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"| Stress: {len(X_stress)} rows")
    print(f"y_raw range: {np.expm1(y_train_log.min()):.2f} – "
          f"{np.expm1(y_train_log.max()):.2f} EUR")

    df_full = load_training_data()
    if not no_plots:
        plot_price_distributions(df_full, save_dir=save_plots)

    print("\n[2/4] Training model on log(price) ...")
    model = train_model(X_train, y_train_log)
    print(f"Number of trees: {model.n_estimators}")

    print("\n[3/4] Evaluating ...")
    # Also report in log space for reference
    y_pred_log_test = model.predict(X_test)
    r2_log = r2_score(y_test_log, y_pred_log_test)
    mae_log = mean_absolute_error(y_test_log, y_pred_log_test)
    print(f"Internal Test (log space): MAE={mae_log:.4f}  R²={r2_log:.4f}")

    mae_t, rmse_t, r2_t, y_pred_eur = evaluate_in_eur(
        model, X_test, y_test_raw, "Internal Test Set (EUR)"
    )
    if not no_plots:
        plot_actual_vs_predicted(y_test_raw, y_pred_eur, "Internal Test", "steelblue", save_plots)
        plot_feature_importance(model, save_dir=save_plots)

    mae_s, rmse_s, r2_s, y_stress_pred_eur = evaluate_in_eur(
        model, X_stress, y_stress, "Stress-Test Set (EUR)"
    )
    if not no_plots:
        plot_actual_vs_predicted(y_stress, y_stress_pred_eur, "Stress Test", "darkorange", save_plots)

    print("\n[4/4] Summary ...")
    comparison = pd.DataFrame({
        "Experiment":  ["RF raw price (baseline)", "RF log(price) (this)"],
        "MAE Test":    ["1.80", f"{mae_t:.2f}"],
        "R² Test":     ["0.9249", f"{r2_t:.4f}"],
        "MAE Stress":  ["8.68", f"{mae_s:.2f}"],
        "R² Stress":   ["-0.3288", f"{r2_s:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF Regressor with log-price target transformation."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
