"""
XGBoost Regression – Baseline (NB1)
=====================================

Purpose
-------
Train an XGBRegressor on 8 raw features plus an engineered protein-to-fat
ratio to predict price_eur_per_kg. Establishes the XGBoost regression baseline
and evaluates on both an internal 80/20 test split and the stress-test dataset.
Direct port of `xgb_regression.ipynb`.

Step-by-step explanation
------------------------
1. Load training data, inspect shape, missing values, and summary statistics.
2. Engineer protein_fat_ratio = protein_pct / (fat_content_pct + 1e-5).
3. 80/20 train/test split (unstratified for regression); train XGBRegressor
   (n_estimators=150, learning_rate=0.08, max_depth=5).
4. Report train and test RMSE, MAE, R² to gauge overfitting.
5. Plot feature importance (XGBoost gain metric).
6. Load stress-test dataset; show descriptive statistics and boxplots
   to identify outlier and missing-value patterns.
7. Impute stress NaN with X_train medians; add protein_fat_ratio.
   Evaluate on the 91 rows that have a known price.
8. Comparison table with previous Random Forest baselines.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURES : list of str
    Eight raw input columns (target excluded).
- ALL_FEATURES : list of str
    FEATURES plus 'protein_fat_ratio' (no label leakage).
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Medium. Full run takes under 15 seconds.

Usage
-----
    python -m regression.xgb_regression
    python -m regression.xgb_regression --no-plots
    python -m regression.xgb_regression --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from shared.data_loading import load_stress_data, load_training_data


# --------------------------------------------------------
# Constants
# --------------------------------------------------------

TARGET = "price_eur_per_kg"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "meat_type",
    "fat_content_pct",
    "protein_pct",
    "marbling_score",
    "animal_age_months",
    "storage_days",
    "organic",
    "cut_quality",
]

ALL_FEATURES = FEATURES + ["protein_fat_ratio"]


# --------------------------------------------------------
# Feature engineering
# --------------------------------------------------------

def add_protein_fat_ratio(df):
    """
    Add protein_fat_ratio = protein_pct / (fat_content_pct + 1e-5).

    The 1e-5 epsilon prevents division by zero. This feature captures lean-to-fat
    composition without involving the price target, so there is no label leakage.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy of df with 'protein_fat_ratio' appended.
    """
    out = df.copy()
    out["protein_fat_ratio"] = out["protein_pct"] / (out["fat_content_pct"] + 1e-5)
    return out


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load training data, engineer features, split, and prepare stress data.

    Stress features are imputed with X_train (training-split) medians to avoid
    data leakage. protein_fat_ratio is recomputed from imputed protein/fat values.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress, df_stress_raw)
    """
    df = load_training_data()
    df = add_protein_fat_ratio(df)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    df_stress_raw = load_stress_data()

    # Record which rows have a valid target before imputing anything
    valid_mask = df_stress_raw[TARGET].notna()

    # Fill NaN in the 8 raw features with X_train medians only
    train_medians = X_train[FEATURES].median()
    df_stress = df_stress_raw.copy()
    for col in FEATURES:
        df_stress[col] = df_stress[col].fillna(train_medians[col])

    # Recompute ratio after imputation
    df_stress = add_protein_fat_ratio(df_stress)

    X_stress = df_stress.loc[valid_mask, ALL_FEATURES]
    y_stress = df_stress_raw.loc[valid_mask, TARGET]  # original (non-imputed) prices

    return X_train, X_test, y_train, y_test, X_stress, y_stress, df_stress_raw


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train an XGBRegressor with the baseline hyperparameters.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    XGBRegressor
    """
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------
# Evaluation
# --------------------------------------------------------

def evaluate_regressor(model, X, y, label):
    """
    Compute and print RMSE, MAE, R².

    Parameters
    ----------
    model : fitted XGBRegressor
    X : pd.DataFrame
    y : pd.Series
    label : str

    Returns
    -------
    tuple
        (rmse: float, mae: float, r2: float, y_pred: np.ndarray)
    """
    y_pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\n=== {label} ===")
    print(f"RMSE: {rmse:.4f} EUR")
    print(f"MAE:  {mae:.4f} EUR")
    print(f"R²:   {r2:.4f}")
    print(f"Rows: {len(y)}")

    return rmse, mae, r2, y_pred


# --------------------------------------------------------
# Plot helpers
# --------------------------------------------------------

def _save_or_show(fig, name, save_dir):
    """Save figure to disk or display it, then close."""
    if save_dir is not None:
        path = Path(save_dir) / f"{name}.png"
        fig.savefig(path)
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_importance(model, save_dir=None):
    """
    Horizontal bar chart of normalised feature importance.

    Parameters
    ----------
    model : XGBRegressor
    save_dir : str or None
    """
    importance = pd.Series(
        model.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", color="teal", ax=ax)
    ax.set_xlabel("Feature Importance (gain, normalised)")
    ax.set_title("XGBoost Feature Importance – Regression Baseline")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance", save_dir)

    print("\nFeature importances:")
    print(importance.sort_values(ascending=False).round(4))


def plot_stress_boxplots(df_stress_raw, save_dir=None):
    """
    3×3 grid of boxplots showing the distribution of each stress-test column.

    Outliers appear as individual points beyond the whiskers.

    Parameters
    ----------
    df_stress_raw : pd.DataFrame
    save_dir : str or None
    """
    num_cols = df_stress_raw.select_dtypes(include=["float64", "int64"]).columns.tolist()
    ncols = 3
    nrows = (len(num_cols) + ncols - 1) // ncols

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df_stress_raw[col], color="tomato", ax=axes_flat[i])
        axes_flat[i].set_title(col, fontsize=10)
        axes_flat[i].set_ylabel("")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Stress Test – Distribution & Outlier Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "stress_boxplots", save_dir)


def plot_actual_vs_predicted(y_true, y_pred, label, color, save_dir=None):
    """
    Scatter plot of actual vs predicted prices and a residual plot.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
    y_pred : np.ndarray
    label : str
    color : str
    save_dir : str or None
    """
    residuals = np.array(y_true) - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.5, s=25, color=color)
    lo, hi = min(np.min(y_true), y_pred.min()), max(np.max(y_true), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Price (EUR)")
    axes[0].set_ylabel("Predicted Price (EUR)")
    axes[0].set_title(f"Actual vs Predicted – {label}")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.5, s=25, color="coral")
    axes[1].axhline(y=0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted Price (EUR)")
    axes[1].set_ylabel("Residual (EUR)")
    axes[1].set_title(f"Residuals – {label}")

    plt.tight_layout()
    _save_or_show(
        fig, f"actual_vs_predicted_{label.replace(' ', '_').lower()}", save_dir
    )


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, train, evaluate internally and on stress.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None -> show interactively.
    """
    print("=" * 60)
    print("XGBoost Regression – Baseline (NB1)")
    print("=" * 60)

    print("\n[1/4] Loading and preparing data ...")
    (X_train, X_test, y_train, y_test,
     X_stress, y_stress, df_stress_raw) = prepare_data()

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"| Stress: {len(X_stress)} rows")
    print("Stress descriptive statistics:")
    print(df_stress_raw.describe().round(2).to_string())
    print("\nMissing values (stress):")
    print(df_stress_raw.isna().sum())

    if not no_plots:
        plot_stress_boxplots(df_stress_raw, save_dir=save_plots)

    print("\n[2/4] Training model ...")
    model = train_model(X_train, y_train)
    print(f"Trees: {model.n_estimators}  |  max_depth: {model.max_depth}")

    print("\n[3/4] Evaluating ...")
    rmse_tr, _, _, _ = evaluate_regressor(model, X_train, y_train, "Train Set")
    rmse_te, mae_te, r2_te, y_pred_te = evaluate_regressor(
        model, X_test, y_test, "Internal Test Set"
    )

    if not no_plots:
        plot_actual_vs_predicted(y_test, y_pred_te, "Internal Test", "steelblue", save_plots)
        plot_feature_importance(model, save_dir=save_plots)

    rmse_s, mae_s, r2_s, y_pred_s = evaluate_regressor(
        model, X_stress, y_stress, "Stress-Test Set"
    )

    if not no_plots:
        plot_actual_vs_predicted(y_stress, y_pred_s, "Stress Test", "darkorange", save_plots)

    print("\n[4/4] Sample stress predictions (first 10 rows) ...")
    sample = pd.DataFrame({
        "true_price": y_stress.values[:10],
        "predicted_price": y_pred_s[:10],
    })
    print(sample.round(2).to_string(index=False))

    print("\nComparison with previous models:")
    comparison = pd.DataFrame({
        "Experiment": [
            "DT regression (8 raw features)",
            "RF regression (8 raw features)",
            "RF log-price (8 raw features)",
            "XGB baseline + protein_fat_ratio (this)",
        ],
        "MAE Test":    ["2.38", "1.80", "1.80", f"{mae_te:.2f}"],
        "R² Test":     ["0.8737", "0.9249", "0.9252", f"{r2_te:.4f}"],
        "MAE Stress":  ["8.59", "8.68", "8.80", f"{mae_s:.2f}"],
        "R² Stress":   ["-0.2544", "-0.3288", "-0.3161", f"{r2_s:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Regressor baseline with protein_fat_ratio feature."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
