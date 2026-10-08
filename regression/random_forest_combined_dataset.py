"""
Random Forest Regression – Combined Dataset Training
====================================================

Purpose
-------
Combine both datasets (1200 training + 100 stress-test = 1300 rows), impute
missing values using column-type-appropriate strategies, and train a Random
Forest that learns from both clean patterns and edge cases. Direct port of
`random_forest_combined_dataset.ipynb`.

Step-by-step explanation
------------------------
1. Load both raw CSVs from the shared data paths and tag each row with
   its source ('main' or 'stress').
2. Concatenate into a single 1300-row DataFrame.
3. Impute missing values: mode for categorical/binary columns (meat_type,
   organic), median for continuous/ordinal columns.
4. Define the eight raw feature columns and split the combined set 80/20.
5. Train a RandomForestRegressor (200 trees, max_depth=7).
6. Evaluate on the full test set (mixed main + stress rows).
7. Break down test-set performance by source row: main vs stress.
8. Plot actual vs predicted colour-coded by source and feature importances.
9. Compare R² with the single-dataset RF baseline.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURE_COLS : list of str
    Eight raw input columns; 'source' is used only for tracking.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- TEST_SIZE : float
    Fraction held out for the internal test split (0.20).

Runtime
-------
Fast. Full run takes under 15 seconds.

Usage
-----
    python -m regression.random_forest_combined_dataset
    python -m regression.random_forest_combined_dataset --no-plots
    python -m regression.random_forest_combined_dataset --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from shared.data_loading import STRESS_FILE, TRAIN_FILE


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

# Mixed imputation strategy: categorical/binary → mode, others → median
MODE_COLS   = ["meat_type", "organic"]
MEDIAN_COLS = [
    "fat_content_pct", "protein_pct", "marbling_score",
    "animal_age_months", "storage_days", "cut_quality",
    TARGET,
]


# ------------------------------------------------------------
# Data loading and imputation
# ------------------------------------------------------------

def load_and_combine():
    """
    Load both CSVs, tag rows by source, and concatenate.

    Returns
    -------
    pd.DataFrame
        Combined 1300-row DataFrame with a 'source' column.
    """
    df_main   = pd.read_csv(TRAIN_FILE)
    df_stress = pd.read_csv(STRESS_FILE)

    df_main["source"]   = "main"
    df_stress["source"] = "stress"

    return pd.concat([df_main, df_stress], ignore_index=True)


def impute_combined(df):
    """
    Impute missing values in the combined DataFrame.

    Mode for categorical/binary; median for continuous/ordinal.
    Imputation statistics are computed on the combined data.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame (may contain NaN).

    Returns
    -------
    pd.DataFrame
        Copy of df with all NaN filled.
    """
    df = df.copy()
    for col in MODE_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in MEDIAN_COLS:
        df[col] = df[col].fillna(df[col].median())
    return df


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

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
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_regressor(model, X, y, label):
    """
    Compute and print MAE, RMSE, and R² for a regression model.

    Parameters
    ----------
    model : fitted sklearn regressor
    X : pd.DataFrame
    y : pd.Series
    label : str

    Returns
    -------
    tuple
        (mae: float, rmse: float, r2: float, y_pred: np.ndarray)
    """
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2   = r2_score(y, y_pred)

    print(f"\n=== {label} ===")
    print(f"MAE:  {mae:.2f} EUR")
    print(f"RMSE: {rmse:.2f} EUR")
    print(f"R²:   {r2:.4f}")
    if len(y) > 0:
        print(f"Relative error: {mae / y.mean() * 100:.1f}%")

    return mae, rmse, r2, y_pred


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


def plot_actual_vs_predicted(y_test, y_pred, test_source, save_dir=None):
    """
    Scatter and residual plots colour-coded by source (main vs stress).

    Parameters
    ----------
    y_test : pd.Series
    y_pred : np.ndarray
    test_source : pd.Series
        'main' or 'stress' label per test row.
    save_dir : str or None
    """
    colors = ["steelblue" if s == "main" else "darkorange" for s in test_source]
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5, s=25, c=colors)
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", linestyle="--", linewidth=2,
    )
    axes[0].set_xlabel("Actual Price (EUR)")
    axes[0].set_ylabel("Predicted Price (EUR)")
    axes[0].set_title("Actual vs Predicted (blue=main, orange=stress)")

    axes[1].scatter(y_pred, residuals, alpha=0.5, s=25, c=colors)
    axes[1].axhline(y=0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted Price (EUR)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals (blue=main, orange=stress)")

    plt.tight_layout()
    _save_or_show(fig, "actual_vs_predicted", save_dir)


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
    ax.set_title("Feature Importance – Combined Dataset")
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
    Run the full pipeline: combine, impute, train, evaluate, compare.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("Random Forest Regression – Combined Dataset Training")
    print("=" * 60)

    print("\n[1/4] Loading and combining datasets ...")
    df = load_and_combine()
    print(f"Main: 1200 rows | Stress: 100 rows | Combined: {len(df)} rows")
    print(f"Missing values before imputation:\n{df[FEATURE_COLS + [TARGET]].isnull().sum()}")

    print("\n[2/4] Imputing missing values ...")
    df = impute_combined(df)
    print("Missing values after imputation:")
    print(df[FEATURE_COLS + [TARGET]].isnull().sum())

    source_col = df["source"]
    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    test_source = source_col.loc[X_test.index]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Test split by source: {test_source.value_counts().to_dict()}")

    print("\n[3/4] Training model ...")
    model = train_model(X_train, y_train)
    print(f"Number of trees: {model.n_estimators} | Max depth: {model.max_depth}")

    print("\n[4/4] Evaluating ...")
    mae_full, rmse_full, r2_full, y_pred = evaluate_regressor(
        model, X_test, y_test, "Full Test Set (main + stress)"
    )

    # Separate evaluation by source
    main_mask   = test_source == "main"
    stress_mask = test_source == "stress"

    if main_mask.sum() > 0:
        evaluate_regressor(
            model,
            X_test[main_mask.values],
            y_test[main_mask],
            f"Main-Data Rows in Test ({main_mask.sum()} rows)",
        )
    if stress_mask.sum() > 0:
        evaluate_regressor(
            model,
            X_test[stress_mask.values],
            y_test[stress_mask],
            f"Stress-Data Rows in Test ({stress_mask.sum()} rows)",
        )

    if not no_plots:
        plot_actual_vs_predicted(y_test, y_pred, test_source, save_dir=save_plots)
        plot_feature_importance(model, save_dir=save_plots)

    # Comparison with single-dataset RF baseline
    comparison = pd.DataFrame({
        "Experiment":  ["RF baseline (single dataset)", "RF combined dataset (this)"],
        "R² Test":     ["0.9249", f"{r2_full:.4f}"],
        "R² Stress":   ["-0.3288", "(see source breakdown above)"],
    })
    print("\n" + comparison.to_string(index=False))
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF Regressor trained on combined main+stress dataset."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
