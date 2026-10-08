"""
Decision Tree Regression – Meat Price Prediction
=================================================

Purpose
-------
Train a Decision Tree Regressor to predict `price_eur_per_kg` (continuous)
from the eight raw nutritional and quality features. Direct port of
`decision_tree_regression_price_per_kilo.ipynb`.

Step-by-step explanation
------------------------
1. Load the training data via `shared.data_loading.load_training_data()`;
   use only the eight raw feature columns to match the original notebook
   and avoid label leakage from engineered features.
2. Explore the price distribution and feature correlations.
3. Split into 80 % train / 20 % test (no stratification for regression).
4. Train a DecisionTreeRegressor with max_depth=7 (chosen after depth
   experiments in the original notebook).
5. Explain the regression metrics (MAE, RMSE, R²) and evaluate on the
   internal test set.
6. Plot actual vs predicted and residuals.
7. Visualise the tree structure (first 3 levels) and feature importances.
8. Load, impute, and evaluate on the stress-test dataset.
9. Print a side-by-side comparison of both evaluation sets.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURE_COLS : list of str
    The eight raw input columns; engineered features that derive from
    TARGET are excluded to prevent label leakage.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- TEST_SIZE : float
    Fraction held out for the internal test split (0.20).

Runtime
-------
Medium. Full run takes under 5 seconds.

Usage
-----
    python -m regression.decision_tree_regression_price_per_kilo
    python -m regression.decision_tree_regression_price_per_kilo --no-plots
    python -m regression.decision_tree_regression_price_per_kilo --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

from shared.data_loading import load_stress_data, load_training_data
from shared.modeling import impute_with_train_medians


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TARGET = "price_eur_per_kg"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Original notebook used only the eight raw columns.
# price_per_protein and price_x_marbling are excluded: both are computed
# from TARGET and would let the model see the answer during training.
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
    Load, split, and prepare training and stress-test data.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress)
    """
    df = load_training_data()

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    df_stress = load_stress_data()
    # Impute using the training-split medians
    df_stress = impute_with_train_medians(X_train, df_stress, FEATURE_COLS)

    # Drop rows where the target is NaN (can't evaluate without a label)
    df_stress = df_stress.dropna(subset=[TARGET])

    X_stress = df_stress[FEATURE_COLS]
    y_stress = df_stress[TARGET]

    return X_train, X_test, y_train, y_test, X_stress, y_stress


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train a Decision Tree Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    DecisionTreeRegressor
        Fitted regressor.
    """
    # max_depth=7 chosen after depth-grid experiments in the original notebook
    model = DecisionTreeRegressor(max_depth=7, random_state=RANDOM_STATE)
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
        True target values.
    label : str
        Header shown in the printed output.

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
    print(f"MAE  : {mae:.2f} EUR")
    print(f"RMSE : {rmse:.2f} EUR")
    print(f"R²   : {r2:.4f}")
    print(f"Mean price: {y.mean():.2f} EUR")
    print(f"Relative error (MAE / mean): {mae / y.mean() * 100:.1f}%")

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


def plot_eda(df, save_dir=None):
    """
    Plot price distribution and feature correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataframe (raw columns only).
    save_dir : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df[TARGET], bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price (EUR/kg)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df[TARGET].mean(), color="red", linestyle="--", label="Mean")
    axes[0].legend()

    corr_with_price = (
        df[FEATURE_COLS + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values()
    )
    corr_with_price.plot(kind="barh", ax=axes[1], color="teal", edgecolor="black")
    axes[1].set_title("Feature Correlation with Price")
    axes[1].set_xlabel("Correlation")

    plt.tight_layout()
    _save_or_show(fig, "eda", save_dir)

    print("Correlation values:")
    print(corr_with_price.round(3))


def plot_scatter_price(df, save_dir=None):
    """
    Scatter plots of price vs the three most correlated features.

    Parameters
    ----------
    df : pd.DataFrame
    save_dir : str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ["meat_type", "cut_quality", "fat_content_pct"]):
        ax.scatter(df[col], df[TARGET], alpha=0.3, s=15)
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
        ax.set_title(f"Price vs {col}")
    plt.tight_layout()
    _save_or_show(fig, "scatter_price", save_dir)


def plot_actual_vs_predicted(y_true, y_pred, label, color, save_dir=None):
    """
    Actual vs predicted scatter and residual plot.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    label : str
        Title suffix.
    color : str
        Dot colour for the scatter.
    save_dir : str or None
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.4, s=20, color=color)
    axes[0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red", linestyle="--", linewidth=2, label="Perfect prediction",
    )
    axes[0].set_xlabel("Actual Price (EUR)")
    axes[0].set_ylabel("Predicted Price (EUR)")
    axes[0].set_title(f"Actual vs Predicted – {label}")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.4, s=20, color="coral")
    axes[1].axhline(y=0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted Price (EUR)")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title(f"Residual Plot – {label}")

    plt.tight_layout()
    _save_or_show(fig, f"actual_vs_predicted_{label.replace(' ', '_').lower()}", save_dir)

    print(f"Mean residual: {residuals.mean():.2f}")
    print(f"Std  residual: {residuals.std():.2f}")


def plot_tree_viz(model, save_dir=None):
    """
    Visualise the first 3 levels of the fitted decision tree.

    Parameters
    ----------
    model : DecisionTreeRegressor
    save_dir : str or None
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=FEATURE_COLS,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision Tree Regressor (first 3 levels)", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, "tree_viz", save_dir)


def plot_feature_importance(model, save_dir=None):
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : DecisionTreeRegressor
    save_dir : str or None
    """
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="teal", edgecolor="black")
    ax.set_title("Feature Importance – Decision Tree Regressor")
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
    Run the full pipeline: prepare, train, evaluate, compare.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("Decision Tree Regression – Meat Price Prediction")
    print("=" * 60)

    print("\n[1/5] Loading and preparing data ...")
    X_train, X_test, y_train, y_test, X_stress, y_stress = prepare_data()

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"| Stress: {len(X_stress)} rows")
    print(f"Features: {FEATURE_COLS}")
    print(f"Target stats:")
    print(y_train.describe().round(2))

    print("\n[2/5] Exploratory data analysis ...")
    df_raw = pd.concat([X_train, y_train], axis=1)
    if not no_plots:
        plot_eda(df_raw, save_dir=save_plots)
        plot_scatter_price(df_raw, save_dir=save_plots)

    print("\n[3/5] Training model ...")
    model = train_model(X_train, y_train)
    print(f"Tree depth : {model.get_depth()}")
    print(f"Leaf nodes : {model.get_n_leaves()}")

    print("\n[4/5] Evaluating ...")
    mae_t, rmse_t, r2_t, y_pred_test = evaluate_regressor(
        model, X_test, y_test, "Internal Test Set"
    )
    if not no_plots:
        plot_actual_vs_predicted(y_test, y_pred_test, "Internal Test", "steelblue", save_plots)
        plot_tree_viz(model, save_dir=save_plots)
        plot_feature_importance(model, save_dir=save_plots)

    mae_s, rmse_s, r2_s, y_pred_stress = evaluate_regressor(
        model, X_stress, y_stress, "Stress-Test Set"
    )
    if not no_plots:
        plot_actual_vs_predicted(y_stress, y_pred_stress, "Stress Test", "darkorange", save_plots)

    print("\n[5/5] Summary ...")
    comparison = pd.DataFrame({
        "Dataset":    ["Internal Test (20%)", "Stress Test (100 rows)"],
        "Rows":       [len(y_test), len(y_stress)],
        "MAE (EUR)":  [f"{mae_t:.2f}", f"{mae_s:.2f}"],
        "RMSE (EUR)": [f"{rmse_t:.2f}", f"{rmse_s:.2f}"],
        "R²":         [f"{r2_t:.4f}", f"{r2_s:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decision Tree Regressor – predict price_eur_per_kg."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Suppress all plot windows.",
    )
    parser.add_argument(
        "--save-plots",
        metavar="DIR",
        default=None,
        help="Save all plots as PNG files to DIR instead of showing them.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
