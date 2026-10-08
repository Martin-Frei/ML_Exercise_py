"""
XGBoost Regression – Combined Training Dataset (NB3)
======================================================

Purpose
-------
Combine the clean training data (1 200 rows) with the pre-processed stress
data (91 rows with valid prices) into a single training set and let the model
learn from both distributions. Uses is_stress as a stratification key.
Direct port of `xgb_regression_combined.ipynb`.

Step-by-step explanation
------------------------
1. Load the clean training set (1 200 rows); add protein_fat_ratio.
2. Load the stress-test set; apply full preprocessing:
   a. Group-wise imputation (fill NaN per meat_type subgroup).
   b. Drop rows with NaN meat_type or NaN price (9 rows removed -> 91 usable).
   c. Clip animal_age_months, storage_days, and fat_content_pct to clean
      training-set bounds.
   d. Add protein_fat_ratio.
3. Add is_stress flag to both datasets (0 = clean, 1 = stress).
4. Concatenate -> 1 291 rows.
5. Stratified 80/20 split on is_stress so both sets appear in train and test.
6. Train XGBRegressor on the combined training split.
7. Evaluate overall performance and then separately for clean and stress rows.
8. Plot feature importance and actual-vs-predicted.
9. Comparison with NB1 and NB2 stress-test results.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURES : list of str
    Eight raw input columns (target and is_stress excluded).
- ALL_FEATURES : list of str
    FEATURES plus 'protein_fat_ratio'.
- CLIP_COLS : list of str
    Columns clipped to clean-training-set bounds.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Fast. Full run takes under 10 seconds.

Usage
-----
    python -m regression.xgb_regression_combined
    python -m regression.xgb_regression_combined --no-plots
    python -m regression.xgb_regression_combined --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
CLIP_COLS = ["animal_age_months", "storage_days", "fat_content_pct"]


# --------------------------------------------------------
# Feature engineering
# --------------------------------------------------------

def add_protein_fat_ratio(df):
    """
    Add protein_fat_ratio = protein_pct / (fat_content_pct + 1e-5).

    No label leakage because neither input column is derived from price.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    out["protein_fat_ratio"] = out["protein_pct"] / (out["fat_content_pct"] + 1e-5)
    return out


# --------------------------------------------------------
# Stress preprocessing
# --------------------------------------------------------

def group_wise_impute(df, group_col="meat_type"):
    """
    Fill numeric NaN with the subgroup median keyed by group_col.

    Rows whose group_col is itself NaN fall back to the global column median.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col == group_col or not out[col].isnull().any():
            continue
        out[col] = out[col].fillna(
            out.groupby(group_col)[col].transform("median")
        )
        out[col] = out[col].fillna(out[col].median())
    return out


# --------------------------------------------------------
# Evaluation helper
# --------------------------------------------------------

def print_metrics(label, y_true, y_pred):
    """
    Compute and print RMSE, MAE, R² for a subset of predictions.

    Parameters
    ----------
    label : str
    y_true : array-like
    y_pred : array-like
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:<22}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  n={len(y_true)}")
    return rmse, mae, r2


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load, preprocess, combine, and split training + stress data.

    Steps performed:
    - Clean set: add protein_fat_ratio, add is_stress=0.
    - Stress set: group-wise impute -> drop NaN meat_type / NaN price -> clip
      three columns to clean-training bounds -> add protein_fat_ratio -> is_stress=1.
    - Concatenate -> 1 291 rows.
    - Stratified 80/20 split on is_stress.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, is_stress_test)
        is_stress_test : pd.Series indicating source of each test row.
    """
    df_clean = load_training_data()
    df_clean = add_protein_fat_ratio(df_clean)
    df_clean["is_stress"] = 0

    df_stress_raw = load_stress_data()

    # Group-wise impute first (uses stress data's own subgroup statistics)
    df_stress = group_wise_impute(df_stress_raw)

    # Drop rows with NaN meat_type (cannot be group-imputed) or NaN price
    n_before = len(df_stress)
    df_stress = df_stress.dropna(subset=["meat_type", TARGET])
    n_after = len(df_stress)
    print(f"Dropped {n_before - n_after} stress rows with NaN meat_type or price "
          f"-> {n_after} usable stress rows")

    # Clip to clean training bounds
    for col in CLIP_COLS:
        upper = df_clean[col].max()
        df_stress[col] = df_stress[col].clip(upper=upper)

    df_stress = add_protein_fat_ratio(df_stress)
    df_stress["is_stress"] = 1

    # Combine: 1 200 clean + 91 stress = 1 291 rows
    df_combined = pd.concat([df_clean, df_stress], ignore_index=True)
    print(f"Combined dataset: {len(df_combined)} rows "
          f"({df_combined['is_stress'].value_counts().to_dict()})")

    X = df_combined[ALL_FEATURES]
    y = df_combined[TARGET]
    strat = df_combined["is_stress"]

    X_train, X_test, y_train, y_test, strat_train, strat_test = train_test_split(
        X, y, strat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )

    is_stress_test = strat_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, is_stress_test


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train an XGBRegressor on the combined dataset.

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


def plot_actual_vs_predicted_colored(y_true, y_pred, is_stress, save_dir=None):
    """
    Scatter of actual vs predicted with clean/stress rows in different colours.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    is_stress : pd.Series (0 or 1)
    save_dir : str or None
    """
    y_arr = np.array(y_true)
    is_s = np.array(is_stress)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_arr[is_s == 0], y_pred[is_s == 0],
               alpha=0.4, s=20, color="steelblue", label="Clean")
    ax.scatter(y_arr[is_s == 1], y_pred[is_s == 1],
               alpha=0.7, s=40, color="darkorange", marker="^", label="Stress")

    lo = min(y_arr.min(), y_pred.min())
    hi = max(y_arr.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Price (EUR)")
    ax.set_ylabel("Predicted Price (EUR)")
    ax.set_title("Combined Model – Actual vs Predicted (Test Set)")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "actual_vs_predicted_combined", save_dir)


def plot_feature_importance(model, save_dir=None):
    """
    Horizontal bar chart of XGBoost feature importances.

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
    ax.set_title("XGBoost Feature Importance – Combined Model")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance_combined", save_dir)

    print("\nFeature importances:")
    print(importance.sort_values(ascending=False).round(4))


def plot_residuals_by_source(y_true, y_pred, is_stress, save_dir=None):
    """
    Residual distributions for clean vs stress rows as overlapping histograms.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    is_stress : pd.Series
    save_dir : str or None
    """
    residuals = np.array(y_true) - y_pred
    is_s = np.array(is_stress)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals[is_s == 0], bins=30, alpha=0.6, color="steelblue",
            label=f"Clean ({(is_s == 0).sum()} rows)")
    ax.hist(residuals[is_s == 1], bins=15, alpha=0.6, color="darkorange",
            label=f"Stress ({(is_s == 1).sum()} rows)")
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Residual (EUR)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution by Source")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "residuals_by_source", save_dir)


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: combine, train, evaluate overall and by source.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None -> show interactively.
    """
    print("=" * 60)
    print("XGBoost Regression – Combined Training Dataset (NB3)")
    print("=" * 60)

    print("\n[1/4] Preparing combined dataset ...")
    X_train, X_test, y_train, y_test, is_stress_test = prepare_data()

    n_stress_train = (X_train.index.isin(
        X_train.index[X_train.index >= 1200]
    )).sum()
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Test set composition:")
    print(f"  Clean rows:  {(is_stress_test == 0).sum()}")
    print(f"  Stress rows: {(is_stress_test == 1).sum()}")

    print("\n[2/4] Training combined model ...")
    model = train_model(X_train, y_train)

    if not no_plots:
        plot_feature_importance(model, save_dir=save_plots)

    print("\n[3/4] Evaluating ...")
    y_pred_test = model.predict(X_test)
    is_s = is_stress_test.values

    print("\nOverall and per-source metrics (test set):")
    rmse_all, mae_all, r2_all = print_metrics("Overall", y_test, y_pred_test)

    mask_clean = is_s == 0
    mask_stress = is_s == 1
    if mask_clean.any():
        print_metrics("Clean subset",
                      y_test.values[mask_clean],
                      y_pred_test[mask_clean])
    if mask_stress.any():
        print_metrics("Stress subset",
                      y_test.values[mask_stress],
                      y_pred_test[mask_stress])

    if not no_plots:
        plot_actual_vs_predicted_colored(y_test, y_pred_test, is_stress_test, save_plots)
        plot_residuals_by_source(y_test, y_pred_test, is_stress_test, save_plots)

    print("\n[4/4] Comparison table ...")
    comparison = pd.DataFrame({
        "Experiment": [
            "NB1 – XGB baseline (naive fill)",
            "NB2 – XGB group impute + clip",
            "NB3 – XGB combined training (this)",
        ],
        "RMSE Stress": ["~11.2", "~10.8", f"{rmse_all:.4f}*"],
        "R² Stress":   ["~-0.36", "~-0.31", f"{r2_all:.4f}*"],
    })
    print(comparison.to_string(index=False))
    print("* NB3 evaluates on a mixed test set; stress subset R² differs from "
          "evaluating stress alone.")
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Regressor trained on combined clean + stress data."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
