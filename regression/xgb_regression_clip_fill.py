"""
XGBoost Regression – Group-Wise Imputation and Outlier Clipping (NB2)
=======================================================================

Purpose
-------
Improve the XGBoost regression baseline by applying smarter stress-data
preprocessing: group-wise median imputation (fill NaN with the subgroup
median for the row's meat_type) and outlier clipping (cap extreme values
at training-set bounds). Direct port of `xgb_regression_clip_fill.ipynb`.

Step-by-step explanation
------------------------
1. Load training data, engineer protein_fat_ratio, perform 80/20 split.
2. Train XGBRegressor with the same parameters as NB1.
3. Evaluate on internal test set (RMSE, MAE, R²).
4. Prepare the stress dataset in two steps:
   a. Group-wise imputation: fill NaN in each column with the median of
      the stress rows that share the same meat_type value. Rows with
      NaN meat_type fall back to the global stress median. Note: using
      within-stress subgroup statistics is valid for regression because
      meat_type is a feature, not the target.
   b. Outlier clipping: cap animal_age_months, storage_days, and
      fat_content_pct at their respective X_train maximums to suppress
      the most extreme out-of-distribution values.
5. Add protein_fat_ratio to the preprocessed stress set.
6. Evaluate on the 91 stress rows with a known price.
7. Comparison with NB1 baseline.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURES : list of str
    Eight raw input columns (target excluded).
- ALL_FEATURES : list of str
    FEATURES plus 'protein_fat_ratio'.
- CLIP_COLS : list of str
    Three columns clipped to training-set maximum.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Fast. Full run takes under 10 seconds.

Usage
-----
    python -m regression.xgb_regression_clip_fill
    python -m regression.xgb_regression_clip_fill --no-plots
    python -m regression.xgb_regression_clip_fill --save-plots DIR
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

# Columns with values far outside the training distribution in the stress set
CLIP_COLS = ["animal_age_months", "storage_days", "fat_content_pct"]


# --------------------------------------------------------
# Feature engineering
# --------------------------------------------------------

def add_protein_fat_ratio(df):
    """
    Add protein_fat_ratio = protein_pct / (fat_content_pct + 1e-5).

    Epsilon prevents division by zero. No label leakage since neither
    protein_pct nor fat_content_pct is derived from price.

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
# Stress preprocessing
# --------------------------------------------------------

def group_wise_impute(df, group_col="meat_type"):
    """
    Fill numeric NaN with the subgroup median keyed by group_col.

    Algorithm:
    1. For each column that has at least one NaN, compute the median
       within each group_col level and use it to fill NaN values in
       that group.
    2. After step 1, fill any remaining NaN (rows whose group_col is
       itself NaN) with the global column median.

    This strategy is valid for regression where group_col (meat_type)
    is a feature, not the regression target. It would be leaky for
    classification when group_col is the target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to impute (modified in place, returns copy).
    group_col : str
        Column used as grouping key.

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
        # Rows where group_col itself is NaN can't be group-filled; use global
        out[col] = out[col].fillna(out[col].median())
    return out


def clip_to_train_max(df, X_train):
    """
    Clip CLIP_COLS to their X_train maximum.

    Reduces the impact of extreme out-of-distribution stress values
    without filling NaN (NaN stays NaN after clipping).

    Parameters
    ----------
    df : pd.DataFrame
    X_train : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    for col in CLIP_COLS:
        upper = X_train[col].max()
        out[col] = out[col].clip(upper=upper)
    return out


def prepare_stress(df_stress_raw, X_train):
    """
    Full stress preprocessing pipeline.

    Order: group-wise impute -> clip -> add protein_fat_ratio.

    Parameters
    ----------
    df_stress_raw : pd.DataFrame
        Raw stress dataset (from load_stress_data()).
    X_train : pd.DataFrame
        Training features (used for clip bounds).

    Returns
    -------
    tuple
        (X_stress: pd.DataFrame, y_stress: pd.Series)
        Both contain only the 91 rows with a known price.
    """
    valid_mask = df_stress_raw[TARGET].notna()

    df_s = group_wise_impute(df_stress_raw)
    df_s = clip_to_train_max(df_s, X_train)
    df_s = add_protein_fat_ratio(df_s)

    X_stress = df_s.loc[valid_mask, ALL_FEATURES]
    y_stress = df_stress_raw.loc[valid_mask, TARGET]
    return X_stress, y_stress


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load training data, engineer protein_fat_ratio, split 80/20.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, df_stress_raw)
    """
    df = load_training_data()
    df = add_protein_fat_ratio(df)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    df_stress_raw = load_stress_data()
    return X_train, X_test, y_train, y_test, df_stress_raw


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train an XGBRegressor with the same parameters as NB1.

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


def plot_clip_comparison(df_stress_raw, X_train, save_dir=None):
    """
    Side-by-side boxplots of the three clipped columns before and after clipping.

    Visualises how clipping reduces extreme outlier values.

    Parameters
    ----------
    df_stress_raw : pd.DataFrame
    X_train : pd.DataFrame
    save_dir : str or None
    """
    df_clipped = clip_to_train_max(df_stress_raw, X_train)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, CLIP_COLS):
        ax.boxplot(
            [df_stress_raw[col].dropna(), df_clipped[col].dropna()],
            labels=["Before clip", "After clip"],
            patch_artist=True,
        )
        ax.axhline(y=X_train[col].max(), color="red", linestyle="--", label="Train max")
        ax.set_title(col)
        ax.legend()

    plt.suptitle("Outlier Clipping Effect on Stress-Test Columns", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, "clip_comparison", save_dir)


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
    lo = min(np.min(y_true), y_pred.min())
    hi = max(np.max(y_true), y_pred.max())
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
    Run the full pipeline: train, evaluate internally, preprocess stress, evaluate stress.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None -> show interactively.
    """
    print("=" * 60)
    print("XGBoost Regression – Group Impute + Clip (NB2)")
    print("=" * 60)

    print("\n[1/4] Loading and preparing data ...")
    X_train, X_test, y_train, y_test, df_stress_raw = prepare_data()
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Stress raw: {len(df_stress_raw)} rows "
          f"| Valid price: {df_stress_raw[TARGET].notna().sum()}")

    print("\nMissing values in stress dataset:")
    print(df_stress_raw.isna().sum())

    print("\n[2/4] Training model ...")
    model = train_model(X_train, y_train)

    print("\n[3/4] Internal evaluation ...")
    _, _, _, _ = evaluate_regressor(model, X_train, y_train, "Train Set")
    rmse_te, mae_te, r2_te, y_pred_te = evaluate_regressor(
        model, X_test, y_test, "Internal Test Set"
    )

    if not no_plots:
        plot_actual_vs_predicted(y_test, y_pred_te, "Internal Test", "steelblue", save_plots)

    print("\n[4/4] Stress evaluation (group impute + clip) ...")

    # Diagnostic: show clipping effect
    if not no_plots:
        plot_clip_comparison(df_stress_raw, X_train, save_dir=save_plots)

    for col in CLIP_COLS:
        n_clipped = (
            df_stress_raw[col].notna() & (df_stress_raw[col] > X_train[col].max())
        ).sum()
        print(f"  {col}: {n_clipped} values clipped to <= {X_train[col].max():.1f}")

    X_stress, y_stress = prepare_stress(df_stress_raw, X_train)
    rmse_s, mae_s, r2_s, y_pred_s = evaluate_regressor(
        model, X_stress, y_stress, "Stress-Test Set (group impute + clip)"
    )

    if not no_plots:
        plot_actual_vs_predicted(y_stress, y_pred_s, "Stress Test", "darkorange", save_plots)

    print("\nComparison with NB1 baseline:")
    comparison = pd.DataFrame({
        "Experiment": [
            "NB1 – XGB baseline (naive median fill)",
            "NB2 – XGB group impute + clip (this)",
        ],
        "RMSE Stress":  ["~11.2", f"{rmse_s:.4f}"],
        "MAE Stress":   ["~8.76", f"{mae_s:.4f}"],
        "R² Stress":    ["~-0.36", f"{r2_s:.4f}"],
        "R² Test":      [f"~{r2_te:.2f}", f"{r2_te:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Regressor with group-wise imputation and outlier clipping."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
