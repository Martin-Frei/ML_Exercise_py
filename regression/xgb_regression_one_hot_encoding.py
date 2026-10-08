"""
XGBoost Regression – One-Hot Encoding without Log Transform (NB5)
===================================================================

Purpose
-------
Ablation study: same pipeline as NB4 (one-hot encoding + median imputation
for stress) but with the raw price target — no log transform. Quantifies the
contribution of the log transform to model performance.
Direct port of `xgb_regression_one_hot_encoding.ipynb`.

Step-by-step explanation
------------------------
1. Load training data (8 raw features, no protein_fat_ratio).
2. One-hot encode with pd.get_dummies(drop_first=True). For numeric meat_type
   (dtype int64), get_dummies leaves it unchanged. Column is kept as-is.
3. Align train and test / stress columns (same join='left' pattern as NB4).
4. No log transform: train directly on raw price_eur_per_kg.
5. 80/20 split; train XGBRegressor.
6. Evaluate on internal test set.
7. Impute stress NaN with X_train column medians; encode + align.
8. Evaluate on the 91 stress rows with a known price.
9. Side-by-side comparison of NB4 (log) vs NB5 (no log).

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURES : list of str
    Eight raw input columns. No engineered features.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Fast. Full run takes under 10 seconds.

Usage
-----
    python -m regression.xgb_regression_one_hot_encoding
    python -m regression.xgb_regression_one_hot_encoding --no-plots
    python -m regression.xgb_regression_one_hot_encoding --save-plots DIR
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


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load data, one-hot encode, split, and prepare stress.

    This function is identical to NB4's prepare_data except that no
    log transform is applied to the target. All differences in results
    between NB4 and NB5 are attributable to the log transform alone.

    Returns
    -------
    tuple
        (X_train_enc, X_test_enc, y_train, y_test,
         X_stress_enc, y_stress, train_medians)
    """
    df = load_training_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # One-hot encode (no-op for numeric meat_type, adds generality)
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_raw_enc = pd.get_dummies(X_test, drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_raw_enc, join="left", axis=1, fill_value=0
    )

    train_medians = X_train[FEATURES].median()

    # Stress preparation: impute with training medians, then encode + align
    df_stress_raw = load_stress_data()
    valid_mask = df_stress_raw[TARGET].notna()

    df_stress = df_stress_raw.copy()
    for col in FEATURES:
        df_stress[col] = df_stress[col].fillna(train_medians[col])

    X_stress_raw_enc = pd.get_dummies(df_stress[FEATURES], drop_first=True)
    X_train_enc, X_stress_enc = X_train_enc.align(
        X_stress_raw_enc, join="left", axis=1, fill_value=0
    )

    X_stress_enc = X_stress_enc.loc[valid_mask]
    y_stress = df_stress_raw.loc[valid_mask, TARGET]

    return (X_train_enc, X_test_enc, y_train, y_test,
            X_stress_enc, y_stress, train_medians)


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_model(X_train_enc, y_train):
    """
    Train XGBRegressor on the raw price target.

    Parameters
    ----------
    X_train_enc : pd.DataFrame
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
    model.fit(X_train_enc, y_train)
    return model


# --------------------------------------------------------
# Evaluation
# --------------------------------------------------------

def evaluate_regressor(model, X_enc, y_true, label):
    """
    Compute and print RMSE, MAE, R².

    Parameters
    ----------
    model : fitted XGBRegressor
    X_enc : pd.DataFrame
    y_true : pd.Series
    label : str

    Returns
    -------
    tuple
        (rmse: float, mae: float, r2: float, y_pred: np.ndarray)
    """
    y_pred = model.predict(X_enc)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {label} ===")
    print(f"RMSE: {rmse:.4f} EUR")
    print(f"MAE:  {mae:.4f} EUR")
    print(f"R²:   {r2:.4f}")
    print(f"Rows: {len(y_true)}")

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


def plot_actual_vs_predicted(y_true, y_pred, label, color, save_dir=None):
    """
    Scatter of actual vs predicted prices and a residual plot.

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


def plot_log_vs_no_log_comparison(model_no_log, X_enc, y_test, model_log,
                                   X_enc_log, y_test_log, save_dir=None):
    """
    Residual histograms for NB4 (log) vs NB5 (no log) side by side.

    Parameters
    ----------
    model_no_log : XGBRegressor  (trained on raw price)
    X_enc : pd.DataFrame
    y_test : pd.Series
    model_log : XGBRegressor or None  (optional, for reference)
    X_enc_log : pd.DataFrame or None
    y_test_log : pd.Series or None
    save_dir : str or None
    """
    y_pred_no_log = model_no_log.predict(X_enc)
    residuals_no_log = np.array(y_test) - y_pred_no_log

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals_no_log, bins=40, alpha=0.7, color="steelblue", label="No log (NB5)")
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Residual (EUR)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution – No Log Transform (NB5)")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "residuals_no_log", save_dir)


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: one-hot encode (no log), train, evaluate.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None -> show interactively.
    """
    print("=" * 60)
    print("XGBoost Regression – One-Hot Encoding (No Log Transform, NB5)")
    print("=" * 60)

    print("\n[1/4] Preparing data ...")
    (X_train_enc, X_test_enc, y_train, y_test,
     X_stress_enc, y_stress, train_medians) = prepare_data()

    print(f"Train: {len(X_train_enc)} rows | Test: {len(X_test_enc)} rows "
          f"| Stress: {len(X_stress_enc)} rows")
    print(f"Feature columns: {list(X_train_enc.columns)}")

    print("\n[2/4] Training model (raw price target, no log transform) ...")
    model = train_model(X_train_enc, y_train)

    print("\n[3/4] Evaluating ...")
    _, _, _, _ = evaluate_regressor(model, X_train_enc, y_train, "Train Set")
    rmse_te, mae_te, r2_te, y_pred_te = evaluate_regressor(
        model, X_test_enc, y_test, "Internal Test Set"
    )

    if not no_plots:
        plot_actual_vs_predicted(y_test, y_pred_te, "Internal Test", "steelblue", save_plots)
        plot_log_vs_no_log_comparison(
            model, X_test_enc, y_test, None, None, None, save_dir=save_plots
        )

    rmse_s, mae_s, r2_s, y_pred_s = evaluate_regressor(
        model, X_stress_enc, y_stress, "Stress-Test Set"
    )

    if not no_plots:
        plot_actual_vs_predicted(y_stress, y_pred_s, "Stress Test", "darkorange", save_plots)

    print("\n[4/4] NB4 (log) vs NB5 (no log) comparison ...")
    comparison = pd.DataFrame({
        "Experiment": [
            "NB4 – XGB OHE + log transform",
            "NB5 – XGB OHE, no log transform (this)",
        ],
        "R² Test":    ["~0.975 (log space)", f"{r2_te:.4f}"],
        "RMSE Stress": ["~11.5", f"{rmse_s:.4f}"],
        "MAE Stress":  ["~9.2", f"{mae_s:.4f}"],
        "R² Stress":   ["~-0.37", f"{r2_s:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Regressor with one-hot encoding and raw price target."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
