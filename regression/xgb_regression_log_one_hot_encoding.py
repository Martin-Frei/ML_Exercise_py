"""
XGBoost Regression – Log Target + One-Hot Encoding (NB4)
==========================================================

Purpose
-------
Investigate two preprocessing choices: log-transforming the price target
(to reduce right-skew) and one-hot encoding meat_type. Compares stress
performance against the baseline and group-impute variants.
Direct port of `xgb_regression_log_one_hot_encoding.ipynb`.

Step-by-step explanation
------------------------
1. Load training data (8 raw features, no protein_fat_ratio engineering).
2. One-hot encode with pd.get_dummies(drop_first=True). For numeric meat_type
   (dtype int64), get_dummies leaves it unchanged — the column stays numeric.
   The call is kept for faithfulness to the notebook and generality if meat_type
   were ever passed as a categorical/object column.
3. Align train and test (or stress) columns so that any new dummy columns
   in one split are filled with 0 in the other.
4. Log-transform the target: y_log = np.log1p(y). Predictions are inverted
   with np.expm1() before computing EUR metrics.
5. 80/20 split; train XGBRegressor on X_train with log target.
6. Evaluate on internal test set (metrics in EUR after expm1).
7. Prepare stress: impute NaN with X_train column medians, then one-hot
   encode and align to the training column layout.
8. Evaluate on the 91 stress rows with a known price.
9. Comparison with previous XGB regression variants.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURES : list of str
    Eight raw input columns. No engineered features in this notebook.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Fast. Full run takes under 10 seconds.

Usage
-----
    python -m regression.xgb_regression_log_one_hot_encoding
    python -m regression.xgb_regression_log_one_hot_encoding --no-plots
    python -m regression.xgb_regression_log_one_hot_encoding --save-plots DIR
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
    Load data, one-hot encode, split, log-transform target, prepare stress.

    One-hot encoding note:
    pd.get_dummies operates on object / category columns by default; numeric
    (int64) columns like meat_type are left as-is. The call + align pattern
    ensures that train and test / stress always have identical columns even
    when future data has meat_type as a categorical type.

    Returns
    -------
    tuple
        (X_train_enc, X_test_enc, y_train_log, y_test,
         X_stress_enc, y_stress, train_medians)
    """
    df = load_training_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Log-transform the training target; metrics will be computed in EUR space
    y_train_log = np.log1p(y_train)

    # One-hot encode: no-op for numeric meat_type, but makes the pipeline robust
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_raw_enc = pd.get_dummies(X_test, drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_raw_enc, join="left", axis=1, fill_value=0
    )

    # Store training column medians before encoding for stress imputation
    train_medians = X_train[FEATURES].median()

    # Stress preparation: impute NaN with training medians, then encode + align
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

    return (X_train_enc, X_test_enc, y_train_log, y_test,
            X_stress_enc, y_stress, train_medians)


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_model(X_train_enc, y_train_log):
    """
    Train XGBRegressor on log(price).

    Parameters
    ----------
    X_train_enc : pd.DataFrame
    y_train_log : pd.Series

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
    model.fit(X_train_enc, y_train_log)
    return model


# --------------------------------------------------------
# Evaluation
# --------------------------------------------------------

def evaluate_regressor(model, X_enc, y_true, label, use_log_inverse=True):
    """
    Compute RMSE, MAE, R² after reversing the log transform.

    Parameters
    ----------
    model : fitted XGBRegressor
    X_enc : pd.DataFrame
    y_true : pd.Series
    label : str
    use_log_inverse : bool
        If True, predictions are inverse-transformed with np.expm1().

    Returns
    -------
    tuple
        (rmse: float, mae: float, r2: float, y_pred_eur: np.ndarray)
    """
    y_pred_log = model.predict(X_enc)
    y_pred_eur = np.expm1(y_pred_log) if use_log_inverse else y_pred_log

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_eur)))
    mae = mean_absolute_error(y_true, y_pred_eur)
    r2 = r2_score(y_true, y_pred_eur)

    print(f"\n=== {label} ===")
    print(f"RMSE: {rmse:.4f} EUR")
    print(f"MAE:  {mae:.4f} EUR")
    print(f"R²:   {r2:.4f}")
    print(f"Rows: {len(y_true)}")

    return rmse, mae, r2, y_pred_eur


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


def plot_log_transform_effect(y_train, save_dir=None):
    """
    Histograms of raw and log-transformed price to show the skew reduction.

    Parameters
    ----------
    y_train : pd.Series
    save_dir : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y_train, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Price (EUR)")
    axes[0].set_title("Raw Target Distribution")

    axes[1].hist(np.log1p(y_train), bins=40, color="teal", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("log1p(Price)")
    axes[1].set_title("Log-Transformed Target Distribution")

    plt.suptitle("Effect of Log Transform on Target Skew", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, "log_transform_effect", save_dir)


def plot_actual_vs_predicted(y_true, y_pred, label, color, save_dir=None):
    """
    Scatter of actual vs predicted prices (EUR, after inverse log transform).

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
    y_pred : np.ndarray (EUR)
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
    Run the full pipeline: log-transform, train, evaluate internally and on stress.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None -> show interactively.
    """
    print("=" * 60)
    print("XGBoost Regression – Log Target + One-Hot Encoding (NB4)")
    print("=" * 60)

    print("\n[1/4] Preparing data ...")
    (X_train_enc, X_test_enc, y_train_log, y_test,
     X_stress_enc, y_stress, train_medians) = prepare_data()

    print(f"Train: {len(X_train_enc)} rows | Test: {len(X_test_enc)} rows "
          f"| Stress: {len(X_stress_enc)} rows")
    print(f"Feature columns after get_dummies: {list(X_train_enc.columns)}")

    if not no_plots:
        # Show the log transform effect on the raw training target
        from shared.data_loading import load_training_data as _ld
        _df = _ld()
        plot_log_transform_effect(_df[TARGET], save_dir=save_plots)

    print("\n[2/4] Training model (target = log1p(price)) ...")
    model = train_model(X_train_enc, y_train_log)

    print("\n[3/4] Evaluating (metrics in EUR after expm1 inverse) ...")
    # Train-set eval: compare log predictions vs log truth for overfitting gauge
    y_pred_train_log = model.predict(X_train_enc)
    rmse_tr_log = float(np.sqrt(mean_squared_error(y_train_log, y_pred_train_log)))
    print(f"\n=== Train Set (log space) ===")
    print(f"RMSE (log): {rmse_tr_log:.4f}")

    rmse_te, mae_te, r2_te, y_pred_te = evaluate_regressor(
        model, X_test_enc, y_test, "Internal Test Set (EUR)", use_log_inverse=True
    )

    if not no_plots:
        plot_actual_vs_predicted(y_test, y_pred_te, "Internal Test", "steelblue", save_plots)

    rmse_s, mae_s, r2_s, y_pred_s = evaluate_regressor(
        model, X_stress_enc, y_stress, "Stress-Test Set (EUR)", use_log_inverse=True
    )

    if not no_plots:
        plot_actual_vs_predicted(y_stress, y_pred_s, "Stress Test", "darkorange", save_plots)

    print("\n[4/4] Comparison ...")
    comparison = pd.DataFrame({
        "Experiment": [
            "NB1 – XGB baseline (raw price)",
            "NB2 – XGB group impute + clip (raw price)",
            "NB4 – XGB log price + OHE (this)",
        ],
        "R² Test":    ["~0.975", "~0.975", f"{r2_te:.4f}"],
        "RMSE Stress": ["~11.2", "~10.8", f"{rmse_s:.4f}"],
        "R² Stress":   ["~-0.36", "~-0.31", f"{r2_s:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Regressor with log-transformed target and one-hot encoding."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
