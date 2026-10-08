"""
Random Forest Regression – Outlier Clipping and NaN Imputation
==============================================================

Purpose
-------
Improve stress-test performance by attacking the root cause — poor
preprocessing — rather than hyperparameter tuning. Three strategies are
compared: median imputation (baseline), KNN imputation, and KNN imputation
with outlier clipping. Direct port of
`random_forest_regression_outlier_imputation.ipynb`.

Step-by-step explanation
------------------------
1. Load training data with eight raw features and split 80/20.
2. Train the RF baseline (200 trees, max_depth=7) and record internal test
   metrics.
3. Load the stress-test dataset and detect out-of-range outliers relative
   to training bounds.
4. Strategy 1 – Median imputation: fill NaN with training-split medians.
5. Strategy 2 – KNN imputation: fill NaN using the 5 nearest training
   rows (fitted on training data only, no leakage).
6. Strategy 3 – KNN + outlier clipping: after KNN fill, clip values to
   the 1st–99th percentile range of the training data.
7. Compare all three strategies and identify the best preprocessing step.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURE_COLS : list of str
    Eight raw input columns; engineered features excluded (label leakage).
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- TEST_SIZE : float
    Fraction held out for the internal test split (0.20).
- CLIP_LOWER_PCTILE / CLIP_UPPER_PCTILE : float
    Percentile boundaries for outlier clipping (1st and 99th).

Runtime
-------
Medium. Full run takes under 20 seconds.

Usage
-----
    python -m regression.random_forest_regression_outlier_imputation
    python -m regression.random_forest_regression_outlier_imputation --no-plots
    python -m regression.random_forest_regression_outlier_imputation --save-plots DIR
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
from shared.modeling import impute_with_train_medians


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TARGET = "price_eur_per_kg"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLIP_LOWER_PCTILE = 0.01
CLIP_UPPER_PCTILE = 0.99

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
    Load, split, and return training + raw stress data.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress_raw, y_stress)
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
    X_stress_raw = df_stress[FEATURE_COLS].copy()
    y_stress = df_stress[TARGET].copy()

    return X_train, X_test, y_train, y_test, X_stress_raw, y_stress


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
# Preprocessing strategies
# ------------------------------------------------------------

def strategy_median(X_stress_raw, y_stress, X_train):
    """
    Strategy 1: impute NaN with training-set medians.

    Parameters
    ----------
    X_stress_raw : pd.DataFrame
    y_stress : pd.Series
    X_train : pd.DataFrame

    Returns
    -------
    tuple
        (X_filled, y_valid)
    """
    df_stress = X_stress_raw.copy()
    df_stress[TARGET] = y_stress

    df_stress = impute_with_train_medians(X_train, df_stress, FEATURE_COLS)
    df_stress = df_stress.dropna(subset=[TARGET])

    return df_stress[FEATURE_COLS], df_stress[TARGET]


def strategy_knn(X_stress_raw, y_stress, X_train):
    """
    Strategy 2: impute NaN with KNN (5 nearest training neighbours).

    Parameters
    ----------
    X_stress_raw : pd.DataFrame
    y_stress : pd.Series
    X_train : pd.DataFrame

    Returns
    -------
    tuple
        (X_filled, y_valid, knn_imputer)
    """
    knn = KNNImputer(n_neighbors=5)
    knn.fit(X_train)

    X_filled = pd.DataFrame(
        knn.transform(X_stress_raw),
        columns=FEATURE_COLS,
        index=X_stress_raw.index,
    )

    valid_mask = y_stress.notna()
    return X_filled[valid_mask], y_stress[valid_mask], knn


def strategy_knn_clip(X_stress_raw, y_stress, X_train):
    """
    Strategy 3: KNN imputation then clip outliers to training percentiles.

    Parameters
    ----------
    X_stress_raw : pd.DataFrame
    y_stress : pd.Series
    X_train : pd.DataFrame

    Returns
    -------
    tuple
        (X_clipped, y_valid, clip_bounds)
    """
    X_filled, y_valid, knn = strategy_knn(X_stress_raw, y_stress, X_train)

    # Boundaries from training data
    clip_bounds = pd.DataFrame({
        "lower": X_train.quantile(CLIP_LOWER_PCTILE),
        "upper": X_train.quantile(CLIP_UPPER_PCTILE),
    })

    X_clipped = X_filled.copy()
    for col in FEATURE_COLS:
        X_clipped[col] = X_clipped[col].clip(
            clip_bounds.loc[col, "lower"],
            clip_bounds.loc[col, "upper"],
        )

    return X_clipped, y_valid, clip_bounds


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_regressor(model, X, y, label):
    """
    Compute and print MAE, RMSE, and R².

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
    print(f"Rows: {len(y)}")

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


def plot_outlier_summary(X_train, X_stress_raw, save_dir=None):
    """
    Print a table of training range vs stress range per feature.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_stress_raw : pd.DataFrame
    save_dir : str or None
        Unused; included for interface consistency.
    """
    print(f"\n{'Feature':<22} {'Train Min':>10} {'Train Max':>10} "
          f"{'Stress Min':>10} {'Stress Max':>10} {'Outliers?':>10}")
    print("-" * 74)
    for col in FEATURE_COLS:
        t_min, t_max = X_train[col].min(), X_train[col].max()
        s_min = X_stress_raw[col].min()
        s_max = X_stress_raw[col].max()
        flag = "YES" if (s_min < t_min or s_max > t_max) else "no"
        print(f"{col:<22} {t_min:>10.2f} {t_max:>10.2f} "
              f"{s_min:>10.2f} {s_max:>10.2f} {flag:>10}")


def plot_three_strategies(results, save_dir=None):
    """
    Side-by-side actual vs predicted scatter for all three strategies.

    Parameters
    ----------
    results : list of (y_true, y_pred, title, color)
    save_dir : str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (y_true, y_pred, title, color) in zip(axes, results):
        ax.scatter(y_true, y_pred, alpha=0.5, s=25, color=color)
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            color="red", linestyle="--", linewidth=2,
        )
        ax.set_xlabel("Actual Price (EUR)")
        ax.set_ylabel("Predicted Price (EUR)")
        ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, "three_strategies", save_dir)


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, train, and compare three strategies.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("RF Regression – Outlier Clipping & NaN Imputation")
    print("=" * 60)

    print("\n[1/4] Loading data and splitting ...")
    X_train, X_test, y_train, y_test, X_stress_raw, y_stress = prepare_data()
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\n[2/4] Training model ...")
    model = train_model(X_train, y_train)
    mae_t, rmse_t, r2_t, _ = evaluate_regressor(
        model, X_test, y_test, "Internal Test Set"
    )

    print("\n[3/4] Stress-test outlier analysis ...")
    plot_outlier_summary(X_train, X_stress_raw)

    print("\nEvaluating three preprocessing strategies on stress test ...")

    # Strategy 1 – Median
    X_s1, y_s1 = strategy_median(X_stress_raw, y_stress, X_train)
    mae_s1, rmse_s1, r2_s1, y_p1 = evaluate_regressor(
        model, X_s1, y_s1, "Strategy 1: Median Imputation"
    )

    # Strategy 2 – KNN
    X_s2, y_s2, _ = strategy_knn(X_stress_raw, y_stress, X_train)
    mae_s2, rmse_s2, r2_s2, y_p2 = evaluate_regressor(
        model, X_s2, y_s2, "Strategy 2: KNN Imputation"
    )

    # Strategy 3 – KNN + clip
    X_s3, y_s3, clip_bounds = strategy_knn_clip(X_stress_raw, y_stress, X_train)
    print(f"\nClipping bounds (1st–99th pctile from training):\n{clip_bounds.round(2)}")
    mae_s3, rmse_s3, r2_s3, y_p3 = evaluate_regressor(
        model, X_s3, y_s3, "Strategy 3: KNN + Outlier Clipping"
    )

    if not no_plots:
        plot_three_strategies(
            [
                (y_s1, y_p1, "Strategy 1: Median", "steelblue"),
                (y_s2, y_p2, "Strategy 2: KNN", "darkorange"),
                (y_s3, y_p3, "Strategy 3: KNN + Clip", "green"),
            ],
            save_dir=save_plots,
        )

    print("\n[4/4] Summary ...")
    comparison = pd.DataFrame({
        "Strategy":   [
            "Internal Test",
            "Stress: Median (baseline)",
            "Stress: KNN Imputation",
            "Stress: KNN + Outlier Clip",
        ],
        "MAE (EUR)":  [f"{mae_t:.2f}", f"{mae_s1:.2f}", f"{mae_s2:.2f}", f"{mae_s3:.2f}"],
        "RMSE (EUR)": [f"{rmse_t:.2f}", f"{rmse_s1:.2f}", f"{rmse_s2:.2f}", f"{rmse_s3:.2f}"],
        "R²":         [f"{r2_t:.4f}", f"{r2_s1:.4f}", f"{r2_s2:.4f}", f"{r2_s3:.4f}"],
    })
    print(comparison.to_string(index=False))

    stress_r2s = {"Median": r2_s1, "KNN": r2_s2, "KNN+Clip": r2_s3}
    best = max(stress_r2s, key=stress_r2s.get)
    print(f"Best stress-test strategy: {best} (R² = {stress_r2s[best]:.4f})")
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF Regressor comparing three stress-test preprocessing strategies."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
