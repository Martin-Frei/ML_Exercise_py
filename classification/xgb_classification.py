"""
XGBoost Classification – Baseline (NB1)
========================================

Purpose
-------
Predict meat_type (classes 1–5) from 8 raw features using XGBClassifier.
Compares two stress-test variants: native NaN handling vs training-median fill.
Direct port of `xgb_classification.ipynb`.

Step-by-step explanation
------------------------
1. Load data; verify class balance and missing-value pattern in stress set.
2. Shift labels 1–5 → 0–4 (XGBoost requires 0-indexed consecutive labels).
3. Stratified 80/20 split; identical random state to all earlier notebooks.
4. Train XGBClassifier (n_estimators=150, learning_rate=0.08, max_depth=7).
   Uses multi:softprob objective – outputs one probability per class.
5. Evaluate on internal test: accuracy, gap, per-class F1, confusion matrix.
6. Feature importance using XGBoost's gain metric.
7. Prepare stress test: drop rows where meat_type is missing (100 → 91).
8. Two stress variants:
   - A (native NaN): feed NaN rows directly; XGBoost uses its default direction.
   - B (median fill): replace NaN with training-set median before predicting.
9. Detailed per-class report on the better variant.
10. Out-of-range table: how many stress values exceed the training feature range.
11. Comparison table with all previous classification experiments.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- FEATURES : list of str
    Eight raw input columns (target excluded, no engineered features).
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- CLASS_NAMES : list of str
    Original meat_type labels as strings ('1'–'5'), used in reports.

Runtime
-------
Medium. Full run takes under 15 seconds.

Usage
-----
    python -m classification.xgb_classification
    python -m classification.xgb_classification --no-plots
    python -m classification.xgb_classification --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from shared.data_loading import load_stress_data, load_training_data


# --------------------------------------------------------
# Constants
# --------------------------------------------------------

TARGET = "meat_type"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "fat_content_pct",
    "protein_pct",
    "marbling_score",
    "animal_age_months",
    "storage_days",
    "organic",
    "cut_quality",
    "price_eur_per_kg",
]

CLASS_NAMES = ["1", "2", "3", "4", "5"]


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load training data, shift labels 1–5 → 0–4, split 80/20, and load stress.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress)
    """
    df = load_training_data()

    X = df[FEATURES]
    y = df[TARGET].astype(int) - 1  # 1-5 → 0-4 for XGBoost

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    df_stress = load_stress_data()
    stress = df_stress.dropna(subset=[TARGET]).copy()
    X_stress = stress[FEATURES]
    y_stress = stress[TARGET].astype(int) - 1

    return X_train, X_test, y_train, y_test, X_stress, y_stress


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train XGBClassifier with multi:softprob objective.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series  (labels 0–4)

    Returns
    -------
    XGBClassifier
    """
    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=150,
        learning_rate=0.08,
        max_depth=7,
        eval_metric="mlogloss",
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


def plot_confusion_matrix(y_true, y_pred, label, cmap, save_dir=None):
    """
    Heatmap confusion matrix using original meat_type labels.

    Parameters
    ----------
    y_true, y_pred : array-like  (values 0–4)
    label : str
    cmap : str
    save_dir : str or None
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted meat_type")
    ax.set_ylabel("True meat_type")
    ax.set_title(f"Confusion Matrix – {label}")
    plt.tight_layout()
    _save_or_show(fig, f"confusion_{label.replace(' ', '_').lower()}", save_dir)


def plot_feature_importance(model, save_dir=None):
    """
    Horizontal bar chart of XGBoost gain-based feature importance.

    Parameters
    ----------
    model : XGBClassifier
    save_dir : str or None
    """
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_xlabel("Importance (gain, normalised)")
    ax.set_title("XGBoost Feature Importance – Baseline")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance", save_dir)

    print("\nFeature importances:")
    print((importance.sort_values(ascending=False) * 100).round(1).astype(str) + " %")


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, train, evaluate, compare stress variants.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("XGBoost Classification – Baseline (NB1)")
    print("=" * 60)

    print("\n[1/5] Loading data ...")
    X_train, X_test, y_train, y_test, X_stress, y_stress = prepare_data()
    print(f"Train: {len(X_train)} rows  Test: {len(X_test)} rows  Stress: {len(X_stress)} rows")
    print("Class distribution (training):")
    print((y_train + 1).value_counts().sort_index())
    print("\nMissing values (stress features):")
    print(X_stress.isna().sum())

    print("\n[2/5] Training model ...")
    model = train_model(X_train, y_train)
    print("Model trained.")

    print("\n[3/5] Evaluating on internal test set ...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    print(f"Train accuracy: {acc_train:.2%}")
    print(f"Test accuracy:  {acc_test:.2%}")
    print(f"Gap:            {acc_train - acc_test:.2%}")
    print("\nClassification report (internal test):")
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, digits=2))

    if not no_plots:
        plot_confusion_matrix(y_test, y_test_pred, "Internal Test", "Blues", save_plots)
        plot_feature_importance(model, save_dir=save_plots)

    print("\n[4/5] Stress test: native NaN vs median fill ...")

    # Variant A – native NaN: XGBoost uses its trained default direction
    y_pred_native = model.predict(X_stress)
    acc_native = accuracy_score(y_stress, y_pred_native)

    # Variant B – median fill from training data only (no target leakage)
    train_medians = X_train.median()
    X_stress_median = X_stress.fillna(train_medians)
    y_pred_median = model.predict(X_stress_median)
    acc_median = accuracy_score(y_stress, y_pred_median)

    results_stress = pd.DataFrame({
        "Variant": ["A – Native NaN", "B – Median fill"],
        "Correct / 91": [
            int((y_pred_native == y_stress.values).sum()),
            int((y_pred_median == y_stress.values).sum()),
        ],
        "Accuracy": [f"{acc_native * 100:.2f} %", f"{acc_median * 100:.2f} %"],
        "Macro F1": [
            round(f1_score(y_stress, y_pred_native, average="macro"), 3),
            round(f1_score(y_stress, y_pred_median, average="macro"), 3),
        ],
    })
    print(results_stress.to_string(index=False))

    # Detailed report on the better variant
    if acc_native >= acc_median:
        best_name, y_pred_best = "Native NaN", y_pred_native
    else:
        best_name, y_pred_best = "Median fill", y_pred_median

    print(f"\nBest stress variant: {best_name}")
    print(classification_report(
        y_stress, y_pred_best, labels=range(5),
        target_names=CLASS_NAMES, digits=2, zero_division=0,
    ))

    if not no_plots:
        plot_confusion_matrix(
            y_stress, y_pred_best, f"Stress Test ({best_name})", "Oranges", save_plots
        )

    # Out-of-range analysis: how many stress values lie outside the training bounds
    train_min, train_max = X_train.min(), X_train.max()
    out_of_range = pd.DataFrame({
        "train_min": train_min,
        "train_max": train_max,
        "stress_min": X_stress.min(),
        "stress_max": X_stress.max(),
        "rows_below": (X_stress < train_min).sum(),
        "rows_above": (X_stress > train_max).sum(),
    })
    print("\nOut-of-range values (stress vs training bounds):")
    print(out_of_range)

    print("\n[5/5] Comparison with previous models ...")
    comparison = pd.DataFrame({
        "Model": [
            "DT baseline",
            "RF baseline (KNN)",
            "RF + target encoding",
            "XGB NB1 – native NaN",
            "XGB NB1 – median fill",
        ],
        "Acc Test": [
            "53.0 %", "56.0 %", "62.5 %",
            f"{acc_test * 100:.1f} %", f"{acc_test * 100:.1f} %",
        ],
        "Acc Stress": [
            "27 %", "24 %", "25 %",
            f"{acc_native * 100:.1f} %", f"{acc_median * 100:.1f} %",
        ],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Classifier baseline – native NaN vs median fill on stress test."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
