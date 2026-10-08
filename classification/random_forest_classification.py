"""
Random Forest Classification – Meat Type Prediction
====================================================

Purpose
-------
Train a Random Forest Classifier (200 trees) to predict `meat_type`
(classes 1–5) and compare it with the Decision Tree baseline. Direct
port of `random_forest_classification.ipynb`.

Step-by-step explanation
------------------------
1. Load the training data via `shared.data_loading.load_training_data()`;
   use only the eight raw feature columns to match the original notebook.
2. Explore class distribution and correlations with `meat_type`.
3. Split into 80 % train / 20 % test (stratified).
4. Train a RandomForestClassifier with 200 trees and max_depth=7.
5. Evaluate on the internal test set: accuracy, classification report,
   and confusion matrix.
6. Compare feature importances with the Decision Tree.
7. Load the stress-test dataset and impute with training-set medians.
8. Evaluate on the stress-test data.
9. Print a side-by-side comparison with the Decision Tree results.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- FEATURE_COLS : list of str
    The eight raw input columns; engineered features are excluded to
    match the original notebook's feature set.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- TEST_SIZE : float
    Fraction held out for the internal test split (0.20).

Runtime
-------
Medium. Full run takes under 15 seconds.

Usage
-----
    python -m classification.random_forest_classification
    python -m classification.random_forest_classification --no-plots
    python -m classification.random_forest_classification --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from shared.data_loading import load_stress_data, load_training_data
from shared.modeling import impute_with_train_medians


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TARGET = "meat_type"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Original notebook did not add engineered features; we match that set.
FEATURE_COLS = [
    "fat_content_pct",
    "protein_pct",
    "marbling_score",
    "animal_age_months",
    "storage_days",
    "organic",
    "cut_quality",
    "price_eur_per_kg",
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

    # stratify preserves class proportions in both halves
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    df_stress = load_stress_data()
    # Impute using the training-split medians
    df_stress = impute_with_train_medians(X_train, df_stress, FEATURE_COLS)

    # Drop rows where the target itself is NaN
    df_stress = df_stress.dropna(subset=[TARGET])

    X_stress = df_stress[FEATURE_COLS]
    y_stress = df_stress[TARGET].astype(int)

    return X_train, X_test, y_train, y_test, X_stress, y_stress


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    RandomForestClassifier
        Fitted classifier.
    """
    # 200 trees, max_depth=7 mirrors the Decision Tree for fair comparison
    model = RandomForestClassifier(
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

def evaluate_classifier(model, X, y, label):
    """
    Print accuracy and classification report for a fitted classifier.

    Parameters
    ----------
    model : fitted sklearn classifier
    X : pd.DataFrame
    y : pd.Series
    label : str

    Returns
    -------
    tuple
        (accuracy: float, y_pred: np.ndarray)
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, zero_division=0))

    return acc, y_pred


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
    Plot class distribution and feature correlations with meat_type.

    Parameters
    ----------
    df : pd.DataFrame
    save_dir : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df[TARGET].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="black"
    )
    axes[0].set_title("Class Distribution – meat_type")
    axes[0].set_xlabel("Meat Type")
    axes[0].set_ylabel("Count")

    corr = df[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    corr.plot(kind="barh", ax=axes[1], color="teal", edgecolor="black")
    axes[1].set_title("Feature Correlation with meat_type")
    axes[1].set_xlabel("Correlation")

    plt.tight_layout()
    _save_or_show(fig, "eda", save_dir)

    print("Correlation values:")
    print(corr.round(3))


def plot_confusion_matrix(y_true, y_pred, label, cmap, save_dir=None):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    label : str
    cmap : str
    save_dir : str or None
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap=cmap)
    ax.set_title(f"Confusion Matrix – {label}")
    plt.tight_layout()
    _save_or_show(fig, f"confusion_{label.replace(' ', '_').lower()}", save_dir)


def plot_feature_importance(model, save_dir=None):
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : RandomForestClassifier
    save_dir : str or None
    """
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="teal", edgecolor="black")
    ax.set_title("Feature Importance – Random Forest Classifier")
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
    print("Random Forest Classification – Meat Type Prediction")
    print("=" * 60)

    print("\n[1/5] Loading and preparing data ...")
    X_train, X_test, y_train, y_test, X_stress, y_stress = prepare_data()

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"| Stress: {len(X_stress)} rows")
    print(f"Classes: {sorted(y_train.unique().tolist())}")

    print("\n[2/5] Exploratory data analysis ...")
    df_full = pd.concat([X_train, y_train], axis=1)
    if not no_plots:
        plot_eda(df_full, save_dir=save_plots)

    print("\n[3/5] Training model ...")
    model = train_model(X_train, y_train)
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max depth:       {model.max_depth}")

    print("\n[4/5] Evaluating ...")
    acc_test, y_pred_test = evaluate_classifier(
        model, X_test, y_test, "Internal Test Set"
    )
    if not no_plots:
        plot_confusion_matrix(y_test, y_pred_test, "Internal Test", "Blues", save_plots)
        plot_feature_importance(model, save_dir=save_plots)

    acc_stress, y_pred_stress = evaluate_classifier(
        model, X_stress, y_stress, "Stress-Test Set"
    )
    if not no_plots:
        plot_confusion_matrix(y_stress, y_pred_stress, "Stress Test", "Oranges", save_plots)

    print("\n[5/5] Summary ...")
    # Decision Tree baseline from the original notebook
    comparison = pd.DataFrame({
        "Metric":   ["Accuracy (Internal Test)", "Accuracy (Stress Test)"],
        "DT Baseline": ["0.5292", "0.2637"],
        "Random Forest": [f"{acc_test:.4f}", f"{acc_stress:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Forest Classifier – predict meat_type."
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
