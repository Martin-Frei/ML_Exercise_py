"""
Decision Tree Classification – Meat Type Prediction
====================================================

Purpose
-------
Train a Decision Tree Classifier to predict `meat_type` (classes 1–5)
from nutritional and quality features. Direct port of
`decision_tree_classification_meat_type.ipynb`.

Step-by-step explanation
------------------------
1. Load the training data with three engineered features via
   `shared.data_loading.load_training_data()`.
2. Explore class distribution and feature correlations with `meat_type`.
3. Split into 80 % train / 20 % test (stratified so each class keeps
   its proportion in both sets).
4. Train a DecisionTreeClassifier with max_depth=5.
5. Evaluate on the internal test set: accuracy, classification report,
   and confusion matrix.
6. Visualise the tree structure (first 3 levels) and feature importances.
7. Load the stress-test dataset and impute missing values with
   training-set medians.
8. Evaluate on the stress-test data and print a side-by-side comparison.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).
- TEST_SIZE : float
    Fraction held out for the internal test split (0.20).
- feature_cols : list of str
    All columns except 'meat_type'; includes 'price_eur_per_kg' and the
    three engineered columns. No label leakage because the target is
    'meat_type', not 'price_eur_per_kg'.

Runtime
-------
Medium. Full run takes under 5 seconds.

Usage
-----
    python -m classification.decision_tree_classification_meat_type
    python -m classification.decision_tree_classification_meat_type --no-plots
    python -m classification.decision_tree_classification_meat_type --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from shared.data_loading import load_stress_data, load_training_data
from shared.modeling import impute_with_train_medians


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TARGET = "meat_type"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def prepare_data():
    """
    Load, split, and prepare training and stress-test data.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress, feature_cols)
    """
    df = load_training_data()

    # All columns except target (includes price_eur_per_kg + engineered cols)
    feature_cols = [c for c in df.columns if c != TARGET]

    X = df[feature_cols]
    y = df[TARGET]

    # stratify preserves class proportions in both halves
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    df_stress = load_stress_data()
    # Impute using the training-split medians (not full-dataset medians)
    df_stress = impute_with_train_medians(X_train, df_stress, feature_cols)

    # Drop rows where the target itself is NaN
    df_stress = df_stress.dropna(subset=[TARGET])

    X_stress = df_stress[feature_cols]
    y_stress = df_stress[TARGET].astype(int)

    return X_train, X_test, y_train, y_test, X_stress, y_stress, feature_cols


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train a Decision Tree Classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    DecisionTreeClassifier
        Fitted classifier.
    """
    model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
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
        True labels.
    label : str
        Header shown in the printed output.

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


def plot_eda(df, feature_cols, save_dir=None):
    """
    Plot class distribution and feature correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that includes TARGET and feature_cols.
    feature_cols : list of str
    save_dir : str or None
        Directory to save PNG; None → display on screen.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df[TARGET].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="black"
    )
    axes[0].set_title("Class Distribution – meat_type")
    axes[0].set_xlabel("Meat Type")
    axes[0].set_ylabel("Count")

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1],
    )
    axes[1].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    _save_or_show(fig, "eda", save_dir)


def plot_confusion_matrix(y_true, y_pred, label, cmap, save_dir=None):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    label : str
        Title suffix.
    cmap : str
        Matplotlib colormap.
    save_dir : str or None
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap=cmap)
    ax.set_title(f"Confusion Matrix – {label}")
    plt.tight_layout()
    _save_or_show(fig, f"confusion_{label.replace(' ', '_').lower()}", save_dir)


def plot_tree_viz(model, feature_cols, classes, save_dir=None):
    """
    Visualise the first 3 levels of the fitted decision tree.

    Parameters
    ----------
    model : DecisionTreeClassifier
    feature_cols : list of str
    classes : list of str
        Class labels for the leaf nodes.
    save_dir : str or None
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=classes,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision Tree (showing first 3 levels)", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, "tree_viz", save_dir)


def plot_feature_importance(model, feature_cols, save_dir=None):
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : fitted tree-based model
    feature_cols : list of str
    save_dir : str or None
    """
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="teal", edgecolor="black")
    ax.set_title("Feature Importance – Decision Tree")
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
    print("Decision Tree Classification – Meat Type Prediction")
    print("=" * 60)

    print("\n[1/5] Loading and preparing data ...")
    (X_train, X_test, y_train, y_test,
     X_stress, y_stress, feature_cols) = prepare_data()

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"| Stress: {len(X_stress)} rows")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    print("\n[2/5] Exploratory data analysis ...")
    df_full = pd.concat([X_train, y_train], axis=1)
    corr = df_full.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    print("Correlations with meat_type:")
    print(corr.round(3))
    if not no_plots:
        plot_eda(df_full, feature_cols, save_dir=save_plots)

    print("\n[3/5] Training model ...")
    model = train_model(X_train, y_train)
    print(f"Tree depth : {model.get_depth()}")
    print(f"Leaf nodes : {model.get_n_leaves()}")

    print("\n[4/5] Evaluating ...")
    acc_test, y_pred_test = evaluate_classifier(
        model, X_test, y_test, "Internal Test Set"
    )
    if not no_plots:
        classes = [str(c) for c in sorted(y_train.unique())]
        plot_confusion_matrix(y_test, y_pred_test, "Internal Test Set", "Blues", save_plots)
        plot_tree_viz(model, feature_cols, classes, save_plots)
        plot_feature_importance(model, feature_cols, save_plots)

    acc_stress, y_pred_stress = evaluate_classifier(
        model, X_stress, y_stress, "Stress-Test Set"
    )
    if not no_plots:
        plot_confusion_matrix(y_stress, y_pred_stress, "Stress Test", "Oranges", save_plots)

    print("\n[5/5] Summary ...")
    comparison = pd.DataFrame({
        "Dataset":  ["Internal Test (20%)", "Stress Test (100 rows)"],
        "Rows":     [len(y_test), len(y_stress)],
        "Accuracy": [f"{acc_test:.4f}", f"{acc_stress:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decision Tree Classifier – predict meat_type."
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
