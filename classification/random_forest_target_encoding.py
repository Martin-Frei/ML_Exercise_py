"""
Random Forest Classification – Target Encoding
===============================================

Purpose
-------
Improve classification accuracy by replacing raw feature values with
target-encoded versions — each value is replaced with the mean `meat_type`
for that group, computed on training data only. Direct port of
`random_forest_target_encoding.ipynb`.

Step-by-step explanation
------------------------
1. Load data and split 80/20 (stratified).
2. Compute target encoding maps from training data:
   - cut_quality, organic, marbling_score → mean meat_type per group
   - price_eur_per_kg → binned into 5 quantile groups, then mean meat_type
3. Apply encodings to create four new `_te` columns on train and test.
4. Visualise per-class distributions of the encoded features.
5. Train RandomForestClassifier (200 trees, max_depth=7) on the 12
   combined features (8 original + 4 encoded).
6. Evaluate on the internal test set and report class-level F1 scores.
7. Load the stress-test dataset and impute missing values with
   training-set medians, then apply the same target encodings.
8. Evaluate on the stress-test data.
9. Compare with all previous classification experiments.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- FEATURE_COLS : list of str
    Eight raw input columns used as base features.
- ENCODE_COLS : list of str
    Categorical/ordinal features to target-encode directly.
- N_PRICE_BINS : int
    Number of quantile bins for price_eur_per_kg encoding (5).
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Medium. Full run takes under 15 seconds.

Usage
-----
    python -m classification.random_forest_target_encoding
    python -m classification.random_forest_target_encoding --no-plots
    python -m classification.random_forest_target_encoding --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
N_PRICE_BINS = 5

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

# Categorical/ordinal features to target-encode directly
ENCODE_COLS = ["cut_quality", "organic", "marbling_score"]


# ------------------------------------------------------------
# Target encoding helpers
# ------------------------------------------------------------

def build_encoding_maps(X_train, y_train):
    """
    Compute target encoding maps and price bin edges from training data.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    tuple
        (encoding_maps: dict, bin_edges: np.ndarray)
    """
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values

    encoding_maps = {}

    for col in ENCODE_COLS:
        encoding_maps[col] = train_df.groupby(col)[TARGET].mean()

    # Bin price into N_PRICE_BINS quantile groups and encode
    price_bins = pd.qcut(train_df["price_eur_per_kg"], q=N_PRICE_BINS, labels=False)
    train_df["price_bin"] = price_bins
    encoding_maps["price_bin"] = train_df.groupby("price_bin")[TARGET].mean()

    _, bin_edges = pd.qcut(
        train_df["price_eur_per_kg"], q=N_PRICE_BINS, retbins=True
    )
    bin_edges[0]  = -np.inf  # handle values below training min
    bin_edges[-1] =  np.inf  # handle values above training max

    return encoding_maps, bin_edges


def apply_target_encoding(X_data, encoding_maps, bin_edges):
    """
    Add target-encoded columns to a feature DataFrame.

    New columns created: cut_quality_te, organic_te,
    marbling_score_te, price_bin_te.

    Parameters
    ----------
    X_data : pd.DataFrame
        Feature matrix (must contain FEATURE_COLS).
    encoding_maps : dict
        Maps returned by build_encoding_maps().
    bin_edges : np.ndarray
        Price quantile bin edges from build_encoding_maps().

    Returns
    -------
    pd.DataFrame
        Copy of X_data with four additional `_te` columns appended.
    """
    X_encoded = X_data.copy()
    global_mean = {
        col: encoding_maps[col].mean() for col in ENCODE_COLS + ["price_bin"]
    }

    for col in ENCODE_COLS:
        te_map = encoding_maps[col]
        X_encoded[f"{col}_te"] = X_data[col].map(te_map).fillna(global_mean[col])

    price_bins = pd.cut(X_data["price_eur_per_kg"], bins=bin_edges, labels=False)
    X_encoded["price_bin_te"] = price_bins.map(
        encoding_maps["price_bin"]
    ).fillna(global_mean["price_bin"])

    return X_encoded


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def prepare_data():
    """
    Load, split, build encoding maps, and prepare stress-test data.

    Returns
    -------
    tuple
        (X_train_te, X_test_te, y_train, y_test,
         X_stress_te, y_stress, encoding_maps, bin_edges)
    """
    df = load_training_data()

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    encoding_maps, bin_edges = build_encoding_maps(X_train, y_train)

    X_train_te = apply_target_encoding(X_train, encoding_maps, bin_edges)
    X_test_te  = apply_target_encoding(X_test,  encoding_maps, bin_edges)

    # Stress data: impute with training medians, then apply encoding
    df_stress = load_stress_data()
    df_stress = impute_with_train_medians(X_train, df_stress, FEATURE_COLS)
    df_stress = df_stress.dropna(subset=[TARGET])

    X_stress = df_stress[FEATURE_COLS]
    y_stress = df_stress[TARGET].astype(int)

    X_stress_te = apply_target_encoding(X_stress, encoding_maps, bin_edges)

    return (X_train_te, X_test_te, y_train, y_test,
            X_stress_te, y_stress, encoding_maps, bin_edges)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(X_train_te, y_train):
    """
    Train a Random Forest Classifier on the encoded feature set.

    Parameters
    ----------
    X_train_te : pd.DataFrame
        Training features including target-encoded columns.
    y_train : pd.Series

    Returns
    -------
    RandomForestClassifier
        Fitted classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_te, y_train)
    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_classifier(model, X, y, label):
    """
    Print accuracy and classification report.

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


def plot_encoded_distributions(X_train_te, y_train, save_dir=None):
    """
    Histogram of each target-encoded feature split by meat_type.

    Parameters
    ----------
    X_train_te : pd.DataFrame
    y_train : pd.Series
    save_dir : str or None
    """
    te_cols = ["cut_quality_te", "organic_te", "marbling_score_te", "price_bin_te"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, col in zip(axes, te_cols):
        for mt in sorted(y_train.unique()):
            mask = (y_train == mt).values
            ax.hist(X_train_te.loc[mask, col], bins=15, alpha=0.5, label=f"Type {mt}")
        ax.set_title(col)
        ax.set_xlabel("Encoded Value")
        ax.legend(fontsize=7)

    plt.suptitle("Target-Encoded Features by Meat Type", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, "encoded_distributions", save_dir)


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


def plot_feature_importance(model, feature_names, save_dir=None):
    """
    Plot feature importances, colouring original vs encoded features.

    Parameters
    ----------
    model : RandomForestClassifier
    feature_names : list of str
    save_dir : str or None
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    # Orange for encoded features, teal for originals
    colors = ["darkorange" if "_te" in f else "teal" for f in importances.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Feature Importance (teal=original, orange=target-encoded)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance", save_dir)

    orig_sum = importances[[c for c in importances.index if "_te" not in c]].sum()
    te_sum   = importances[[c for c in importances.index if "_te" in c]].sum()
    print(f"Original features total: {orig_sum:.3f}")
    print(f"Encoded features total:  {te_sum:.3f}")


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, encode, train, evaluate, compare.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("Random Forest Classification – Target Encoding")
    print("=" * 60)

    print("\n[1/5] Loading and preparing data with target encoding ...")
    (X_train_te, X_test_te, y_train, y_test,
     X_stress_te, y_stress, encoding_maps, bin_edges) = prepare_data()

    te_cols = [c for c in X_train_te.columns if "_te" in c]
    print(f"Base features:    {len(FEATURE_COLS)}")
    print(f"Encoded features: {len(te_cols)} ({te_cols})")
    print(f"Total features:   {X_train_te.shape[1]}")

    print("\n[2/5] Target encoding maps (from training data):")
    for col in ENCODE_COLS:
        print(f"  {col}: {encoding_maps[col].round(4).to_dict()}")
    print(f"  price_bin: {encoding_maps['price_bin'].round(4).to_dict()}")
    print(f"  price bin edges: {[round(b, 2) for b in bin_edges]}")

    if not no_plots:
        plot_encoded_distributions(X_train_te, y_train, save_dir=save_plots)

    print("\n[3/5] Training model ...")
    model = train_model(X_train_te, y_train)
    print(f"Number of trees: {model.n_estimators}")

    print("\n[4/5] Evaluating ...")
    acc_test, y_pred_test = evaluate_classifier(
        model, X_test_te, y_test, "Internal Test Set"
    )
    if not no_plots:
        plot_confusion_matrix(y_test, y_pred_test, "Internal Test", "Blues", save_plots)
        plot_feature_importance(model, list(X_train_te.columns), save_dir=save_plots)

    acc_stress, y_pred_stress = evaluate_classifier(
        model, X_stress_te, y_stress, "Stress-Test Set"
    )
    if not no_plots:
        plot_confusion_matrix(y_stress, y_pred_stress, "Stress Test", "Oranges", save_plots)

    print("\n[5/5] Summary ...")
    comparison = pd.DataFrame({
        "Experiment":    [
            "DT + feature engineering",
            "RF baseline (global median imputation)",
            "RF + target encoding (this)",
        ],
        "Acc Test":   ["0.5292", "0.5625", f"{acc_test:.4f}"],
        "Acc Stress": ["0.2637", "0.1978", f"{acc_stress:.4f}"],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF Classifier with target-encoded features."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
