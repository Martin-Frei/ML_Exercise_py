"""
XGBoost Classification – Imputation and Outlier Clipping (NB2)
===============================================================

Purpose
-------
Test whether better stress-data preparation (outlier clipping + smarter
imputation) improves stress-test accuracy for the XGBoost classifier.
Direct port of `xgb_classification_imputation_clip.ipynb`.

Step-by-step explanation
------------------------
1. Load data, shift labels 1–5 → 0–4, stratified 80/20 split.
2. Train XGBClassifier with max_depth=3 (best from NB1 depth analysis).
3. Step A – Outlier clipping: limit stress feature values to the training
   min/max. NaN rows are not touched by clipping.
4. Step B – Four leakage-free imputation methods, each with and without
   clipping (8 combinations total):
   - Native NaN: XGBoost uses its default direction, no filling done.
   - Global median: fill NaN with the training-set median per feature.
   - KNN (k=5): StandardScaler + KNNImputer fitted on training data only.
     Integer-valued features are rounded after imputation.
   - Iterative: IterativeImputer (each feature predicted from all others,
     10 rounds). Integer features are rounded after imputation.
5. Step C – Leakage demonstration: fill NaN with the median of the row's
   TRUE meat_type group. INVALID in production — label is the answer we
   are trying to predict. Shows how much leakage inflates the score.
6. Best valid variant: per-class report and confusion matrix.
7. Recall per class across all 8 valid variants.
8. Comparison with previous models.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- FEATURES : list of str
    Eight raw input columns.
- INT_FEATURES : list of str
    Features that are integers; rounded after KNN/Iterative imputation.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Medium. Full run takes under 30 seconds (IterativeImputer adds ~10 s).

Usage
-----
    python -m classification.xgb_classification_imputation_clip
    python -m classification.xgb_classification_imputation_clip --no-plots
    python -m classification.xgb_classification_imputation_clip --save-plots DIR
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# These features are integers; round after KNN / Iterative imputation
INT_FEATURES = ["marbling_score", "animal_age_months", "storage_days", "organic", "cut_quality"]

IMPUTATION_METHODS = ["Native NaN", "Global median", "KNN", "Iterative"]


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load training data, shift labels, split, and load stress data.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress)
    """
    df = load_training_data()

    X = df[FEATURES]
    y = df[TARGET].astype(int) - 1  # 1-5 → 0-4

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
    Train XGBClassifier with max_depth=3.

    max_depth=3 outperforms depth 5 and 7 on the internal test set while
    having a smaller train/test gap.

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
        max_depth=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------
# Preprocessing helpers
# --------------------------------------------------------

def clip_to_train(X_in, train_min, train_max):
    """Clip each feature to [train_min, train_max]. NaN stays NaN."""
    return X_in.clip(lower=train_min, upper=train_max, axis=1)


def round_int_features(X_in):
    """Round integer-valued features to the nearest integer after imputation."""
    X_out = X_in.copy()
    X_out[INT_FEATURES] = X_out[INT_FEATURES].round()
    return X_out


def build_imputers(X_train):
    """
    Fit all imputers on training data only (no target leakage).

    KNN uses StandardScaler so that distance is not dominated by large-range
    features (e.g. animal_age_months vs organic).

    Parameters
    ----------
    X_train : pd.DataFrame

    Returns
    -------
    dict
        Keys: 'train_medians', 'scaler', 'knn', 'iterative'.
    """
    train_medians = X_train.median()
    scaler = StandardScaler().fit(X_train)
    knn = KNNImputer(n_neighbors=5).fit(scaler.transform(X_train))
    iterative = IterativeImputer(max_iter=10, random_state=RANDOM_STATE).fit(X_train)
    return {
        "train_medians": train_medians,
        "scaler": scaler,
        "knn": knn,
        "iterative": iterative,
    }


def impute(X_in, method, imputers):
    """
    Return a filled copy of X_in using the specified method.

    Parameters
    ----------
    X_in : pd.DataFrame
    method : str
        One of 'Native NaN', 'Global median', 'KNN', 'Iterative'.
    imputers : dict
        Fitted objects from build_imputers().

    Returns
    -------
    pd.DataFrame
    """
    if method == "Native NaN":
        return X_in.copy()
    if method == "Global median":
        return X_in.fillna(imputers["train_medians"])
    if method == "KNN":
        scaler = imputers["scaler"]
        knn = imputers["knn"]
        # NaN propagates through scaling; KNNImputer fills it in scaled space
        filled = scaler.inverse_transform(knn.transform(scaler.transform(X_in)))
        return round_int_features(pd.DataFrame(filled, columns=FEATURES, index=X_in.index))
    if method == "Iterative":
        filled = imputers["iterative"].transform(X_in)
        return round_int_features(pd.DataFrame(filled, columns=FEATURES, index=X_in.index))
    raise ValueError(f"Unknown imputation method: {method!r}")


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
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted meat_type")
    ax.set_ylabel("True meat_type")
    ax.set_title(f"Confusion Matrix – {label}")
    plt.tight_layout()
    _save_or_show(fig, f"confusion_{label.replace(' ', '_').lower()}", save_dir)


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, train, compare 8 preprocessing variants.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("XGBoost Classification – Imputation & Clipping (NB2)")
    print("=" * 60)

    print("\n[1/5] Loading data ...")
    X_train, X_test, y_train, y_test, X_stress, y_stress = prepare_data()
    print(f"Train: {len(X_train)} rows  Test: {len(X_test)} rows  Stress: {len(X_stress)} rows")

    print("\n[2/5] Training baseline model (max_depth=3) ...")
    model = train_model(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    print(f"Train accuracy: {acc_train:.2%}")
    print(f"Test accuracy:  {acc_test:.2%}")
    print(f"Gap:            {acc_train - acc_test:.2%}")

    print("\n[3/5] Fitting imputers and preparing stress variants ...")
    imputers = build_imputers(X_train)

    train_min, train_max = X_train.min(), X_train.max()
    X_stress_clipped = clip_to_train(X_stress, train_min, train_max)
    changed = (X_stress.fillna(-999) != X_stress_clipped.fillna(-999)).sum()
    print(f"Total values changed by clipping: {changed.sum()}")
    print(changed[changed > 0])

    print("\n[4/5] Evaluating all 8 variants (4 methods × clip/no-clip) ...")
    rows = []
    predictions = {}

    for clip in [False, True]:
        X_base = X_stress_clipped if clip else X_stress
        for method in IMPUTATION_METHODS:
            X_ready = impute(X_base, method, imputers)
            y_pred = model.predict(X_ready)
            name = f"{method}{' + clip' if clip else ''}"
            predictions[name] = y_pred
            rows.append({
                "Variant": name,
                "Clipping": "yes" if clip else "no",
                "Correct / 91": int((y_pred == y_stress.values).sum()),
                "Accuracy": accuracy_score(y_stress, y_pred),
                "Macro F1": f1_score(y_stress, y_pred, average="macro"),
            })

    results = (
        pd.DataFrame(rows)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    display_r = results.copy()
    display_r["Accuracy"] = (display_r["Accuracy"] * 100).round(2).astype(str) + " %"
    display_r["Macro F1"] = display_r["Macro F1"].round(3)
    print(display_r.to_string(index=False))

    # Leakage demonstration – uses the true meat_type label to guide imputation
    print("\n--- Leakage demonstration (INVALID – reference only) ---")
    group_medians = (
        X_train.assign(meat_type=y_train + 1).groupby("meat_type").median()
    )

    def leaky_fill(X_in, true_labels):
        """Fill NaN with training median of the row's TRUE meat_type. Uses the target."""
        X_out = X_in.copy()
        for mt in group_medians.index:
            mask = (true_labels + 1) == mt
            X_out.loc[mask] = X_out.loc[mask].fillna(group_medians.loc[mt])
        return X_out

    y_pred_leaky = model.predict(leaky_fill(X_stress, y_stress))
    acc_leaky = accuracy_score(y_stress, y_pred_leaky)
    best_valid = results.iloc[0]

    print(f"Best valid variant:  {best_valid['Variant']:<22} "
          f"{best_valid['Accuracy']:.2%} ({best_valid['Correct / 91']}/91)")
    print(f"Leaky group median:  {'(INVALID)':<22} {acc_leaky:.2%} "
          f"({int((y_pred_leaky == y_stress.values).sum())}/91)")
    print(f"Inflation from leakage: {(acc_leaky - best_valid['Accuracy']) * 100:+.2f} pp")

    # Detailed analysis of the best valid variant
    best_name = best_valid["Variant"]
    y_pred_best = predictions[best_name]
    print(f"\nBest variant: {best_name}")
    print(classification_report(
        y_stress, y_pred_best, labels=range(5),
        target_names=CLASS_NAMES, digits=2, zero_division=0,
    ))

    if not no_plots:
        plot_confusion_matrix(
            y_stress, y_pred_best, f"Stress Test ({best_name})", "Oranges", save_plots
        )

    # Recall per class across all variants
    recall_table = pd.DataFrame({
        name: {
            f"Recall class {c}": classification_report(
                y_stress, pred, labels=range(5), target_names=CLASS_NAMES,
                output_dict=True, zero_division=0,
            )[c]["recall"]
            for c in CLASS_NAMES
        }
        for name, pred in predictions.items()
    }).T.round(2)
    print("\nRecall per class across all variants:")
    print(recall_table.to_string())

    print("\n[5/5] Comparison with previous models ...")
    comparison = pd.DataFrame({
        "Model": [
            "DT baseline",
            "RF baseline (KNN)",
            "RF domain imputation (leaky)",
            "RF + target encoding",
            "XGB NB1 – native NaN (depth 7)",
            f"XGB NB2 best – {best_name}",
        ],
        "Acc Test": [
            "53.0 %", "56.0 %", "56.0 %", "62.5 %",
            "65.4 %", f"{acc_test * 100:.1f} %",
        ],
        "Acc Stress": [
            "27 %", "24 %", "27.5 %", "25 %",
            "22.0 %", f"{best_valid['Accuracy'] * 100:.1f} %",
        ],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Classifier – imputation and clipping comparison on stress test."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
