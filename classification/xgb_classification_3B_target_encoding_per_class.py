"""
XGBoost Classification – Per-Class Target Encoding (NB3b)
==========================================================

Purpose
-------
Test a multi-class variant of target encoding. Instead of one mean class value
per group (NB3a), each group gets one share per class (5 new columns per
encoded feature). Direct port of
`xgb_classification_3B_target_encoding_per_class.ipynb`.

Why per-class encoding?
-----------------------
Mean encoding squeezes class information into one number, so "price bin 2 →
mean 3.4" could mean "mostly class 3-4" or "half class 1, half class 5".
Per-class encoding keeps the full distribution:

    price bin 2 → c1: 0.02 | c2: 0.10 | c3: 0.41 | c4: 0.38 | c5: 0.09

This could help weak classes (especially class 3) that sit between neighbours.

Step-by-step explanation
------------------------
1. Load data, shift labels 1–5 → 0–4, stratified 80/20 split.
2. Bin price into 5 quantile groups (bin edges from X_train only).
3. Per-class OOF smoothed encoding:
   - For each encoded feature and each class k, compute the smoothed share
     of class k among rows with that feature value.
   - Uses 5-fold StratifiedKFold OOF to avoid label leakage.
   - Produces 5 new columns per feature: {feat}_te_c1 … {feat}_te_c5.
4. Build feature sets: 8 original + 20 encoded = 28 features.
5. Train baseline (8 features) and TE model (28 features), both max_depth=3.
6. Evaluate on internal test; compare per-class F1 with NB3a.
7. Feature importance grouped by original feature (sum of 5 class columns).
8. Stress test: baseline and TE model.
9. Comparison with NB3a and all previous models.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- FEATURES : list of str
    Eight raw input columns.
- TE_FEATURES : list of str
    Columns to target-encode ('price_bin' is derived).
- N_BINS : int
    Number of quantile price bins (5).
- N_CLASSES : int
    Number of meat_type classes (5).
- SMOOTHING : int
    Smoothing strength; pulls rare groups toward global class share (10).

Runtime
-------
Medium. Full run takes under 25 seconds.

Usage
-----
    python -m classification.xgb_classification_3B_target_encoding_per_class
    python -m classification.xgb_classification_3B_target_encoding_per_class --no-plots
    python -m classification.xgb_classification_3B_target_encoding_per_class --save-plots DIR
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from shared.data_loading import load_stress_data, load_training_data


# --------------------------------------------------------
# Constants
# --------------------------------------------------------

TARGET = "meat_type"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BINS = 5
N_CLASSES = 5
SMOOTHING = 10

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

TE_FEATURES = ["cut_quality", "organic", "marbling_score", "price_bin"]

CLASS_NAMES = ["1", "2", "3", "4", "5"]


# --------------------------------------------------------
# Price binning
# --------------------------------------------------------

def make_bin_edges(X_train):
    """
    Compute 5-quantile bin edges from training prices.

    Open outer edges (−∞, +∞) ensure out-of-range stress prices land in
    the nearest bin rather than producing NaN.

    Parameters
    ----------
    X_train : pd.DataFrame

    Returns
    -------
    np.ndarray  (length N_BINS + 1)
    """
    inner = X_train["price_eur_per_kg"].quantile(
        np.linspace(0, 1, N_BINS + 1)[1:-1]
    ).values
    return np.concatenate([[-np.inf], inner, [np.inf]])


def price_to_bin(price_series, bin_edges):
    """Map a price series to bin indices (0–4). NaN stays NaN."""
    return pd.cut(price_series, bins=bin_edges, labels=False)


# --------------------------------------------------------
# Per-class encoding helpers
# --------------------------------------------------------

def fit_encoding(values, target, m=SMOOTHING):
    """
    Smoothed mean of a 0/1 target per unique feature value.

    Used with target = (y == k).astype(float) to encode class-k share.
    Rare values are pulled toward the global share via the smoothing term.

    Parameters
    ----------
    values : pd.Series
    target : pd.Series  (0.0 or 1.0)
    m : int

    Returns
    -------
    tuple
        (encoding: pd.Series, prior: float)
    """
    prior = target.mean()
    stats = target.groupby(values).agg(["mean", "count"])
    encoding = (stats["count"] * stats["mean"] + m * prior) / (stats["count"] + m)
    return encoding, prior


def apply_encoding(values, encoding, prior):
    """
    Map values to their encoding. Unseen values → prior. NaN stays NaN.

    Parameters
    ----------
    values : pd.Series
    encoding : pd.Series
    prior : float

    Returns
    -------
    pd.Series (float)
    """
    out = values.map(encoding)
    out[values.notna() & out.isna()] = prior
    return out.astype(float)


def oof_encode(col, target_k, skf, y_train):
    """
    Out-of-fold encoding for a single binary-target column (class k).

    Parameters
    ----------
    col : pd.Series      (feature to encode, training rows)
    target_k : pd.Series (1 if row belongs to class k, else 0)
    skf : StratifiedKFold
    y_train : pd.Series  (shifted labels 0–4, used for stratification)

    Returns
    -------
    pd.Series
    """
    out = pd.Series(np.nan, index=col.index, dtype=float)
    for fit_idx, enc_idx in skf.split(col, y_train):
        enc, prior = fit_encoding(col.iloc[fit_idx], target_k.iloc[fit_idx])
        out.iloc[enc_idx] = apply_encoding(col.iloc[enc_idx], enc, prior).values
    return out


def build_te_features(X_tr, X_other_list, y_train, bin_edges, skf):
    """
    Build per-class OOF-encoded training set and full-map test/stress sets.

    For each encoded feature and each of the 5 classes, adds one
    `{feat}_te_c{k+1}` column. The intermediate `price_bin` column is
    dropped before returning.

    Result: 8 original + 4 × 5 = 28 features.

    Parameters
    ----------
    X_tr : pd.DataFrame
    X_other_list : list of pd.DataFrame
    y_train : pd.Series  (shifted 0–4)
    bin_edges : np.ndarray
    skf : StratifiedKFold

    Returns
    -------
    tuple
        (X_tr_encoded, [X_other_encoded, ...])
    """
    def add_price_bin(X_in):
        X_out = X_in.copy()
        X_out["price_bin"] = price_to_bin(X_out["price_eur_per_kg"], bin_edges)
        return X_out

    X_tr = add_price_bin(X_tr)
    others = [add_price_bin(X_o) for X_o in X_other_list]

    for feat in TE_FEATURES:
        for k in range(N_CLASSES):
            target_k = (y_train == k).astype(float)  # 1 if row is class k
            new_col = f"{feat}_te_c{k + 1}"
            # OOF for training rows; full-training map for test/stress
            X_tr[new_col] = oof_encode(X_tr[feat], target_k, skf, y_train)
            enc, prior = fit_encoding(X_tr[feat], target_k)
            for X_o in others:
                X_o[new_col] = apply_encoding(X_o[feat], enc, prior)

    X_tr = X_tr.drop(columns="price_bin")
    others = [X_o.drop(columns="price_bin") for X_o in others]
    return X_tr, others


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def make_model():
    """XGBClassifier with max_depth=3."""
    return XGBClassifier(
        objective="multi:softprob",
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )


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
    """Heatmap confusion matrix with original class labels."""
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted meat_type")
    ax.set_ylabel("True meat_type")
    ax.set_title(f"Confusion Matrix – {label}")
    plt.tight_layout()
    _save_or_show(fig, f"confusion_{label.replace(' ', '_').lower()}", save_dir)


def plot_grouped_importance(model_te, feature_names, save_dir=None):
    """
    Importance grouped by original feature (sum across the 5 class columns).

    Parameters
    ----------
    model_te : XGBClassifier
    feature_names : list of str
    save_dir : str or None
    """
    importance = pd.Series(model_te.feature_importances_, index=feature_names)

    # Remove class suffix (e.g. 'price_bin_te_c3' → 'price_bin_te')
    group = importance.index.str.replace(r"_c\d$", "", regex=True)
    grouped = importance.groupby(group).sum().sort_values(ascending=False)

    encoded_share = importance[importance.index.str.contains("_te_c")].sum()
    print("\nImportance grouped per feature:")
    print((grouped * 100).round(1).astype(str) + " %")
    print(f"\nOriginal features total: {(1 - encoded_share) * 100:.1f} %")
    print(f"Encoded features total:  {encoded_share * 100:.1f} %")

    fig, ax = plt.subplots(figsize=(7, 5))
    grouped.sort_values().plot(kind="barh", color="steelblue", ax=ax)
    ax.set_xlabel("Importance (gain, normalised, summed per feature)")
    ax.set_title("Feature Importance – Per-Class Encoding (NB3b)")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance_per_class", save_dir)


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, per-class encode, train, evaluate, compare.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("XGBoost Classification – Per-Class Target Encoding (NB3b)")
    print("=" * 60)

    print("\n[1/5] Loading data ...")
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

    print(f"Train: {len(X_train)} rows  Test: {len(X_test)} rows  Stress: {len(X_stress)} rows")

    print("\n[2/5] Building price bins and OOF per-class encoding ...")
    bin_edges = make_bin_edges(X_train)
    inner_edges = bin_edges[1:-1]
    print(f"Inner bin edges (EUR): {np.round(inner_edges, 2)}")

    # Show class shares per price bin as a sanity check
    bins_tr = price_to_bin(X_train["price_eur_per_kg"], bin_edges)
    example = pd.DataFrame({
        f"class {k + 1}": fit_encoding(bins_tr, (y_train == k).astype(float))[0]
        for k in range(N_CLASSES)
    })
    example.index.name = "price_bin"
    print("\nShare of each class per price bin (training):")
    print(example.round(2))
    print("\nRow sums (should be ~1):", example.sum(axis=1).round(2).tolist())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train_te, (X_test_te, X_stress_te) = build_te_features(
        X_train, [X_test, X_stress], y_train, bin_edges, skf
    )
    n_encoded = X_train_te.shape[1] - len(FEATURES)
    print(f"\nFeature count: {X_train_te.shape[1]}  "
          f"(8 original + {len(TE_FEATURES)} features × {N_CLASSES} classes = "
          f"{n_encoded} encoded)")

    print("\n[3/5] Training baseline and per-class TE models ...")
    model_base = make_model().fit(X_train, y_train)
    model_te = make_model().fit(X_train_te, y_train)
    print("Both models trained.")

    print("\n[4/5] Evaluating on internal test set ...")
    acc_tr_base = accuracy_score(y_train, model_base.predict(X_train))
    pred_te_base = model_base.predict(X_test)
    acc_te_base = accuracy_score(y_test, pred_te_base)

    acc_tr_te = accuracy_score(y_train, model_te.predict(X_train_te))
    pred_te_te = model_te.predict(X_test_te)
    acc_te_te = accuracy_score(y_test, pred_te_te)

    summary = pd.DataFrame({
        "Model": ["Baseline", "Per-class encoding"],
        "Train acc": [f"{acc_tr_base * 100:.2f} %", f"{acc_tr_te * 100:.2f} %"],
        "Test acc": [f"{acc_te_base * 100:.2f} %", f"{acc_te_te * 100:.2f} %"],
        "Gap": [f"{(acc_tr_base - acc_te_base) * 100:.2f} %",
                f"{(acc_tr_te - acc_te_te) * 100:.2f} %"],
        "Correct / 240": [
            int((pred_te_base == y_test.values).sum()),
            int((pred_te_te == y_test.values).sum()),
        ],
    })
    print(summary.to_string(index=False))

    f1_table = pd.DataFrame({
        "Baseline F1": f1_score(y_test, pred_te_base, average=None),
        "TE F1": f1_score(y_test, pred_te_te, average=None),
    }, index=[f"Class {c}" for c in CLASS_NAMES])
    f1_table["Change"] = f1_table["TE F1"] - f1_table["Baseline F1"]
    print("\nF1 per class:")
    print(f1_table.round(2))

    print("\nClassification report – per-class TE model (internal test):")
    print(classification_report(y_test, pred_te_te, target_names=CLASS_NAMES, digits=2))

    if not no_plots:
        plot_confusion_matrix(
            y_test, pred_te_te, "Internal Test (Per-Class TE)", "Blues", save_plots
        )
        plot_grouped_importance(model_te, list(X_train_te.columns), save_dir=save_plots)

    print("\n[5/5] Stress test ...")
    pred_st_base = model_base.predict(X_stress)
    pred_st_te = model_te.predict(X_stress_te)

    acc_st_base = accuracy_score(y_stress, pred_st_base)
    acc_st_te = accuracy_score(y_stress, pred_st_te)

    stress_summary = pd.DataFrame({
        "Model": ["Baseline", "Per-class encoding"],
        "Correct / 91": [
            int((pred_st_base == y_stress.values).sum()),
            int((pred_st_te == y_stress.values).sum()),
        ],
        "Accuracy": [f"{acc_st_base * 100:.2f} %", f"{acc_st_te * 100:.2f} %"],
        "Macro F1": [
            round(f1_score(y_stress, pred_st_base, average="macro"), 3),
            round(f1_score(y_stress, pred_st_te, average="macro"), 3),
        ],
    })
    print(stress_summary.to_string(index=False))

    print("\nClassification report – per-class TE model (stress test):")
    print(classification_report(
        y_stress, pred_st_te, labels=range(5),
        target_names=CLASS_NAMES, digits=2, zero_division=0,
    ))

    if not no_plots:
        plot_confusion_matrix(
            y_stress, pred_st_te, "Stress Test (Per-Class TE)", "Oranges", save_plots
        )

    print("\nComparison with previous models:")
    comparison = pd.DataFrame({
        "Model": [
            "DT baseline",
            "RF baseline (KNN)",
            "RF + target encoding",
            "XGB NB1 depth 3 – native NaN",
            "XGB NB2 best – Iterative",
            "XGB NB3b – per-class encoding",
        ],
        "Acc Test": [
            "53.0 %", "56.0 %", "62.5 %", "67.1 %", "67.1 %",
            f"{acc_te_te * 100:.1f} %",
        ],
        "Acc Stress": [
            "27 %", "24 %", "25 %", "22.0 %", "23.1 %",
            f"{acc_st_te * 100:.1f} %",
        ],
    })
    print(comparison.to_string(index=False))
    print("\n--- Done! ---")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Classifier with 5-fold OOF per-class target encoding."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
