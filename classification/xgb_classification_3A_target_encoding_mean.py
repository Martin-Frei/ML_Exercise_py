"""
XGBoost Classification – Mean Target Encoding (NB3a)
=====================================================

Purpose
-------
Add target-encoded features using 5-fold out-of-fold (OOF) mean encoding and
test whether they improve XGBoost accuracy over the NB2 baseline. Direct port
of `xgb_classification_3A_target_encoding_mean.ipynb`.

Step-by-step explanation
------------------------
1. Load data, shift labels 1–5 → 0–4, stratified 80/20 split.
2. Bin price_eur_per_kg into 5 quantile bins; bin edges from X_train only.
   Outer edges are open (−∞ / +∞) so out-of-range prices land in the
   nearest bin instead of producing a NaN.
3. Mean encoding with smoothing (m=10):
   - For each group value, replace it with the smoothed mean meat_type
     (original labels 1–5).
   - Rare groups are pulled toward the global mean via the smoothing term.
4. Out-of-fold encoding for training rows:
   - 5-fold StratifiedKFold; each row is encoded from the other 4 folds.
   - Prevents the row's own label from inflating its encoding.
   - Test and stress rows use the full-training-set map.
5. Build 12-feature sets: 8 original + 4 encoded (_te columns).
6. Train two identical models (max_depth=3):
   - Baseline: 8 original features
   - TE model: 12 features
7. Evaluate both on internal test: accuracy, gap, per-class F1.
8. Feature importance: original vs encoded contribution.
9. Stress test: baseline and TE model (native NaN for missing features).
10. Price-bin crosstab: visualises the price shift that hurts TE on stress.
11. Comparison table.

Variables glossary
------------------
- TARGET : str
    Column to predict ('meat_type').
- FEATURES : list of str
    Eight raw input columns.
- TE_FEATURES : list of str
    Columns to target-encode ('price_bin' is a derived helper column).
- N_BINS : int
    Number of quantile price bins (5).
- SMOOTHING : int
    Smoothing strength; pulls rare groups toward global mean (10).

Runtime
-------
Medium. Full run takes under 20 seconds.

Usage
-----
    python -m classification.xgb_classification_3A_target_encoding_mean
    python -m classification.xgb_classification_3A_target_encoding_mean --no-plots
    python -m classification.xgb_classification_3A_target_encoding_mean --save-plots DIR
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

# 'price_bin' is a derived helper; it is encoded and then dropped
TE_FEATURES = ["cut_quality", "organic", "marbling_score", "price_bin"]

CLASS_NAMES = ["1", "2", "3", "4", "5"]


# --------------------------------------------------------
# Price binning
# --------------------------------------------------------

def make_bin_edges(X_train):
    """
    Compute 5-quantile bin edges from training prices.

    Outer edges are open (−∞, +∞) so stress prices outside the training
    range still land in the nearest bin.

    Parameters
    ----------
    X_train : pd.DataFrame

    Returns
    -------
    np.ndarray
        Length-6 array of bin boundaries.
    """
    inner = X_train["price_eur_per_kg"].quantile(
        np.linspace(0, 1, N_BINS + 1)[1:-1]
    ).values
    return np.concatenate([[-np.inf], inner, [np.inf]])


def price_to_bin(price_series, bin_edges):
    """Map a price series to bin indices (0–4). NaN stays NaN."""
    return pd.cut(price_series, bins=bin_edges, labels=False)


# --------------------------------------------------------
# Target encoding helpers
# --------------------------------------------------------

def fit_encoding(values, target, m=SMOOTHING):
    """
    Smoothed mean encoding: mean(target) per group value.

    Rare groups (few rows) are pulled toward the global mean with
    strength m to prevent overfit on small groups.

    Parameters
    ----------
    values : pd.Series  (categorical or binned feature)
    target : pd.Series  (original meat_type labels 1–5)
    m : int             (smoothing strength)

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
    Map values to their smoothed encoding.

    NaN values stay NaN; unseen values (not in encoding) get the prior.

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


def oof_encode(col, target, skf, y_train):
    """
    Out-of-fold encoding for training rows.

    Each training row is encoded using the other 4 folds only, so its own
    label does not leak into its encoded value.

    Parameters
    ----------
    col : pd.Series    (feature column to encode, training rows)
    target : pd.Series (original labels 1–5 for training rows)
    skf : StratifiedKFold
    y_train : pd.Series (shifted labels 0–4, used for stratification)

    Returns
    -------
    pd.Series
    """
    out = pd.Series(np.nan, index=col.index, dtype=float)
    for fit_idx, enc_idx in skf.split(col, y_train):
        enc, prior = fit_encoding(col.iloc[fit_idx], target.iloc[fit_idx])
        out.iloc[enc_idx] = apply_encoding(col.iloc[enc_idx], enc, prior).values
    return out


def build_te_features(X_tr, X_other_list, y_train, bin_edges, skf):
    """
    Build OOF-encoded training set and full-map encoded test/stress sets.

    Adds 4 `_te` columns (one per TE_FEATURES entry). The intermediate
    `price_bin` column is dropped before returning.

    Parameters
    ----------
    X_tr : pd.DataFrame
    X_other_list : list of pd.DataFrame  (test, stress, ...)
    y_train : pd.Series  (shifted 0–4)
    bin_edges : np.ndarray
    skf : StratifiedKFold

    Returns
    -------
    tuple
        (X_tr_encoded, [X_other_encoded, ...])
    """
    target_original = y_train + 1  # 1-5 for mean encoding maps

    def add_price_bin(X_in):
        X_out = X_in.copy()
        X_out["price_bin"] = price_to_bin(X_out["price_eur_per_kg"], bin_edges)
        return X_out

    X_tr = add_price_bin(X_tr)
    others = [add_price_bin(X_o) for X_o in X_other_list]

    for feat in TE_FEATURES:
        new_col = f"{feat}_te"
        X_tr[new_col] = oof_encode(X_tr[feat], target_original, skf, y_train)
        enc, prior = fit_encoding(X_tr[feat], target_original)
        for X_o in others:
            X_o[new_col] = apply_encoding(X_o[feat], enc, prior)

    X_tr = X_tr.drop(columns="price_bin")
    others = [X_o.drop(columns="price_bin") for X_o in others]
    return X_tr, others


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def make_model():
    """XGBClassifier with max_depth=3 (best depth from NB1 analysis)."""
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


def plot_feature_importance(model_te, feature_names, save_dir=None):
    """
    Feature importance for the TE model, split by original vs encoded.

    Parameters
    ----------
    model_te : XGBClassifier
    feature_names : list of str
    save_dir : str or None
    """
    importance = pd.Series(
        model_te.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    encoded_share = importance[importance.index.str.endswith("_te")].sum()

    print("\nTop feature importances:")
    print((importance * 100).round(1).astype(str) + " %")
    print(f"\nOriginal features total: {(1 - encoded_share) * 100:.1f} %")
    print(f"Encoded features total:  {encoded_share * 100:.1f} %")

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.sort_values().plot(kind="barh", color="steelblue", ax=ax)
    ax.set_xlabel("Importance (gain, normalised)")
    ax.set_title("Feature Importance – Mean Encoding (NB3a)")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance_mean_te", save_dir)


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None):
    """
    Run the full pipeline: prepare, encode (OOF), train, evaluate, compare.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None → show interactively.
    """
    print("=" * 60)
    print("XGBoost Classification – Mean Target Encoding (NB3a)")
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

    print("\n[2/5] Building price bins and OOF mean encoding ...")
    bin_edges = make_bin_edges(X_train)
    inner_edges = bin_edges[1:-1]
    print(f"Inner bin edges (EUR): {np.round(inner_edges, 2)}")
    print("Training rows per price bin:")
    print(price_to_bin(X_train["price_eur_per_kg"], bin_edges).value_counts().sort_index())

    # Show the encoding maps (full-training version, for reference)
    target_original = y_train + 1
    price_bins_tr = price_to_bin(X_train["price_eur_per_kg"], bin_edges)
    enc_price, prior_price = fit_encoding(price_bins_tr, target_original)
    print(f"\nGlobal mean meat_type: {prior_price:.2f}")
    print("Mean meat_type per price bin:")
    print(enc_price.round(2))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train_te, (X_test_te, X_stress_te) = build_te_features(
        X_train, [X_test, X_stress], y_train, bin_edges, skf
    )
    print(f"\nFeature count: {X_train_te.shape[1]}  Columns: {list(X_train_te.columns)}")

    print("\n[3/5] Training baseline and TE models ...")
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
        "Model": ["Baseline", "Target encoding"],
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

    print("\nClassification report – TE model (internal test):")
    print(classification_report(y_test, pred_te_te, target_names=CLASS_NAMES, digits=2))

    if not no_plots:
        plot_confusion_matrix(y_test, pred_te_te, "Internal Test (Mean TE)", "Blues", save_plots)
        plot_feature_importance(model_te, list(X_train_te.columns), save_dir=save_plots)

    print("\n[5/5] Stress test ...")
    pred_st_base = model_base.predict(X_stress)
    pred_st_te = model_te.predict(X_stress_te)

    acc_st_base = accuracy_score(y_stress, pred_st_base)
    acc_st_te = accuracy_score(y_stress, pred_st_te)

    stress_summary = pd.DataFrame({
        "Model": ["Baseline", "Target encoding"],
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

    # Recall per class for both stress variants
    def recalls(pred):
        rep = classification_report(
            y_stress, pred, labels=range(5), target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        return [rep[c]["recall"] for c in CLASS_NAMES]

    recall_table = pd.DataFrame(
        [recalls(pred_st_base), recalls(pred_st_te)],
        index=["Baseline", "Target encoding"],
        columns=[f"Recall class {c}" for c in CLASS_NAMES],
    ).round(2)
    print("\nRecall per class (stress test):")
    print(recall_table.to_string())

    if not no_plots:
        plot_confusion_matrix(
            y_stress, pred_st_te, "Stress Test (Mean TE)", "Oranges", save_plots
        )

    # Price-bin crosstab reveals the price shift that breaks TE on stress
    bins_train = price_to_bin(X_train["price_eur_per_kg"], bin_edges)
    bins_stress = price_to_bin(X_stress["price_eur_per_kg"], bin_edges)
    print("\nTraining: share of each price bin per meat_type")
    print(pd.crosstab(y_train + 1, bins_train, normalize="index").round(2))
    print("\nStress: share of each price bin per meat_type (NaN prices excluded)")
    print(pd.crosstab(y_stress + 1, bins_stress, normalize="index").round(2))

    print("\nComparison with previous models:")
    comparison = pd.DataFrame({
        "Model": [
            "DT baseline",
            "RF baseline (KNN)",
            "RF + target encoding",
            "XGB NB1 depth 3 – native NaN",
            "XGB NB2 best – Iterative",
            "XGB NB3a – mean encoding",
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
        description="XGBoost Classifier with 5-fold OOF mean target encoding."
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots)
