"""
XGBoost Regression – Hyperparameter Tuning (NB6)
=================================================

Purpose
-------
Systematically search for the best XGBRegressor hyperparameters using
combined-dataset training (clean + stress rows). Picks up from NB3 (#17),
which achieved the best stress R² so far (+0.197), and tries to push
further with RandomizedSearchCV followed by a fine search.
Direct port of `xgb_hyperparameter_tuning.ipynb`.

Step-by-step explanation
------------------------
1. Load training data (1 200 rows) and stress data (100 rows).
2. Preprocess stress data:
   a. Group-wise imputation using TRAINING-set subgroup medians
      (meat_type groups). More principled than using stress-data's own
      group medians because it never looks at the stress distribution.
   b. Fallback to global training median for rows with NaN meat_type.
   c. Clip ALL numeric features to the [1st, 99th] percentile range of
      the training data (more conservative than clipping to max).
   d. Add protein_fat_ratio = protein_pct / fat_content_pct.clip(0.1).
      Note: protein_fat_ratio IS added to df_combined but the model uses
      only the 8 base features (feature_cols computed before engineering).
3. Add is_stress flag; concatenate -> 1 291-row combined dataset.
4. Stratified 80/20 split on is_stress.
5. Train NB5 baseline (n_estimators=150, lr=0.08, max_depth=5).
6. RandomizedSearchCV (n_iter=100, cv=3, 11-parameter search space).
7. Fine RandomizedSearchCV (n_iter=50, cv=3) with narrowed ranges
   centred on the best parameters from step 6.
8. Compare all three models on the mixed test set and the full stress set
   (91 rows with valid prices).
9. Feature importance plot and actual-vs-predicted scatter.

Variables glossary
------------------
- TARGET : str
    Column to predict ('price_eur_per_kg').
- FEATURES : list of str
    Eight raw input columns. The actual model feature set.
    protein_fat_ratio is added to df_combined but intentionally excluded
    from FEATURES to replicate the notebook's behaviour.
- N_ITER_RANDOM : int
    Number of parameter settings sampled in step 6 (100).
- N_ITER_FINE : int
    Number of parameter settings sampled in step 7 (50).
- PARAM_DIST : dict
    Full hyperparameter search space for step 6.
- RANDOM_STATE : int
    Fixed seed for reproducibility (42).

Runtime
-------
Slow. RandomizedSearchCV (100 iter, 3-fold) + fine search (50 iter, 3-fold)
takes 8–25 minutes depending on hardware. Use --save-model to persist the
best model and avoid rerunning.

Usage
-----
    python -m regression.xgb_hyperparameter_tuning --no-plots
    python -m regression.xgb_hyperparameter_tuning
    python -m regression.xgb_hyperparameter_tuning --save-model
    python -m regression.xgb_hyperparameter_tuning --save-plots DIR --save-model
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from shared.data_loading import load_stress_data, load_training_data

warnings.filterwarnings("ignore")


# --------------------------------------------------------
# Constants
# --------------------------------------------------------

TARGET = "price_eur_per_kg"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER_RANDOM = 100  # RandomizedSearchCV iterations (broad search)
N_ITER_FINE = 50     # Fine search iterations (narrow search around best)

# Model feature set: 8 raw columns.
# protein_fat_ratio is added to df_combined but excluded here to faithfully
# replicate the notebook (feature_cols was computed before engineering).
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

# Full hyperparameter search space for RandomizedSearchCV
PARAM_DIST = {
    "n_estimators": [100, 150, 200, 250, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1.0, 2.0],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1.0],
    "objective": ["reg:squarederror"],
}


# --------------------------------------------------------
# Preprocessing helpers
# --------------------------------------------------------

def group_wise_impute_stress(df_stress, df_train):
    """
    Fill NaN in stress data using training-set subgroup medians.

    For each column with missing values, fill using the median of training rows
    that share the same meat_type value. Falls back to the global training
    median for rows where meat_type itself is NaN or the subgroup is too small.

    Using training-data statistics (not stress-data statistics) prevents any
    information from the test distribution from influencing imputation.

    Parameters
    ----------
    df_stress : pd.DataFrame
        Stress dataset to be imputed (copy is made internally).
    df_train : pd.DataFrame
        Training dataset — source of group and global medians.

    Returns
    -------
    pd.DataFrame
        Stress data with NaN feature values filled. Target column is
        intentionally left unchanged so valid_mask can be applied later.
    """
    out = df_stress.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        if col == TARGET or not out[col].isnull().any():
            continue
        for mt_val in out["meat_type"].dropna().unique():
            mask = (out["meat_type"] == mt_val) & out[col].isnull()
            grp_median = df_train.loc[df_train["meat_type"] == mt_val, col].median()
            if not np.isnan(grp_median):
                out.loc[mask, col] = grp_median
        # Fallback for remaining NaN (rows whose meat_type is NaN)
        out[col] = out[col].fillna(df_train[col].median())

    return out


def compute_clip_bounds(df_train):
    """
    Compute per-feature [1st, 99th]-percentile clip bounds from training data.

    Clips ALL numeric feature columns (not just the three outlier columns
    as in NB3). This is a more conservative strategy that prevents any
    single extreme value from dominating.

    Parameters
    ----------
    df_train : pd.DataFrame

    Returns
    -------
    dict
        {column_name: (lower_bound, upper_bound)}
    """
    bounds = {}
    for col in FEATURES:
        if df_train[col].dtype in ["float64", "int64"]:
            bounds[col] = (
                df_train[col].quantile(0.01),
                df_train[col].quantile(0.99),
            )
    return bounds


def clip_features(df, clip_bounds):
    """
    Clip feature columns to the precomputed training-data bounds.

    Parameters
    ----------
    df : pd.DataFrame
    clip_bounds : dict

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    for col, (lo, hi) in clip_bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lo, hi)
    return out


def add_protein_fat_ratio(df):
    """
    Add protein_fat_ratio = protein_pct / fat_content_pct.clip(lower=0.1).

    Clipping fat_content_pct to a minimum of 0.1 prevents extreme ratios
    from near-zero fat values. This column is added to df_combined for
    reference but is NOT included in the model's FEATURES list.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    out["protein_fat_ratio"] = out["protein_pct"] / out["fat_content_pct"].clip(lower=0.1)
    return out


# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

def prepare_data():
    """
    Load, preprocess, combine, and split training + stress data.

    Preprocessing pipeline:
    1. Impute stress features using training-data group medians.
    2. Clip all numeric features to training 1st–99th percentile.
    3. Add protein_fat_ratio to both datasets (for reference, not model input).
    4. Mark is_stress flag; concatenate; drop rows with NaN target.
    5. Stratified 80/20 split on is_stress.

    Stress evaluation uses only the 91 rows with a genuinely known price
    (valid_mask captured before any imputation occurs).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, X_stress, y_stress,
         is_stress_test, df_train_raw)
    """
    df_train = load_training_data()
    df_stress_raw = load_stress_data()

    # Record which stress rows have a known price (before imputing anything)
    valid_mask = df_stress_raw[TARGET].notna()

    # Clip bounds from training data
    clip_bounds = compute_clip_bounds(df_train)

    # Preprocess stress
    df_stress_imp = group_wise_impute_stress(df_stress_raw, df_train)
    df_stress_proc = clip_features(df_stress_imp, clip_bounds)
    df_stress_proc = add_protein_fat_ratio(df_stress_proc)
    df_stress_proc["is_stress"] = 1

    # Augment training data
    df_train_aug = add_protein_fat_ratio(df_train.copy())
    df_train_aug["is_stress"] = 0

    # Combine and drop rows with NaN target
    df_combined = pd.concat([df_train_aug, df_stress_proc], ignore_index=True)
    df_combined = df_combined.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"Combined dataset: {len(df_combined)} rows "
          f"(clean: {(df_combined['is_stress'] == 0).sum()}, "
          f"stress: {(df_combined['is_stress'] == 1).sum()})")

    # Stratified split on is_stress
    X = df_combined[FEATURES]
    y = df_combined[TARGET]
    strat = df_combined["is_stress"]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, strat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )
    is_stress_test = s_test.reset_index(drop=True)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} "
          f"(clean: {(is_stress_test == 0).sum()}, stress: {(is_stress_test == 1).sum()})")
    print(f"Stress ratio in train: {s_train.mean():.3f}")

    # Stress evaluation set (91 rows with valid prices only)
    X_stress = df_stress_proc.loc[valid_mask, FEATURES]
    y_stress = df_stress_raw.loc[valid_mask, TARGET]  # original prices, not imputed

    return X_train, X_test, y_train, y_test, X_stress, y_stress, is_stress_test, df_train


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_baseline(X_train, y_train):
    """
    Train the NB5/combined-training baseline model.

    Uses the same parameters as the combined model from #17 (NB3 in this
    project's numbering): n_estimators=150, learning_rate=0.08, max_depth=5.

    Parameters
    ----------
    X_train : pd.DataFrame
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
    model.fit(X_train, y_train)
    return model


def run_random_search(X_train, y_train):
    """
    RandomizedSearchCV over PARAM_DIST (N_ITER_RANDOM iterations, 3-fold CV).

    Uses neg_root_mean_squared_error scoring. n_jobs=-1 uses all CPU cores.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    tuple
        (best_estimator: XGBRegressor, best_params: dict, best_cv_rmse: float)
    """
    xgb = XGBRegressor(
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER_RANDOM,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    print(f"Random search: {N_ITER_RANDOM} iterations x 3-fold CV "
          f"= {N_ITER_RANDOM * 3} fits ...")
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Done in {elapsed / 60:.1f} min")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, -search.best_score_


def run_fine_search(X_train, y_train, best_params):
    """
    Narrow RandomizedSearchCV centred on best_params from the random search.

    Numeric params (n_estimators, max_depth, learning_rate, min_child_weight)
    get a three-value range [best-delta, best, best+delta]. The regularisation
    and subsampling params keep their original ranges. Only params that appear
    in best_params are included in the fine grid.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    best_params : dict

    Returns
    -------
    tuple
        (best_estimator: XGBRegressor, best_params: dict, best_cv_rmse: float)
    """
    n_est = best_params.get("n_estimators", 150)
    depth = best_params.get("max_depth", 5)
    lr = best_params.get("learning_rate", 0.08)
    mcw = best_params.get("min_child_weight", 1)

    fine_param_grid = {
        "n_estimators":      [max(1, n_est - 50), n_est, n_est + 50],
        "max_depth":         [max(1, depth - 1), depth, depth + 1],
        "learning_rate":     [max(0.001, lr - 0.03), lr, lr + 0.03],
        "min_child_weight":  [max(1, mcw - 1), mcw, mcw + 1],
        "subsample":         [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
        "gamma":             [0, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha":         [0, 0.01, 0.1],
        "reg_lambda":        [1.0, 1.5, 2.0, 3.0],
    }
    # Keep only params that appear in best_params
    fine_grid = {k: v for k, v in fine_param_grid.items() if k in best_params}

    print(f"\nFine search around best params ({len(fine_grid)} dimensions, "
          f"{N_ITER_FINE} iterations x 3-fold CV = {N_ITER_FINE * 3} fits) ...")
    for k, v in fine_grid.items():
        print(f"  {k}: {v}")

    search = RandomizedSearchCV(
        estimator=XGBRegressor(
            random_state=RANDOM_STATE,
            eval_metric="rmse",
            verbosity=0,
        ),
        param_distributions=fine_grid,
        n_iter=N_ITER_FINE,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Done in {elapsed / 60:.1f} min")
    print(f"Fine-tuned best CV RMSE: {-search.best_score_:.4f}")
    print(f"Fine-tuned best params:  {search.best_params_}")
    return search.best_estimator_, search.best_params_, -search.best_score_


# --------------------------------------------------------
# Evaluation
# --------------------------------------------------------

def evaluate_model(model, X_test, y_test, X_stress, y_stress, name="Model"):
    """
    Evaluate a fitted model on the mixed test set and the stress set.

    Reports RMSE, MAE, R² for both evaluation sets.

    Parameters
    ----------
    model : fitted XGBRegressor
    X_test : pd.DataFrame (mixed clean + stress rows from the split)
    y_test : pd.Series
    X_stress : pd.DataFrame (91 stress rows with valid prices)
    y_stress : pd.Series
    name : str

    Returns
    -------
    dict with keys: name, rmse_test, mae_test, r2_test,
                    rmse_stress, mae_stress, r2_stress
    """
    y_pred_test = model.predict(X_test)
    y_pred_stress = model.predict(X_stress)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    rmse_stress = float(np.sqrt(mean_squared_error(y_stress, y_pred_stress)))
    mae_stress = mean_absolute_error(y_stress, y_pred_stress)
    r2_stress = r2_score(y_stress, y_pred_stress)

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")
    print(f"Mixed Test  -- RMSE: {rmse_test:.4f}  MAE: {mae_test:.4f}  R2: {r2_test:.4f}")
    print(f"Stress Test -- RMSE: {rmse_stress:.4f}  MAE: {mae_stress:.4f}  R2: {r2_stress:.4f}")

    return {
        "name": name,
        "rmse_test": rmse_test, "mae_test": mae_test, "r2_test": r2_test,
        "rmse_stress": rmse_stress, "mae_stress": mae_stress, "r2_stress": r2_stress,
    }


# --------------------------------------------------------
# Plot helpers
# --------------------------------------------------------

def _save_or_show(fig, name, save_dir):
    """Save figure to disk or display it, then close."""
    if save_dir is not None:
        path = Path(save_dir) / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_importance(model, save_dir=None):
    """
    Horizontal bar chart of feature importances for the best model.

    Parameters
    ----------
    model : XGBRegressor
    save_dir : str or None
    """
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nFeature importance (best model):")
    print(importance.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("Feature Importance -- Tuned XGBoost")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    _save_or_show(fig, "feature_importance_tuned", save_dir)


def plot_stress_predictions(y_stress, y_pred, r2, save_dir=None):
    """
    Actual vs predicted scatter and error distribution for the stress set.

    Parameters
    ----------
    y_stress : pd.Series
    y_pred : np.ndarray
    r2 : float
    save_dir : str or None
    """
    errors = np.array(y_stress) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lo = min(y_stress.min(), y_pred.min())
    hi = max(y_stress.max(), y_pred.max())
    axes[0].scatter(y_stress, y_pred, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect")
    axes[0].set_xlabel("Actual Price (EUR/kg)")
    axes[0].set_ylabel("Predicted Price (EUR/kg)")
    axes[0].set_title(f"Predicted vs Actual (Stress Test)\nR2 = {r2:.4f}")
    axes[0].legend()

    axes[1].hist(errors, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Error (EUR/kg)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Error Distribution\nMean: {errors.mean():.2f}  Std: {errors.std():.2f}"
    )

    plt.tight_layout()
    _save_or_show(fig, "stress_test_results", save_dir)


# --------------------------------------------------------
# Main workflow
# --------------------------------------------------------

def main(no_plots=False, save_plots=None, save_model=False):
    """
    Run the full hyperparameter-tuning pipeline.

    Parameters
    ----------
    no_plots : bool
        Suppress all plot windows.
    save_plots : str or None
        Directory to write PNG files; None shows interactively.
    save_model : bool
        Persist the fine-tuned model to disk.
    """
    print("=" * 60)
    print("XGBoost Regression -- Hyperparameter Tuning (NB6)")
    print("=" * 60)

    import xgboost as _xgb
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"xgboost: {_xgb.__version__}")

    # -- Data --
    print("\n[1/6] Preparing combined dataset ...")
    (X_train, X_test, y_train, y_test,
     X_stress, y_stress, is_stress_test,
     df_train_raw) = prepare_data()

    # -- Baseline --
    print("\n[2/6] Training NB5 baseline model ...")
    t0 = time.time()
    baseline = train_baseline(X_train, y_train)
    print(f"Done in {time.time() - t0:.1f} s")

    # -- Random search --
    print("\n[3/6] RandomizedSearchCV (broad search) ...")
    best_random, best_random_params, cv_rmse_random = run_random_search(X_train, y_train)

    # -- Fine search --
    print("\n[4/6] Fine RandomizedSearchCV (narrow search) ...")
    best_fine, best_fine_params, cv_rmse_fine = run_fine_search(
        X_train, y_train, best_random_params
    )

    # -- Evaluate all three --
    print("\n[5/6] Evaluating all models ...")
    res_baseline = evaluate_model(baseline, X_test, y_test, X_stress, y_stress,
                                  "NB5 Baseline")
    res_random = evaluate_model(best_random, X_test, y_test, X_stress, y_stress,
                                "Tuned XGBoost (random search)")
    res_fine = evaluate_model(best_fine, X_test, y_test, X_stress, y_stress,
                              "Fine-Tuned XGBoost")

    # Comparison table
    summary = pd.DataFrame([res_baseline, res_random, res_fine]).set_index("name")
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(summary.round(4).to_string())

    # Improvement vs baseline
    print("\nImprovement vs NB5 Baseline:")
    for row_name in ["Tuned XGBoost (random search)", "Fine-Tuned XGBoost"]:
        r2_test_imp = summary.loc[row_name, "r2_test"] - summary.loc["NB5 Baseline", "r2_test"]
        r2_stress_imp = (
            summary.loc[row_name, "r2_stress"] - summary.loc["NB5 Baseline", "r2_stress"]
        )
        mae_stress_imp = (
            summary.loc[row_name, "mae_stress"] - summary.loc["NB5 Baseline", "mae_stress"]
        )
        print(f"  {row_name}:")
        print(f"    Test R2 delta:     {r2_test_imp:+.4f}")
        print(f"    Stress R2 delta:   {r2_stress_imp:+.4f}")
        print(f"    Stress MAE delta:  {mae_stress_imp:+.4f} EUR")

    # -- Plots --
    print("\n[6/6] Plots ...")
    if not no_plots:
        plot_feature_importance(best_fine, save_dir=save_plots)
        y_pred_stress_fine = best_fine.predict(X_stress)
        r2_stress_fine = res_fine["r2_stress"]
        plot_stress_predictions(y_stress, y_pred_stress_fine, r2_stress_fine,
                                save_dir=save_plots)

    # -- Save model --
    if save_model:
        import pickle
        model_dir = Path(__file__).parent / "saved_models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "xgb_hyperparameter_tuning.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_fine, f)
        print(f"\nModel saved: {model_path}")
        # Also save as .joblib so the Streamlit page can load it
        from shared.modeling import save_model as _save_model_joblib
        _save_model_joblib(best_fine, "xgb_hyperparameter_tuning")
        print("Model also saved: models/xgb_hyperparameter_tuning.joblib")

    print("\n--- Done! ---")
    print("\nNext steps:")
    print("  1. Feature interaction engineering (meat_type x fat_content, age x storage)")
    print("  2. Ensemble stacking (XGBoost + RF + Linear)")
    print("  3. GroupKFold cross-validation by meat_type")


# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "XGBoost Regressor with RandomizedSearchCV hyperparameter tuning. "
            "WARNING: Full run takes 8-25 minutes."
        )
    )
    parser.add_argument("--no-plots", action="store_true", help="Suppress all plots.")
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Save plots as PNG to DIR.",
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Persist the fine-tuned model to regression/saved_models/.",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots, save_plots=args.save_plots, save_model=args.save_model)
