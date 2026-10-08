"""
Shared modeling utilities: metrics, imputation, and saving/loading.

Purpose
-------
This module centralizes the evaluation metrics and the imputation
strategy that appear in every regression notebook. It also provides
helpers to save a trained model to disk and load it back, which is
essential for the "slow" notebooks that must be pre-trained.

Step-by-step explanation
------------------------
The original notebooks did the following in every file:

1. Prepare the stress-test dataframe by imputing missing values
   using medians from the training set.
2. Compute MAE, RMSE, and R² on both internal test and stress test.
3. Print a comparison.

Here we factor out:

- `impute_with_train_medians()`: fill NaN in the stress set using
  the training set medians (never the stress set's own statistics,
  to avoid data leakage).
- `evaluate_regression()`: compute MAE, RMSE, R² and return them
  in a dict.
- `save_model()` / `load_model()`: persist and restore models.

Variables glossary
------------------
- MODEL_DIR : Path
    Folder where trained models are stored as .joblib files.

See each function's docstring for its specific parameters.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# Folder where .joblib files are stored (created if missing)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------

def impute_with_train_medians(
    df_train: pd.DataFrame,
    df_stress: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Fill missing values in the stress set using training-set medians.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (should not contain NaN in the feature columns).
    df_stress : pd.DataFrame
        Stress data, possibly with NaN in the feature columns.
    feature_cols : list of str
        Names of the columns to consider as features.

    Returns
    -------
    pd.DataFrame
        A copy of df_stress with all feature NaNs filled.
    """
    # Compute the median of every feature column on the training set
    train_medians = df_train[feature_cols].median()

    # Work on a copy and fill NaNs in one pass
    df_filled = df_stress.copy()
    df_filled[feature_cols] = df_filled[feature_cols].fillna(train_medians)

    return df_filled


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_regression(y_true, y_pred) -> dict:
    """
    Compute MAE, RMSE, and R² for a regression model.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth target values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ------------------------------------------------------------
# Model persistence
# ------------------------------------------------------------

def save_model(model, name: str) -> Path:
    """
    Save a model to disk as a .joblib file in the models/ folder.

    Parameters
    ----------
    model : sklearn-like estimator
        The trained model to save.
    name : str
        Base name (without extension) for the file.

    Returns
    -------
    Path
        Full path to the saved file.
    """
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """
    Load a previously saved model from the models/ folder.

    Parameters
    ----------
    name : str
        Base name (without extension) of the file to load.

    Returns
    -------
    model
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If no file with that name exists in models/.
    """
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Pre-train it first by running the corresponding script."
        )
    return joblib.load(path)


def model_exists(name: str) -> bool:
    """
    Check whether a pre-trained model file exists.

    Parameters
    ----------
    name : str
        Base name (without extension) of the file.

    Returns
    -------
    bool
        True if the file exists in models/, otherwise False.
    """
    return (MODEL_DIR / f"{name}.joblib").exists()


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Model folder:", MODEL_DIR)
    print("Example: save_model() would write to:", MODEL_DIR / "example.joblib")