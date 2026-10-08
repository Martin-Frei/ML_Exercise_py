"""
Data loading and feature engineering utilities.

Purpose
-------
This module provides functions to load the raw CSV files and to
engineer the three derived features that were created in the
original notebooks. It is imported by every training script and
by the Streamlit app, so the data preparation logic lives in
exactly one place.

Step-by-step explanation
------------------------
The original notebooks repeated this logic in every file:

1. Load `meat_price_dataset.csv` and `meat_price_100_stress_test.csv`.
2. Add three engineered features:
   - `price_per_protein` = price / protein percentage
   - `fat_to_protein_ratio` = fat / protein
   - `price_x_marbling` = price * marbling score
3. Return the enriched DataFrames.

Here that logic is wrapped into two small functions, so both the
training scripts and the app can call them without copy-paste.

Variables glossary
------------------
- DATA_DIR : Path
    Folder that contains the CSV files (relative to the project root).
- TRAIN_FILE : Path
    Path to the training CSV.
- STRESS_FILE : Path
    Path to the stress-test CSV.
"""

from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Paths (relative to the project root)
# ------------------------------------------------------------

# Folder containing the CSV files
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Full path to the training dataset
TRAIN_FILE = DATA_DIR / "meat_price_dataset.csv"

# Full path to the stress-test dataset
STRESS_FILE = DATA_DIR / "meat_price_100_stress_test.csv"


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the three engineered features used in the original notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with at least the columns:
        'price_eur_per_kg', 'protein_pct', 'fat_content_pct',
        'marbling_score'.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with three new columns:
        - price_per_protein
        - fat_to_protein_ratio
        - price_x_marbling
    """
    # Work on a copy so we never modify the caller's dataframe
    df = df.copy()

    # Price per gram of protein (higher = more expensive per protein)
    df["price_per_protein"] = df["price_eur_per_kg"] / df["protein_pct"]

    # Fat-to-protein ratio (higher = fattier relative to protein)
    df["fat_to_protein_ratio"] = df["fat_content_pct"] / df["protein_pct"]

    # Price times marbling score (captures premium for marbled meat)
    df["price_x_marbling"] = df["price_eur_per_kg"] * df["marbling_score"]

    return df


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """
    Load the training CSV and apply feature engineering.

    Returns
    -------
    pd.DataFrame
        Training data with engineered features.
    """
    df = pd.read_csv(TRAIN_FILE)
    return engineer_features(df)


def load_stress_data() -> pd.DataFrame:
    """
    Load the stress-test CSV and apply feature engineering.

    Returns
    -------
    pd.DataFrame
        Stress-test data with engineered features (may contain NaNs).
    """
    df = pd.read_csv(STRESS_FILE)
    return engineer_features(df)


# ------------------------------------------------------------
# Standalone test (runs only when executed directly)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check: load both files and print their shapes
    train = load_training_data()
    stress = load_stress_data()

    print("Training data shape:", train.shape)
    print("Stress data shape:  ", stress.shape)
    print()
    print("Training columns:")
    for col in train.columns:
        print("  -", col)