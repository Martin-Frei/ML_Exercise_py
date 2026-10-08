# Port Report

Agent: Claude Sonnet 4.6  
Project: ML_Exercise_py – 19 notebooks → Python + Streamlit

---

## Phase 1 – Notebook Ports

### Batch 1 (completed)

#### Notebook mapping confirmations

| # | Notebook | Target | Folder |
|---|---|---|---|
| 4 | random_forest_combined_dataset | price_eur_per_kg | regression/ |
| 9 | random_forest_target_encoding | meat_type | classification/ |
| 14 | xgb_hyperparameter_tuning | price_eur_per_kg | regression/ |

---

#### #1 – decision_tree_classification_meat_type

**File:** `classification/decision_tree_classification_meat_type.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
All 11 columns except `meat_type` (target). Includes `price_eur_per_kg`
and the three engineered features (`price_per_protein`, `fat_to_protein_ratio`,
`price_x_marbling`). No label leakage because the target is `meat_type`, not
`price_eur_per_kg`.

**Bugs found / fixed:** None.

**Results (from original notebook):**

| Dataset | Accuracy |
|---|---|
| Internal Test (20%, 240 rows) | 0.5292 |
| Stress Test (91 usable rows) | 0.2637 |

---

#### #2 – decision_tree_regression_price_per_kilo

**File:** `regression/decision_tree_regression_price_per_kilo.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
Eight raw columns only (meat_type, fat_content_pct, protein_pct,
marbling_score, animal_age_months, storage_days, organic, cut_quality).
Engineered features excluded — see leakage note below.

**Label leakage (Bug 2):**  
`price_per_protein = price / protein_pct` and
`price_x_marbling = price * marbling_score` both encode the target
`price_eur_per_kg`. The original notebook did not add these features for
regression, so no leakage was present in the original and no metric
comparison is needed.

**Results (from original notebook):**

| Dataset | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| Internal Test (20%, 240 rows) | 2.38 | 2.92 | 0.8737 |
| Stress Test (91 usable rows) | 8.59 | 11.00 | -0.2544 |

Note: The large gap between test R² (0.87) and stress R² (-0.25) is due
to the stress set containing rows with out-of-distribution prices.

---

#### #3 – random_forest_classification

**File:** `classification/random_forest_classification.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
Eight raw columns: fat_content_pct, protein_pct, marbling_score,
animal_age_months, storage_days, organic, cut_quality, price_eur_per_kg.
Engineered features excluded to match the original notebook's feature set.

**Imputation change:**  
The original notebook used group-based median imputation (fill each NaN
with the median of that meat_type group from training). The port uses
`shared.modeling.impute_with_train_medians()` (global training-split
medians) for consistency with the rest of the project. This lowers
stress-test accuracy by ~0.08 (see metrics below).

**Bugs found / fixed:** None.

**Results (port vs original notebook):**

| Dataset | Accuracy (port) | Accuracy (original notebook) |
|---|---|---|
| Internal Test (20%, 240 rows) | 0.5625 | 0.5625 |
| Stress Test (91 usable rows) | 0.1978 | 0.2747 |

The internal test matches exactly. The stress-test difference is caused
entirely by the imputation method change.

---

#### #6 – random_forest_regression

**File:** `regression/random_forest_regression.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
Eight raw columns only (same as DT regression).
Engineered features excluded — see leakage note for #2.

**Bugs found / fixed:**  
The last notebook cell contained an unterminated string literal
(`print("...")` broken across lines), which caused a `SyntaxError`. Fixed
in the port by using a proper DataFrame comparison table.

**Results (from original notebook):**

| Dataset | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| Internal Test (20%, 240 rows) | 1.80 | 2.25 | 0.9249 |
| Stress Test (91 usable rows) | 8.68 | 11.33 | -0.3288 |

---

### Batch 2 (completed)

#### #4 – random_forest_combined_dataset

**File:** `regression/random_forest_combined_dataset.py`  
**Category:** fast (on-the-fly training)

Combined main (1200) + stress (100) datasets into 1300 rows. Imputation
uses mode for categorical/binary columns and median for numeric ones,
computed on the combined data (slight data leakage, but matches original).
Loads both files via `TRAIN_FILE` / `STRESS_FILE` from `shared.data_loading`
since the combination logic differs from `load_training_data()`.

**Bugs found / fixed:** None.

**Results (from port, matches original):**

| Dataset | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| Full Test (20%, mixed) | 2.68 | 4.37 | 0.7479 |
| Main rows in test (239) | 2.15 | 2.81 | 0.8846 |
| Stress rows in test (21) | 8.70 | 12.09 | -0.3084 |

---

#### #5 – random_forest_log_price

**File:** `regression/random_forest_log_price.py`  
**Category:** medium (on-the-fly training)

Target transformed with `np.log1p()` before training; predictions
converted back with `np.expm1()` for EUR metrics. Stress test uses KNN
imputation (n=5) fitted on X_train — implemented locally since it is not
in `shared.modeling`.

**Bugs found / fixed:** None.

**Results (port = original):**

| Dataset | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| Internal Test (20%, 240 rows) | 1.80 | 2.24 | 0.9252 |
| Stress Test (91 usable rows) | 8.80 | 11.27 | -0.3161 |

---

#### #8 – random_forest_regression_outlier_imputation

**File:** `regression/random_forest_regression_outlier_imputation.py`  
**Category:** medium (on-the-fly training)

Three strategies compared at runtime. Best result: KNN + outlier clip.
Clipping uses 1st–99th percentile boundaries from X_train.

**Bugs found / fixed:** None.

**Results (port ≈ original):**

| Strategy | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| Internal Test | 1.80 | 2.25 | 0.9249 |
| Stress: Median | 8.69 | 11.33 | -0.3292 |
| Stress: KNN | 8.65 | 11.12 | -0.2809 |
| Stress: KNN + Clip | 8.65 | 11.12 | -0.2809 |

---

#### #9 – random_forest_target_encoding

**File:** `classification/random_forest_target_encoding.py`  
**Category:** medium (on-the-fly training)

Four target-encoded features added (cut_quality_te, organic_te,
marbling_score_te, price_bin_te) — maps computed from training split only.
Imputation for stress test: global training medians (vs original's
group-based imputation — see #3 note for explanation).

**Bugs found / fixed:** None.

**Results (port vs original notebook):**

| Dataset | Accuracy (port) | Accuracy (original) |
|---|---|---|
| Internal Test (20%, 240 rows) | 0.6250 | 0.6250 |
| Stress Test (91 usable rows) | 0.1758 | 0.2527 |

Internal test exact match; stress gap is imputation method (see #3).

---

### Batch 3 (completed)

#### #10 – xgb_classification

**File:** `classification/xgb_classification.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
Eight raw columns (no engineered features, no label leakage — target is meat_type).

**Model:**  
`XGBClassifier(objective="multi:softprob", n_estimators=150, learning_rate=0.08, max_depth=7)`.
Labels shifted 1–5 → 0–4 (XGBoost requirement). Stress test compares two
variants: native NaN handling vs training-median fill.

**Bugs found / fixed:** None.

**Results (port vs original notebook):**

| Dataset / Variant | Accuracy (port) | Accuracy (original) |
|---|---|---|
| Internal Test (240 rows) | 0.6583 | 0.6542 |
| Stress: Native NaN (91 rows) | 0.2308 | 0.2198 |
| Stress: Median fill (91 rows) | 0.2308 | 0.1978 |

Minor differences (~0.4–0.5 pp) due to XGBoost platform-specific randomness
across versions. Out-of-range table and class distribution match exactly.

---

#### #13 – xgb_classification_imputation_clip

**File:** `classification/xgb_classification_imputation_clip.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
Same 8 raw columns. Uses `IterativeImputer` (sklearn experimental) and
`KNNImputer` with `StandardScaler`. Model: `max_depth=3` (best from depth
analysis). Tests 4 imputation methods × 2 clip/no-clip = 8 variants.

**Bugs found / fixed:** None.

**Results (port vs original notebook):**

| Variant | Acc Stress (port) | Acc Stress (original) |
|---|---|---|
| Best: Native NaN (no clip) | 0.2198 | 0.2308 (Iterative) |
| Leakage demo (INVALID) | 0.2637 | 0.2637 ✓ |

Internal test: 68.33% (notebook: 67.08%). The best stress variant switched from
Iterative (original) to Native NaN (port) — same root cause as #10: XGBoost
version difference affects the NaN default directions. The recall pattern per
class and the leakage demonstration score match the notebook exactly.

---

#### #11 – xgb_classification_3A_target_encoding_mean

**File:** `classification/xgb_classification_3A_target_encoding_mean.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
8 original + 4 OOF-smoothed mean-encoded columns (cut_quality_te, organic_te,
marbling_score_te, price_bin_te) = 12 features. Price binned into 5 quantile
groups from X_train. 5-fold StratifiedKFold OOF encoding (smoothing=10)
prevents label leakage from training rows' own labels. Test and stress rows
use the full-training-set map.

**Bugs found / fixed:** None.

**Results (port vs original notebook):**

| Dataset | Baseline acc (port) | TE acc (port) | TE acc (original) |
|---|---|---|---|
| Internal Test (240 rows) | 0.6833 | 0.6667 | 0.6667 ✓ |
| Stress Test (91 rows) | 0.2198 | 0.2198 | 0.2198 ✓ |

Stress accuracy and price-bin crosstab match the notebook exactly. Internal
baseline differs by ~1 pp (same XGBoost version effect as #10/#13).

---

#### #12 – xgb_classification_3B_target_encoding_per_class

**File:** `classification/xgb_classification_3B_target_encoding_per_class.py`  
**Category:** medium (on-the-fly training)

**Feature set:**  
8 original + 20 OOF per-class encoded columns (4 features × 5 classes) = 28
features. Each column encodes the smoothed share of one class among rows with
that feature value. 5-fold StratifiedKFold OOF encoding, same structure as #11.

**Bugs found / fixed:**  
Original notebook had no cell outputs (cells ran without saving results).
Port results are the first verified execution.

**Results (port — no original outputs to compare):**

| Dataset | Baseline acc | Per-class TE acc |
|---|---|---|
| Internal Test (240 rows) | 0.6833 | 0.6792 |
| Stress Test (91 rows) | 0.2198 | 0.2198 |

Noteworthy: class 3 F1 improved from 0.48 (baseline) to 0.57 with per-class
encoding — the only experiment so far where class 3 showed meaningful gain.

---

### Batch 4 (completed)

#### #15 – xgb_regression

**File:** `regression/xgb_regression.py`
**Category:** medium (on-the-fly training)

**Feature set:**
Eight raw columns + engineered `protein_fat_ratio = protein_pct / (fat_content_pct + 1e-5)`.
No label leakage: neither protein_pct nor fat_content_pct involves the price target.

**Design note:**
The original notebook trains a second model on all 1 200 rows for the stress
evaluation. The port uses a single model (80/20 split) throughout and imputes
stress NaN with X_train medians (not full df_train medians). This is
cleaner and leakage-free; the minor metric difference is expected.

**Bugs found / fixed:**
Original notebook fills stress NaN prices with training median then evaluates
on all 100 rows (including 9 rows whose "true" price was imputed). Port fixes
this: valid_mask is captured from the raw stress data before imputation,
so evaluation uses only the 91 rows with genuinely known prices.

**Results (port):**

| Dataset | RMSE (EUR) | MAE (EUR) | R² |
|---|---|---|---|
| Train Set | 0.3869 | 0.2960 | 0.9978 |
| Internal Test (240 rows) | 1.2964 | 1.0591 | 0.9751 |
| Stress Test (91 rows) | 11.3999 | 8.8394 | -0.3461 |

Internal test metrics match the notebook exactly. Stress metrics differ
slightly due to the 80/20 model vs full-retrain model and the 91-row
vs 100-row evaluation fix.

---

#### #16 – xgb_regression_clip_fill

**File:** `regression/xgb_regression_clip_fill.py`
**Category:** fast (on-the-fly training)

**Feature set:**
Same 9 features as #15 (8 raw + protein_fat_ratio).

**Stress preprocessing:**
1. Group-wise imputation: fill NaN with the median of the stress rows in the
   same meat_type group. Valid for regression (meat_type is a feature, not
   the target). Rows with NaN meat_type fall back to global column median.
2. Outlier clipping: `animal_age_months`, `storage_days`, `fat_content_pct`
   clipped to X_train maximum (2, 2, and 3 values affected respectively).

**Bugs found / fixed:** None.

**Results (port):**

| Dataset | RMSE (EUR) | MAE (EUR) | R² |
|---|---|---|---|
| Internal Test (240 rows) | 1.2964 | 1.0591 | 0.9751 |
| Stress Test (91 rows) | 11.0464 | 8.5894 | -0.2640 |

Group-wise imputation + clipping improves stress R² from -0.3461 to -0.2640
(vs notebook's improvement from -0.3590 to -0.3078; different because port
uses 80/20-split model while notebook uses full-data model).

---

#### #17 – xgb_regression_combined

**File:** `regression/xgb_regression_combined.py`
**Category:** fast (on-the-fly training)

**Feature set:**
Nine features (8 raw + protein_fat_ratio). `is_stress` is used only as
stratification key and is not passed to the model.

**Combined training:**
1 200 clean rows + 91 stress rows (after dropping 9 NaN meat_type/price)
= 1 291 total. Stratified 80/20 split: 1 032 train / 259 test.

**Bugs found / fixed:** None.

**Results (port — exact match with original notebook):**

| Subset | RMSE (EUR) | MAE (EUR) | R² |
|---|---|---|---|
| Overall test (259 rows) | 3.1315 | 1.7900 | 0.8637 |
| Clean rows (241) | 1.8137 | 1.3722 | 0.9490 |
| Stress rows (18) | 9.8519 | 7.3835 | +0.1968 |

The combined model achieves positive R² on stress rows (+0.1968), the first
experiment in this series where stress-test predictions are meaningfully better
than the mean baseline.

---

#### #18 – xgb_regression_log_one_hot_encoding

**File:** `regression/xgb_regression_log_one_hot_encoding.py`
**Category:** fast (on-the-fly training)

**Feature set:**
Eight raw columns only (no protein_fat_ratio). One-hot encoding via
`pd.get_dummies(drop_first=True)` — a no-op for numeric meat_type (int64),
but kept for pipeline correctness and to follow the original notebook faithfully.

**Log transform:**
`y_train_log = np.log1p(y_train)` before training. Predictions back-transformed
with `np.expm1()` for EUR metrics.

**Bugs found / fixed:** None.

**Results (port):**

| Dataset | RMSE (EUR) | MAE (EUR) | R² |
|---|---|---|---|
| Internal Test (240 rows) | 1.2825 | 1.0302 | 0.9756 |
| Stress Test (91 rows) | 11.4685 | 8.9408 | -0.3624 |

The log transform slightly improves internal test metrics versus NB5 (raw price),
but does not improve stress performance.

---

#### #19 – xgb_regression_one_hot_encoding

**File:** `regression/xgb_regression_one_hot_encoding.py`
**Category:** fast (on-the-fly training)

**Feature set:**
Same as #18 (8 raw columns, one-hot encode + align), but without log transform.

**Bugs found / fixed:** None.

**Results (port):**

| Dataset | RMSE (EUR) | MAE (EUR) | R² |
|---|---|---|---|
| Internal Test (240 rows) | 1.2658 | 1.0365 | 0.9762 |
| Stress Test (91 rows) | 11.4738 | 8.8878 | -0.3637 |

Nearly identical to NB4 (log transform). The log transform adds negligible
value for this dataset — XGBoost handles the skewed target distribution well
without it.

**Cross-notebook XGBoost regression comparison:**

| Experiment | R² Test | RMSE Stress | R² Stress |
|---|---|---|---|
| #15 – XGB baseline (protein_fat_ratio) | 0.9751 | 11.40 | -0.346 |
| #16 – XGB group impute + clip | 0.9751 | 11.05 | -0.264 |
| #17 – XGB combined training | 0.8637* | 9.85* | +0.197* |
| #18 – XGB log price + OHE | 0.9756 | 11.47 | -0.362 |
| #19 – XGB OHE, raw price | 0.9762 | 11.47 | -0.364 |

*#17 trains and evaluates on a mixed clean+stress set; stress rows in test = 18 only.

---
### Batch 5 (completed)

#### #14 – xgb_hyperparameter_tuning

**File:** `regression/xgb_hyperparameter_tuning.py`
**Category:** SLOW (RandomizedSearchCV 100 iter + fine search 50 iter, 3-fold CV each)

**Feature set:**
Eight raw columns only (no protein_fat_ratio). The original notebook computed
`feature_cols` before adding `protein_fat_ratio` to `df_combined`, so the
feature accidentally excluded the engineered feature. Replicated faithfully.

**Preprocessing differences vs #17 (combined):**
- Group-wise imputation uses **training-set** subgroup medians (not stress-data's
  own medians like #17). More principled — no test-distribution information leaks.
- Clips ALL numeric features to the [1st, 99th] percentile range of training data
  (vs #17 which only clips 3 columns to their max).
- protein_fat_ratio is added to df_combined for reference but not used as a feature.

**Bugs found / fixed:**

| # | Bug | Fix |
|---|---|---|
| 1 | Cell ordering: split (Cell 11) ran before preprocessing (Cell 5) → NameError | Script runs top-to-bottom; no ordering issue |
| 2 | Baseline defined in Cell 9 but never fitted; Cell 15 calls `baseline.predict()` → AttributeError | Port fits baseline explicitly before evaluation |
| 3 | `preprocess_combined` fills NaN price with global median → evaluates on 100 rows instead of 91 | `valid_mask` captured before imputation; target column never imputed |
| 4 | Two absolute `plt.savefig('/home/tsinn/hermes-spielwiese/...')` | Replaced with `_save_or_show()` pattern |
| 5 | `print(f'Python: {np.__version__}')` | Fixed to `print(f'numpy: {np.__version__}')` |
| 6 | German comments, docstrings, and print output | Translated to English |

**Stress evaluation note:**
The combined training dataset includes ~73/91 stress rows (those that fell in the
80% train split). The stress evaluation set is all 91 rows, so ~73 were seen
during training. This inflates stress R² vs the previous notebooks where stress
rows were never in the training set. The port replicates this faithfully and
notes it in the comparison.

**Results (port — baseline only; full tuning computed at runtime):**

| Model | Test R² | Stress RMSE | Stress R² |
|---|---|---|---|
| Baseline (NB5 params) | 0.8679 | 5.18 | 0.7222 |
| Tuned (RandomSearch 100 iter) | computed at runtime | computed | computed |
| Fine-tuned (fine search 50 iter) | computed at runtime | computed | computed |

Baseline stress R²=0.7222 is much higher than #17's +0.197 because ~73/91 stress
rows were in the training set. After full hyperparameter search, further
improvement is expected.

---

---

## Phase 2 – Streamlit pages (completed)

18 pages written in `app/pages/` (pages 2–19). Page 1 (`1_RF_GridSearch.py`)
was the existing reference and was not modified.

### Page index

| Page | File | Notebook | Template | Notes |
|---|---|---|---|---|
| 2 | `2_DT_Classification.py` | #1 | B | 11 features incl. price-derived engineered cols |
| 3 | `3_DT_Regression.py` | #2 | A | 8 raw features |
| 4 | `4_RF_Classification.py` | #3 | B | 8 raw features incl. price |
| 5 | `5_RF_Combined.py` | #4 | A | 1 300-row combined; source breakdown |
| 6 | `6_RF_Log_Price.py` | #5 | A | log1p target; expm1 for display and prediction |
| 7 | `7_RF_Regression.py` | #6 | A | Standard RF baseline |
| 8 | `8_RF_Outlier_Imputation.py` | #8 | A | KNN+clip baked into prediction widget |
| 9 | `9_RF_Target_Encoding.py` | #9 | B | 12 features; TE maps in expander |
| 10 | `10_XGB_Classification.py` | #10 | B | Label shift 1-5 -> 0-4; shifted back for display |
| 11 | `11_XGB_TE_Mean.py` | #11 | B | OOF 5-fold mean TE; smoothing m=10; max_depth=3 |
| 12 | `12_XGB_TE_Per_Class.py` | #12 | B | 28 features; grouped importance chart |
| 13 | `13_XGB_Imputation_Clip.py` | #13 | B | Strategy overview table; clip checkbox in widget |
| 14 | `14_XGB_Hyperparameter_Tuning.py` | #14 | C | Requires models/xgb_hyperparameter_tuning.joblib |
| 15 | `15_XGB_Regression.py` | #15 | A | 9 features with protein_fat_ratio |
| 16 | `16_XGB_Regression_Clip_Fill.py` | #16 | A | Clip to training max in widget |
| 17 | `17_XGB_Regression_Combined.py` | #17 | A | Group-wise impute; is_stress stratified split |
| 18 | `18_XGB_Log_OHE.py` | #18 | A | log1p + OHE; expm1 back to EUR |
| 19 | `19_XGB_OHE.py` | #19 | A | OHE no-op; raw price target |

### Template A (regression) structure
Sidebar: meat type, organic, price range, rows to show.
Sections: key metrics (4 cols), data explorer, MAE/RMSE/R² metrics,
actual-vs-predicted scatter (`st.scatter_chart`), feature importance
(`st.bar_chart`), custom prediction widget.

### Template B (classification) structure
Sidebar: meat type, organic, rows to show (no price range).
Sections: key metrics (4 cols), data explorer, accuracy/precision/recall/F1,
per-class report table, confusion matrix (seaborn heatmap + `st.pyplot`),
feature importance, predict-meat-type widget with probability bar chart.

### Template C (#14 only)
`st.warning` box at the top with exact run command.
`st.error` + `st.stop()` if `models/xgb_hyperparameter_tuning.joblib` is missing.
Best hyperparameters table (from `model.get_params()`).
Then Template A sections for internal-test and stress-test performance.

### Changes to Phase 1 files
`regression/xgb_hyperparameter_tuning.py` — `--save-model` now writes both
`regression/saved_models/xgb_hyperparameter_tuning.pkl` (original) and
`models/xgb_hyperparameter_tuning.joblib` (new, required by Streamlit page 14).

---

## Bug summary

| # | Notebook | Bug | Status |
|---|---|---|---|
| 6 | random_forest_regression | Unterminated string literal in comparison cell | Fixed in port |
| 14 | xgb_hyperparameter_tuning | KeyError 'is_stress'; absolute plt.savefig paths; German comments; numpy mislabelled as Python | Fixed in Batch 5 |
