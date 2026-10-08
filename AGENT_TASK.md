# Task: Port 19 ML Notebooks to Python + Streamlit App

## Context

I have a Jupyter notebook collection (19 notebooks) in a source folder.
I want you to port every notebook into a well-documented Python file
and then build a Streamlit app on top of them.

- Source folder (READ-ONLY): C:\Users\tsinn\VSCode\Repos\ML_Ecercise\
- Target folder (WRITE HERE): C:\Users\tsinn\VSCode\Repos\ML_Exercise_py\

Both folders are on my local disk. Both Git repos exist. Do NOT push
to Git yourself. I handle Git commits after each batch.

## Language

EVERYTHING in English: code, comments, docstrings, filenames, the
REPORT.md file, and all chat with me. No German anywhere in the output.

## Source folder rules (READ-ONLY)

The folder C:\Users\tsinn\VSCode\Repos\ML_Ecercise\ contains:
- 19 .ipynb notebooks
- meat_price_dataset.csv
- meat_price_100_stress_test.csv
- Various helper scripts

You may READ any file in this folder to understand the logic.
You may NEVER write, edit, delete, rename, or move anything here.
You may NEVER run git commands in this folder.

## Target folder rules (WRITE HERE)

The folder C:\Users\tsinn\VSCode\Repos\ML_Exercise_py\ already contains
a working reference prototype:

- shared/__init__.py
- shared/data_loading.py          <- REFERENCE: data loading pattern
- shared/modeling.py              <- REFERENCE: metrics + model save/load
- regression/__init__.py
- regression/random_forest_grid_search.py  <- REFERENCE: slow model port
- app/__init__.py
- app/app.py                      <- REFERENCE: Streamlit home page
- app/pages/1_RF_GridSearch.py    <- REFERENCE: Streamlit model page
- data/meat_price_dataset.csv
- data/meat_price_100_stress_test.csv
- models/random_forest_grid_search.joblib
- requirements.txt
- .gitignore
- .venv/ (do not touch)

READ these reference files FIRST. Every new file you write must match
their style exactly:
- Module docstring length and structure
- Comment density
- Naming conventions
- Import order
- Function docstring format
- `if __name__ == "__main__":` pattern

## Target folder structure (final state)

ML_Exercise_py/
├── .venv/                              do not touch
├── .gitignore                          do not touch
├── requirements.txt                    do not touch
├── AGENT_TASK.md                       this file
├── REPORT.md                           you create and update
├── data/                               do not touch
├── models/                             you write .joblib files here
├── shared/                             do not touch (references)
├── classification/
│   ├── __init__.py                     create
│   └── <one .py per classification notebook>
├── regression/
│   ├── __init__.py                     exists
│   ├── random_forest_grid_search.py    exists (reference)
│   └── <one .py per regression notebook>
└── app/
    ├── __init__.py                     exists
    ├── app.py                          exists (reference)
    └── pages/
        ├── 1_RF_GridSearch.py          exists (reference)
        └── <numbered pages for all remaining notebooks>

## Notebook inventory and category

The 19 notebooks have been analyzed for runtime. Their category
decides how the ported model is handled:

| # | Notebook | Category | Action |
|---|---|---|---|
| 1 | decision_tree_classification_meat_type | medium | on-the-fly |
| 2 | decision_tree_regression_price_per_kilo | medium | on-the-fly |
| 3 | random_forest_classification | medium | on-the-fly |
| 4 | random_forest_combined_dataset | fast | on-the-fly |
| 5 | random_forest_log_price | medium | on-the-fly |
| 6 | random_forest_regression | medium | on-the-fly |
| 7 | random_forest_regression_grid_search | SLOW | ALREADY PORTED (reference) |
| 8 | random_forest_regression_outlier_imputation | medium | on-the-fly |
| 9 | random_forest_target_encoding | medium | on-the-fly |
| 10 | xgb_classification | medium | on-the-fly |
| 11 | xgb_classification_3A_target_encoding_mean | medium | on-the-fly |
| 12 | xgb_classification_3B_target_encoding_per_class | medium | on-the-fly |
| 13 | xgb_classification_imputation_clip | medium | on-the-fly |
| 14 | xgb_hyperparameter_tuning | FAILED | fix bug, then treat as slow |
| 15 | xgb_regression | medium | on-the-fly |
| 16 | xgb_regression_clip_fill | fast | on-the-fly |
| 17 | xgb_regression_combined | fast | on-the-fly |
| 18 | xgb_regression_log_one_hot_encoding | fast | on-the-fly |
| 19 | xgb_regression_one_hot_encoding | fast | on-the-fly |

### Category rules

**fast / medium**:
- The .py file trains the model in its `if __name__ == "__main__"`
  block. No .joblib file.
- The Streamlit page uses `@st.cache_resource` to train once on first
  visit and cache it in memory.

**slow**:
- The .py file accepts a `--save-model` flag (like
  `regression/random_forest_grid_search.py`).
- The Streamlit page checks for the .joblib file with
  `shared.modeling.model_exists()`. If missing, it shows an error
  with the exact command to run and calls `st.stop()`.

**failed**:
- Fix the bug first (see "Known bugs" section below).
- Report the fix in REPORT.md.
- Then treat the notebook as slow.

## Known bugs to fix

These bugs were identified before the port. Fix them in the .py
version. Document each fix in REPORT.md.

### Bug 1: `xgb_hyperparameter_tuning.ipynb`

- **Bug**: `KeyError: 'is_stress'`. The preprocessing cell that adds
  the `is_stress` column is not executed before it is used.
- **Fix**: Reorder cells so preprocessing runs first. The function
  `preprocess_combined()` must run and return the combined dataframe
  before `X = df_combined[feature_cols]` is called.
- **Bug**: Absolute paths like `/home/tsinn/hermes-spielwiese/...`
  inside `plt.savefig()` calls.
- **Fix**: Remove all `plt.savefig()` calls with absolute paths.
  Keep `plt.show()` only.
- **Bug**: German comments throughout the notebook.
- **Fix**: Translate all comments to English.
- **Bug**: `print(f'Python: {np.__version__}')` mislabels numpy as
  "Python".
- **Fix**: Change to `print(f'numpy: {np.__version__}')`.

### Additional bugs

If you find ADDITIONAL bugs while porting other notebooks:
1. Fix them in the .py version.
2. Add a header comment at the top of the file:
   `# TODO: BUG FOUND - see REPORT.md for details`
3. Document the fix in REPORT.md.
4. Continue with the next notebook. Do NOT stop.

## Phase 1: Convert each notebook to its own .py file

Work in 5 batches. After each batch, STOP and report to me.
Wait for my explicit approval ("approved" or "continue") before
starting the next batch.

### Batch 1 (4 notebooks)
- decision_tree_classification_meat_type
- decision_tree_regression_price_per_kilo
- random_forest_classification
- random_forest_regression

### Batch 2 (4 notebooks)
- random_forest_combined_dataset
- random_forest_log_price
- random_forest_regression_outlier_imputation
- random_forest_target_encoding

### Batch 3 (4 notebooks)
- xgb_classification
- xgb_classification_3A_target_encoding_mean
- xgb_classification_3B_target_encoding_per_class
- xgb_classification_imputation_clip

### Batch 4 (5 notebooks)
- xgb_regression
- xgb_regression_clip_fill
- xgb_regression_combined
- xgb_regression_log_one_hot_encoding
- xgb_regression_one_hot_encoding

### Batch 5 (1 notebook, after bug fix)
- xgb_hyperparameter_tuning

Note: `random_forest_regression_grid_search.py` is already done as
a reference. Do NOT redo it.

## Conversion rules (Phase 1, per notebook)

1. **Module docstring** (20-40 lines at the top):
   - Purpose (1 paragraph)
   - Step-by-step explanation (numbered list, one item per notebook
     section)
   - Variables glossary
   - Runtime note (fast / medium / slow)
   - Usage (how to run standalone)

2. **Import order**: stdlib, third-party, local (`from shared...`).
   Use `from shared.data_loading import ...` and
   `from shared.modeling import ...` — never reimplement those.

3. **Markdown cells** become numbered sections inside the module
   docstring. Preserve all important explanations, translate any
   German to English.

4. **Code cells** become plain Python in the same order.

5. **`!pip install X` cells**: skip. If X is not in requirements.txt,
   add it. Otherwise ignore.

6. **`%matplotlib inline`**: skip.

7. **Plot cells**: keep `plt.show()`. Add `plt.tight_layout()` before.
   NEVER use `plt.savefig()` with absolute paths.

8. **`display(df)` or bare `df.head()`**: replace with
   `print(df.head().to_string())` so output appears when run.

9. **Every function** gets a docstring with Purpose, Parameters,
   Returns.

10. **Non-trivial variables** get a short inline comment.

11. **`if __name__ == "__main__":`** block in every file. It runs
    the full pipeline and prints key results (metrics, tables) like
    the notebook did. For slow models, use argparse with a
    `--save-model` flag.

12. **`random_state=42`** everywhere a random seed is needed.

13. **Relative paths only**. Never absolute paths.

14. **Every .py file must be runnable** standalone via:
    `python -m classification.<name>` or `python -m regression.<name>`
    (module syntax, not file path).

15. **File naming**: same as the notebook name, but snake_case and
    .py. Example: `decision_tree_classification_meat_type.ipynb`
    becomes `classification/decision_tree_classification_meat_type.py`.

## Phase 2: Build the Streamlit pages

Only start Phase 2 after ALL Phase 1 batches are approved.

Create one Streamlit page per notebook in `app/pages/`, numbered
2 to 19 (page 1 already exists as `1_RF_GridSearch.py`).

### Three templates

**Template A: Regression pages**
For all regression models.

Sections (top to bottom):
1. Title + short description
2. **Sidebar filters**:
   - Meat type (multi-select, default all)
   - Organic (radio: All / Only organic / Only non-organic)
   - Price range (slider)
   - Rows to show (slider 5-50)
3. **Key metrics** (4 columns): filtered rows, mean price, min price,
   max price
4. **Data explorer**: table of filtered data (top N rows)
5. **Model performance** (3 columns): MAE, RMSE, R²
6. **Actual vs Predicted**: `st.scatter_chart`
7. **Feature importance**: `st.bar_chart`
8. **Predict a custom price**: 8 sliders + predict button

Copy the exact structure from `app/pages/1_RF_GridSearch.py`.

**Template B: Classification pages**
For all classification models.

Sections:
1. Title + short description
2. **Sidebar filters**:
   - Meat type (multi-select, default all)
   - Organic (radio)
   - Rows to show (slider)
3. **Key metrics** (3-4 columns): filtered rows, number of classes,
   class balance
4. **Data explorer**: table
5. **Model performance** (4 columns): accuracy, precision, recall, F1
6. **Classification report**: table per class
7. **Confusion matrix**: heatmap using matplotlib + st.pyplot
8. **Feature importance**: `st.bar_chart`
9. **Predict meat type**: sliders for input features, button shows
   predicted class + probability per class

**Template C: Special page**
Only for `xgb_hyperparameter_tuning.py`.

Sections:
1. Title + short description
2. **Warning box** (`st.warning`) at the top explaining the original
   bug and the fix
3. **Best parameters table** from the tuned model
4. Then follow Template A (regression) structure

### Rules for ALL Streamlit pages

1. **sys.path fix at the top** (needed because Streamlit adds the
   page folder to sys.path, not the project root):

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

    ## Additional rules

### Git
- Do NOT run any git commands in either folder: no commit, no push,
  no branch, no checkout. I handle all Git operations myself.

### Running files
- You MAY run the ported .py files to verify they work, using the
  project venv:
  `.venv\Scripts\python -m classification.<name>`
  `.venv\Scripts\python -m regression.<name>`
- Compare the printed metrics with the original notebook outputs
  and report any differences in REPORT.md.

### Folder mapping
- classification/: #1, #3, #9, #10, #11, #12, #13
- regression/: #2, #4, #5, #6, #8, #14, #15, #16, #17, #18, #19

### Naming exception
- Notebook #7 (random_forest_regression_grid_search) was ported as
  regression/random_forest_grid_search.py. Keep this name.
  Do not rename or redo it.