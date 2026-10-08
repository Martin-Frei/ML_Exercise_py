**English** | [Deutsch](README.de.md)

# ML Exercises — Python & Streamlit Edition

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-testing-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **19 machine learning notebooks, ported to clean Python modules and an interactive Streamlit app with 19 pages: Decision Tree → Random Forest → XGBoost on one meat price dataset.**

This repository is the Python version of [ml_exercises](https://github.com/Martin-Frei/ml_exercises). Every Jupyter notebook from that repository became:

1. **A standalone Python module** in `classification/` or `regression/`, with a docstring that explains each step, and
2. **A Streamlit page** in `app/pages/`, where you can explore the data, inspect the model metrics and make your own predictions with sliders.

The original notebooks stay the documentation of the learning journey. This repository is the runnable, interactive version.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The Streamlit App](#the-streamlit-app)
3. [Streamlit Command Reference](#streamlit-command-reference)
4. [App Pages](#app-pages)
5. [Running the Scripts Without Streamlit](#running-the-scripts-without-streamlit)
6. [Project Structure](#project-structure)
7. [How the Port Differs From the Notebooks](#how-the-port-differs-from-the-notebooks)
8. [Known Issues](#known-issues)
9. [Testing](#testing)
10. [Roadmap](#roadmap)
11. [Acknowledgments](#acknowledgments)
12. [License](#license)

---

## Quick Start

Three steps from clone to running app:

```bash
# 1. Clone and enter the project
git clone https://github.com/Martin-Frei/ML_Exercise_py.git
cd ML_Exercise_py

# 2. Create a virtual environment and install the packages
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

# 3. Start the app
streamlit run app/app.py
```

Your browser opens at **http://localhost:8501**. Use the sidebar on the left to switch between the 19 model pages.

**Important:** always start the app from the **project root** (the folder that contains `app/`, `shared/`, `regression/` and `classification/`). The app adds this folder to Python's import path. If you start it from somewhere else, you get `ModuleNotFoundError: No module named 'shared'`.

---

## The Streamlit App

### What the app is

Streamlit turns a plain Python script into a web app. There is no HTML, no JavaScript and no web server to configure. Every widget (slider, dropdown, button) is one line of Python. When you move a slider, Streamlit reruns the script from top to bottom and redraws the page.

The app has a **home page** (`app/app.py`) and **19 model pages** (`app/pages/`). Streamlit finds the pages automatically and sorts them in the sidebar by the number at the start of the filename: `1_DT_Classification.py` comes first, `19_XGB_Hyperparameter_Tuning.py` comes last. The order follows the learning path: Decision Tree → Random Forest → XGBoost.

### What every page shows

The pages are built from three templates.

**Regression pages** (predicting `price_eur_per_kg`):

| Section | What it shows |
|---|---|
| Sidebar filters | Meat type, organic, price range, number of rows to show |
| Key metrics | Filtered rows, mean / min / max price |
| Data explorer | The filtered data as a table |
| Model performance | MAE, RMSE and R² |
| Actual vs predicted | Scatter chart of real against predicted prices |
| Feature importance | Which features the model relies on |
| Predict a custom price | Sliders for every feature, button returns a price |

**Classification pages** (predicting `meat_type`):

| Section | What it shows |
|---|---|
| Sidebar filters | Meat type, organic, number of rows to show |
| Key metrics | Filtered rows, number of classes, class balance |
| Data explorer | The filtered data as a table |
| Model performance | Accuracy, precision, recall, F1 |
| Classification report | Per-class metrics as a table |
| Confusion matrix | Heatmap of predicted against real classes |
| Feature importance | Which features the model relies on |
| Predict meat type | Sliders for every feature, button returns the class and the probability of each class |

**Special page** (page 19, hyperparameter tuning): a warning box, the table of the best hyperparameters, then the regression sections.

### Two kinds of pages: trained live or loaded from disk

| Kind | Pages | What happens |
|---|---|---|
| **Trained on first visit** | 17 pages | The model is trained the first time you open the page, then kept in memory (`@st.cache_resource`). The first visit takes a few seconds; every visit after that is instant. |
| **Loaded from a saved model** | 6 (RF GridSearch), 19 (XGB Hyperparameter Tuning) | Training takes too long for a page visit, so the page loads a `.joblib` file from `models/`. If the file is missing, the page shows an error with the command to create it and stops. |

To create the saved models:

```bash
python -m regression.random_forest_grid_search --save-model
python -m regression.xgb_hyperparameter_tuning --save-model --no-plots
```

Use `--no-plots`. Without it, every plot window blocks the script until you close it.

### Caching: when does a model get retrained?

The trained models live in the memory of the running Streamlit process. They are retrained when:

- you **restart** the app (`Ctrl+C`, then `streamlit run app/app.py` again),
- you **clear the cache** in the app (press `C`, or use the menu ⋮ in the top right → *Clear cache*), or
- you **change the code** of the cached training function. Streamlit notices the change and retrains.

If you change the CSV files in `data/`, clear the cache or restart. Otherwise the pages keep showing the old model.

### Configuration: `.streamlit/config.toml`

```toml
[server]
headless = false     # false = open the browser automatically on start

[theme]
base = "light"       # light theme; "dark" is the alternative
```

Streamlit reads this file on start. Every setting can also be given on the command line (see the next section). The command line wins over the file.

---

## Streamlit Command Reference

All commands are run from the project root with the virtual environment activated.

### Starting and stopping

| Command | What it does |
|---|---|
| `streamlit run app/app.py` | Starts the app at http://localhost:8501 and opens the browser |
| `python -m streamlit run app/app.py` | Same, but uses the Streamlit of the active Python. Use this if `streamlit` is "not recognized" or the wrong version starts |
| `Ctrl+C` (in the terminal) | Stops the app |

### Useful options for `streamlit run`

| Command | What it does |
|---|---|
| `streamlit run app/app.py --server.port 8502` | Uses port 8502 instead of 8501. Needed when 8501 is already taken, e.g. by a second app |
| `streamlit run app/app.py --server.headless true` | Starts without opening a browser. Useful on a server, or when the browser tab is already open |
| `streamlit run app/app.py --server.runOnSave true` | Reloads the page automatically every time you save a `.py` file. Practical while editing a page |
| `streamlit run app/app.py --theme.base dark` | Starts with the dark theme, overriding `config.toml` |
| `streamlit run app/app.py --logger.level debug` | Prints detailed log output in the terminal. Helpful when a page fails without a clear error |
| `streamlit run app/app.py --browser.gatherUsageStats false` | Turns off Streamlit's anonymous usage statistics |

Options can be combined:

```bash
streamlit run app/app.py --server.port 8502 --server.runOnSave true
```

### Other Streamlit commands

| Command | What it does |
|---|---|
| `streamlit --version` | Shows the installed Streamlit version (should match `requirements.txt`) |
| `streamlit hello` | Starts Streamlit's built-in demo app. A quick check that the installation works, independent of this project |
| `streamlit config show` | Prints every configuration option with its current value, including the values from `config.toml` |
| `streamlit cache clear` | Clears Streamlit's cache on disk. This app keeps its models **in memory**, so to force retraining use `C` in the app or restart it |
| `streamlit docs` | Opens the Streamlit documentation in the browser |
| `streamlit help` | Lists all commands |

### Inside the running app

| Key / menu | What it does |
|---|---|
| `R` | Reruns the current page |
| `C` | Clears the cache, so models are retrained on the next run |
| Menu ⋮ (top right) → *Rerun* / *Clear cache* | Same as `R` / `C` |
| Menu ⋮ → *Settings* | Switch theme, turn on "Run on save" |
| *Always rerun* (appears after you save a file) | Reruns automatically after every save from now on |

---

## App Pages

| # | Page | Task | Model | Python module | Kind |
|---|---|---|---|---|---|
| 1 | DT Classification | Classification | Decision Tree | `classification/decision_tree_classification_meat_type.py` | trained live |
| 2 | DT Regression | Regression | Decision Tree | `regression/decision_tree_regression_price_per_kilo.py` | trained live |
| 3 | RF Classification | Classification | Random Forest | `classification/random_forest_classification.py` | trained live |
| 4 | RF Target Encoding | Classification | Random Forest | `classification/random_forest_target_encoding.py` | trained live |
| 5 | RF Regression | Regression | Random Forest | `regression/random_forest_regression.py` | trained live |
| 6 | RF GridSearch | Regression | Random Forest | `regression/random_forest_grid_search.py` | **saved model** |
| 7 | RF Outlier Imputation | Regression | Random Forest | `regression/random_forest_regression_outlier_imputation.py` | trained live |
| 8 | RF Combined | Regression | Random Forest | `regression/random_forest_combined_dataset.py` | trained live |
| 9 | RF Log Price | Regression | Random Forest | `regression/random_forest_log_price.py` | trained live |
| 10 | XGB Classification | Classification | XGBoost | `classification/xgb_classification.py` | trained live |
| 11 | XGB TE Mean | Classification | XGBoost | `classification/xgb_classification_3A_target_encoding_mean.py` | trained live |
| 12 | XGB TE Per Class | Classification | XGBoost | `classification/xgb_classification_3B_target_encoding_per_class.py` | trained live |
| 13 | XGB Imputation Clip | Classification | XGBoost | `classification/xgb_classification_imputation_clip.py` | trained live |
| 14 | XGB Regression | Regression | XGBoost | `regression/xgb_regression.py` | trained live |
| 15 | XGB Regression Clip Fill | Regression | XGBoost | `regression/xgb_regression_clip_fill.py` | trained live |
| 16 | XGB OHE | Regression | XGBoost | `regression/xgb_regression_one_hot_encoding.py` | trained live |
| 17 | XGB Log OHE | Regression | XGBoost | `regression/xgb_regression_log_one_hot_encoding.py` | trained live |
| 18 | XGB Regression Combined | Regression | XGBoost | `regression/xgb_regression_combined.py` | trained live |
| 19 | XGB Hyperparameter Tuning | Regression | XGBoost | `regression/xgb_hyperparameter_tuning.py` | **saved model** |

Each module has the same name as the original notebook, so you can always find the matching `.ipynb` in [ml_exercises](https://github.com/Martin-Frei/ml_exercises).

---

## Running the Scripts Without Streamlit

Every module can run on its own and prints the same results the notebook showed. Use **module syntax** (`-m`, dots instead of slashes, no `.py`):

```bash
python -m classification.xgb_classification
python -m regression.xgb_regression_combined
```

| Flag | What it does | Available in |
|---|---|---|
| *(none)* | Runs the full pipeline, prints metrics and opens plot windows | all modules |
| `--no-plots` | Skips all plots. Use this for quick runs and for long scripts | modules that create plots |
| `--save-plots DIR` | Saves every plot as a PNG file into `DIR` instead of opening a window | modules that create plots |
| `--save-model` | Saves the trained model to `models/` so the Streamlit page can load it | `random_forest_grid_search`, `xgb_hyperparameter_tuning` |
| `--help` | Lists the flags a module accepts | all modules with flags |

Examples:

```bash
python -m regression.xgb_regression --no-plots
python -m regression.xgb_regression --save-plots plots
python -m regression.xgb_hyperparameter_tuning --save-model --no-plots
```

Why module syntax? The modules import from `shared/` (data loading, metrics, saving models). `python -m` runs them as part of the project package, so those imports work. `python regression/xgb_regression.py` would fail with `ModuleNotFoundError`.

---

## Project Structure

```
ML_Exercise_py/
│
├── app/
│   ├── app.py                      # Streamlit home page (entry point)
│   └── pages/                      # 19 model pages, 1_ ... 19_, auto-discovered by Streamlit
│
├── classification/                 # 7 modules, target = meat_type
├── regression/                     # 12 modules, target = price_eur_per_kg
│
├── shared/
│   ├── data_loading.py             # Loads the two CSV files
│   └── modeling.py                 # Metrics, imputation, save/load models
│
├── data/
│   ├── meat_price_dataset.csv      # 1,200 clean rows
│   └── meat_price_100_stress_test.csv  # 100 out-of-distribution rows
│
├── models/                         # Saved models for pages 6 and 19 (.joblib)
├── .streamlit/config.toml          # Streamlit settings (browser, theme)
│
├── requirements.txt
├── REPORT.md                       # Port log: per-notebook results and bug fixes
├── TESTING.md                      # Test checklist with expected values
├── README.md                       # English
└── README.de.md                    # German
```

The dataset itself is described in detail in the [ml_exercises README](https://github.com/Martin-Frei/ml_exercises#the-dataset).

---

## How the Port Differs From the Notebooks

The port was done with Claude Code in five batches. Each batch was reviewed before the next one started. The full log is in [REPORT.md](REPORT.md). The most important changes:

| Change | Where | Effect |
|---|---|---|
| Stress test evaluated on the **91 rows with a real price** only, not on 100 rows of which 9 had an imputed price | XGB regression modules | More honest stress metrics; small differences to the notebooks (e.g. XGB baseline stress R² −0.346 instead of −0.359) |
| Leaky imputation removed: missing values are no longer filled with the median of the row's own `meat_type`, because `meat_type` is the target | RF classification (pages 3, 4) | Stress accuracy drops (27.5% → 19.8%, 25.3% → 17.6%). The lower value is the honest one |
| Syntax error fixed (unterminated string) | RF regression | Module runs again |
| Cell-order bug, unfitted baseline, absolute file paths, German comments fixed | XGB hyperparameter tuning | Module runs again |
| Plots controlled with `--no-plots` / `--save-plots` | all modules with plots | No blocking windows, no hard-coded paths |

XGBoost results can differ by about 0.5–1 percentage point from the notebooks because the XGBoost version changed.

---

## Known Issues

1. **The one-hot encoding pages do not one-hot encode** (pages 16 and 17). `pd.get_dummies` does nothing on the integer column `meat_type`. The original notebooks had the same problem, so their conclusion that "one-hot encoding fragments the strongest feature" is not backed by the experiment. Fix: convert `meat_type` to a category before `get_dummies`.
2. **The stress-test numbers on page 19 are not valid.** About 73 of the 91 stress rows are in the training set. The stress R² therefore measures how much a model memorized, not how well it generalizes: the flexible baseline scores 0.72, the strongly regularized tuned models 0.03–0.08. The valid result of page 19 is the gain on the held-out mixed test set (R² 0.868 → 0.895, MAE 1.89 → 1.56 EUR).
3. **Page 19's runtime note is wrong.** It says 8–25 minutes; the tuning actually finishes in seconds.
4. **Fine-search grid on page 19:** `learning_rate` includes `0.001`, which cannot learn anything useful with 250–350 trees, and `colsample_bylevel` (0.7 in the broad search) is dropped and silently falls back to 1.0.
5. **RF Combined (page 8)** computes its fill values on the combined data, including the test rows. This is a small leak, kept to match the original notebook.
6. **Small samples.** Positive stress results (page 18: R² +0.197) are measured on 18–21 stress rows. They are informative, but noisy.

---

## Testing

The app has not been fully tested yet. [TESTING.md](TESTING.md) contains:

- an environment check,
- a script that runs every module once,
- a checklist for all 19 pages with the expected metric values from the port, and
- the most common errors and how to fix them.

---

## Roadmap

- [x] Port all 19 notebooks to Python modules
- [x] Streamlit app with 19 pages
- [x] Pages sorted in learning-path order
- [ ] Complete the test checklist in TESTING.md
- [ ] Fix the one-hot encoding on pages 16 and 17
- [ ] Page 19: evaluate the stress test on the 18 held-out stress rows only, with repeated splits
- [ ] Page 19: correct the runtime note and the fine-search grid
- [ ] Add a `LICENSE` file
- [ ] Deploy to Streamlit Community Cloud

---

## Acknowledgments

This project was developed as part of ML tutoring sessions with [Adeena](https://github.com/Adeenasamoo), who guided the experiments and reviewed the results.

---

## License

MIT License.

**Author:** Martin Freimuth — [GitHub](https://github.com/Martin-Frei)
