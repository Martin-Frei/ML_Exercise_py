# Testing — ML_Exercise_py

A step-by-step checklist to verify the port: first the environment, then every module, then every Streamlit page.

All commands run from the **project root** in PowerShell, with the virtual environment activated (`.venv\Scripts\activate`).

Expected values come from [REPORT.md](REPORT.md). XGBoost results may differ by about ±0.01 because of version differences. Anything larger is worth a note.

---

## Step 1 — Environment

| # | Check | Command | Expected |
|---|---|---|---|
| 1.1 | Python version | `python --version` | 3.11.x |
| 1.2 | Packages installed | `python -m pip install -r requirements.txt` | No errors |
| 1.3 | Streamlit works | `streamlit --version` | Version from `requirements.txt` |
| 1.4 | Streamlit demo runs | `streamlit hello` | Demo opens in the browser; stop with `Ctrl+C` |
| 1.5 | Project imports work | `python -c "import shared.data_loading, shared.modeling; print('OK')"` | `OK` |
| 1.6 | Data files present | `Test-Path data\meat_price_dataset.csv, data\meat_price_100_stress_test.csv` | `True` `True` |
| 1.7 | Saved models present | `Test-Path models\random_forest_grid_search.joblib, models\xgb_hyperparameter_tuning.joblib` | `True` `True` |
| 1.8 | No leftover temp pages | `Get-ChildItem app\pages -Filter tmp_*` | No output |
| 1.9 | No hard-coded page names | `Select-String -Path app\*.py, app\pages\*.py -Pattern "switch_page\|page_link\|pages[/\\]"` | No output |

If 1.7 shows `False`, create the missing model:

```powershell
python -m regression.random_forest_grid_search --save-model
python -m regression.xgb_hyperparameter_tuning --save-model --no-plots
```

---

## Step 2 — Run every module once

This loop runs every module with `--no-plots` and writes all output to `smoke_test.log`. The two slow modules are skipped because they were already run in step 1.7. `*.log` is in `.gitignore`.

```powershell
$skip = "random_forest_grid_search", "xgb_hyperparameter_tuning"
Get-ChildItem classification, regression -Filter *.py |
  Where-Object { $_.Name -ne "__init__.py" -and $_.BaseName -notin $skip } |
  ForEach-Object {
    $m = "$($_.Directory.Name).$($_.BaseName)"
    Write-Host "=== $m ===" -ForegroundColor Cyan
    python -m $m --no-plots
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $m" -ForegroundColor Red }
  } *>&1 | Tee-Object smoke_test.log
```

Then search the log for failures:

```powershell
Select-String -Path smoke_test.log -Pattern "FAILED|Traceback|Error"
```

**If a module fails with `unrecognized arguments: --no-plots`**, it has no plots and therefore no flag. Run it again without the flag. That is not a bug.

| # | Check | Expected |
|---|---|---|
| 2.1 | All 17 modules ran | 17 `=== ... ===` headers in the log |
| 2.2 | No `FAILED` lines | No output from the `Select-String` above |
| 2.3 | Metrics match REPORT.md | Compare with the tables in step 4 |

---

## Step 3 — Start the app

```powershell
streamlit run app/app.py
```

| # | Check | Expected |
|---|---|---|
| 3.1 | Browser opens | http://localhost:8501 shows "Meat Analytics Suite" |
| 3.2 | Home page metrics | Training rows 1.200, stress-test rows 100, notebooks ported 19 |
| 3.3 | Sidebar order | Pages 1 → 19 in learning-path order (DT Classification first, XGB Hyperparameter Tuning last) |
| 3.4 | No errors in the terminal | No `Traceback` while clicking through the pages |

---

## Step 4 — Every page

For each page, check four things:

- **Loads:** the page opens without a red error box. The first visit can take a few seconds (training).
- **Metrics:** the model performance values match the expected values below (with the default filters).
- **Filters:** changing the sidebar filters updates the data explorer and key metrics.
- **Prediction:** the prediction widget returns a plausible value (a price between about 5 and 45 EUR, or a class 1–5 with probabilities that add up to 100%).

### Classification pages (accuracy)

| # | Page | Test | Stress | Loads | Metrics | Filters | Prediction |
|---|---|---|---|---|---|---|---|
| 1 | DT Classification | 0.5292 | 0.2637 | [ ] | [ ] | [ ] | [ ] |
| 3 | RF Classification | 0.5625 | 0.1978 | [ ] | [ ] | [ ] | [ ] |
| 4 | RF Target Encoding | 0.6250 | 0.1758 | [ ] | [ ] | [ ] | [ ] |
| 10 | XGB Classification | 0.6583 | 0.2308 | [ ] | [ ] | [ ] | [ ] |
| 11 | XGB TE Mean | 0.6667 | 0.2198 | [ ] | [ ] | [ ] | [ ] |
| 12 | XGB TE Per Class | 0.6792 | 0.2198 | [ ] | [ ] | [ ] | [ ] |
| 13 | XGB Imputation Clip | 0.6833 | 0.2198 | [ ] | [ ] | [ ] | [ ] |

Notes:

- Page 11 baseline (without target encoding): test 0.6833.
- Page 12: class 3 F1 should rise from about 0.48 (baseline) to about 0.57.
- Page 13: the invalid leakage demonstration should show 0.2637.

### Regression pages (MAE in EUR / R²)

| # | Page | MAE test | R² test | MAE stress | R² stress | Loads | Metrics | Filters | Prediction |
|---|---|---|---|---|---|---|---|---|---|
| 2 | DT Regression | 2.38 | 0.8737 | 8.59 | −0.2544 | [ ] | [ ] | [ ] | [ ] |
| 5 | RF Regression | 1.80 | 0.9249 | 8.68 | −0.3288 | [ ] | [ ] | [ ] | [ ] |
| 6 | RF GridSearch | — | ~0.84 | ~8.46 | ~−0.27 | [ ] | [ ] | [ ] | [ ] |
| 7 | RF Outlier Imputation | 1.80 | 0.9249 | 8.65 | −0.2809 | [ ] | [ ] | [ ] | [ ] |
| 8 | RF Combined | 2.15 | 0.8846 | 8.70 | −0.3084 | [ ] | [ ] | [ ] | [ ] |
| 9 | RF Log Price | 1.80 | 0.9252 | 8.80 | −0.3161 | [ ] | [ ] | [ ] | [ ] |
| 14 | XGB Regression | 1.06 | 0.9751 | 8.84 | −0.3461 | [ ] | [ ] | [ ] | [ ] |
| 15 | XGB Regression Clip Fill | 1.06 | 0.9751 | 8.59 | −0.2640 | [ ] | [ ] | [ ] | [ ] |
| 16 | XGB OHE | 1.04 | 0.9762 | 8.89 | −0.3637 | [ ] | [ ] | [ ] | [ ] |
| 17 | XGB Log OHE | 1.03 | 0.9756 | 8.94 | −0.3624 | [ ] | [ ] | [ ] | [ ] |
| 18 | XGB Regression Combined | 1.37 | 0.9490 | 7.38 | **+0.1968** | [ ] | [ ] | [ ] | [ ] |
| 19 | XGB Hyperparameter Tuning | 1.56 | 0.8952 | 7.44 | 0.0260 ⚠ | [ ] | [ ] | [ ] | [ ] |

Notes:

- **Page 6:** REPORT.md has no ported values for this page. The approximate values are from the original notebook (`max_features=0.3`). Write down what the page shows.
- **Page 8:** test values are for the 239 clean rows in the test split, stress values for the 21 stress rows in the test split.
- **Page 18:** test values are for the 241 clean rows, stress values for the 18 stress rows in the test split.
- **Page 19:** values are for the fine-tuned model on the mixed test set. The stress values are **not valid** (training data leaked into the stress evaluation, see README → Known Issues). Only check that they match the script output.
- **Page 19:** the warning box still says the tuning takes 8–25 minutes. It actually takes seconds. That is a known text issue, not a test failure.

---

## Step 5 — Cache behaviour

| # | Check | How | Expected |
|---|---|---|---|
| 5.1 | Second visit is fast | Open a page, go to another page, come back | The page appears instantly |
| 5.2 | Clear cache retrains | Press `C` on a page, then `R` | The page trains again (short delay), metrics stay the same |
| 5.3 | Restart retrains | `Ctrl+C`, then `streamlit run app/app.py` | First visit to each page is slow again |

---

## Common Problems

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'shared'` | App or script started from the wrong folder, or run as a file path | Start from the project root; run scripts with `python -m regression.<name>`, not `python regression/<name>.py` |
| `FileNotFoundError` for a CSV file | Started from the wrong folder | Start from the project root |
| `streamlit` is not recognized | Virtual environment not active | `.venv\Scripts\activate`, or use `python -m streamlit run app/app.py` |
| `Port 8501 is already in use` | Another Streamlit app is running | Stop it with `Ctrl+C`, or use `--server.port 8502` |
| Page 6 or 19 shows "Model file not found" | The `.joblib` file is missing | Run the `--save-model` command from step 1.7 |
| A script hangs after printing results | A plot window is open (possibly behind other windows) | Close the window, or use `--no-plots` |
| Pages show old results after changing data or code | Cached model | Press `C` in the app, or restart |
| Sidebar order is wrong | Page files not renamed | Check `Get-ChildItem app\pages -Name`; filenames must start with 1_ … 19_ |
| Files named `tmp_*.py` in `app\pages` | The rename script stopped halfway | Rename them back by hand, removing `tmp_` and using the new number |
| Metrics differ by more than ±0.01 | Different package versions or a real bug | Compare `pip freeze` with `requirements.txt`, then note the page and both values |

---

## Reporting Results

When the checklist is done, note for every failed check:

1. the page or module,
2. what you did,
3. the full error message (from the terminal, not only the red box in the browser), and
4. expected value vs. actual value.

That is enough to fix the problem in a follow-up Claude Code session.
