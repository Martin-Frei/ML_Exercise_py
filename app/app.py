"""
Meat Analytics Suite – Streamlit Home Page.

Purpose
-------
Entry point of the Streamlit app. Introduces the project to visitors
(what machine learning is, what this project shows, where to start).
The individual model pages are auto-discovered by Streamlit from the
pages/ folder and shown in the sidebar.

Note on imports
---------------
Streamlit adds the folder of each page to sys.path, but NOT the
project root. To let the app import from `shared.*` and
`regression.*`, we manually add the project root to sys.path here.
"""

import sys
from pathlib import Path

# Project root = the folder that contains app/, shared/, regression/
# __file__ is app/app.py, so parent.parent is ML_Exercise_py/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Only add once, to avoid duplicates when the script reruns
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Now the regular Streamlit code
# ------------------------------------------------------------

import streamlit as st

# Page configuration: must be the first Streamlit call in the app
st.set_page_config(
    page_title="Meat Analytics Suite",
    page_icon="🥩",
    layout="wide",
)

# ------------------------------------------------------------
# Step 1: Header and intro text
# ------------------------------------------------------------

st.title("🥩 Meat Analytics Suite")

st.subheader(
    "Predicting meat prices with machine learning — "
    "from a single decision tree to XGBoost"
)

st.markdown(
    """
    This app shows **19 machine learning experiments** on one dataset of meat cuts.
    Every page trains a model, shows how well it works, and lets you make
    your own predictions with sliders.

    Built by **Martin Freimuth** — master butcher turned ML developer.
    """
)


# ------------------------------------------------------------
# Step 2: Key results as metric tiles
# ------------------------------------------------------------

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Experiments",
    value="19",
    help="Each experiment is one notebook from the original project, now one page in this app.",
)

col2.metric(
    label="Algorithms",
    value="3",
    help="Decision Tree, Random Forest and XGBoost.",
)

col3.metric(
    label="Best R² on clean data",
    value="0.975",
    delta="+0.10 vs. single tree",
    help="XGBoost explains 97.5% of the price variation. A single decision tree reached 0.87.",
)

col4.metric(
    label="R² on unusual data",
    value="+0.20",
    delta="+0.56 vs. baseline",
    help=(
        "On the stress test (100 unusual cuts), every model was worse than guessing "
        "until the model saw such cases during training. Baseline: -0.36."
    ),
)

# ------------------------------------------------------------
# Step 3: Explanations in tabs
# ------------------------------------------------------------

st.divider()

tab_ml, tab_project, tab_metrics, tab_csv = st.tabs(
    ["🤖 What is machine learning?", "🔬 What this project shows", "📏 How to read the numbers", "📊 Inside the CSV Files"]
)

with tab_ml:
    st.markdown(
        """
        Instead of writing rules by hand (*"beef costs more than chicken"*), we show
        a computer **1,200 real examples**: meat type, fat content, marbling, age,
        organic label — and the price. The model finds the patterns on its own.
        Afterwards it can estimate the price of a cut it has never seen.

        This project asks two kinds of questions:

        - **Regression:** *What does this cut cost per kilo?* → the answer is a number.
        - **Classification:** *Which meat type is this?* → the answer is one of five categories.
        """
    )

with tab_project:
    st.markdown(
        """
        1. **Better models learn better.** Decision Tree → Random Forest → XGBoost
           improved the price prediction from R² 0.87 to 0.97.
        2. **But the real world is harder.** A second dataset of 100 unusual cuts —
           missing values, outliers, higher prices — broke every model.
           Thousands of tuning attempts did not fix it.
        3. **The fix was data, not tuning.** Only when the model saw such unusual
           cases during training did it beat guessing on them (R² +0.20).
        4. **Honest evaluation matters.** Along the way we found and fixed hidden
           data leaks that made some results look better than they were.
        """
    )

    with st.expander("Why meat prices?"):
        st.markdown(
            """
            I am a master butcher (*Metzgermeister*) by trade. Knowing how meat is
            graded and priced helped directly: filling missing values per meat type
            gave better results than generic statistical methods.
            """
        )

with tab_metrics:
    st.markdown(
        """
        | Metric | Used for | In one sentence |
        |---|---|---|
        | **R²** | Regression | Share of the price differences the model explains: 1.0 = perfect, 0 = no better than always guessing the average, below 0 = worse than that |
        | **MAE** | Regression | Average error in euros: *"the prediction is X € off per kilo"* |
        | **Accuracy** | Classification | How often the model picks the right meat type (5 classes, so guessing gives about 20%) |
        """
    )
    
with tab_csv:
    st.markdown(
        """
        The project uses **two CSV files** with the same nine columns.
        Both are **synthetic**: they were generated for this project and do not come
        from real purchasing or sales data. The value ranges are realistic, but no
        row describes a real cut of meat.

        | File | Rows | Purpose |
        |---|---|---|
        | `meat_price_dataset.csv` | 1,200 | Clean data for training and testing the models |
        | `meat_price_100_stress_test.csv` | 100 | Hard cases that challenge the models — and usually overwhelm them |

        **The columns:**

        | Column | Meaning |
        |---|---|
        | `meat_type` | Meat category, 1 = cheapest to 5 = most expensive |
        | `fat_content_pct` | Fat content in percent |
        | `protein_pct` | Protein content in percent |
        | `marbling_score` | Marbling grade |
        | `animal_age_months` | Age of the animal in months |
        | `storage_days` | Days in storage |
        | `organic` | Organic label (0 = no, 1 = yes) |
        | `cut_quality` | Quality grade of the cut, 1 to 5 |
        | `price_eur_per_kg` | Price in euros per kilo — the value the regression models predict |

        **What makes the stress test hard:**

        - **Missing values** in every column (7–14 per column)
        - **Outliers** far outside the clean range, e.g. an animal age of 189 months
          instead of at most 71
        - **Shifted prices:** on average about 7 € per kilo higher than in the clean data.
          This is the main reason every model trained only on clean data fails here.
        """
    )

    with st.expander("Value ranges: clean data vs. stress test"):
        st.markdown(
            """
            | Column | Clean data | Stress test | Missing (stress) | Outside clean range |
            |---|---|---|---|---|
            | `meat_type` | 1–5 | 1–5 | 9 | 0 |
            | `fat_content_pct` | 2–35 | 2–**58.5** | 14 | 3 |
            | `protein_pct` | 15–26 | 15.1–25 | 10 | 0 |
            | `marbling_score` | 1–10 | **0**–**12** | 11 | 2 |
            | `animal_age_months` | 2–71 | 2–**189** | 8 | 2 |
            | `storage_days` | 0–20 | 0–**80** | 9 | 2 |
            | `organic` | 0/1 | 0/1 | 9 | 0 |
            | `cut_quality` | 1–5 | 1–5 | 7 | 0 |
            | `price_eur_per_kg` | 2.50–43.33 | 12.14–**65.40** | 9 | 6 |

            **Mean price:** 23.05 € (clean) vs. 29.97 € (stress).
            Bold values lie outside the range the models saw during training.
            """
        )