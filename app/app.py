"""
Meat Analytics Suite – Streamlit Home Page.

Purpose
-------
Entry point of the Streamlit app. Shows a short introduction and
the metrics overview. The individual model pages are auto-discovered
by Streamlit from the pages/ folder and shown in the sidebar.

Note on imports
---------------
Streamlit adds the folder of each page to sys.path, but NOT the
project root. To let every page import from `shared.*` and
`regression.*`, we manually add the project root to sys.path here.
Because Streamlit loads app.py first, this fix applies to all pages.
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

st.title("🥩 Meat Analytics Suite")

st.markdown(
    """
    Interactive demos of machine learning models trained on the
    **meat price dataset**.

    Use the sidebar on the left to navigate between pages.
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Training rows", "1.200")
col2.metric("Stress-test rows", "100")
col3.metric("Notebooks ported", "19")