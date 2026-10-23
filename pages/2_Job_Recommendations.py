"""Job Recommendations Streamlit page (multipage entry).

Reuses the same recommendation UI as the main dashboard sidebar page.
Run via: streamlit run app.py  then open Job Recommendations from sidebar,
or open this page from Streamlit's multipage menu if enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when Streamlit loads pages/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.dashboard import page_job_recommendations

st.set_page_config(
    page_title="Job Recommendations",
    page_icon="🔎",
    layout="wide",
)

page_job_recommendations()
