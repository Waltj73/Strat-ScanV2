# app.py

import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(page_title="STRAT Scan V2", layout="wide")

# --- Safe imports (won't crash app if a page file isn't there yet) ---
def _missing_page(title: str):
    st.title(title)
    st.warning("This page file isn't created yet (or the function name doesn't match).")

try:
    from app_pages.dashboard import dashboard_main
except Exception:
    dashboard_main = lambda: _missing_page("Dashboard")

try:
    from app_pages.scanner import scanner_main
except Exception:
    scanner_main = lambda: _missing_page("Scanner")

try:
    from app_pages.ticker_analyzer import analyzer_main
except Exception:
    analyzer_main = lambda: _missing_page("Ticker Analyzer")

try:
    from app_pages.guide import guide_main
except Exception:
    guide_main = lambda: _missing_page("Guide")

try:
    from app_pages.cheat_sheet import cheat_sheet_main
except Exception:
    cheat_sheet_main = lambda: _missing_page("Cheat Sheet")

# --- Sidebar nav ---
st.sidebar.title("STRAT Scan V2")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Scanner", "Ticker Analyzer", "Guide", "Cheat Sheet"],
    index=0,
)

# --- Router ---
if page == "Dashboard":
    dashboard_main()
elif page == "Scanner":
    scanner_main()
elif page == "Ticker Analyzer":
    analyzer_main()
elif page == "Guide":
    guide_main()
else:
    cheat_sheet_main()
