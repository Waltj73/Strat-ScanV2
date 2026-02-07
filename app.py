# app.py (clean, working)

import os
import sys
import streamlit as st

# --- Ensure project root is importable ---
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Streamlit config ---
st.set_page_config(
    page_title="STRAT Scanner (Modular)",
    layout="wide",
)

st.sidebar.title("STRAT App")

# --- Safe imports (so you see errors instead of a blank screen) ---
try:
    from app_pages.scanner import show as scanner_page
except Exception as e:
    scanner_page = None
    scanner_err = e

try:
    from app_pages.dashboard import dashboard_main as dashboard_page
except Exception as e:
    dashboard_page = None
    dashboard_err = e

try:
    from app_pages.ticker_analyzer import analyzer_main as analyzer_page
except Exception as e:
    analyzer_page = None
    analyzer_err = e

try:
    from app_pages.guide import guide_main as guide_page
except Exception as e:
    guide_page = None
    guide_err = e

try:
    from app_pages.cheat_sheet import cheat_sheet_main as cheat_sheet_page
except Exception as e:
    cheat_sheet_page = None
    cheat_sheet_err = e


PAGES = {
    "Scanner": ("Scanner", scanner_page, locals().get("scanner_err", None)),
    "Dashboard": ("Dashboard", dashboard_page, locals().get("dashboard_err", None)),
    "Ticker Analyzer": ("Ticker Analyzer", analyzer_page, locals().get("analyzer_err", None)),
    "Guide": ("Guide", guide_page, locals().get("guide_err", None)),
    "Cheat Sheet": ("Cheat Sheet", cheat_sheet_page, locals().get("cheat_sheet_err", None)),
}

choice = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)

title, fn, err = PAGES[choice]

# --- Render ---
st.title(title)

if fn is None:
    st.error(f"❌ This page failed to load: {title}")
    st.code(str(err))
    st.stop()

# Run the page
fn()
