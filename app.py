# app.py

import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from app_pages.scanner import show as scanner_main
from app_pages.dashboard import dashboard_main
from app_pages.ticker_analyzer import analyzer_main
from app_pages.guide import guide_main
from app_pages.cheat_sheet import cheat_sheet_main

st.set_page_config(page_title="STRAT-only Scanner", layout="wide")

PAGES = {
    "Scanner": scanner_main,
    "Dashboard": dashboard_main,
    "Ticker Analyzer": analyzer_main,
    "Guide": guide_main,
    "Cheat Sheet": cheat_sheet_main,
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)

PAGES[choice]()
