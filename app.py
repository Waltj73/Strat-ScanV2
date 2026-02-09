import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from app_pages.dashboard import dashboard_main
from app_pages.scanner import scanner_main
from app_pages.ticker_analyzer import analyzer_main
from app_pages.rsi_dashboard import rsi_dashboard_main
from app_pages.share_calculator import share_calculator_main
from app_pages.guide import guide_main
from app_pages.cheat_sheet import cheat_sheet_main
from app_pages.strat_market_dashboard import strat_market_dashboard_main


st.set_page_config(page_title="STRAT Scanner", layout="wide")

PAGES = {
    "Dashboard": dashboard_main,
    "Scanner": scanner_main,
    "Ticker Analyzer": analyzer_main,
    "RSI Dashboard": rsi_dashboard_main,
    "Share Calculator": share_calculator_main,
    "Guide": guide_main,
    "Cheat Sheet": cheat_sheet_main,
}

choice = st.sidebar.radio("Navigation", list(PAGES.keys()))

PAGES[choice]()
