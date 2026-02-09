import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# Pages
from app_pages.dashboard import dashboard_main
from app_pages.scanner import scanner_main
from app_pages.ticker_analyzer import analyzer_main
from app_pages.guide import guide_main
from app_pages.cheat_sheet import cheat_sheet_main
from app_pages.rsi_dashboard import rsi_dashboard_main
from app_pages.strat_market_dashboard import strat_market_dashboard_main

st.set_page_config(
    page_title="STRAT Scanner",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Scanner",
        "Ticker Analyzer",
        "Guide",
        "Cheat Sheet",
        "RSI Dashboard",
        "STRAT Market Dashboard",
    ],
)

if page == "Dashboard":
    dashboard_main()

elif page == "Scanner":
    scanner_main()

elif page == "Ticker Analyzer":
    analyzer_main()

elif page == "Guide":
    guide_main()

elif page == "Cheat Sheet":
    cheat_sheet_main()

elif page == "RSI Dashboard":
    rsi_dashboard_main()

elif page == "STRAT Market Dashboard":
    strat_market_dashboard_main()
