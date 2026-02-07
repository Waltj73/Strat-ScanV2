# app.py

import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from app_pages.scanner import scanner_main
from app_pages.dashboard import dashboard_main

st.set_page_config(page_title="STRAT Scan V2", layout="wide")

st.sidebar.title("STRAT Scan V2")
page = st.sidebar.radio("Go to", ["Dashboard", "Scanner"], index=0)

if page == "Dashboard":
    dashboard_main()
else:
    scanner_main()
