import os
import sys

# Ensure the project root is on PYTHONPATH (Streamlit Cloud can be weird)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from pages.scanner import scanner_main
from pages.dashboard import dashboard_main
from pages.ticker_analyzer import analyzer_main
from pages.guide import guide_main
from pages.cheat_sheet import cheat_sheet_main


st.set_page_config(page_title="STRAT Scan V2", layout="wide")

def main():
    scanner_main()

if __name__ == "__main__":
    main()
