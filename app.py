import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from app_pages.scanner import scanner_main
from app_pages.dashboard import dashboard_main
from app_pages.ticker_analyzer import analyzer_main
from app_pages.guide import guide_main
from app_pages.cheat_sheet import cheat_sheet_main
