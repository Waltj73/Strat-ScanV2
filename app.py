# app.py — STRAT-only Scanner (modular, crash-proof navigation)

import os
import sys
import importlib

import streamlit as st

# ----------------------------
# Ensure project root is importable
# ----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="STRAT Scanner (Modular)", layout="wide")


def _module_exists(module_path: str) -> bool:
    """
    module_path like: 'app_pages.scanner'
    Checks if a corresponding .py file exists.
    """
    parts = module_path.split(".")
    rel_path = os.path.join(ROOT, *parts) + ".py"
    return os.path.exists(rel_path)


def _safe_import(module_path: str, func_name: str):
    """
    Returns callable or None. Shows a nice error in-app if import fails.
    """
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name, None)
        if fn is None:
            st.error(f"Missing function `{func_name}()` in `{module_path}.py`")
            return None
        return fn
    except Exception as e:
        st.error(f"Failed to import `{module_path}` → {e}")
        return None


# ----------------------------
# Page registry (only what exists)
# ----------------------------
PAGES = []  # (label, module_path, function_name)

# Scanner
if _module_exists("app_pages.scanner"):
    PAGES.append(("📡 Scanner", "app_pages.scanner", "scanner_main"))

# Ticker Analyzer
if _module_exists("app_pages.ticker_analyzer"):
    PAGES.append(("🔎 Ticker Analyzer", "app_pages.ticker_analyzer", "analyzer_main"))

# Cheat Sheet
if _module_exists("app_pages.cheat_sheet"):
    PAGES.append(("🧾 Cheat Sheet", "app_pages.cheat_sheet", "cheat_sheet_main"))

# Guide
if _module_exists("app_pages.guide"):
    PAGES.append(("📘 Guide", "app_pages.guide", "guide_main"))

# Dashboard (optional)
if _module_exists("app_pages.dashboard"):
    PAGES.append(("📊 Dashboard", "app_pages.dashboard", "dashboard_main"))

st.sidebar.title("STRAT Scanner")
st.sidebar.caption("Modular build • STRAT-only • crash-proof")

if not PAGES:
    st.error("No pages found. Make sure you have at least `app_pages/scanner.py` with `scanner_main()`.")
    st.stop()

labels = [p[0] for p in PAGES]
choice = st.sidebar.radio("Navigation", labels, index=0)

# Run selected page
label, module_path, func_name = next(p for p in PAGES if p[0] == choice)
page_fn = _safe_import(module_path, func_name)

if page_fn:
    page_fn()
else:
    st.info("Fix the error shown above, then refresh.")
