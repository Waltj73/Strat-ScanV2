import streamlit as st

from app_pages.scanner import scanner_main

st.set_page_config(
    page_title="STRAT Scanner",
    layout="wide"
)

st.sidebar.title("Navigation")

pages = {
    "Scanner": scanner_main,
}

choice = st.sidebar.radio("Go to", list(pages.keys()))

pages[choice]()
