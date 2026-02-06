import streamlit as st
from pages.scanner import main as scanner_main

st.set_page_config(page_title="STRAT Scanner v2 (Step 1)", layout="wide")

def main():
    scanner_main()

if __name__ == "__main__":
    main()
