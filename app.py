import streamlit as st
from pages.scanner import scanner_main

st.set_page_config(page_title="STRAT Scan V2", layout="wide")

def main():
    scanner_main()

if __name__ == "__main__":
    main()
