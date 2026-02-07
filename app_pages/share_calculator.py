# app_pages/share_calculator.py

import streamlit as st


def share_calculator_main():
    st.title("Share Position Calculator")

    entry = st.number_input("Entry Price", value=100.0)
    stop = st.number_input("Stop Price", value=95.0)
    risk = st.number_input("Risk per Trade ($)", value=200.0)

    risk_per_share = abs(entry - stop)

    if risk_per_share == 0:
        st.warning("Entry and stop cannot be equal.")
        return

    shares = risk / risk_per_share
    position_size = shares * entry

    st.metric("Shares to Buy", int(shares))
    st.metric("Position Size ($)", f"{position_size:,.2f}")
