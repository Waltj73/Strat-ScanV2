# app_pages/rsi_dashboard.py

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.fetch import get_hist
from config.universe import SECTOR_ETFS
from strat.indicators import rsi_wilder


def rsi_dashboard_main():
    st.title("RSI Dashboard — Sector Rotation")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    rows = []

    for sector, etf in SECTOR_ETFS.items():
        df = get_hist(etf)
        if df.empty:
            continue

        close = df["Close"].dropna()
        if len(close) < 20:
            continue

        rsi = rsi_wilder(close, 14).iloc[-1]

        rows.append({
            "Sector": sector,
            "ETF": etf,
            "RSI": round(float(rsi), 1),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        st.warning("No data available.")
        return

    out = out.sort_values("RSI", ascending=False)

    st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption(
        "Higher RSI = rotation strength. Use STRAT scanner for entries."
    )
