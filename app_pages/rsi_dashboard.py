# app_pages/rsi_dashboard.py

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from data.fetch import get_hist
from config.universe import SECTOR_ETFS


def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI (no external dependencies)."""
    series = series.astype(float)
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50.0)


def rsi_dashboard_main():
    st.title("RSI Dashboard — Sector Rotation (RSI 14)")

    top = st.columns([1, 4])
    with top[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    rows = []
    for sector, etf in SECTOR_ETFS.items():
        df = get_hist(etf)
        if df is None or df.empty:
            continue

        close = df["Close"].dropna()
        if len(close) < 30:
            continue

        rsi_val = float(rsi_wilder(close, 14).iloc[-1])

        # simple “rotation” label
        state = "HOT (70+)" if rsi_val >= 70 else "STRONG (55+)" if rsi_val >= 55 else "WEAK (<45)" if rsi_val < 45 else "NEUTRAL"

        rows.append({
            "Sector": sector,
            "ETF": etf,
            "RSI(14)": round(rsi_val, 1),
            "State": state,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        st.warning("No data returned. Try Refresh.")
        return

    out = out.sort_values("RSI(14)", ascending=False)
    st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption("Use this to spot rotation, then go to the STRAT Scanner to find leaders + triggers.")
