# app_pages/dashboard.py

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS, SECTOR_ETFS
from strat.core import (
    candle_type_label,
    compute_flags,
    score_regime,
    market_bias_and_strength,
)

def dashboard_main():
    st.title("STRAT Dashboard")
    st.caption("STRAT-only market regime + sector alignment")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    # =====================
    # MARKET REGIME
    # =====================
    st.subheader("Market Regime")

    market_rows = []

    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)

        if d.empty:
            bull = bear = 0
            d_type = w_type = m_type = "n/a"
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")

            d_type = candle_type_label(d_tf)
            w_type = candle_type_label(w_tf)
            m_type = candle_type_label(m_tf)

            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "Daily": d_type,
            "Weekly": w_type,
            "Monthly": m_type,
            "BullScore": bull,
            "BearScore": bear,
        })

    market_df = pd.DataFrame(market_rows)
    st.dataframe(market_df, use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)

    if bias == "LONG":
        st.success(
            f"Market Bias: LONG 🟢 | Strength: {strength}/100 | Diff: {diff}"
        )
    elif bias == "SHORT":
        st.error(
            f"Market Bias: SHORT 🔴 | Strength: {strength}/100 | Diff: {diff}"
        )
    else:
        st.warning(
            f"Market Bias: MIXED 🟠 | Strength: {strength}/100 | Diff: {diff}"
        )

    # =====================
    # SECTOR ALIGNMENT
    # =====================
    st.subheader("Sector Alignment")

    sector_rows = []

    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)

        if d.empty:
            bull = bear = 0
            d_type = w_type = m_type = "n/a"
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")

            d_type = candle_type_label(d_tf)
            w_type = candle_type_label(w_tf)
            m_type = candle_type_label(m_tf)

            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "Daily": d_type,
            "Weekly": w_type,
            "Monthly": m_type,
            "BullScore": bull,
            "BearScore": bear,
        })

    sector_df = pd.DataFrame(sector_rows)
    sector_df["Diff"] = sector_df["BullScore"] - sector_df["BearScore"]

    if bias == "LONG":
        sector_df = sector_df.sort_values("Diff", ascending=False)
    elif bias == "SHORT":
        sector_df = sector_df.sort_values("Diff", ascending=True)

    st.dataframe(sector_df, use_container_width=True, hide_index=True)

    st.caption("Top sectors align with market bias.")
