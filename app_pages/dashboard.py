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

def _checkify(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out


def dashboard_main():
    st.title("STRAT Dashboard")
    st.caption("Market Bias + Sector Alignment (STRAT-only)")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET BIAS
    # =========================
    st.subheader("Market Bias")

    market_rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
        if d.empty:
            flags = {}
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
            "D": d_type,
            "W": w_type,
            "M": m_type,
            "Bull": bull,
            "Bear": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
        })

    mdf = pd.DataFrame(market_rows)
    mdf = _checkify(mdf, ["D_Inside","W_Inside","M_Inside"])
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)

    if bias == "LONG":
        st.success(f"Market Bias: LONG 🟢 | Strength {strength}/100 | Diff {diff}")
    elif bias == "SHORT":
        st.error(f"Market Bias: SHORT 🔴 | Strength {strength}/100 | Diff {diff}")
    else:
        st.warning(f"Market Bias: MIXED 🟠 | Strength {strength}/100 | Diff {diff}")

    # =========================
    # SECTOR ALIGNMENT
    # =========================
    st.subheader("Sector Alignment")

    sector_rows = []
    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            flags = {}
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
            "D": d_type,
            "W": w_type,
            "M": m_type,
            "Bull": bull,
            "Bear": bear,
            "Diff": bull - bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
        })

    sdf = pd.DataFrame(sector_rows)

    if bias == "LONG":
        sdf = sdf.sort_values(["Diff","Bull"], ascending=[False, False])
    elif bias == "SHORT":
        sdf = sdf.sort_values(["Diff","Bear"], ascending=[True, False])
    else:
        sdf = sdf.sort_values("Diff", ascending=False)

    sdf = _checkify(sdf, ["D_Inside","W_Inside"])

    st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.caption("Goal: trade strongest sectors in bias direction.")
