import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS, SECTOR_ETFS, SECTOR_TICKERS
from strat.core import (
    candle_type_label,
    compute_flags,
    score_regime,
    market_bias_and_strength,
    best_trigger,
)

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def _checkify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out


def _rotation_lists(sectors_df: pd.DataFrame, bias: str, n: int = 3):
    df = sectors_df.copy()
    df["Diff"] = df["BullScore"] - df["BearScore"]

    if bias == "LONG":
        rot_in = df.sort_values(["Diff","BullScore"], ascending=[False, False]).head(n)
        rot_out = df.sort_values(["Diff","BearScore"], ascending=[True, False]).head(n)

    elif bias == "SHORT":
        rot_in = df.sort_values(["Diff","BearScore"], ascending=[True, False]).head(n)
        rot_out = df.sort_values(["Diff","BullScore"], ascending=[False, False]).head(n)

    else:
        rot_in = df.sort_values("Diff", ascending=False).head(n)
        rot_out = df.sort_values("Diff", ascending=True).head(n)

    return rot_in, rot_out


# ---------------------------------------------------
# Main Scanner Page
# ---------------------------------------------------

def show():

    st.title("Scanner (STRAT-only)")

    cols = st.columns([1, 6])
    with cols[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    # =========================
    # MARKET REGIME
    # =========================

    st.subheader("Market Regime — SPY / QQQ / IWM / DIA")

    market_rows = []

    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)

        if d.empty:
            flags = {}
            bull = bear = 0
            d_type = w_type = m_type = "n/a"

        else:
            w = resample_ohlc(d, "W-FRI")
            m = resample_ohlc(d, "M")

            d_type = candle_type_label(d)
            w_type = candle_type_label(w)
            m_type = candle_type_label(m)

            flags = compute_flags(d, w, m)
            bull, bear = score_regime(flags)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
        })

    market_df = pd.DataFrame(market_rows)
    market_df = _checkify(market_df, ["D_Inside","W_Inside","M_Inside"])
    st.dataframe(market_df, use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)

    if bias == "LONG":
        st.success(f"Bias: LONG | Strength: {strength}/100 | Diff: {diff}")
    elif bias == "SHORT":
        st.error(f"Bias: SHORT | Strength: {strength}/100 | Diff: {diff}")
    else:
        st.warning(f"Bias: MIXED | Strength: {strength}/100 | Diff: {diff}")

    # =========================
    # SECTOR ROTATION
    # =========================

    st.subheader("Sector Rotation")

    sector_rows = []

    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)

        if d.empty:
            flags = {}
            bull = bear = 0
            d_type = w_type = m_type = "n/a"

        else:
            w = resample_ohlc(d, "W-FRI")
            m = resample_ohlc(d, "M")

            d_type = candle_type_label(d)
            w_type = candle_type_label(w)
            m_type = candle_type_label(m)

            flags = compute_flags(d, w, m)
            bull, bear = score_regime(flags)

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
        })

    sectors_df = pd.DataFrame(sector_rows)
    sectors_df["Diff"] = sectors_df["BullScore"] - sectors_df["BearScore"]

    st.dataframe(
        _checkify(sectors_df, ["D_Inside","W_Inside","M_Inside"]),
        use_container_width=True,
        hide_index=True,
    )

    # =========================
    # Rotation IN / OUT
    # =========================

    st.subheader("Rotation IN / OUT")

    rot_in, rot_out = _rotation_lists(sectors_df, bias)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Rotation IN")
        for _, r in rot_in.iterrows():
            st.write(f"✅ {r['Sector']} ({r['ETF']})")

    with c2:
        st.markdown("### Rotation OUT")
        for _, r in rot_out.iterrows():
            st.write(f"❌ {r['Sector']} ({r['ETF']})")

    # =========================
    # SECTOR DRILLDOWN
    # =========================

    st.subheader("Drill Into Sector")

    sector_choice = st.selectbox(
        "Choose sector:",
        options=list(SECTOR_TICKERS.keys()),
        index=0,
    )

    tickers = SECTOR_TICKERS.get(sector_choice, [])

    rows = []

    for t in tickers:
        d = get_hist(t)
        if d.empty:
            continue

        w = resample_ohlc(d, "W-FRI")
        m = resample_ohlc(d, "M")

        flags = compute_flags(d, w, m)

        tf, entry, stop = best_trigger(bias, d, w)

        rows.append({
            "Ticker": t,
            "D_Type": candle_type_label(d),
            "W_Type": candle_type_label(w),
            "M_Type": candle_type_label(m),
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "TriggerTF": tf if tf else "—",
            "Entry": entry,
            "Stop": stop,
        })

    scan_df = pd.DataFrame(rows)
    scan_df = _checkify(scan_df, ["D_Inside","W_Inside"])

    st.dataframe(scan_df, use_container_width=True, hide_index=True)

    st.caption("LONG = break inside bar high. SHORT = break inside bar low.")


# ---------------------------------------------------
# Required entry for app.py
# ---------------------------------------------------

def scanner_main():
    show()
