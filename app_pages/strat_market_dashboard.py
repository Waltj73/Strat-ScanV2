import numpy as np
import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS
from strat.core import candle_type_label, compute_flags, score_regime


def _in_force_label(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return "—"

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    c = float(cur["Close"])
    ph = float(prev["High"])
    pl = float(prev["Low"])

    if c > ph:
        return "IF↑"
    if c < pl:
        return "IF↓"
    return "—"


def _grade_from_strength(strength: int) -> str:
    if strength >= 75:
        return "A"
    if strength >= 60:
        return "B"
    return "C"


def strat_market_dashboard_main():
    st.title("STRAT Market Dashboard")

    rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
        if d.empty:
            continue

        d_tf = d
        w_tf = resample_ohlc(d, "W-FRI")
        m_tf = resample_ohlc(d, "M")

        flags = compute_flags(d_tf, w_tf, m_tf)
        bull, bear = score_regime(flags)

        rows.append({
            "Name": name,
            "Ticker": sym,
            "Close": float(d_tf["Close"].iloc[-1]),

            "D": candle_type_label(d_tf),
            "W": candle_type_label(w_tf),
            "M": candle_type_label(m_tf),

            "D_IF": _in_force_label(d_tf),
            "W_IF": _in_force_label(w_tf),
            "M_IF": _in_force_label(m_tf),

            "BullScore": bull,
            "BearScore": bear,
        })

    if not rows:
        st.warning("Market data unavailable.")
        return

    df = pd.DataFrame(rows)

    bull_total = int(df["BullScore"].sum())
    bear_total = int(df["BearScore"].sum())
    diff = bull_total - bear_total

    strength = int(np.clip(50 + diff * 8, 0, 100))

    if diff >= 3:
        bias = "LONG"
        badge = "🟢"
    elif diff <= -3:
        bias = "SHORT"
        badge = "🔴"
    else:
        bias = "MIXED"
        badge = "🟠"

    grade = _grade_from_strength(strength)

    cols = st.columns([1, 1, 1, 1, 1.4])

    for i, r in enumerate(rows[:4]):
        with cols[i]:
            st.metric(r["Ticker"], f"{r['Close']:.2f}")
            st.write(f"D: {r['D']} ({r['D_IF']})")
            st.write(f"W: {r['W']} ({r['W_IF']})")
            st.write(f"M: {r['M']} ({r['M_IF']})")
            st.write(f"Score: B{r['BullScore']} / S{r['BearScore']}")

    with cols[4]:
        st.markdown("### Market Grade")
        st.write(f"Bias: {badge} {bias}")
        st.write(f"Grade: {grade}")
        st.write(f"Strength: {strength}/100")
        st.write(f"Diff: {diff}")

    st.divider()
    st.dataframe(df, use_container_width=True, hide_index=True)
