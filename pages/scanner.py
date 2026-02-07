import streamlit as st
import pandas as pd

from config.universe import MARKET_ETFS
from data.fetch import get_hist, resample_ohlc
from strat.signals import compute_flags, best_trigger
from strat.scoring import score_regime, market_bias

def tf_frames(daily: pd.DataFrame):
    d = daily.copy()
    w = resample_ohlc(daily, "W-FRI")
    m = resample_ohlc(daily, "M")
    return d, w, m

def scanner_main():
    st.title("Market Regime (STRAT-only) — SPY / QQQ / IWM / DIA")

    rows = []
    for name, etf in MARKET_ETFS.items():
        d = get_hist(etf)
        if d.empty:
            flags = {}
            bull, bear = 0, 0
        else:
            d_tf, w_tf, m_tf = tf_frames(d)
            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        row = {"Market": name, "ETF": etf, "BullScore": bull, "BearScore": bear}
        row.update(flags)
        rows.append(row)

    df = pd.DataFrame(rows).fillna(False)

    bull_total = int(df["BullScore"].sum())
    bear_total = int(df["BearScore"].sum())
    bias = market_bias(bull_total, bear_total)

    st.subheader("Regime Summary")
    st.write(f"**Bias:** {bias} | **Bull:** {bull_total} | **Bear:** {bear_total}")

    st.subheader("Market Table (Signals)")
    # show nice ✅
    show_cols = ["Market","ETF","BullScore","BearScore","W_Inside","D_Inside","W_212Up","D_212Up","W_212Dn","D_212Dn"]
    for c in ["W_Inside","D_Inside","W_212Up","D_212Up","W_212Dn","D_212Dn"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: "✅" if bool(v) else "")

    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
