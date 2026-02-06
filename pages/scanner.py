import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple

from data.fetch import get_hist, resample_ohlc
from strat.candles import last_candle_type
from strat.signals import compute_flags
from strat.triggers import best_trigger, trigger_status

# Minimal universe (you can expand later)
MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

SECTOR_TICKERS: Dict[str, List[str]] = {
    "Technology": ["AAPL","MSFT","NVDA","AMD","AVGO","ORCL","ADBE"],
    "Financials": ["JPM","BAC","GS","MS","C","SCHW"],
    "Energy": ["XOM","CVX","COP","EOG","SLB"],
    "Industrials": ["CAT","DE","GE","HON","BA"],
    "Metals": ["GLD","SLV","CPER","PPLT","PALL"],
}

def tf_frames(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = daily.copy()
    w = resample_ohlc(daily, "W-FRI")
    m = resample_ohlc(daily, "M")
    return d, w, m

def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:
    """
    STRAT-only regime score:
      Monthly (3), Weekly (2), Daily (1)
      plus 2-1-2 continuation weight
    """
    bull = 0
    bear = 0

    bull += 3 if flags.get("M_Bull") else 0
    bull += 2 if flags.get("W_Bull") else 0
    bull += 1 if flags.get("D_Bull") else 0
    bull += 2 if flags.get("W_212Up") else 0
    bull += 1 if flags.get("D_212Up") else 0

    bear += 3 if flags.get("M_Bear") else 0
    bear += 2 if flags.get("W_Bear") else 0
    bear += 1 if flags.get("D_Bear") else 0
    bear += 2 if flags.get("W_212Dn") else 0
    bear += 1 if flags.get("D_212Dn") else 0

    return bull, bear

def market_bias(market_rows: List[Dict]) -> Tuple[str, int, int]:
    bull_total = sum(r["BullScore"] for r in market_rows)
    bear_total = sum(r["BearScore"] for r in market_rows)
    diff = bull_total - bear_total

    # STRAT-only “strength”: 50 baseline, diff * 8 points
    strength = int(np.clip(50 + diff * 8, 0, 100))

    if diff >= 3:
        bias = "LONG"
    elif diff <= -3:
        bias = "SHORT"
    else:
        bias = "MIXED"

    return bias, strength, diff

def main():
    st.title("🟩 STRAT Scanner v2 — Step 1 (STRAT-only foundation)")
    st.caption("Candle types • Inside Bars • 2-1-2 • Weekly-first triggers • STRAT-only market regime. No RSI. No RS/Rotation yet.")

    topbar = st.columns([1, 1, 6])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    # -------------------------
    # MARKET REGIME
    # -------------------------
    st.subheader("Market Regime (STRAT-only) — SPY / QQQ / IWM / DIA")

    market_rows: List[Dict] = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
        if d.empty or len(d) < 40:
            flags = {k: False for k in [
                "D_Bull","W_Bull","M_Bull","D_Bear","W_Bear","M_Bear",
                "D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"
            ]}
            bull, bear = 0, 0
            d_type = w_type = m_type = None
        else:
            d_tf, w_tf, m_tf = tf_frames(d)
            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)
            d_type = last_candle_type(d_tf)
            w_type = last_candle_type(w_tf)
            m_type = last_candle_type(m_tf)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "D_Type": d_type or "—",
            "W_Type": w_type or "—",
            "M_Type": m_type or "—",
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
        })

    bias, strength, diff = market_bias(market_rows)

    st.dataframe(pd.DataFrame(market_rows), use_container_width=True, hide_index=True)
    badge = "🟢" if bias == "LONG" else "🔴" if bias == "SHORT" else "🟠"
    st.success(f"Bias: **{bias}** {badge} | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")

    # -------------------------
    # SCANNER INPUT
    # -------------------------
    st.subheader("Scanner")

    left, right = st.columns([1.2, 2.0])
    with left:
        mode = st.radio("Ticker source", ["Sector list", "Custom list"], index=0)
        if mode == "Sector list":
            sector = st.selectbox("Choose sector", list(SECTOR_TICKERS.keys()), index=0)
            tickers = SECTOR_TICKERS.get(sector, [])
        else:
            raw = st.text_area("Tickers (comma or space separated)", value="AAPL, MSFT, NVDA, AMD")
            tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]

        scan_n = st.slider("Scan count", 1, max(1, len(tickers)), min(15, len(tickers)))
        tickers = tickers[:scan_n]

    with right:
        st.markdown("### What this scanner shows (Step 1)")
        st.write("• D/W/M candle type (1, 2U, 2D, 3)")
        st.write("• Inside bars (D/W/M)")
        st.write("• 2-1-2 Up/Down (D/W)")
        st.write("• Weekly-first trigger + Entry/Stop (LONG levels)")
        st.write("• TriggerStatus = D/W/M snapshot")

    if not tickers:
        st.info("Add some tickers and scan.")
        return

    rows = []
    for t in tickers:
        d = get_hist(t)
        if d.empty or len(d) < 40:
            continue

        d_tf, w_tf, m_tf = tf_frames(d)
        flags = compute_flags(d_tf, w_tf, m_tf)

        d_type = last_candle_type(d_tf) or "—"
        w_type = last_candle_type(w_tf) or "—"
        m_type = last_candle_type(m_tf) or "—"

        tf, entry, stop = best_trigger(d_tf, w_tf)
        status = trigger_status(d_tf, w_tf, m_tf)

        rows.append({
            "Ticker": t,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "D_Inside": flags["D_Inside"],
            "W_Inside": flags["W_Inside"],
            "M_Inside": flags["M_Inside"],
            "D_212Up": flags["D_212Up"],
            "W_212Up": flags["W_212Up"],
            "D_212Dn": flags["D_212Dn"],
            "W_212Dn": flags["W_212Dn"],
            "TriggerTF": tf or "—",
            "Entry": None if entry is None else round(float(entry), 2),
            "Stop": None if stop is None else round(float(stop), 2),
            "TriggerStatus": status,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No data returned for the selected tickers (yfinance empty). Try Refresh or different tickers.")
        return

    # Show booleans as checkmarks (Step 1)
    flag_cols = ["D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"]
    display_df = df.copy()
    for c in flag_cols:
        display_df[c] = display_df[c].apply(lambda v: "✅" if v is True else "")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("Note: Entry/Stop are LONG trigger levels (break of Inside Bar high / stop below low). SHORT triggers come in Step 2+ when we formalize bias-direction triggers.")
