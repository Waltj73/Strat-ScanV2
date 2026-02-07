# app_pages/ticker_analyzer.py

import math
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS
from strat.core import (
    candle_type_label,
    compute_flags,
    score_regime,
    market_bias_and_strength,
    best_trigger,
)

# -------------------------
# Helpers
# -------------------------
def _fmt(v, nd=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "—"

def atr14(d: pd.DataFrame):
    if d is None or d.empty or len(d) < 20:
        return None
    h, l, c = d["High"], d["Low"], d["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out = tr.rolling(14).mean().iloc[-1]
    return None if not math.isfinite(out) else float(out)

def targets_20_63(d: pd.DataFrame, bias: str):
    if d is None or d.empty or len(d) < 70:
        return None, None
    hi20 = float(d["High"].rolling(20).max().iloc[-1])
    lo20 = float(d["Low"].rolling(20).min().iloc[-1])
    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])
    if bias == "SHORT":
        return lo20, lo63
    return hi20, hi63

def rr_to_t2(entry, stop, t2, bias: str):
    if entry is None or stop is None or t2 is None:
        return None
    try:
        entry = float(entry); stop = float(stop); t2 = float(t2)
    except Exception:
        return None

    if bias == "SHORT":
        risk = stop - entry
        reward = entry - t2
    else:
        risk = entry - stop
        reward = t2 - entry

    if risk <= 0:
        return None
    return max(0.0, reward / risk)

def _market_bias():
    """
    Derive a simple STRAT bias using MARKET_ETFS (SPY/QQQ/IWM/DIA).
    Uses the same scoring approach as scanner/dashboard.
    """
    market_rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
        if d.empty:
            flags = {}
            bull = bear = 0
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")
            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "BullScore": bull,
            "BearScore": bear,
        })

    return market_bias_and_strength(market_rows)


# -------------------------
# Main page
# -------------------------
def analyzer_main():
    st.title("STRAT Ticker Analyzer")
    st.caption("STRAT-only: candle types, alignment, inside-bar triggers, and a clean trade plan.")

    topbar = st.columns([1, 1, 6])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Bias selection
    derived_bias, strength, diff = _market_bias()

    with st.expander("Analyzer Settings", expanded=True):
        c1, c2, c3 = st.columns([1.3, 1.1, 1.2])
        with c1:
            ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        with c2:
            use_market_bias = st.checkbox("Use Market Bias", value=True)
        with c3:
            manual_bias = st.selectbox("Manual Bias", ["LONG", "SHORT", "MIXED"], index=0)

    bias = derived_bias if use_market_bias else manual_bias

    # Bias status
    if bias == "LONG":
        st.success(f"Bias: LONG 🟢 | STRAT Strength {strength}/100 | Diff {diff}")
    elif bias == "SHORT":
        st.error(f"Bias: SHORT 🔴 | STRAT Strength {strength}/100 | Diff {diff}")
    else:
        st.warning(f"Bias: MIXED 🟠 | STRAT Strength {strength}/100 | Diff {diff}")

    if not ticker:
        st.info("Type a ticker to analyze.")
        return

    d = get_hist(ticker)
    if d.empty:
        st.warning("No data returned (bad symbol or yfinance empty). Try another ticker.")
        return

    w = resample_ohlc(d, "W-FRI")
    m = resample_ohlc(d, "M")

    # Candle types
    d_type = candle_type_label(d)
    w_type = candle_type_label(w)
    m_type = candle_type_label(m)

    flags = compute_flags(d, w, m)

    # Decide effective bias for triggers if MIXED
    eff_bias = bias if bias in ("LONG", "SHORT") else "LONG"

    tf, entry, stop = best_trigger(eff_bias, d, w)

    d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
    w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
    m_inside = bool(flags.get("M_Inside"))

    trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

    # Plan metrics
    last_close = float(d["Close"].iloc[-1]) if "Close" in d.columns and len(d) else None
    atr = atr14(d)
    atrp = (atr / last_close * 100.0) if (atr is not None and last_close and last_close > 0) else None

    t1, t2 = targets_20_63(d, eff_bias)
    rr = rr_to_t2(entry, stop, t2, eff_bias)

    # -------------------------
    # Display
    # -------------------------
    st.subheader(f"{ticker} — STRAT Snapshot")

    a, b, c, dd = st.columns(4)
    with a:
        st.metric("Daily Type", d_type)
    with b:
        st.metric("Weekly Type", w_type)
    with c:
        st.metric("Monthly Type", m_type)
    with dd:
        st.metric("Trigger", trigger_status)

    st.subheader("Actionable Trigger (Inside Bar)")
    if entry is None or stop is None or tf is None:
        st.info("No Inside Bar trigger available right now (WAIT). That’s fine — your edge is waiting for the trigger.")
    else:
        if eff_bias == "LONG":
            st.success(
                f"**{tf} Trigger (LONG)** → Entry: **{_fmt(entry)}** (break IB High) | Stop: **{_fmt(stop)}** (below IB Low)"
            )
        else:
            st.error(
                f"**{tf} Trigger (SHORT)** → Entry: **{_fmt(entry)}** (break IB Low) | Stop: **{_fmt(stop)}** (above IB High)"
            )

    st.subheader("Plan Notes (simple, STRAT-friendly)")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Last Close", _fmt(last_close))
    with p2:
        st.metric("ATR(14)", _fmt(atr))
    with p3:
        st.metric("ATR%", _fmt(atrp, nd=2))
    with p4:
        st.metric("RR to T2", _fmt(rr, nd=2))

    st.write(
        f"**Targets ({eff_bias})** → "
        f"T1 (20d): **{_fmt(t1)}** | "
        f"T2 (63d): **{_fmt(t2)}**"
    )

    if eff_bias == "LONG":
        st.write("**Invalidation:** break below your **Stop** (or close-based stop if that’s your rule).")
    else:
        st.write("**Invalidation:** break above your **Stop** (or close-based stop if that’s your rule).")

    # Flags summary
    st.subheader("STRAT Flags (what’s currently true)")
    rows = []
    rows.append({"TF": "D", "Bull": bool(flags.get("D_Bull")), "Bear": bool(flags.get("D_Bear")), "Inside": bool(flags.get("D_Inside")),
                 "212Up": bool(flags.get("D_212Up")), "212Dn": bool(flags.get("D_212Dn"))})
    rows.append({"TF": "W", "Bull": bool(flags.get("W_Bull")), "Bear": bool(flags.get("W_Bear")), "Inside": bool(flags.get("W_Inside")),
                 "212Up": bool(flags.get("W_212Up")), "212Dn": bool(flags.get("W_212Dn"))})
    rows.append({"TF": "M", "Bull": bool(flags.get("M_Bull")), "Bear": bool(flags.get("M_Bear")), "Inside": bool(flags.get("M_Inside")),
                 "212Up": False, "212Dn": False})

    fdf = pd.DataFrame(rows)
    for col in ["Bull","Bear","Inside","212Up","212Dn"]:
        fdf[col] = fdf[col].apply(lambda v: "✅" if v else "")
    st.dataframe(fdf, use_container_width=True, hide_index=True)

    st.caption("If you want: next we’ll add a ‘Trade Grade’ that is STRAT-only (alignment + trigger quality + room to target).")
