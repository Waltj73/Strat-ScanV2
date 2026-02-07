# app_pages/ticker_analyzer.py
# STRAT-only Ticker Analyzer (no RSI / no RS / no rotation systems)

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from strat.core import (
    candle_type_label,
    compute_flags,
    best_trigger,
    market_bias_and_strength,
)

from config.universe import MARKET_ETFS


def _check(v: bool) -> str:
    return "✅" if bool(v) else "—"


def _bias_badge(bias: str) -> str:
    if bias == "LONG":
        return "🟢 LONG"
    if bias == "SHORT":
        return "🔴 SHORT"
    return "🟠 MIXED"


def _trigger_status(flags: dict, tf: str | None, entry, stop) -> str:
    d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
    w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
    m_inside = bool(flags.get("M_Inside"))
    return f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"


def _market_bias_strat_only() -> tuple[str, int, int]:
    """
    Compute STRAT bias from SPY/QQQ/IWM/DIA using strat.core.market_bias_and_strength
    without needing RSI/RS systems.
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

            # market_bias_and_strength expects BullScore/BearScore in each row;
            # your strat.core.score_regime is already used in Scanner, but we can
            # compute bias from inside compute_flags via best_trigger logic + types.
            # To keep it consistent with your Scanner, we’ll infer scores similarly:
            bull = 0
            bear = 0
            bull += 3 if flags.get("M_Bull") else 0
            bull += 2 if flags.get("W_Bull") else 0
            bull += 1 if flags.get("D_Bull") else 0
            bear += 3 if flags.get("M_Bear") else 0
            bear += 2 if flags.get("W_Bear") else 0
            bear += 1 if flags.get("D_Bear") else 0
            bull += 2 if flags.get("W_212Up") else 0
            bull += 1 if flags.get("D_212Up") else 0
            bear += 2 if flags.get("W_212Dn") else 0
            bear += 1 if flags.get("D_212Dn") else 0

        row = {"Market": name, "ETF": sym, "BullScore": bull, "BearScore": bear}
        market_rows.append(row)

    bias, strength, diff = market_bias_and_strength(market_rows)
    return bias, strength, diff


def analyzer_main():
    st.title("🔎 Ticker Analyzer (STRAT-only)")
    st.caption("D/W/M candle types → STRAT flags → actionable trigger levels → READY today / week / month.")

    topbar = st.columns([1, 6])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # ===== Market Bias (STRAT-only) =====
    with st.expander("Market Bias (STRAT-only)", expanded=True):
        bias, strength, diff = _market_bias_strat_only()
        st.write(f"Bias: **{_bias_badge(bias)}** | Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
        st.caption("This is the same STRAT regime idea as your Scanner—no RSI/RS involved.")

    # ===== Inputs =====
    c1, c2 = st.columns([1.2, 1.2])
    with c1:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    with c2:
        # If market is MIXED, let you force direction for trigger levels
        forced = st.selectbox(
            "Trigger Direction (used for Entry/Stop)",
            ["AUTO (market bias)", "LONG", "SHORT"],
            index=0
        )

    if not ticker:
        st.info("Type a ticker to analyze.")
        return

    d = get_hist(ticker)
    if d.empty:
        st.warning("No data returned (bad ticker or yfinance empty). Try another symbol.")
        return

    d_tf = d
    w_tf = resample_ohlc(d, "W-FRI")
    m_tf = resample_ohlc(d, "M")

    # ===== Candle Types =====
    d_type = candle_type_label(d_tf)
    w_type = candle_type_label(w_tf)
    m_type = candle_type_label(m_tf)

    # ===== Flags =====
    flags = compute_flags(d_tf, w_tf, m_tf)

    # Direction choice
    if forced == "LONG":
        eff_bias = "LONG"
    elif forced == "SHORT":
        eff_bias = "SHORT"
    else:
        eff_bias = bias if bias in ("LONG", "SHORT") else "LONG"

    # Trigger levels (weekly first, then daily) based on your strat.core.best_trigger
    tf, entry, stop = best_trigger(eff_bias, d_tf, w_tf)
    entry_r = None if entry is None else round(float(entry), 2)
    stop_r = None if stop is None else round(float(stop), 2)

    # READY labels
    ready_today = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
    ready_week = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
    ready_month = bool(flags.get("M_Inside"))

    # ===== Header / Summary =====
    st.subheader(f"{ticker} — STRAT Summary")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Direction", eff_bias)
    with s2:
        st.metric("D / W / M Types", f"{d_type} / {w_type} / {m_type}")
    with s3:
        st.metric("READY Today", "✅" if ready_today else "—")
    with s4:
        st.metric("READY This Week", "✅" if ready_week else "—")

    # Month readiness (informational)
    st.write(f"**READY This Month (Monthly Inside):** {'✅' if ready_month else '—'}")

    # ===== Trigger Box =====
    st.subheader("🎯 Actionable Trigger Levels")
    st.write(_trigger_status(flags, tf, entry, stop))

    if entry_r is None or stop_r is None or tf is None:
        st.info("No actionable Inside Bar trigger found (Daily/Weekly). That’s normal — **WAIT** is a valid output.")
    else:
        if eff_bias == "LONG":
            st.success(
                f"**{tf} Trigger (LONG):** Entry **{entry_r}** (break IB HIGH) | Stop **{stop_r}** (below IB LOW)"
            )
        else:
            st.error(
                f"**{tf} Trigger (SHORT):** Entry **{entry_r}** (break IB LOW) | Stop **{stop_r}** (above IB HIGH)"
            )

    # ===== Flag Checklist =====
    st.subheader("✅ STRAT Conditions (what’s currently true)")

    chk = pd.DataFrame([{
        "TF": "Daily",
        "Inside": _check(flags.get("D_Inside")),
        "Bull": _check(flags.get("D_Bull")),
        "Bear": _check(flags.get("D_Bear")),
        "2-1-2 Up": _check(flags.get("D_212Up")),
        "2-1-2 Down": _check(flags.get("D_212Dn")),
    },{
        "TF": "Weekly",
        "Inside": _check(flags.get("W_Inside")),
        "Bull": _check(flags.get("W_Bull")),
        "Bear": _check(flags.get("W_Bear")),
        "2-1-2 Up": _check(flags.get("W_212Up")),
        "2-1-2 Down": _check(flags.get("W_212Dn")),
    },{
        "TF": "Monthly",
        "Inside": _check(flags.get("M_Inside")),
        "Bull": _check(flags.get("M_Bull")),
        "Bear": _check(flags.get("M_Bear")),
        "2-1-2 Up": "—",
        "2-1-2 Down": "—",
    }])

    st.dataframe(chk, use_container_width=True, hide_index=True)

    # ===== What to do next =====
    st.subheader("📌 What You Do With This (STRAT-only)")
    if ready_week:
        st.success("Weekly is READY → this is the cleanest trigger. Consider prioritizing this over Daily.")
    elif ready_today:
        st.warning("Daily is READY → valid, but Weekly still preferred if/when it sets up.")
    else:
        st.info("No trigger → WAIT. Let it form a Daily or Weekly Inside Bar for clean Entry/Stop levels.")

    with st.expander("Show raw recent bars (D/W/M)"):
        st.markdown("**Daily (last 12):**")
        st.dataframe(d_tf.tail(12), use_container_width=True)
        st.markdown("**Weekly (last 12):**")
        st.dataframe(w_tf.tail(12), use_container_width=True)
        st.markdown("**Monthly (last 12):**")
        st.dataframe(m_tf.tail(12), use_container_width=True)
