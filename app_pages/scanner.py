# app_pages/scanner.py

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS, SECTOR_ETFS, SECTOR_TICKERS
from strat.core import (
    candle_type_label,
    candle_color_last,
    compute_flags,
    score_regime,
    market_bias_and_strength,
    best_trigger,
    setup_failed2m_2uW_actionableD,
)

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
        rot_in = df.sort_values(["Diff", "BullScore"], ascending=[False, False]).head(n)
        rot_out = df.sort_values(["Diff", "BearScore"], ascending=[True, False]).head(n)
    elif bias == "SHORT":
        rot_in = df.sort_values(["Diff", "BearScore"], ascending=[True, False]).head(n)
        rot_out = df.sort_values(["Diff", "BullScore"], ascending=[False, False]).head(n)
    else:
        rot_in = df.sort_values("Diff", ascending=False).head(n)
        rot_out = df.sort_values("Diff", ascending=True).head(n)

    return rot_in, rot_out

def scanner_main():
    st.title("Scanner (STRAT-only)")
    st.caption("Market Regime → Sector Rotation → Drilldown. Now includes Failed-2-Month setup scan.")

    topbar = st.columns([1, 1, 6])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET REGIME (STRAT-only)
    # =========================
    st.subheader("Market Regime (STRAT-only) — SPY / QQQ / IWM / DIA")

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
    market_df = _checkify(market_df, ["D_Inside", "W_Inside", "M_Inside"])
    st.dataframe(market_df, use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)

    if bias == "LONG":
        st.success(f"Bias: **LONG** 🟢 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
    elif bias == "SHORT":
        st.error(f"Bias: **SHORT** 🔴 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
    else:
        st.warning(f"Bias: **MIXED** 🟠 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")

    # =========================
    # SECTOR ROTATION (STRAT-only)
    # =========================
    st.subheader("Sector Rotation (STRAT-only) — ranked by bias")

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
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
            "D_212Up": flags.get("D_212Up", False),
            "W_212Up": flags.get("W_212Up", False),
            "D_212Dn": flags.get("D_212Dn", False),
            "W_212Dn": flags.get("W_212Dn", False),
        })

    sectors_df = pd.DataFrame(sector_rows)
    sectors_df["Diff"] = sectors_df["BullScore"] - sectors_df["BearScore"]

    if bias == "LONG":
        sectors_df = sectors_df.sort_values(["Diff", "BullScore"], ascending=[False, False])
    elif bias == "SHORT":
        sectors_df = sectors_df.sort_values(["Diff", "BearScore"], ascending=[True, False])
    else:
        sectors_df = sectors_df.sort_values(["Diff"], ascending=False)

    show_cols = [
        "Sector", "ETF", "D_Type", "W_Type", "M_Type",
        "BullScore", "BearScore", "Diff",
        "D_Inside", "W_Inside", "M_Inside",
        "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"
    ]
    out_df = sectors_df[show_cols].copy()
    out_df = _checkify(out_df, ["D_Inside", "W_Inside", "M_Inside", "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"])
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    # =========================
    # QUICK MARKET READ + ROTATION IN/OUT
    # =========================
    st.subheader("Quick Market Read + Rotation IN/OUT")
    rot_in, rot_out = _rotation_lists(sectors_df, bias, n=3)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔁 Rotation IN")
        for _, r in rot_in.iterrows():
            st.write(f"✅ **{r['Sector']}** ({r['ETF']}) — Bull {int(r['BullScore'])} / Bear {int(r['BearScore'])} | Diff {int(r['Diff'])}")
    with c2:
        st.markdown("### 🔁 Rotation OUT")
        for _, r in rot_out.iterrows():
            st.write(f"❌ **{r['Sector']}** ({r['ETF']}) — Bull {int(r['BullScore'])} / Bear {int(r['BearScore'])} | Diff {int(r['Diff'])}")

    # =========================
    # DRILL INTO A SECTOR + NEW SETUP FILTER
    # =========================
    st.subheader("Drill into a Sector (scan tickers inside that group)")

    with st.expander("Setup Scan Filters", expanded=True):
        failed2m_setup = st.checkbox(
            "Scan: Failed 2 Month (2D green) + Weekly 2U + Daily actionable",
            value=False
        )
        st.caption("Daily actionable = Daily Inside Day OR Daily 2-1-2 Up. Monthly must be 2D but GREEN. Weekly must be 2U.")

    sector_choice = st.selectbox("Choose sector:", options=list(SECTOR_TICKERS.keys()), index=0)
    tickers = SECTOR_TICKERS.get(sector_choice, [])
    st.write(f"Selected: **{sector_choice}** — tickers: **{len(tickers)}**")

    scan_n = st.slider("Scan count", 1, max(1, len(tickers)), value=min(15, len(tickers)))
    scan_list = tickers[:scan_n]

    rows = []
    for t in scan_list:
        d = get_hist(t)
        if d.empty:
            continue

        d_tf = d
        w_tf = resample_ohlc(d, "W-FRI")
        m_tf = resample_ohlc(d, "M")

        if w_tf is None or w_tf.empty or m_tf is None or m_tf.empty:
            continue

        # NEW: setup filter
        if failed2m_setup:
            if not setup_failed2m_2uW_actionableD(d_tf, w_tf, m_tf):
                continue

        flags = compute_flags(d_tf, w_tf, m_tf)

        tf, entry, stop = best_trigger(bias, d_tf, w_tf)

        d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
        w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
        m_inside = bool(flags.get("M_Inside"))

        trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

        rows.append({
            "Ticker": t,
            "M_Type": candle_type_label(m_tf),
            "M_Color": candle_color_last(m_tf),
            "W_Type": candle_type_label(w_tf),
            "D_Type": candle_type_label(d_tf),
            "D_Actionable": (candle_type_label(d_tf) == "1") or flags.get("D_212Up", False),
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
            "D_212Up": flags.get("D_212Up", False),
            "W_212Up": flags.get("W_212Up", False),
            "D_212Dn": flags.get("D_212Dn", False),
            "W_212Dn": flags.get("W_212Dn", False),
            "TriggerTF": tf if tf else "—",
            "Entry": None if entry is None else round(float(entry), 2),
            "Stop": None if stop is None else round(float(stop), 2),
            "TriggerStatus": trigger_status,
        })

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        st.info("No matches for this sector list (or the setup is rare right now). Try a different sector or raise Scan count.")
        return

    scan_df["D_Actionable"] = scan_df["D_Actionable"].apply(lambda v: "✅" if bool(v) else "")
    scan_df = _checkify(scan_df, ["D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"])

    st.dataframe(scan_df, use_container_width=True, hide_index=True)
    st.caption("Note: Monthly 2D-green is a 'failed 2 down' vibe. Weekly 2U suggests weekly momentum up. Daily actionable = Inside or 2-1-2 Up.")
