# app_pages/scanner.py

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Tuple

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

# -----------------------
# Small helpers
# -----------------------
def _checkify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out


def _rotation_lists(sectors_df: pd.DataFrame, bias: str, n: int = 3):
    """
    STRAT-only 'rotation' = which sector ETFs are most aligned with the bias (IN)
    vs most opposed (OUT), based on BullScore/BearScore dominance.
    """
    df = sectors_df.copy()
    df["Diff"] = df["BullScore"] - df["BearScore"]  # + = bullish dominance

    if bias == "LONG":
        rot_in = df.sort_values(["Diff", "BullScore"], ascending=[False, False]).head(n)
        rot_out = df.sort_values(["Diff", "BearScore"], ascending=[True, False]).head(n)
    elif bias == "SHORT":
        # For short bias, "IN" means bearish dominance (most negative Diff)
        rot_in = df.sort_values(["Diff", "BearScore"], ascending=[True, False]).head(n)
        rot_out = df.sort_values(["Diff", "BullScore"], ascending=[False, False]).head(n)
    else:
        # MIXED: show extremes both ways
        rot_in = df.sort_values("Diff", ascending=False).head(n)
        rot_out = df.sort_values("Diff", ascending=True).head(n)

    return rot_in, rot_out


def _atr14(d: pd.DataFrame) -> Optional[float]:
    if d is None or d.empty or len(d) < 20:
        return None
    h, l, c = d["High"], d["Low"], d["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        return None
    return float(atr)


def _magnitude(
    bias: str,
    d: pd.DataFrame,
    entry: Optional[float],
    stop: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (Room, RR, ATR%, Compression)

    Room:
      LONG  -> entry to 63d high
      SHORT -> entry to 63d low

    RR:
      reward / risk to that 63d target (rough but useful)

    ATR%:
      ATR(14) / close * 100

    Compression:
      if last bar is inside bar: (bar_range / ATR)
    """
    if d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = _atr14(d)
    if atr is None or close <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100.0

    if entry is None or stop is None:
        return None, None, atrp, None

    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])

    if bias == "SHORT":
        target = lo63
        reward = max(0.0, entry - target)
        risk = max(0.0, stop - entry)
        room = reward
    else:
        target = hi63
        reward = max(0.0, target - entry)
        risk = max(0.0, entry - stop)
        room = reward

    rr = None
    if risk > 0:
        rr = reward / risk

    # Compression: last bar range vs ATR (nice for “tight” triggers)
    compression = None
    try:
        last_rng = float(d["High"].iloc[-1] - d["Low"].iloc[-1])
        compression = last_rng / atr if atr > 0 else None
    except Exception:
        compression = None

    return room, rr, atrp, compression


# -----------------------
# Page
# -----------------------
def show():
    st.title("Scanner (STRAT-only)")
    st.caption("Market Regime + Sector Rotation + Rotation IN/OUT + Sector Drilldown (now with Magnitude).")

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
            d_type = w_type = m_type = "n/a"
            flags = {}
            bull = bear = 0
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
    # DRILL INTO A SECTOR (bias-aware triggers + MAGNITUDE)
    # =========================
    st.subheader("Drill into a Sector (scan tickers inside that group)")

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

        flags = compute_flags(d_tf, w_tf, m_tf)

        tf, entry, stop = best_trigger(bias, d_tf, w_tf)

        d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
        w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
        m_inside = bool(flags.get("M_Inside"))

        trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

        room, rr, atrp, compression = _magnitude(bias, d_tf, entry, stop)

        rows.append({
            "Ticker": t,
            "D_Type": candle_type_label(d_tf),
            "W_Type": candle_type_label(w_tf),
            "M_Type": candle_type_label(m_tf),
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
            "Room": None if room is None else round(float(room), 2),
            "RR": None if rr is None else round(float(rr), 2),
            "ATR%": None if atrp is None else round(float(atrp), 2),
            "Compression": None if compression is None else round(float(compression), 2),
            "TriggerStatus": trigger_status,
        })

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        st.info("No data returned for this sector list right now.")
        return

    scan_df = _checkify(scan_df, ["D_Inside", "W_Inside", "M_Inside", "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"])

    # Put magnitude columns closer to Entry/Stop
    ordered_cols = [
        "Ticker",
        "D_Type", "W_Type", "M_Type",
        "D_Inside", "W_Inside", "M_Inside",
        "D_212Up", "W_212Up", "D_212Dn", "W_212Dn",
        "TriggerTF", "Entry", "Stop",
        "Room", "RR", "ATR%", "Compression",
        "TriggerStatus",
    ]
    ordered_cols = [c for c in ordered_cols if c in scan_df.columns]
    scan_df = scan_df[ordered_cols].copy()

    st.dataframe(scan_df, use_container_width=True, hide_index=True)

    st.caption(
        "Magnitude notes: Room/RR are rough targets to the 63-day extreme. "
        "Compression ~ (last bar range / ATR). Lower = tighter trigger."
    )

    st.caption("Trigger levels are bias-aware. LONG = break IB high / stop below. SHORT = break IB low / stop above.")
