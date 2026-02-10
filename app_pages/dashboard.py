# app_pages/dashboard.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS, SECTOR_ETFS, SECTOR_TICKERS
from strat.core import (
    candle_type_label,
    compute_flags,
    score_regime,
    market_bias_and_strength,
)

# -----------------------
# Helpers
# -----------------------
def _checkify(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out


def _grade_row(row: Dict) -> Tuple[str, int]:
    """
    Simple STRAT A+ grading heuristic (safe + explainable).
    You can tune weights later.

    Score components:
      + 3 if Monthly aligns with bias (bull/bear)
      + 2 if Weekly aligns with bias
      + 1 if Daily aligns with bias
      + 1 if W inside
      + 1 if D inside
    """
    bias = row.get("_bias", "MIXED")

    bull = int(row.get("BullScore", 0))
    bear = int(row.get("BearScore", 0))
    d_inside = bool(row.get("D_Inside", False))
    w_inside = bool(row.get("W_Inside", False))

    score = 0
    if bias == "LONG":
        score += 3 if row.get("M_Bull", False) else 0
        score += 2 if row.get("W_Bull", False) else 0
        score += 1 if row.get("D_Bull", False) else 0
    elif bias == "SHORT":
        score += 3 if row.get("M_Bear", False) else 0
        score += 2 if row.get("W_Bear", False) else 0
        score += 1 if row.get("D_Bear", False) else 0
    else:
        # mixed: reward alignment either way but less “conviction”
        score += 2 if (row.get("M_Bull", False) or row.get("M_Bear", False)) else 0
        score += 1 if (row.get("W_Bull", False) or row.get("W_Bear", False)) else 0

    score += 1 if w_inside else 0
    score += 1 if d_inside else 0

    # Grade mapping
    if score >= 7:
        return "A+", score
    if score >= 5:
        return "A", score
    if score >= 3:
        return "B", score
    return "C", score


def _style_grades_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep it dependency-free: we’ll just add an emoji column.
    out = df.copy()
    out["GradeTag"] = out["SetupGrade"].map(
        {"A+": "🔥 A+", "A": "✅ A", "B": "🟡 B", "C": "⚪ C"}
    ).fillna(out["SetupGrade"])
    return out


# -----------------------
# Main Page
# -----------------------
def dashboard_main():
    st.title("Dashboard (STRAT-only)")
    st.caption("Overall STRAT Market Regime + A+ Leaders (with chart viewer)")

    topbar = st.columns([1, 7])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET REGIME
    # =========================
    st.subheader("Market Regime (STRAT-only) — SPY / QQQ / IWM / DIA")

    market_rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
        if d is None or d.empty:
            flags = {}
            bull = bear = 0
            d_type = w_type = m_type = "n/a"
        else:
            w = resample_ohlc(d, "W-FRI")
            m = resample_ohlc(d, "M")
            flags = compute_flags(d, w, m)
            bull, bear = score_regime(flags)
            d_type = candle_type_label(d)
            w_type = candle_type_label(w)
            m_type = candle_type_label(m)

        market_rows.append(
            {
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
            }
        )

    market_df = pd.DataFrame(market_rows)
    st.dataframe(_checkify(market_df, ["D_Inside", "W_Inside", "M_Inside"]),
                 use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)

    if bias == "LONG":
        st.success(f"Bias: **LONG** 🟢 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
    elif bias == "SHORT":
        st.error(f"Bias: **SHORT** 🔴 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
    else:
        st.warning(f"Bias: **MIXED** 🟠 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")

    st.divider()

    # =========================
    # A+ LEADERS (overall market)
    # =========================
    st.subheader("A+ Market Leaders (overall market, not sector-limited)")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        top_n = st.slider("Top count", 5, 30, 15)
    with c2:
        include_sector_tickers = st.toggle("Include sector tickers", value=True)
    with c3:
        max_per_sector = st.slider("Max tickers per sector", 3, 50, 10)

    # Build universe
    universe = []
    universe += list(MARKET_ETFS.values())
    universe += list(SECTOR_ETFS.values())

    if include_sector_tickers:
        for sector, tlist in SECTOR_TICKERS.items():
            universe += tlist[:max_per_sector]

    # De-dupe while preserving order
    seen = set()
    universe = [x for x in universe if not (x in seen or seen.add(x))]

    rows = []
    for sym in universe:
        d = get_hist(sym)
        if d is None or d.empty:
            continue

        w = resample_ohlc(d, "W-FRI")
        m = resample_ohlc(d, "M")
        flags = compute_flags(d, w, m)
        bull, bear = score_regime(flags)

        row = {
            "Ticker": sym,
            "D_Type": candle_type_label(d),
            "W_Type": candle_type_label(w),
            "M_Type": candle_type_label(m),
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
            # keep flags for grading:
            "D_Bull": flags.get("D_Bull", False),
            "W_Bull": flags.get("W_Bull", False),
            "M_Bull": flags.get("M_Bull", False),
            "D_Bear": flags.get("D_Bear", False),
            "W_Bear": flags.get("W_Bear", False),
            "M_Bear": flags.get("M_Bear", False),
            "_bias": bias,
        }

        grade, score = _grade_row(row)
        row["SetupGrade"] = grade
        row["GradeScore"] = score

        rows.append(row)

    if not rows:
        st.info("No leaders found (data fetch may be empty right now).")
        return

    df = pd.DataFrame(rows)

    # Sort best first
    grade_rank = {"A+": 3, "A": 2, "B": 1, "C": 0}
    df["_gr"] = df["SetupGrade"].map(grade_rank).fillna(0)
    df = df.sort_values(["_gr", "GradeScore", "BullScore"], ascending=[False, False, False]).head(top_n)

    show_cols = [
        "SetupGrade", "GradeScore", "Ticker",
        "D_Type", "W_Type", "M_Type",
        "BullScore", "BearScore",
        "D_Inside", "W_Inside", "M_Inside",
    ]
    out = df[show_cols].copy()
    out = _checkify(out, ["D_Inside", "W_Inside", "M_Inside"])
    out = _style_grades_df(out)

    st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption(
        "This ranks the best STRAT-style setups across the whole market universe "
        "(market ETFs + sector ETFs + optional sector constituents)."
    )

    # =========================
    # Chart Viewer (no Plotly)
    # =========================
    st.markdown("### Chart viewer (daily close)")

    picked = st.selectbox("View chart for:", df["Ticker"].tolist(), index=0)

    bars = get_hist(picked)
    if bars is None or bars.empty:
        st.warning("No data for that ticker.")
        return

    bars = bars.tail(220).copy()
    # Streamlit-native chart (fast + no dependencies)
    st.line_chart(bars["Close"], height=320)

    st.caption("Tip: If you want true candles + Strat trigger lines, we can add Plotly later (optional).")


# Back-compat
def show():
    return dashboard_main()
