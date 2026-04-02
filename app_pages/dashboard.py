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
    bias = row.get("_bias", "MIXED")

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
        score += 2 if (row.get("M_Bull", False) or row.get("M_Bear", False)) else 0
        score += 1 if (row.get("W_Bull", False) or row.get("W_Bear", False)) else 0

    score += 1 if row.get("W_Inside", False) else 0
    score += 1 if row.get("D_Inside", False) else 0

    if score >= 7:
        return "A+", score
    if score >= 5:
        return "A", score
    if score >= 3:
        return "B", score
    return "C", score


def _style_grades_df(df: pd.DataFrame) -> pd.DataFrame:
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
    st.caption("Overall STRAT Market Regime + A+ Leaders")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET REGIME
    # =========================
    st.subheader("Market Regime")

    market_rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)

        if d.empty:
            continue

        w = resample_ohlc(d, "W-FRI")
        m = resample_ohlc(d, "ME")  # ✅ FIXED

        flags = compute_flags(d, w, m)
        bull, bear = score_regime(flags)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "D_Type": candle_type_label(d),
            "W_Type": candle_type_label(w),
            "M_Type": candle_type_label(m),
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "M_Inside": flags.get("M_Inside", False),
        })

    market_df = pd.DataFrame(market_rows)
    st.dataframe(_checkify(market_df, ["D_Inside", "W_Inside", "M_Inside"]),
                 use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)

    st.write(f"Bias: {bias} | Strength: {strength}")

    # =========================
    # A+ LEADERS
    # =========================
    st.subheader("A+ Leaders")

    universe = list(MARKET_ETFS.values()) + list(SECTOR_ETFS.values())

    rows = []
    for sym in universe:
        d = get_hist(sym)
        if d.empty:
            continue

        w = resample_ohlc(d, "W-FRI")
        m = resample_ohlc(d, "ME")  # ✅ FIXED HERE TOO

        flags = compute_flags(d, w, m)
        bull, bear = score_regime(flags)

        row = {
            "Ticker": sym,
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": flags.get("D_Inside", False),
            "W_Inside": flags.get("W_Inside", False),
            "_bias": bias,
            "D_Bull": flags.get("D_Bull", False),
            "W_Bull": flags.get("W_Bull", False),
            "M_Bull": flags.get("M_Bull", False),
            "D_Bear": flags.get("D_Bear", False),
            "W_Bear": flags.get("W_Bear", False),
            "M_Bear": flags.get("M_Bear", False),
        }

        grade, score = _grade_row(row)
        row["SetupGrade"] = grade
        row["Score"] = score

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).head(15)

    st.dataframe(df, use_container_width=True, hide_index=True)
