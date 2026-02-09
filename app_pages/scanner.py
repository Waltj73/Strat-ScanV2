# app_pages/scanner.py
# Step 12: A+ Setup Highlighting (STRAT-only)
# - Adds "In Force" columns (color-aware for 2U/2D/3)
# - Adds SetupGrade (A+, A, B, C) + styling highlight
# - Avoids pandas Styler type-hint import errors

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
    candle_color,   # <-- uses your core.py candle_color()
)


# -------------------------
# Helpers
# -------------------------
def _checkify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out


def _tf_in_force(df: pd.DataFrame) -> str:
    """
    STRAT-ish "in force" (simple, practical):
    - 2U + green close => Bull In Force
    - 2U + red close   => Bull (not in force)
    - 2D + red close   => Bear In Force
    - 2D + green close => Bear (not in force)
    - 3 uses candle color as force direction
    - 1 = Inside (neutral)
    """
    if df is None or df.empty or len(df) < 2:
        return "—"

    t = candle_type_label(df)
    cur = df.iloc[-1]
    col = candle_color(cur)

    if t == "1":
        return "Inside"
    if t == "2U":
        return "Bull IF" if col == "green" else "Bull"
    if t == "2D":
        return "Bear IF" if col == "red" else "Bear"
    if t == "3":
        if col == "green":
            return "Bull IF"
        if col == "red":
            return "Bear IF"
        return "3"
    return "—"


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


def _is_tf_supportive(force_label: str, bias: str) -> bool:
    if bias == "LONG":
        return force_label.startswith("Bull")
    if bias == "SHORT":
        return force_label.startswith("Bear")
    return False


def _setup_grade(bias: str, flags: dict, d_force: str, w_force: str, m_force: str, trigger_tf: str, entry, stop) -> tuple[str, int]:
    """
    A+ grading (simple + STRAT-relevant, no magic indicators):
    Points:
      +4 Trigger READY (entry/stop exist and TF is D or W)
      +2 Weekly force aligns with bias
      +2 Monthly force aligns with bias
      +1 2-1-2 in direction (D or W)
      +1 Inside Month present (compression)
      -2 Opposing Weekly force
      -2 Opposing Monthly force
    """
    if bias not in ("LONG", "SHORT"):
        return ("C", 0)

    score = 0

    trigger_ready = (trigger_tf in ("D", "W")) and (entry is not None) and (stop is not None)
    if trigger_ready:
        score += 4

    # Higher timeframe alignment
    if _is_tf_supportive(w_force, bias):
        score += 2
    if _is_tf_supportive(m_force, bias):
        score += 2

    # 2-1-2 bonus (directional)
    if bias == "LONG":
        if flags.get("W_212Up") or flags.get("D_212Up"):
            score += 1
    if bias == "SHORT":
        if flags.get("W_212Dn") or flags.get("D_212Dn"):
            score += 1

    # Compression bonus
    if flags.get("M_Inside"):
        score += 1

    # Penalize opposing HTF
    if bias == "LONG":
        if w_force.startswith("Bear"):
            score -= 2
        if m_force.startswith("Bear"):
            score -= 2
    if bias == "SHORT":
        if w_force.startswith("Bull"):
            score -= 2
        if m_force.startswith("Bull"):
            score -= 2

    # Grade mapping
    if score >= 8:
        return ("A+", score)
    if score >= 6:
        return ("A", score)
    if score >= 4:
        return ("B", score)
    return ("C", score)


def _style_df(df: pd.DataFrame):
    """
    Streamlit accepts Styler. Keep it simple + compatible.
    """
    if df is None or df.empty:
        return df

    def grade_style(val):
        if val == "A+":
            return "background-color: rgba(0, 200, 0, 0.25); font-weight: 800;"
        if val == "A":
            return "background-color: rgba(0, 200, 0, 0.12); font-weight: 700;"
        if val == "B":
            return "background-color: rgba(255, 200, 0, 0.18); font-weight: 700;"
        if val == "C":
            return "background-color: rgba(180, 180, 180, 0.15);"
        return ""

    def force_style(val):
        if isinstance(val, str) and val.endswith("IF"):
            return "font-weight: 800;"
        return ""

    styler = df.style
    if "SetupGrade" in df.columns:
        styler = styler.applymap(grade_style, subset=["SetupGrade"])
    for col in ["D_Force", "W_Force", "M_Force"]:
        if col in df.columns:
            styler = styler.applymap(force_style, subset=[col])

    return styler


# -------------------------
# Main page
# -------------------------
def scanner_main():
    st.title("Scanner (STRAT-only) — A+ Setup Highlighting")
    st.caption("Market regime • Sector rotation • Drilldown • In-Force • A+ grading")

    topbar = st.columns([1, 8])
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
            d_force = w_force = m_force = "—"
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")

            d_type = candle_type_label(d_tf)
            w_type = candle_type_label(w_tf)
            m_type = candle_type_label(m_tf)

            d_force = _tf_in_force(d_tf)
            w_force = _tf_in_force(w_tf)
            m_force = _tf_in_force(m_tf)

            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "D_Force": d_force,
            "W_Force": w_force,
            "M_Force": m_force,
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
    # SECTOR ROTATION
    # =========================
    st.subheader("Sector Rotation (STRAT-only) — ranked by bias")

    sector_rows = []
    for sector, etf in SECTOR_ETFS.items():
        d = get_hist(etf)

        if d.empty:
            flags = {}
            bull = bear = 0
            d_type = w_type = m_type = "n/a"
            d_force = w_force = m_force = "—"
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")

            d_type = candle_type_label(d_tf)
            w_type = candle_type_label(w_tf)
            m_type = candle_type_label(m_tf)

            d_force = _tf_in_force(d_tf)
            w_force = _tf_in_force(w_tf)
            m_force = _tf_in_force(m_tf)

            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "D_Force": d_force,
            "W_Force": w_force,
            "M_Force": m_force,
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
        "Sector", "ETF",
        "D_Type", "W_Type", "M_Type",
        "D_Force", "W_Force", "M_Force",
        "BullScore", "BearScore", "Diff",
        "D_Inside", "W_Inside", "M_Inside",
        "D_212Up", "W_212Up", "D_212Dn", "W_212Dn",
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
    # DRILLDOWN + A+ GRADING
    # =========================
    st.subheader("Drill into a Sector (A+ setups highlighted)")

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

        d_force = _tf_in_force(d_tf)
        w_force = _tf_in_force(w_tf)
        m_force = _tf_in_force(m_tf)

        # Ready flags (bias-aware inside-bar triggers)
        d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
        w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
        m_inside = bool(flags.get("M_Inside"))

        trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

        grade, grade_score = _setup_grade(
            bias=bias,
            flags=flags,
            d_force=d_force,
            w_force=w_force,
            m_force=m_force,
            trigger_tf=tf,
            entry=entry,
            stop=stop,
        )

        rows.append({
            "SetupGrade": grade,
            "GradeScore": grade_score,
            "Ticker": t,
            "D_Type": candle_type_label(d_tf),
            "W_Type": candle_type_label(w_tf),
            "M_Type": candle_type_label(m_tf),
            "D_Force": d_force,
            "W_Force": w_force,
            "M_Force": m_force,
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
        st.info("No data returned for this sector list right now.")
        return

    # Put A+ at top automatically
    grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3}
    scan_df["_g"] = scan_df["SetupGrade"].map(lambda x: grade_order.get(x, 9))
    scan_df = scan_df.sort_values(["_g", "GradeScore"], ascending=[True, False]).drop(columns=["_g"])

    scan_df = _checkify(scan_df, ["D_Inside", "W_Inside", "M_Inside", "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"])

    # Styled render (A+ highlight + bold in-force)
    st.dataframe(_style_df(scan_df), use_container_width=True, hide_index=True)

    st.caption("A+ logic: Trigger READY + weekly/monthly alignment + 2-1-2 bonus + compression bonus − opposing HTF penalty.")
    st.caption("In Force: 2D can be green (Bear, not in force) vs red (Bear IF). Same idea for 2U.")


# Back-compat if you call show() anywhere
def show():
    return scanner_main()
