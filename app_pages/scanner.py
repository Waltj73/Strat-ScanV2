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
    candle_type,
    candle_color,
    compute_flags,
    score_regime,
    market_bias_and_strength,
    best_trigger,
)

# -------------------------
# UI helpers
# -------------------------
def _checkify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out

def _fmt_num(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

# -------------------------
# STRAT: In-Force labeling
# -------------------------
def _in_force_label(df: pd.DataFrame) -> str:
    """
    In-Force (common STRAT lingo):
    - 2D but GREEN candle => IF↑ (down candle "in-force" to upside)
    - 2U but RED candle   => IF↓
    Else blank.
    """
    if df is None or df.empty or len(df) < 2:
        return ""
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    t = candle_type(cur, prev)          # "1", "2U", "2D", "3"
    col = candle_color(cur)             # "green", "red", "doji"

    if t == "2D" and col == "green":
        return "IF↑"
    if t == "2U" and col == "red":
        return "IF↓"
    return ""

def _bias_ok_for_inforce(bias: str, if_label: str) -> bool:
    if bias == "LONG":
        return if_label == "IF↑"
    if bias == "SHORT":
        return if_label == "IF↓"
    return False

# -------------------------
# A+ scoring
# -------------------------
def _a_plus_score(
    bias: str,
    flags: dict,
    d_type: str,
    w_type: str,
    m_type: str,
    d_if: str,
    w_if: str,
    m_if: str,
    trigger_tf: str,
    entry: float | None,
    stop: float | None,
) -> tuple[int, str, str]:
    """
    Simple, STRAT-only A+ scoring.
    Goal: highlight "tradeable now" setups.
    """
    score = 0
    notes = []

    if bias not in ("LONG", "SHORT"):
        return 0, "C", "Bias MIXED (no A+ grades)"

    # 1) Alignment (big weight)
    if bias == "LONG":
        if flags.get("M_Bull"): score += 25; notes.append("M align")
        if flags.get("W_Bull"): score += 20; notes.append("W align")
        if flags.get("D_Bull"): score += 10; notes.append("D align")
    else:
        if flags.get("M_Bear"): score += 25; notes.append("M align")
        if flags.get("W_Bear"): score += 20; notes.append("W align")
        if flags.get("D_Bear"): score += 10; notes.append("D align")

    # 2) In-Force (momentum tells you the "wrong" candle is pushing your way)
    if _bias_ok_for_inforce(bias, m_if):
        score += 18; notes.append("M in-force")
    if _bias_ok_for_inforce(bias, w_if):
        score += 14; notes.append("W in-force")
    if _bias_ok_for_inforce(bias, d_if):
        score += 8; notes.append("D in-force")

    # 3) Trigger readiness (weekly > daily)
    trigger_ready = (entry is not None and stop is not None and trigger_tf in ("W", "D"))
    if trigger_ready and trigger_tf == "W":
        score += 20; notes.append("W trigger ready")
    elif trigger_ready and trigger_tf == "D":
        score += 12; notes.append("D trigger ready")
    else:
        notes.append("No trigger")

    # 4) Compression / patterns (bonuses)
    if flags.get("W_Inside"): score += 8; notes.append("W inside")
    if flags.get("D_Inside"): score += 4; notes.append("D inside")

    if bias == "LONG":
        if flags.get("W_212Up"): score += 8; notes.append("W 212↑")
        if flags.get("D_212Up"): score += 4; notes.append("D 212↑")
    else:
        if flags.get("W_212Dn"): score += 8; notes.append("W 212↓")
        if flags.get("D_212Dn"): score += 4; notes.append("D 212↓")

    # Clamp
    score = int(max(0, min(100, score)))

    # Grade
    if score >= 80:
        grade = "A+"
    elif score >= 65:
        grade = "A"
    elif score >= 50:
        grade = "B"
    else:
        grade = "C"

    # A+ should also basically require a trigger
    if grade == "A+" and not trigger_ready:
        grade = "A"
        notes.append("Downgrade: no trigger")

    return score, grade, " | ".join(notes)

def _style_grades(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Highlight A+ rows.
    """
    def row_style(row):
        if str(row.get("Grade", "")) == "A+":
            return ["background-color: rgba(0, 255, 0, 0.10); font-weight: 700;"] * len(row)
        return [""] * len(row)

    sty = df.style.apply(row_style, axis=1)

    # Make grades visually pop
    def grade_color(v):
        if v == "A+":
            return "background-color: rgba(0, 255, 0, 0.20); font-weight: 800;"
        if v == "A":
            return "background-color: rgba(0, 200, 255, 0.12); font-weight: 700;"
        if v == "B":
            return "background-color: rgba(255, 200, 0, 0.12); font-weight: 700;"
        return ""

    if "Grade" in df.columns:
        sty = sty.applymap(grade_color, subset=["Grade"])

    return sty

# -------------------------
# Rotation IN/OUT (STRAT-only dominance)
# -------------------------
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

# -------------------------
# Main page
# -------------------------
def scanner_main():
    st.title("Scanner (STRAT-only) — A+ Highlighting")
    st.caption("Now highlights A+ candidates using STRAT-only alignment + in-force + trigger readiness.")

    topbar = st.columns([1, 6])
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
        if d.empty:
            d_type = w_type = m_type = "n/a"
            flags = {}
            bull = bear = 0
            d_if = w_if = m_if = ""
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")

            d_type = candle_type_label(d_tf)
            w_type = candle_type_label(w_tf)
            m_type = candle_type_label(m_tf)

            d_if = _in_force_label(d_tf)
            w_if = _in_force_label(w_tf)
            m_if = _in_force_label(m_tf)

            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "D_IF": d_if,
            "W_IF": w_if,
            "M_IF": m_if,
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
            d_if = w_if = m_if = ""
        else:
            d_tf = d
            w_tf = resample_ohlc(d, "W-FRI")
            m_tf = resample_ohlc(d, "M")

            d_type = candle_type_label(d_tf)
            w_type = candle_type_label(w_tf)
            m_type = candle_type_label(m_tf)

            d_if = _in_force_label(d_tf)
            w_if = _in_force_label(w_tf)
            m_if = _in_force_label(m_tf)

            flags = compute_flags(d_tf, w_tf, m_tf)
            bull, bear = score_regime(flags)

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "D_IF": d_if,
            "W_IF": w_if,
            "M_IF": m_if,
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
        "D_IF", "W_IF", "M_IF",
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
    # DRILL INTO A SECTOR (A+ highlighting)
    # =========================
    st.subheader("Drill into a Sector (A+ Highlighting)")

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

        d_type = candle_type_label(d_tf)
        w_type = candle_type_label(w_tf)
        m_type = candle_type_label(m_tf)

        d_if = _in_force_label(d_tf)
        w_if = _in_force_label(w_tf)
        m_if = _in_force_label(m_tf)

        tf, entry, stop = best_trigger(bias, d_tf, w_tf)

        d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
        w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
        m_inside = bool(flags.get("M_Inside"))

        trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

        score, grade, notes = _a_plus_score(
            bias=bias,
            flags=flags,
            d_type=d_type, w_type=w_type, m_type=m_type,
            d_if=d_if, w_if=w_if, m_if=m_if,
            trigger_tf=(tf if tf else ""),
            entry=entry,
            stop=stop
        )

        rows.append({
            "Ticker": t,
            "A+Score": score,
            "Grade": grade,
            "Notes": notes,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "D_IF": d_if,
            "W_IF": w_if,
            "M_IF": m_if,
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

    # sort best first
    scan_df = scan_df.sort_values(["A+Score", "Grade"], ascending=[False, True]).copy()

    # Show Top A+ first
    st.markdown("### ⭐ Top A+ Candidates")
    top_a = scan_df[scan_df["Grade"] == "A+"].head(8)
    if top_a.empty:
        st.info("No A+ setups right now (that’s normal). Look for A’s with W trigger ready + IF in your bias.")
    else:
        display_top = top_a[[
            "Ticker","A+Score","Grade","TriggerTF","Entry","Stop","TriggerStatus","W_Type","W_IF","M_Type","M_IF","Notes"
        ]].copy()
        st.dataframe(_style_grades(display_top), use_container_width=True, hide_index=True)

    st.markdown("### All Scan Results (A+ highlighted)")
    show = scan_df[[
        "Ticker","A+Score","Grade",
        "D_Type","W_Type","M_Type",
        "D_IF","W_IF","M_IF",
        "D_Inside","W_Inside","M_Inside",
        "D_212Up","W_212Up","D_212Dn","W_212Dn",
        "TriggerTF","Entry","Stop","TriggerStatus",
        "Notes"
    ]].copy()

    show = _checkify(show, ["D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"])
    st.dataframe(_style_grades(show), use_container_width=True, hide_index=True)

    st.caption(
        "In-Force: 2D green = IF↑ (bullish pressure), 2U red = IF↓ (bearish pressure). "
        "A+ requires bias (not MIXED) and a usable trigger (W preferred)."
    )
