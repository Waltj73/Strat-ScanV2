# app_pages/scanner.py

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


# -------------------------
# Small utilities
# -------------------------
def _checkify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "✅" if bool(v) else "")
    return out


def _rotation_lists(sectors_df: pd.DataFrame, bias: str, n: int = 3):
    """
    STRAT-only "rotation" proxy:
    - LONG: most bullish dominance (Diff high) = IN
    - SHORT: most bearish dominance (Diff low/negative) = IN
    """
    df = sectors_df.copy()
    if "Diff" not in df.columns:
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


def _atr14(d: pd.DataFrame) -> Optional[float]:
    if d is None or d.empty or len(d) < 20:
        return None
    h = d["High"].astype(float)
    l = d["Low"].astype(float)
    c = d["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    if atr is None or not np.isfinite(atr) or atr <= 0:
        return None
    return float(atr)


def _magnitude_to_target(
    bias: str,
    d: pd.DataFrame,
    entry: Optional[float],
    stop: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (RR, ATR%, RoomToTarget, Compression)
    - Target = 63d high (LONG) or 63d low (SHORT)
    - Room = distance from entry to that target (>=0)
    - Compression = last bar range / ATR (lower is tighter)
    """
    if d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = _atr14(d)
    if atr is None or close <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100.0

    # compression: last bar range vs ATR
    last_rng = float(d["High"].iloc[-1] - d["Low"].iloc[-1])
    compression = (last_rng / atr) if atr > 0 else None

    if entry is None or stop is None:
        return None, atrp, None, compression

    hi63 = float(d["High"].rolling(63).max().iloc[-1])
    lo63 = float(d["Low"].rolling(63).min().iloc[-1])

    if bias == "SHORT":
        target = lo63
        room = max(0.0, entry - target)
        risk = stop - entry
        reward = entry - target
    else:
        target = hi63
        room = max(0.0, target - entry)
        risk = entry - stop
        reward = target - entry

    rr = None
    if risk > 0 and reward > 0:
        rr = reward / risk

    return (None if rr is None else float(rr)), float(atrp), float(room), (None if compression is None else float(compression))


def _alignment_ok(bias: str, flags: dict) -> bool:
    # Simple STRAT alignment proxy
    if bias == "LONG":
        return bool(flags.get("M_Bull") or flags.get("W_Bull"))
    if bias == "SHORT":
        return bool(flags.get("M_Bear") or flags.get("W_Bear"))
    return False


def _grade_setup(
    bias: str,
    flags: dict,
    entry: Optional[float],
    stop: Optional[float],
    rr: Optional[float],
    room: Optional[float],
    atrp: Optional[float],
    compression: Optional[float],
) -> str:
    """
    A/B/C rating:
    A = aligned + trigger + (RR>=2 OR big room) + tighter compression
    B = trigger + (some alignment OR RR>=1.5)
    C = everything else
    """
    has_trigger = (entry is not None and stop is not None)
    aligned = _alignment_ok(bias, flags)

    room_ok = (room is not None and room >= 0)
    rr_ok_a = (rr is not None and rr >= 2.0)
    rr_ok_b = (rr is not None and rr >= 1.5)

    tight = (compression is not None and compression <= 0.9)  # tighter-than-ATR bar
    atr_ok = (atrp is not None and atrp >= 1.0)  # avoids dead tickers

    if aligned and has_trigger and atr_ok and (rr_ok_a or (room_ok and room >= 2.0)) and tight:
        return "A"
    if has_trigger and (aligned or rr_ok_b):
        return "B"
    return "C"


def _grade_style(val: str) -> str:
    if val == "A":
        return "background-color: #114b2b; color: white; font-weight: 700;"
    if val == "B":
        return "background-color: #5a4b11; color: white; font-weight: 700;"
    return "background-color: #5a1111; color: white; font-weight: 700;"


def _writeup_card(row: dict, bias: str):
    """
    Short, STRAT-native writeup for a ticker row from scan_df.
    """
    t = row["Ticker"]
    grade = row.get("Grade", "—")
    trig = row.get("TriggerStatus", "")
    tf = row.get("TriggerTF", "—")
    entry = row.get("Entry")
    stop = row.get("Stop")
    room = row.get("Room")
    rr = row.get("RR")
    atrp = row.get("ATR%", None)
    comp = row.get("Compression", None)

    st.markdown(f"#### {t} — Grade **{grade}** | Bias **{bias}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Types:** D {row.get('D_Type','—')} | W {row.get('W_Type','—')} | M {row.get('M_Type','—')}")
    with c2:
        st.write(f"**Trigger:** {trig}")
        st.write(f"**TF:** {tf}")
    with c3:
        st.write(f"**Entry:** {entry if entry is not None else '—'}")
        st.write(f"**Stop:** {stop if stop is not None else '—'}")
    with c4:
        st.write(f"**Room (to 63d):** {room if room is not None else '—'}")
        st.write(f"**RR:** {rr if rr is not None else '—'}")

    notes = []
    if bias in ("LONG", "SHORT"):
        notes.append("Aligned ✅" if _alignment_ok(bias, row) else "Alignment weak ⚠️")

    if row.get("W_Inside") == "✅":
        notes.append("Weekly inside = best trigger quality ✅")
    elif row.get("D_Inside") == "✅":
        notes.append("Daily inside = valid trigger ✅")
    else:
        notes.append("No inside bar trigger → WAIT")

    if comp is not None:
        if comp <= 0.6:
            notes.append("Very tight compression (good) ✅")
        elif comp <= 0.9:
            notes.append("Tight compression ✅")
        else:
            notes.append("Loose bar (less compression) ⚠️")

    if atrp is not None:
        notes.append(f"ATR% ~ {atrp:.2f}%")

    st.write("**Notes:** " + " | ".join(notes))


# -------------------------
# MAIN PAGE
# -------------------------
def show():
    st.title("Scanner (STRAT-only)")
    st.caption("Market regime → sector dominance → drilldown. Now includes A/B/C rating + watchlist writeups.")

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
        if d is None or d.empty:
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
        if d is None or d.empty:
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
    # DRILL INTO A SECTOR
    # =========================
    st.subheader("Drill into a Sector (scan tickers inside that group)")

    left, right = st.columns([2, 1])
    with left:
        sector_choice = st.selectbox("Choose sector:", options=list(SECTOR_TICKERS.keys()), index=0)
    with right:
        scan_n = st.slider("Scan count", 1, max(1, len(SECTOR_TICKERS.get(sector_choice, []))), value=min(15, len(SECTOR_TICKERS.get(sector_choice, []))))

    tickers = SECTOR_TICKERS.get(sector_choice, [])
    st.write(f"Selected: **{sector_choice}** — tickers: **{len(tickers)}**")
    scan_list = tickers[:scan_n]

    rows = []
    for t in scan_list:
        d = get_hist(t)
        if d is None or d.empty:
            continue

        d_tf = d
        w_tf = resample_ohlc(d, "W-FRI")
        m_tf = resample_ohlc(d, "M")

        flags = compute_flags(d_tf, w_tf, m_tf)

        # bias-aware trigger (your best_trigger should already handle LONG/SHORT)
        tf, entry, stop = best_trigger(bias, d_tf, w_tf)

        rr, atrp, room, compression = _magnitude_to_target(bias if bias in ("LONG","SHORT") else "LONG", d_tf, entry, stop)

        d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
        w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
        m_inside = bool(flags.get("M_Inside"))
        trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

        # build a single dict that includes flags so grade logic can read M/W bull/bear
        full = {}
        full.update(flags)
        full["Ticker"] = t
        full["TriggerTF"] = tf if tf else "—"
        full["Entry"] = None if entry is None else round(float(entry), 2)
        full["Stop"] = None if stop is None else round(float(stop), 2)
        full["Room"] = None if room is None else round(float(room), 2)
        full["RR"] = None if rr is None else round(float(rr), 2)
        full["ATR%"] = None if atrp is None else round(float(atrp), 2)
        full["Compression"] = None if compression is None else round(float(compression), 2)
        full["TriggerStatus"] = trigger_status

        full["D_Type"] = candle_type_label(d_tf)
        full["W_Type"] = candle_type_label(w_tf)
        full["M_Type"] = candle_type_label(m_tf)

        full["D_Inside"] = bool(flags.get("D_Inside", False))
        full["W_Inside"] = bool(flags.get("W_Inside", False))
        full["M_Inside"] = bool(flags.get("M_Inside", False))
        full["D_212Up"] = bool(flags.get("D_212Up", False))
        full["W_212Up"] = bool(flags.get("W_212Up", False))
        full["D_212Dn"] = bool(flags.get("D_212Dn", False))
        full["W_212Dn"] = bool(flags.get("W_212Dn", False))

        # grade
        g = _grade_setup(
            bias=bias,
            flags=full,              # contains M/W Bull/Bear keys
            entry=entry,
            stop=stop,
            rr=rr,
            room=room,
            atrp=atrp,
            compression=compression
        )
        full["Grade"] = g

        rows.append(full)

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        st.info("No data returned for this sector list right now.")
        st.caption("Note: Trigger levels are bias-aware. LONG = break IB high / stop below. SHORT = break IB low / stop above.")
        return

    # For on-screen table: convert bool flags to checkmarks
    table_df = scan_df.copy()
    table_df = _checkify(table_df, ["D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"])

    # Order + display
    show = table_df[
        [
            "Ticker", "Grade",
            "D_Type", "W_Type", "M_Type",
            "W_Inside", "D_Inside",
            "TriggerTF", "Entry", "Stop",
            "Room", "RR", "ATR%", "Compression",
            "TriggerStatus",
        ]
    ].copy()

    # Sort best first: Grade A then B then C, then higher RR/Room
    grade_rank = {"A": 0, "B": 1, "C": 2}
    show["_gr"] = show["Grade"].map(lambda x: grade_rank.get(str(x), 9))
    show["_rr"] = pd.to_numeric(show["RR"], errors="coerce")
    show["_room"] = pd.to_numeric(show["Room"], errors="coerce")
    show = show.sort_values(["_gr", "_rr", "_room"], ascending=[True, False, False]).drop(columns=["_gr","_rr","_room"])

    styled = show.style.applymap(_grade_style, subset=["Grade"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # =========================
    # WATCHLIST + WRITEUPS
    # =========================
    st.subheader("✅ Watchlist (top setups) + Writeups")

    wl_n = st.slider("Watchlist size", 5, 20, 10)

    # Watchlist definition:
    # - Prefer A then B
    # - Prefer names with actual triggers (Entry/Stop present)
    tmp = scan_df.copy()
    tmp["HasTrigger"] = tmp["Entry"].notna() & tmp["Stop"].notna()

    tmp["GradeRank"] = tmp["Grade"].map(lambda x: {"A": 0, "B": 1, "C": 2}.get(str(x), 9))
    tmp["RR_num"] = pd.to_numeric(tmp["RR"], errors="coerce").fillna(-1)
    tmp["Room_num"] = pd.to_numeric(tmp["Room"], errors="coerce").fillna(-1)

    watch = tmp.sort_values(["GradeRank", "HasTrigger", "RR_num", "Room_num"], ascending=[True, False, False, False]).head(int(wl_n))

    if watch.empty:
        st.warning("No watchlist under current scan settings (try increasing Scan count).")
    else:
        wl_table = watch[["Ticker","Grade","TriggerTF","Entry","Stop","Room","RR","ATR%","Compression","TriggerStatus"]].copy()
        wl_table = wl_table.sort_values(["Grade"], ascending=True)
        st.dataframe(wl_table.style.applymap(_grade_style, subset=["Grade"]), use_container_width=True, hide_index=True)

        st.markdown("### 📌 Writeups (click to expand)")
        # Use the already-computed rows (fast). Convert to dict records.
        for rec in watch.to_dict("records"):
            title = f"{rec['Ticker']} | Grade {rec['Grade']} | {rec.get('TriggerStatus','')}"
            with st.expander(title):
                # rec already contains flags + types + magnitude
                _writeup_card(rec, bias)

    st.caption("Note: Trigger levels are bias-aware. LONG = break IB high / stop below. SHORT = break IB low / stop above.")


# Backwards compatibility for app.py routing
def scanner_main():
    show()
