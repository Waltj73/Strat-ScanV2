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


def _strength_trend_label(delta: int) -> str:
    if delta >= 5:
        return "RISING ✅"
    if delta <= -5:
        return "FALLING ⚠️"
    return "FLAT"


def _strength_badge(bias: str) -> str:
    if bias == "LONG":
        return "🟢"
    if bias == "SHORT":
        return "🔴"
    return "🟠"


def _strength_explain(strength: int, bias: str) -> str:
    if bias == "LONG":
        return "Higher = stronger LONG environment"
    if bias == "SHORT":
        return "Higher = stronger SHORT environment"
    return "Higher = less mixed; lower = chop"


def _rotation_lists(sectors_df: pd.DataFrame, bias: str, n: int = 3):
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
    if d is None or d.empty or len(d) < 80:
        return None, None, None, None

    close = float(d["Close"].iloc[-1])
    atr = _atr14(d)
    if atr is None or close <= 0:
        return None, None, None, None

    atrp = (atr / close) * 100.0
    last_rng = float(d["High"].iloc[-1] - d["Low"].iloc[-1])
    compression = (last_rng / atr) if atr > 0 else None

    if entry is None or stop is None:
        return None, float(atrp), None, (None if compression is None else float(compression))

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
    has_trigger = (entry is not None and stop is not None)
    aligned = _alignment_ok(bias, flags)

    rr_ok_a = (rr is not None and rr >= 2.0)
    rr_ok_b = (rr is not None and rr >= 1.5)
    room_ok = (room is not None and room >= 2.0)
    tight = (compression is not None and compression <= 0.9)
    atr_ok = (atrp is not None and atrp >= 1.0)

    if aligned and has_trigger and atr_ok and (rr_ok_a or room_ok) and tight:
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


# -------------------------
# NEW: Universal STRAT strength snapshot for any symbol
# -------------------------
def _symbol_strength(sym: str, drop_last_bars: int = 0) -> Tuple[int, int, int, dict, str, str, str]:
    """
    Returns:
      bull_score, bear_score, strength(0-100), flags, D_type, W_type, M_type
    Strength is derived from bull-bear diff using same mapping as market_bias.
    """
    d = get_hist(sym)
    if d is None or d.empty or len(d) < 10:
        empty_flags = {
            "D_Bull": False, "W_Bull": False, "M_Bull": False,
            "D_Bear": False, "W_Bear": False, "M_Bear": False,
            "D_Inside": False, "W_Inside": False, "M_Inside": False,
            "D_212Up": False, "W_212Up": False, "D_212Dn": False, "W_212Dn": False,
        }
        return 0, 0, 50, empty_flags, "n/a", "n/a", "n/a"

    df = d.copy()
    if drop_last_bars > 0 and len(df) > drop_last_bars:
        df = df.iloc[:-drop_last_bars].copy()

    if df.empty or len(df) < 10:
        empty_flags = {
            "D_Bull": False, "W_Bull": False, "M_Bull": False,
            "D_Bear": False, "W_Bear": False, "M_Bear": False,
            "D_Inside": False, "W_Inside": False, "M_Inside": False,
            "D_212Up": False, "W_212Up": False, "D_212Dn": False, "W_212Dn": False,
        }
        return 0, 0, 50, empty_flags, "n/a", "n/a", "n/a"

    d_tf = df
    w_tf = resample_ohlc(df, "W-FRI")
    m_tf = resample_ohlc(df, "M")

    d_type = candle_type_label(d_tf)
    w_type = candle_type_label(w_tf)
    m_type = candle_type_label(m_tf)

    flags = compute_flags(d_tf, w_tf, m_tf)
    bull, bear = score_regime(flags)

    diff = int(bull - bear)
    strength = int(np.clip(50 + diff * 5, 0, 100))
    return int(bull), int(bear), int(strength), flags, d_type, w_type, m_type


def _writeup_card(row: dict, bias: str):
    t = row["Ticker"]
    grade = row.get("Grade", "—")
    trig = row.get("TriggerStatus", "")
    tf = row.get("TriggerTF", "—")

    st.markdown(f"#### {t} — Grade **{grade}** | Bias **{bias}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Types:** D {row.get('D_Type','—')} | W {row.get('W_Type','—')} | M {row.get('M_Type','—')}")
    with c2:
        st.write(f"**Trigger:** {trig}")
        st.write(f"**TF:** {tf}")
    with c3:
        st.write(f"**Entry:** {row.get('Entry', '—')}")
        st.write(f"**Stop:** {row.get('Stop', '—')}")
    with c4:
        st.write(f"**Room (to 63d):** {row.get('Room', '—')}")
        st.write(f"**RR:** {row.get('RR', '—')}")

    st.write(
        f"**STRAT Strength:** {row.get('Strength', '—')}/100 "
        f"({row.get('StrengthTrend','—')} | Δ {row.get('StrengthDelta','—')})"
    )

    notes = []
    if bias in ("LONG", "SHORT"):
        notes.append("Aligned ✅" if _alignment_ok(bias, row) else "Alignment weak ⚠️")

    if row.get("W_Inside") == "✅":
        notes.append("Weekly inside = best trigger quality ✅")
    elif row.get("D_Inside") == "✅":
        notes.append("Daily inside = valid trigger ✅")
    else:
        notes.append("No inside bar trigger → WAIT")

    comp = row.get("Compression", None)
    if comp is not None and isinstance(comp, (int, float)):
        if comp <= 0.6:
            notes.append("Very tight compression ✅")
        elif comp <= 0.9:
            notes.append("Tight compression ✅")
        else:
            notes.append("Loose bar ⚠️")

    atrp = row.get("ATR%", None)
    if atrp is not None and isinstance(atrp, (int, float)):
        notes.append(f"ATR% ~ {atrp:.2f}%")

    st.write("**Notes:** " + " | ".join(notes))


# -------------------------
# Market snapshot (unchanged idea, reusing market_bias_and_strength)
# -------------------------
def _market_snapshot(drop_last_bars: int = 0) -> Tuple[str, int, int, pd.DataFrame]:
    market_rows = []
    for name, sym in MARKET_ETFS.items():
        bull, bear, strength, flags, d_type, w_type, m_type = _symbol_strength(sym, drop_last_bars=drop_last_bars)
        market_rows.append({
            "Market": name,
            "Ticker": sym,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "BullScore": bull,
            "BearScore": bear,
            "D_Inside": bool(flags.get("D_Inside", False)),
            "W_Inside": bool(flags.get("W_Inside", False)),
            "M_Inside": bool(flags.get("M_Inside", False)),
        })

    bias, strength_out, diff = market_bias_and_strength(market_rows)
    market_df = pd.DataFrame(market_rows)
    market_df = _checkify(market_df, ["D_Inside", "W_Inside", "M_Inside"])
    return bias, int(strength_out), int(diff), market_df


# -------------------------
# MAIN PAGE
# -------------------------
def show():
    st.title("Scanner (STRAT-only)")
    st.caption("Market regime → sector dominance → drilldown. Adds Strength meter + trend at Market, Sector, and Ticker levels.")

    topbar = st.columns([1, 1, 6])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # =========================
    # MARKET REGIME
    # =========================
    st.subheader("Market Regime (STRAT-only) — SPY / QQQ / IWM / DIA")

    bias_now, strength_now, diff_now, market_df = _market_snapshot(drop_last_bars=0)
    bias_prev, strength_prev, diff_prev, _ = _market_snapshot(drop_last_bars=1)

    strength_delta = int(strength_now - strength_prev)
    diff_delta = int(diff_now - diff_prev)

    st.dataframe(market_df, use_container_width=True, hide_index=True)

    badge = _strength_badge(bias_now)
    trend_txt = _strength_trend_label(strength_delta)

    m1, m2, m3, m4 = st.columns([1.2, 1.2, 1.2, 2.4])
    with m1:
        st.metric("Bias", f"{badge} {bias_now}", None)
    with m2:
        st.metric("STRAT Strength (0–100)", strength_now, f"{strength_delta:+d}")
    with m3:
        st.metric("Bull–Bear Diff", diff_now, f"{diff_delta:+d}")
    with m4:
        st.write(f"**Strength trend:** {trend_txt}")
        st.caption(_strength_explain(strength_now, bias_now))

    st.progress(max(0, min(100, strength_now)) / 100.0)

    if bias_now == "LONG":
        st.success(f"Bias: **LONG** 🟢 | Strength: **{strength_now}/100** | Diff: **{diff_now}** | Trend: **{trend_txt}**")
    elif bias_now == "SHORT":
        st.error(f"Bias: **SHORT** 🔴 | Strength: **{strength_now}/100** | Diff: **{diff_now}** | Trend: **{trend_txt}**")
    else:
        st.warning(f"Bias: **MIXED** 🟠 | Strength: **{strength_now}/100** | Diff: **{diff_now}** | Trend: **{trend_txt}**")

    bias = bias_now

    # =========================
    # SECTOR ROTATION (NOW WITH STRENGTH + TREND)
    # =========================
    st.subheader("Sector Rotation (STRAT-only) — ranked by bias + Strength Trend")

    sector_rows = []
    for sector, etf in SECTOR_ETFS.items():
        bull, bear, strength, flags, d_type, w_type, m_type = _symbol_strength(etf, drop_last_bars=0)
        bull_p, bear_p, strength_p, _, _, _, _ = _symbol_strength(etf, drop_last_bars=1)

        diff = int(bull - bear)
        delta = int(strength - strength_p)
        s_trend = _strength_trend_label(delta)

        sector_rows.append({
            "Sector": sector,
            "ETF": etf,
            "D_Type": d_type,
            "W_Type": w_type,
            "M_Type": m_type,
            "BullScore": bull,
            "BearScore": bear,
            "Diff": diff,
            "Strength": strength,
            "StrengthDelta": delta,
            "StrengthTrend": s_trend,
            "D_Inside": bool(flags.get("D_Inside", False)),
            "W_Inside": bool(flags.get("W_Inside", False)),
            "M_Inside": bool(flags.get("M_Inside", False)),
            "D_212Up": bool(flags.get("D_212Up", False)),
            "W_212Up": bool(flags.get("W_212Up", False)),
            "D_212Dn": bool(flags.get("D_212Dn", False)),
            "W_212Dn": bool(flags.get("W_212Dn", False)),
        })

    sectors_df = pd.DataFrame(sector_rows)

    if bias == "LONG":
        sectors_df = sectors_df.sort_values(["Diff", "Strength", "StrengthDelta"], ascending=[False, False, False])
    elif bias == "SHORT":
        sectors_df = sectors_df.sort_values(["Diff", "Strength", "StrengthDelta"], ascending=[True, False, False])
    else:
        sectors_df = sectors_df.sort_values(["Strength", "StrengthDelta"], ascending=[False, False])

    out_df = sectors_df[
        [
            "Sector", "ETF", "D_Type", "W_Type", "M_Type",
            "BullScore", "BearScore", "Diff",
            "Strength", "StrengthDelta", "StrengthTrend",
            "D_Inside", "W_Inside", "M_Inside",
            "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"
        ]
    ].copy()

    out_df = _checkify(out_df, ["D_Inside", "W_Inside", "M_Inside", "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"])
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    # =========================
    # ROTATION IN/OUT (NOW ALSO SHOWS STRENGTH + TREND)
    # =========================
    st.subheader("Quick Market Read + Rotation IN/OUT (STRAT-only)")

    rot_in, rot_out = _rotation_lists(sectors_df, bias, n=3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔁 Rotation IN")
        for _, r in rot_in.iterrows():
            st.write(
                f"✅ **{r['Sector']}** ({r['ETF']}) — "
                f"Diff {int(r['Diff'])} | Strength {int(r['Strength'])}/100 ({r['StrengthTrend']} Δ {int(r['StrengthDelta'])})"
            )

    with c2:
        st.markdown("### 🔁 Rotation OUT")
        for _, r in rot_out.iterrows():
            st.write(
                f"❌ **{r['Sector']}** ({r['ETF']}) — "
                f"Diff {int(r['Diff'])} | Strength {int(r['Strength'])}/100 ({r['StrengthTrend']} Δ {int(r['StrengthDelta'])})"
            )

    # =========================
    # DRILLDOWN (TICKERS NOW GET STRENGTH + TREND TOO)
    # =========================
    st.subheader("Drill into a Sector (tickers inside that group) — includes Strength + Magnitude + Grades")

    left, right = st.columns([2, 1])
    with left:
        sector_choice = st.selectbox("Choose sector:", options=list(SECTOR_TICKERS.keys()), index=0)
    with right:
        scan_n = st.slider(
            "Scan count",
            1,
            max(1, len(SECTOR_TICKERS.get(sector_choice, []))),
            value=min(15, len(SECTOR_TICKERS.get(sector_choice, []))),
        )

    tickers = SECTOR_TICKERS.get(sector_choice, [])
    st.write(f"Selected: **{sector_choice}** — tickers: **{len(tickers)}**")
    scan_list = tickers[:scan_n]

    rows = []
    for t in scan_list:
        d = get_hist(t)
        if d is None or d.empty:
            continue

        # strength + trend for this ticker
        bull, bear, strength, flags, d_type, w_type, m_type = _symbol_strength(t, drop_last_bars=0)
        bull_p, bear_p, strength_p, _, _, _, _ = _symbol_strength(t, drop_last_bars=1)
        s_delta = int(strength - strength_p)
        s_trend = _strength_trend_label(s_delta)

        d_tf = d
        w_tf = resample_ohlc(d, "W-FRI")

        tf, entry, stop = best_trigger(bias, d_tf, w_tf)

        rr, atrp, room, compression = _magnitude_to_target(
            bias if bias in ("LONG", "SHORT") else "LONG",
            d_tf,
            entry,
            stop,
        )

        d_ready = bool(flags.get("D_Inside") and tf == "D" and entry is not None and stop is not None)
        w_ready = bool(flags.get("W_Inside") and tf == "W" and entry is not None and stop is not None)
        m_inside = bool(flags.get("M_Inside"))
        trigger_status = f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"

        rec = {}
        rec.update(flags)
        rec["Ticker"] = t
        rec["D_Type"] = d_type
        rec["W_Type"] = w_type
        rec["M_Type"] = m_type

        rec["BullScore"] = bull
        rec["BearScore"] = bear
        rec["Diff"] = int(bull - bear)

        rec["Strength"] = strength
        rec["StrengthDelta"] = s_delta
        rec["StrengthTrend"] = s_trend

        rec["TriggerTF"] = tf if tf else "—"
        rec["Entry"] = None if entry is None else round(float(entry), 2)
        rec["Stop"] = None if stop is None else round(float(stop), 2)
        rec["Room"] = None if room is None else round(float(room), 2)
        rec["RR"] = None if rr is None else round(float(rr), 2)
        rec["ATR%"] = None if atrp is None else round(float(atrp), 2)
        rec["Compression"] = None if compression is None else round(float(compression), 2)
        rec["TriggerStatus"] = trigger_status

        rec["Grade"] = _grade_setup(
            bias=bias,
            flags=rec,
            entry=entry,
            stop=stop,
            rr=rr,
            room=room,
            atrp=atrp,
            compression=compression,
        )

        rows.append(rec)

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        st.info("No data returned for this sector list right now.")
        st.caption("Note: Trigger levels are bias-aware. LONG = break IB high / stop below. SHORT = break IB low / stop above.")
        return

    table_df = scan_df.copy()
    table_df = _checkify(table_df, ["D_Inside", "W_Inside", "M_Inside", "D_212Up", "W_212Up", "D_212Dn", "W_212Dn"])

    show_tbl = table_df[
        [
            "Ticker", "Grade",
            "Strength", "StrengthDelta", "StrengthTrend",
            "D_Type", "W_Type", "M_Type",
            "W_Inside", "D_Inside",
            "TriggerTF", "Entry", "Stop",
            "Room", "RR", "ATR%", "Compression",
            "TriggerStatus",
        ]
    ].copy()

    grade_rank = {"A": 0, "B": 1, "C": 2}
    show_tbl["_gr"] = show_tbl["Grade"].map(lambda x: grade_rank.get(str(x), 9))
    show_tbl["_rr"] = pd.to_numeric(show_tbl["RR"], errors="coerce")
    show_tbl["_room"] = pd.to_numeric(show_tbl["Room"], errors="coerce")
    show_tbl["_str"] = pd.to_numeric(show_tbl["Strength"], errors="coerce")
    show_tbl = (
        show_tbl.sort_values(["_gr", "_str", "_rr", "_room"], ascending=[True, False, False, False])
        .drop(columns=["_gr", "_rr", "_room", "_str"])
    )

    st.dataframe(show_tbl.style.applymap(_grade_style, subset=["Grade"]), use_container_width=True, hide_index=True)

    # =========================
    # WATCHLIST + WRITEUPS
    # =========================
    st.subheader("✅ Watchlist (top setups) + Writeups")

    wl_n = st.slider("Watchlist size", 5, 20, 10)

    tmp = scan_df.copy()
    tmp["HasTrigger"] = tmp["Entry"].notna() & tmp["Stop"].notna()
    tmp["GradeRank"] = tmp["Grade"].map(lambda x: {"A": 0, "B": 1, "C": 2}.get(str(x), 9))
    tmp["RR_num"] = pd.to_numeric(tmp["RR"], errors="coerce").fillna(-1)
    tmp["Room_num"] = pd.to_numeric(tmp["Room"], errors="coerce").fillna(-1)
    tmp["Str_num"] = pd.to_numeric(tmp["Strength"], errors="coerce").fillna(-1)

    watch = tmp.sort_values(
        ["GradeRank", "HasTrigger", "Str_num", "RR_num", "Room_num"],
        ascending=[True, False, False, False, False]
    ).head(int(wl_n))

    if watch.empty:
        st.warning("No watchlist under current scan settings (try increasing Scan count).")
    else:
        wl_table = watch[
            ["Ticker", "Grade", "Strength", "StrengthDelta", "StrengthTrend",
             "TriggerTF", "Entry", "Stop", "Room", "RR", "ATR%", "Compression", "TriggerStatus"]
        ].copy()
        st.dataframe(wl_table.style.applymap(_grade_style, subset=["Grade"]), use_container_width=True, hide_index=True)

        st.markdown("### 📌 Writeups (click to expand)")
        for rec in watch.to_dict("records"):
            title = f"{rec['Ticker']} | Grade {rec['Grade']} | Strength {rec['Strength']}/100 ({rec['StrengthTrend']} Δ {rec['StrengthDelta']}) | {rec.get('TriggerStatus','')}"
            with st.expander(title):
                _writeup_card(rec, bias)

    st.caption("Note: Trigger levels are bias-aware. LONG = break IB high / stop below. SHORT = break IB low / stop above.")


# Backwards compatibility for app.py routing
def scanner_main():
    show()
