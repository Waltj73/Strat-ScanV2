# app_pages/dashboard.py
# Dashboard: STRAT-only Market Overview + A+ Market Leaders (Top 10–15)

from datetime import datetime, timezone

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
    candle_color,
)


# -------------------------
# Helpers (copied from scanner to keep grading consistent)
# -------------------------
def _tf_in_force(df: pd.DataFrame) -> str:
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


def _is_tf_supportive(force_label: str, bias: str) -> bool:
    if bias == "LONG":
        return isinstance(force_label, str) and force_label.startswith("Bull")
    if bias == "SHORT":
        return isinstance(force_label, str) and force_label.startswith("Bear")
    return False


def _setup_grade(bias: str, flags: dict, d_force: str, w_force: str, m_force: str, trigger_tf: str, entry, stop):
    if bias not in ("LONG", "SHORT"):
        return ("C", 0)

    score = 0
    trigger_ready = (trigger_tf in ("D", "W")) and (entry is not None) and (stop is not None)
    if trigger_ready:
        score += 4

    if _is_tf_supportive(w_force, bias):
        score += 2
    if _is_tf_supportive(m_force, bias):
        score += 2

    if bias == "LONG":
        if flags.get("W_212Up") or flags.get("D_212Up"):
            score += 1
    if bias == "SHORT":
        if flags.get("W_212Dn") or flags.get("D_212Dn"):
            score += 1

    if flags.get("M_Inside"):
        score += 1

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

    if score >= 8:
        return ("A+", score)
    if score >= 6:
        return ("A", score)
    if score >= 4:
        return ("B", score)
    return ("C", score)


def _style_df(df: pd.DataFrame):
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


def _build_market_universe(include_sector_tickers: bool = True, max_sector_tickers: int = 8) -> list[str]:
    """
    Universe for "overall market" scan.
    - Always includes: MARKET_ETFS + SECTOR_ETFS
    - Optionally includes a slice of tickers from each sector list
      (keeps it from being too slow on Streamlit Cloud)
    """
    tickers = set()

    # broad market ETFs
    tickers.update(MARKET_ETFS.values())

    # sector ETFs
    tickers.update(SECTOR_ETFS.values())

    if include_sector_tickers:
        for _, lst in SECTOR_TICKERS.items():
            for t in lst[:max_sector_tickers]:
                tickers.add(t)

    return sorted(tickers)


# -------------------------
# Main
# -------------------------
def dashboard_main():
    st.title("STRAT Dashboard")
    st.caption("Market regime + STRAT-only sentiment + Top A+ setups across the overall market")

    topbar = st.columns([1, 8])
    with topbar[0]:
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # -------------------------
    # Market Regime Summary
    # -------------------------
    st.subheader("Market Regime (STRAT-only) — SPY / QQQ / IWM / DIA")

    market_rows = []
    for name, sym in MARKET_ETFS.items():
        d = get_hist(sym)
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
        })

    market_df = pd.DataFrame(market_rows)
    st.dataframe(market_df, use_container_width=True, hide_index=True)

    bias, strength, diff = market_bias_and_strength(market_rows)
    if bias == "LONG":
        st.success(f"Bias: **LONG** 🟢 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
    elif bias == "SHORT":
        st.error(f"Bias: **SHORT** 🔴 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")
    else:
        st.warning(f"Bias: **MIXED** 🟠 | STRAT Strength: **{strength}/100** | Bull–Bear diff: **{diff}**")

    # -------------------------
    # A+ Market Leaders (Top 10–15)
    # -------------------------
    st.subheader("A+ Market Leaders (overall market, not sector-limited)")

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        top_n = st.slider("Top count", 5, 30, 15)
    with c2:
        include_sector_tickers = st.toggle("Include sector tickers", value=True)
    with c3:
        max_per_sector = st.slider("Max tickers per sector", 1, 25, 8)

    universe = _build_market_universe(include_sector_tickers=include_sector_tickers, max_sector_tickers=max_per_sector)

    rows = []
    for t in universe:
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
            "TriggerTF": tf if tf else "—",
            "Entry": None if entry is None else round(float(entry), 2),
            "Stop": None if stop is None else round(float(stop), 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No market leaders found (no data returned).")
        return

    grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3}
    df["_g"] = df["SetupGrade"].map(lambda x: grade_order.get(x, 9))
    df = df.sort_values(["_g", "GradeScore"], ascending=[True, False]).drop(columns=["_g"])
    df = df.head(top_n)

    st.dataframe(_style_df(df), use_container_width=True, hide_index=True)

st.caption(
    "This ranks the best STRAT-style setups across the whole market universe "
    "(market ETFs + sector ETFs + optional sector constituents)."
)

# ===== Chart Viewer =====
st.markdown("### Chart viewer")

picked = st.selectbox(
    "View chart for:",
    df["Ticker"].tolist(),
    index=0,
)

bars = get_hist(picked)

if bars is None or bars.empty:
    st.warning("No data for that ticker.")
else:
    bars = bars.tail(220).copy()

    cols = {c.lower(): c for c in bars.columns}
    o = cols.get("open", "Open")
    h = cols.get("high", "High")
    l = cols.get("low", "Low")
    c = cols.get("close", "Close")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=bars.index,
                open=bars[o],
                high=bars[h],
                low=bars[l],
                close=bars[c],
                name=picked,
            )
        ]
    )

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        title=f"{picked} — Daily Candles",
    )

    st.plotly_chart(fig, use_container_width=True)



# Back-compat
def show():
    return dashboard_main()
