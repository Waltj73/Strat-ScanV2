# app_pages/dashboard.py

from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from data.fetch import get_hist, resample_ohlc
from config.universe import MARKET_ETFS, SECTOR_ETFS
from strat.core import (
    candle_type_label,
    compute_flags,
    score_regime,
    market_bias_and_strength,
)

def dashboard_main():
    st.title("Dashboard (STRAT-only)")
    st.caption("Top-down view: Market Regime → Sector Rotation → Rotation IN/OUT")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

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
            rot_in = df.sort_values(["Diff","BullScore"], ascending=[False, False]).head(n)
            rot_out = df.sort_values(["Diff","BearScore"], ascending=[True, False]).head(n)
        elif bias == "SHORT":
            rot_in = df.sort_values(["Diff","BearScore"], ascending=[True, False]).head(n)
            rot_out = df.sort_values(["Diff","BullScore"], ascending=[False, False]).head(n)
        else:
            rot_in = df.sort_values("Diff", ascending=False).head(n)
            rot_out = df.sort_values("Diff", ascending=True).head(n)

        return rot_in, rot_out

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
    st.dataframe(_checkify(market_df, ["D_Inside","W_Inside","M_Inside"]), use_container_width=True, hide_index=True)

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
        sectors_df = sectors_df.sort_values(["Diff","BullScore"], ascending=[False, False])
    elif bias == "SHORT":
        sectors_df = sectors_df.sort_values(["Diff","BearScore"], ascending=[True, False])
    else:
        sectors_df = sectors_df.sort_values(["Diff"], ascending=False)

    show_cols = [
        "Sector","ETF","D_Type","W_Type","M_Type",
        "BullScore","BearScore","Diff",
        "D_Inside","W_Inside","M_Inside",
        "D_212Up","W_212Up","D_212Dn","W_212Dn"
    ]
    st.dataframe(
        _checkify(sectors_df[show_cols].copy(), ["D_Inside","W_Inside","M_Inside","D_212Up","W_212Up","D_212Dn","W_212Dn"]),
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # ROTATION IN/OUT
    # =========================
    st.subheader("Rotation IN/OUT (STRAT-only)")

    rot_in, rot_out = _rotation_lists(sectors_df, bias, n=5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔁 Rotation IN")
        for _, r in rot_in.iterrows():
            st.write(f"✅ **{r['Sector']}** ({r['ETF']}) — Bull {int(r['BullScore'])} / Bear {int(r['BearScore'])} | Diff {int(r['Diff'])}")
    with c2:
        st.markdown("### 🔁 Rotation OUT")
        for _, r in rot_out.iterrows():
            st.write(f"❌ **{r['Sector']}** ({r['ETF']}) — Bull {int(r['BullScore'])} / Bear {int(r['BearScore'])} | Diff {int(r['Diff'])}")
