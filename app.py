# app.py
# STRAT Dashboard Scanner
# Thesis: Monthly 2-Down (green) context + Daily entry trigger (Failed 2 Down OR Inside Day)
# Universe: Top 50 mega caps (from S&P 500 + Nasdaq-100), or other universe options
#
# Run:
#   pip install streamlit yfinance pandas numpy lxml html5lib
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="STRAT Scanner Dashboard", layout="wide")

# -----------------------------
# Universe builders
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24)
def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    # Yahoo uses "-" for class shares
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))


@st.cache_data(ttl=60 * 60 * 24)
def get_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)

    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
            best = t
            break
        # some tables have multiindex columns; handle that too
        if isinstance(t.columns, pd.MultiIndex):
            flat_cols = [" ".join(map(str, c)).lower() for c in t.columns]
            if any("ticker" in c for c in flat_cols) or any("symbol" in c for c in flat_cols):
                best = t
                break

    if best is None:
        raise ValueError("Could not find Nasdaq-100 constituents table.")

    # flatten columns if needed
    if isinstance(best.columns, pd.MultiIndex):
        best.columns = [" ".join(map(str, c)).strip() for c in best.columns]

    ticker_col = None
    for c in best.columns:
        cl = str(c).lower()
        if "ticker" in cl or "symbol" in cl:
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("Could not locate a ticker/symbol column on Nasdaq-100 table.")

    tickers = best[ticker_col].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))


@st.cache_data(ttl=60 * 60 * 12)
def get_top50_by_marketcap(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", {}) or {}
            mcap = fi.get("marketCap", None)

            if mcap is None:
                # fallback (slower)
                info = tk.info or {}
                mcap = info.get("marketCap", None)

            if mcap is None:
                continue

            rows.append({"Ticker": t, "MarketCap": float(mcap)})
        except Exception:
            continue

    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("MarketCap", ascending=False).head(50).reset_index(drop=True)
    return df


# -----------------------------
# Data fetch (batch)
# -----------------------------
@st.cache_data(ttl=60 * 15)
def fetch_batch_daily(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """
    Batch download daily OHLCV for many tickers.
    Returns dict[ticker] = DataFrame(Open,High,Low,Close,Volume) with DatetimeIndex.
    """
    if not tickers:
        return {}

    joined = " ".join(tickers)
    data = yf.download(
        joined,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}

    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df = data[t].copy().dropna()
                df = df.rename(columns=str.title)
                # yfinance sometimes returns "Adj Close"; ignore it
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].dropna()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                if len(df) > 5:
                    out[t] = df
    else:
        # single ticker case
        df = data.copy().dropna()
        df = df.rename(columns=str.title)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        out[tickers[0]] = df

    return out


# -----------------------------
# STRAT logic
# -----------------------------
def to_monthly_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily OHLCV to monthly OHLCV (calendar month).
    Uses month-start index (MS).
    """
    d = daily.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d = d.sort_index()

    monthly = pd.DataFrame(
        {
            "Open": d["Open"].resample("MS").first(),
            "High": d["High"].resample("MS").max(),
            "Low": d["Low"].resample("MS").min(),
            "Close": d["Close"].resample("MS").last(),
            "Volume": d["Volume"].resample("MS").sum() if "Volume" in d.columns else np.nan,
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])

    return monthly


def strat_month_2down_green(monthly: pd.DataFrame, use_last_completed_month: bool = True) -> dict:
    """
    Monthly condition:
      2 Down: low < low[1] AND high <= high[1]
      Green: close > open
    Target = evaluated month HIGH (the month you're using as "prior month high")
    """
    if monthly is None or monthly.empty or len(monthly) < 3:
        return {"ok": False}

    now = pd.Timestamp(datetime.now(timezone.utc).date())
    last_idx = monthly.index[-1]
    in_progress = (last_idx.year == now.year) and (last_idx.month == now.month)

    if use_last_completed_month:
        eval_pos = -2 if in_progress else -1
    else:
        eval_pos = -1

    # need eval and prior month
    if len(monthly) < (abs(eval_pos) + 2):
        return {"ok": False}

    m = monthly.iloc[eval_pos]
    m_prev = monthly.iloc[eval_pos - 1]

    month_2down = (m["Low"] < m_prev["Low"]) and (m["High"] <= m_prev["High"])
    month_green = (m["Close"] > m["Open"])
    month_condition = bool(month_2down and month_green)

    return {
        "ok": True,
        "month_condition": month_condition,
        "month_2down": bool(month_2down),
        "month_green": bool(month_green),
        "eval_month_start": monthly.index[eval_pos],
        "target_prior_month_high": float(m["High"]),
        "eval_month_open": float(m["Open"]),
        "eval_month_close": float(m["Close"]),
    }


def strat_daily_triggers(daily: pd.DataFrame) -> dict:
    """
    Daily triggers on the latest daily bar in the dataset:
      Failed 2 Down: low < low[1] AND high > high[1]  (often a 3 bar)
      Inside Day: high < high[1] AND low > low[1]
    """
    if daily is None or daily.empty or len(daily) < 3:
        return {"ok": False}

    d = daily.iloc[-1]
    d_prev = daily.iloc[-2]

    failed2_down = (d["Low"] < d_prev["Low"]) and (d["High"] > d_prev["High"])
    inside_day = (d["High"] < d_prev["High"]) and (d["Low"] > d_prev["Low"])
    failed2_down_green = bool(failed2_down and (d["Close"] > d["Open"]))

    return {
        "ok": True,
        "bar_date": daily.index[-1],
        "last_close": float(d["Close"]),
        "last_open": float(d["Open"]),
        "failed2_down": bool(failed2_down),
        "failed2_down_green": bool(failed2_down_green),
        "inside_day": bool(inside_day),
        "prev_high": float(d_prev["High"]),
        "prev_low": float(d_prev["Low"]),
        "today_high": float(d["High"]),
        "today_low": float(d["Low"]),
    }


def scan_one(
    ticker: str,
    daily: pd.DataFrame,
    use_last_completed_month: bool,
    require_failed2_green: bool,
    min_room_pct: float,
) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Match": False, "Status": "No/insufficient data"}

    monthly = to_monthly_ohlc(daily)

    m = strat_month_2down_green(monthly, use_last_completed_month=use_last_completed_month)
    d = strat_daily_triggers(daily)

    if not m.get("ok") or not d.get("ok"):
        return {"Ticker": ticker, "Match": False, "Status": "Insufficient candles"}

    month_ok = m["month_condition"]

    failed2_ok = d["failed2_down_green"] if require_failed2_green else d["failed2_down"]
    inside_ok = d["inside_day"]
    daily_ok = bool(failed2_ok or inside_ok)

    last_close = d["last_close"]
    target = m["target_prior_month_high"]
    room_pct = (target - last_close) / last_close * 100.0

    match = bool(month_ok and daily_ok and (room_pct >= min_room_pct))

    trigger = "FAILED2D" if failed2_ok else ("INSIDE" if inside_ok else "—")

    return {
        "Ticker": ticker,
        "Match": match,
        "Trigger": trigger,
        "Monthly_2Down": m["month_2down"],
        "Monthly_Green": m["month_green"],
        "Eval_Month": m["eval_month_start"].strftime("%Y-%m"),
        "Daily_Failed2Down": d["failed2_down"],
        "Daily_Failed2Down_Green": d["failed2_down_green"],
        "Daily_InsideDay": d["inside_day"],
        "Last_Bar": pd.to_datetime(d["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PriorMonthHigh": round(target, 2),
        "Room_To_Target_%": round(room_pct, 2),
        "Status": "OK" if match else "No match",
    }


# -----------------------------
# UI
# -----------------------------
st.title("STRAT Scanner Dashboard")
st.caption("Monthly context: **2-Down month that closes green** • Daily entry: **Failed 2 Down OR Inside Day** • Target: **prior month high**")

with st.sidebar:
    st.header("Universe")

    universe = st.selectbox(
        "Select universe",
        ["Top 50 (Mega Caps: S&P500 + Nasdaq-100)", "S&P 500", "Nasdaq-100", "Custom (paste)"],
        index=0,
    )

    period = st.selectbox("History period (yfinance)", ["6mo", "1y", "2y", "5y"], index=2)

    st.header("Rules")
    use_last_completed_month = st.checkbox("Use last COMPLETED month (stable)", value=True)
    require_failed2_green = st.checkbox("Require Failed 2 Down to close green", value=False)
    min_room_pct = st.number_input("Min room to prior month high (%)", value=0.0, step=0.5)

    show_top50_table = st.checkbox("Show Top 50 list table", value=False)

    run = st.button("Run Scan", type="primary")

# Build tickers universe
tickers: list[str] = []
top50_df = None

try:
    if universe.startswith("Top 50"):
        sp = get_sp500_tickers()
        ndx = get_nasdaq100_tickers()
        combined = sorted(set(sp + ndx))
        top50_df = get_top50_by_marketcap(combined)
        tickers = top50_df["Ticker"].tolist()

    elif universe == "S&P 500":
        tickers = get_sp500_tickers()

    elif universe == "Nasdaq-100":
        tickers = get_nasdaq100_tickers()

    else:
        tickers_text = st.text_area(
            "Tickers (comma or newline separated)",
            value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, PLTR",
            height=120,
        )
        tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()]
except Exception as e:
    st.error(f"Universe load error: {e}")
    tickers = []

if universe.startswith("Top 50") and top50_df is not None and show_top50_table:
    st.subheader("Top 50 Mega Caps (by market cap)")
    st.dataframe(top50_df, use_container_width=True, hide_index=True)

if run:
    if not tickers:
        st.warning("No tickers loaded.")
        st.stop()

    st.info(f"Downloading data for {len(tickers)} tickers… (batch)")
    batch = fetch_batch_daily(tickers, period=period)

    results = []
    progress = st.progress(0)
    for i, t in enumerate(tickers, start=1):
        df = batch.get(t, pd.DataFrame())
        results.append(
            scan_one(
                ticker=t,
                daily=df,
                use_last_completed_month=use_last_completed_month,
                require_failed2_green=require_failed2_green,
                min_room_pct=float(min_room_pct),
            )
        )
        progress.progress(i / max(len(tickers), 1))

    res = pd.DataFrame(results)

    # Sort: matches first, then biggest room to target
    if "Room_To_Target_%" in res.columns:
        res = res.sort_values(["Match", "Room_To_Target_%"], ascending=[False, False]).reset_index(drop=True)

    st.subheader("Scan Results")
    st.dataframe(res, use_container_width=True, hide_index=True)

    matches = res[res["Match"] == True]
    st.write(f"Matches: **{len(matches)}** / {len(res)}")

    st.divider()
    st.subheader("Chart Viewer")

    pick = st.selectbox("Select ticker", res["Ticker"].tolist(), index=0)
    ddf = batch.get(pick, pd.DataFrame())

    if ddf is None or ddf.empty:
        st.warning("No chart data for that ticker.")
        st.stop()

    # show close chart
    st.line_chart(ddf["Close"])

    # compute & show key levels and trigger details
    mm = to_monthly_ohlc(ddf)
    m_info = strat_month_2down_green(mm, use_last_completed_month=use_last_completed_month)
    d_info = strat_daily_triggers(ddf)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Monthly Context**")
        if m_info.get("ok"):
            st.write(f"Evaluated Month: **{m_info['eval_month_start'].strftime('%Y-%m')}**")
            st.write(f"2 Down: **{m_info['month_2down']}**")
            st.write(f"Green: **{m_info['month_green']}**")
        else:
            st.write("Monthly: insufficient data")

    with col2:
        st.markdown("**Daily Trigger (latest bar)**")
        if d_info.get("ok"):
            st.write(f"Date: **{pd.to_datetime(d_info['bar_date']).strftime('%Y-%m-%d')}**")
            st.write(f"Failed 2 Down: **{d_info['failed2_down']}**")
            st.write(f"Failed 2 Down (green close): **{d_info['failed2_down_green']}**")
            st.write(f"Inside Day: **{d_info['inside_day']}**")
        else:
            st.write("Daily: insufficient data")

    with col3:
        st.markdown("**Target & Room**")
        if m_info.get("ok") and d_info.get("ok"):
            target = m_info["target_prior_month_high"]
            last_close = d_info["last_close"]
            room_pct = (target - last_close) / last_close * 100.0
            st.write(f"Prior Month High (target): **{target:.2f}**")
            st.write(f"Last Close: **{last_close:.2f}**")
            st.write(f"Room to target: **{room_pct:.2f}%**")
        else:
            st.write("Target: unavailable")

else:
    st.write("Set your universe + rules, then click **Run Scan**.")
